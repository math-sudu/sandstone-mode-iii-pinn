"""Joint interface PINN inversion with a shared law and separate geometry networks."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.func import jacrev

from interface_pinn import ROOT, sha256, validate_input_provenance
from interface_pinn_inverse import ParameterVector, build


class CaseCoordinates:
    """Split verified module coordinates into local neural state and shared law."""
    def __init__(self, model):
        self.mapper = ParameterVector(model)
        offset = 0
        for name, _, count in self.mapper.layout:
            if name == "law.log_parameters":
                assert count == 5
                self.start = offset
                break
            offset += count
        else:
            raise ValueError("Missing five-parameter cohesive law")

    def split(self, vector):
        a = self.start
        return vector[a:a+5], torch.cat((vector[:a], vector[a+5:]), dim=-1)

    def join(self, law, local):
        a = self.start
        return torch.cat((local[:a], law, local[a:]))

    def residual(self, law, local):
        vector = self.join(law, local)
        return torch.cat((self.mapper.main(vector), self.mapper.critical(vector)[None]))

    def linearize(self, law, local, chunk):
        vector = self.join(law, local)
        residual = self.residual(law, local).detach()
        main = jacrev(self.mapper.main, chunk_size=chunk)(vector).detach()
        critical = jacrev(self.mapper.critical)(vector).detach()[None]
        jacobian = torch.cat((main, critical))
        a = self.start
        return (residual, torch.cat((jacobian[:, :a], jacobian[:, a+5:]), dim=1),
                jacobian[:, a:a+5].clone())

    def install(self, law, local):
        self.mapper.install(self.join(law, local))


def block_step(blocks, grams, damping):
    """Exact damped least-squares step after eliminating local neural coordinates.

    Each block is (r, A, B): residual, local-state Jacobian, shared-law Jacobian.
    Minimize sum ||r + A dv + B dt||^2 + damping*(sum ||dv||^2 + ||dt||^2).
    H = A A.T + damping I, so only a five-dimensional Schur system couples cases.
    """
    prototype = blocks[0][0]
    schur = torch.eye(5, device=prototype.device, dtype=prototype.dtype)
    rhs = prototype.new_zeros(5)
    eliminated = []
    for (residual, _, shared), gram in zip(blocks, grams):
        damped = gram.clone()
        damped.diagonal().add_(damping)
        factor = torch.linalg.cholesky(damped)
        solution = torch.cholesky_solve(torch.cat((residual[:, None], shared), dim=1), factor)
        hr, hb = solution[:, 0], solution[:, 1:]
        schur += shared.T @ hb
        rhs -= shared.T @ hr
        eliminated.append((hr, hb))
    law_step = torch.linalg.solve((schur + schur.T) / 2, rhs)
    local_steps = [-local.T @ (hr + hb @ law_step)
                   for (_, local, _), (hr, hb) in zip(blocks, eliminated)]
    return law_step, local_steps


def scale_columns(blocks):
    """Equilibrate local columns and the shared columns across all geometries."""
    def inverse_norm(matrix):
        norms = torch.linalg.vector_norm(matrix, dim=0)
        positive = norms[norms > 0]
        if len(positive) == 0:
            return torch.ones_like(norms)
        return norms.clamp(min=positive.median() * 0.01).reciprocal()

    shared_scale = inverse_norm(torch.cat([shared for _, _, shared in blocks]))
    local_scales = [inverse_norm(local) for _, local, _ in blocks]
    scaled = [(residual, local * local_scale, shared * shared_scale)
              for (residual, local, shared), local_scale in zip(blocks, local_scales)]
    return scaled, shared_scale, local_scales


def build_family(config, recorded_observations=None):
    models, coordinates, data, provenance, case_configs = [], [], [], [], []
    shared = None
    locals_ = []
    depths = [case["notch_depth_mm"] for case in config["cases"]]
    if len(depths) != len(set(depths)) or len(depths) < 2:
        raise ValueError("A family requires distinct geometries")
    if config.get("resume_family_checkpoint") and config.get("initial_family_checkpoint"):
        raise ValueError("Choose either unchanged-objective continuation or family initialization")
    if recorded_observations is not None and len(recorded_observations) != len(config["cases"]):
        raise ValueError("Recorded observations do not cover every family geometry")
    common = {k: v for k, v in config.items()
              if k not in ("cases", "resume_family_checkpoint", "initial_family_checkpoint")}
    for case_index, case in enumerate(config["cases"]):
        case_config = dict(common, **case)
        if case_config["beta"] != config["beta"]:
            raise ValueError("A shared law is fitted separately for each orientation")
        recorded = None if recorded_observations is None else recorded_observations[case_index]
        model, observation, source = build(case_config, recorded_observations=recorded)
        mapper = CaseCoordinates(model)
        law, local = mapper.split(mapper.mapper.pack())
        if shared is None:
            shared = law.clone()
        else:
            torch.testing.assert_close(law, shared, rtol=0, atol=0)
        models.append(model)
        coordinates.append(mapper)
        locals_.append(local)
        data.append(observation)
        provenance.append(source)
        case_configs.append(case_config)
    checkpoint_key = ("resume_family_checkpoint" if config.get("resume_family_checkpoint")
                      else "initial_family_checkpoint")
    if config.get(checkpoint_key):
        checkpoint_path = ROOT / config[checkpoint_key]
        saved = torch.load(checkpoint_path, map_location=config["device"], weights_only=True)
        if checkpoint_key == "resume_family_checkpoint":
            assert saved["data"] == data, "Observation or reference convention changed"
            if saved["config"].get("cutoff_smoothing_fraction", 0.0) != config.get("cutoff_smoothing_fraction", 0.0):
                raise ValueError("Changed cutoff law requires initial_family_checkpoint, not objective continuation")
        assert len(saved["models"]) == len(models)
        for model, source, prior, state in zip(models, provenance, saved["provenance"], saved["models"]):
            validate_input_provenance(source, prior)
            if checkpoint_key == "initial_family_checkpoint":
                # A new objective inherits parameters, not old data or loss buffers.
                with torch.no_grad():
                    for name, parameter in model.named_parameters():
                        parameter.copy_(state[name])
            else:
                missing, unexpected = model.load_state_dict(state, strict=False)
                assert not unexpected and all(key.startswith("physics.") for key in missing)
        shared = saved["shared_log_law"].to(config["device"])
        locals_ = []
        for mapper in coordinates:
            law, local = mapper.split(mapper.mapper.pack())
            torch.testing.assert_close(law, shared, rtol=0, atol=0)
            locals_.append(local)
    return models, coordinates, shared, locals_, data, provenance, case_configs


def physics_targets_met(diagnostics, config):
    return all(row["force_relative_max"] < config["force_tolerance"]
               and row["load_fraction_error_max"] < config["fraction_tolerance"]
               and row["peak_tangent_relative"] < config["critical_tolerance"] for row in diagnostics)


def save_checkpoint(path, models, shared, config, data, provenance):
    states = [{key: value for key, value in model.state_dict().items() if not key.startswith("physics.")}
              for model in models]
    torch.save({"models": states, "shared_log_law": shared, "config": config,
                "data": data, "provenance": provenance}, path)


def run(config_path):
    config = json.loads(config_path.read_text())
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(config.get("cpu_threads", 4))
    torch.manual_seed(config["seed"])
    models, coordinates, shared, locals_, data, provenance, case_configs = build_family(config)
    out = ROOT / config["output_directory"]
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    sources = ("interface_pinn_family.py", "interface_pinn_inverse.py", "interface_pinn.py")
    snapshot = dict(config, source_hashes={name: sha256(Path(__file__).with_name(name)) for name in sources})
    if config.get("resume_family_checkpoint"):
        snapshot["resume_checkpoint_sha256"] = sha256(ROOT / config["resume_family_checkpoint"])
    if config.get("initial_family_checkpoint"):
        snapshot["initial_family_checkpoint_sha256"] = sha256(ROOT / config["initial_family_checkpoint"])
    (out / "run_config.json").write_text(json.dumps(snapshot, indent=2) + "\n")
    (out / "observations.json").write_text(json.dumps(data, indent=2) + "\n")
    for name in sources:
        (out / name).write_bytes(Path(__file__).with_name(name).read_bytes())
    initial = [model.diagnostics() for model in models]
    start = time.perf_counter()
    damping = config["initial_damping"]
    stop = "Configured iteration budget reached"
    law_updates_enabled = False
    for iteration in range(config["lm_iterations"]):
        fixed_law = iteration < config.get("law_warmup_iterations", 0)
        if (iteration == config.get("law_warmup_iterations", 0) and iteration > 0
                and config.get("require_warmup_physics", False)
                and not physics_targets_met([model.diagnostics() for model in models], config)):
            stop = "Neural-state warmup did not meet physics targets; constitutive updates remain disabled"
            break
        law_updates_enabled = law_updates_enabled or not fixed_law
        blocks = [mapper.linearize(shared, local, config["jacobian_chunk"])
                  for mapper, local in zip(coordinates, locals_)]
        if fixed_law:
            blocks = [(residual, local, torch.zeros_like(law)) for residual, local, law in blocks]
        if config.get("column_scaling", False):
            blocks, shared_scale, local_scales = scale_columns(blocks)
        grams = [local @ local.T for _, local, _ in blocks]
        loss = sum(float(r @ r) for r, _, _ in blocks)
        if iteration == 0:
            damping *= max(float((gram.diagonal() + b.square().sum(1)).max())
                           for gram, (_, _, b) in zip(grams, blocks))
        row = {"iteration": iteration, "loss": loss, "damping": damping,
               "law_fixed_for_warmup": fixed_law,
               "seconds": time.perf_counter() - start,
               "cases": [model.diagnostics() for model in models]}
        with (out / "training.jsonl").open("a") as handle:
            handle.write(json.dumps(row) + "\n")
        print(json.dumps({"iteration": iteration, "loss": loss, "seconds": row["seconds"],
                          "law_fixed_for_warmup": fixed_law,
                          "maximum_force_relative": max(r["force_relative_max"] for r in row["cases"]),
                          "maximum_peak_tangent_relative": max(r["peak_tangent_relative"] for r in row["cases"]),
                          "data_sum_squares": sum(r["data_sum_squares"] for r in row["cases"])}), flush=True)
        accepted = False
        for _ in range(14):
            try:
                law_step, local_steps = block_step(blocks, grams, damping)
                if config.get("column_scaling", False):
                    law_step = shared_scale * law_step
                    local_steps = [scale * update for scale, update in zip(local_scales, local_steps)]
            except torch.linalg.LinAlgError:
                damping *= 10
                continue
            candidate_law = shared + law_step
            candidates = [local + update for local, update in zip(locals_, local_steps)]
            with torch.no_grad():
                residuals = [mapper.residual(candidate_law, local)
                             for mapper, local in zip(coordinates, candidates)]
                candidate_loss = sum(float(r @ r) for r in residuals)
            if np.isfinite(candidate_loss) and candidate_loss < loss:
                shared = candidate_law.detach()
                locals_ = [local.detach() for local in candidates]
                damping = max(damping / 3, 1e-16)
                accepted = True
                break
            damping *= 10
        del blocks, grams
        for mapper, local in zip(coordinates, locals_):
            mapper.install(shared, local)
        if (iteration + 1) % config.get("checkpoint_every", 10) == 0:
            save_checkpoint(out / "latest.pt", models, shared, snapshot, data, provenance)
        if not accepted:
            stop = "No decreasing shared-law neural-parameter step at tested damping values"
            break
    final = [model.diagnostics() for model in models]
    met = physics_targets_met(final, config)
    metrics = {"kind": "shared_law_interface_pinn_inverse" if law_updates_enabled else "interface_pinn_inverse_state_initialization",
               "status": "physics_targets_met" if met else "needs_refinement",
               "initial": initial, "final": final, "stop_reason": stop,
               "elapsed_seconds": time.perf_counter() - start, "provenance": provenance,
               "geometry_depths_mm": [case["notch_depth_mm"] for case in case_configs],
               "trainable_parameter_count": len(shared) + sum(len(local) for local in locals_),
               "shared_law_coordinate_count": 5,
               "optimized_law_coordinate_count": 5 if law_updates_enabled else 0,
               "scope": "Shared law across the listed geometries at fitted identification states; no uncertainty, uniqueness or off-grid accuracy claim."}
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    save_checkpoint(out / "checkpoint.pt", models, shared, snapshot, data, provenance)
    for model, case in zip(models, case_configs):
        with torch.no_grad():
            q, d, load, rf, gauges, _ = model.quantities()
        np.savez_compressed(out / f"fields_a{case['notch_depth_mm']}.npz",
                            control=q.cpu().numpy(), slip_mm=d.cpu().numpy(), load_kN=load.cpu().numpy(),
                            force_residual_N=rf.cpu().numpy(), gauge_um=gauges.cpu().numpy(),
                            fractions=np.array(model.data["fractions"]))
    print(json.dumps({k: v for k, v in metrics.items() if k != "provenance"}, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    run(parser.parse_args().config)
