"""Joint neural-state and cohesive-parameter fitting for one instrumented geometry."""
import argparse
import copy
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.func import functional_call, jacrev

from interface_pinn import ROOT, load_inputs, network_from_config, sha256
from baxi_pinn.dic_frames import FRAME_EXCLUSIONS_PATH, frame_is_excluded, load_frame_exclusions


def observations(config, physics):
    paths = {name: (ROOT / config[name]).resolve() for name in ("split_path", "peak_path", "gauge_path")}
    paths["frame_exclusions_path"] = FRAME_EXCLUSIONS_PATH
    exclusions = load_frame_exclusions()
    ids = set(json.loads(paths["split_path"].read_text())["identification_set"])
    beta, depth = config["beta"], config["notch_depth_mm"]
    with paths["peak_path"].open(newline="") as handle:
        peaks = [r for r in csv.DictReader(handle) if r["specimen"] in ids
                 and int(r["beta_deg"]) == beta and int(r["a_mm"]) == depth]
    with paths["gauge_path"].open(newline="") as handle:
        gauges = [r for r in csv.DictReader(handle) if r["specimen_id"] in ids
                  and int(r["beta_deg"]) == beta and int(r["a_mm"]) == depth
                  and r["level_kind"] == "measured" and float(r["load_fraction"]) <= 0.98
                  and not frame_is_excluded(r["specimen_id"], r["frame"], exclusions)]
    if len(peaks) < 2 or (not gauges and not config.get("allow_peak_only", False)):
        raise ValueError("This joint pilot needs identification-set peak replicates and valid gauges")
    fractions = sorted({float(r["load_fraction"]) for r in gauges} | {1.0})
    index = {q: i for i, q in enumerate(fractions)}
    peak_values = np.array([float(r["peak_force_kN"]) for r in peaks])
    profiles = []
    for sid in sorted({r["specimen_id"] for r in gauges}):
        rows = sorted([r for r in gauges if r["specimen_id"] == sid], key=lambda r: int(r["frame"]))
        if len(rows) < 3 or sid not in physics.gauge_keys:
            raise ValueError(f"Missing gauge levels or recorded-station operator: {sid}")
        if any(float(right["load_fraction"]) <= float(left["load_fraction"])
               for left, right in zip(rows, rows[1:])):
            raise ValueError(f"Gauge frames do not follow a strictly rising load branch: {sid}")
        profiles.append({"specimen": sid, "station_index": physics.gauge_keys.index(sid),
                         "reference_index": index[float(rows[0]["load_fraction"])],
                         "reference_frame": int(rows[0]["frame"]),
                         "frames": [int(r["frame"]) for r in rows[1:]],
                         "level_indices": [index[float(r["load_fraction"])] for r in rows[1:]],
                         "measured_um": [float(r["CSD_W_um"]) - float(rows[0]["CSD_W_um"])
                                         for r in rows[1:]],
                         "recorded_first_frame_um": float(rows[0]["CSD_W_um"]),
                         "station_x_mm": float(rows[0]["ext_xn_mm"])})
    return {"fractions": fractions, "peak_values_kN": peak_values.tolist(),
            "peak_specimens": [r["specimen"] for r in peaks],
            "sigma_peak_kN": max(float(peak_values.std(ddof=1)), 0.04 * float(peak_values.mean())),
            "sigma_gauge_um": config["sigma_gauge_um"], "profiles": profiles,
            "source_hashes": {key: sha256(path) for key, path in paths.items()},
            "reference_convention": "Measured and model increments both subtract their first retained measured state at the recorded load fraction. Raw CSV values and station rows remain unchanged; one mounting sign applies to each complete specimen profile.",
            "split": "Only identification replicates supply residual observations and peak-noise estimates. The fixed gauge scale was pooled in the preceding analysis across all DIC-valid replicates, including assessment specimens."}


class InverseModel(nn.Module):
    def __init__(self, physics, law, network, data, initial_controls, weight):
        super().__init__()
        self.physics, self.law, self.network = physics, law, network
        self.law.log_parameters.requires_grad_(True)
        q = physics.f.new_tensor(initial_controls)
        gaps = torch.diff(torch.cat((q.new_zeros(1), q)))
        if not (gaps > 0).all():
            raise ValueError("Initial neural work coordinates must be strictly ordered")
        self.log_gaps = nn.Parameter(gaps.log())
        self.data = data
        self.weight = weight
        self.register_buffer("fractions", physics.f.new_tensor(data["fractions"]))
        self.register_buffer("peak_observed", physics.f.new_tensor(data["peak_values_kN"]))
        tp, d0, _, _, _ = law.physical()
        pscale = (tp * physics.area.sum() / abs(physics.direction @ physics.f)).detach()
        self.register_buffer("load_scale", pscale)
        self.register_buffer("energy_scale", (pscale.square() * (physics.f @ physics.C @ physics.f)).sqrt())
        self.register_buffer("stiffness_scale", (tp / d0 * physics.area.mean()).detach())
        self.measurements = [physics.f.new_tensor(p["measured_um"]) for p in data["profiles"]]

    def controls(self):
        return self.log_gaps.exp().cumsum(0)

    def critical(self):
        d = self.network(self.controls()[-1:], self.physics)[0]
        tangent = self.physics.K + self.law.tangent(d, self.physics.area)[self.physics.active][:, self.physics.active]
        return torch.linalg.eigvalsh((tangent + tangent.T) / 2)[0]

    def quantities(self):
        q = self.controls()
        d = self.network(q, self.physics)
        load, rf, gauges, rd = self.physics.readout(d, self.law)
        return q, d, load, rf, gauges, rd

    def forward(self, mode="main"):
        if mode == "critical":
            return self.weight * self.critical() / self.stiffness_scale
        q, d, load, rf, gauges, _ = self.quantities()
        white = rf @ self.physics.chol_C / self.energy_scale / self.fractions[:, None]
        load_rows = (load[:-1] - self.fractions[:-1] * load[-1]) / self.load_scale / self.fractions[:-1]
        peak_rows = (load[-1] - self.peak_observed) / self.data["sigma_peak_kN"]
        gauge_rows = []
        for profile, measured in zip(self.data["profiles"], self.measurements):
            prediction = gauges[profile["level_indices"], profile["station_index"]] - gauges[profile["reference_index"], profile["station_index"]]
            plus, minus = prediction - measured, -prediction - measured
            signed = torch.where(plus.square().sum() <= minus.square().sum(), plus, minus)
            gauge_rows.append(signed / self.data["sigma_gauge_um"])
        return torch.cat((self.weight * white.flatten(), self.weight * load_rows, peak_rows, *gauge_rows))

    @torch.no_grad()
    def diagnostics(self):
        q, d, load, rf, gauge, rd = self.quantities()
        traction = self.law.force(d, self.physics.area)[:, self.physics.active]
        denom = torch.maximum((load[:, None] * self.physics.f).norm(dim=1), traction.norm(dim=1))
        relative = rf.norm(dim=1) / denom.clamp(min=1e-20)
        main = self()
        count_data = len(self.data["peak_values_kN"]) + sum(len(p["measured_um"]) for p in self.data["profiles"])
        peak_lambda = self.critical()
        physical_values = self.law.physical()
        return {"force_relative_max": float(relative.max()),
                "load_fraction_error_max": float(abs(load[:-1] / load[-1] - self.fractions[:-1]).max()) if len(load) > 1 else 0.0,
                "peak_tangent_min_N_per_mm": float(peak_lambda),
                "peak_tangent_relative": float(abs(peak_lambda) / self.stiffness_scale),
                "peak_load_kN": float(load[-1]), "data_sum_squares": float(main[-count_data:].square().sum()),
                "law_values": {key: float(value) for key, value in zip(
                    ("tau_p_MPa", "delta_0_mm", "delta_c_mm", "p_r", "p_s"), physical_values)},
                "control_delta0": q.cpu().tolist()}


@torch.no_grad()
def initialize_local_from_branch(network, physics, controls, directory):
    """Interpolate previously solved neural fields into the local feature head."""
    from interface_pinn_continuation import read_neural_branch
    rows, fields = read_neural_branch(directory)
    path_controls = np.array([row["control_delta0"] for row in rows])
    if np.any(np.diff(path_controls) <= 0) or min(controls) < path_controls[0] or max(controls) > path_controls[-1]:
        raise ValueError("Neural branch interpolation requires covered, ordered work coordinates")
    target = physics.f.new_tensor(np.stack([np.interp(controls, path_controls, column) for column in fields.T], axis=1))
    q = physics.f.new_tensor(controls)
    temporal = network.local_load_features(q)
    if temporal.shape != (len(q), len(q)):
        raise ValueError("Branch interpolation requires one local load center per fitted state")
    difference = target - network(q, physics)
    envelope = q.pow(network.envelope_power)
    scale = network.correction_scale * envelope / (1 + envelope)
    dx = difference[:, :physics.n] / scale[:, None]
    z = network.coordinates[:, 1]
    dz = difference[:, physics.n:] / scale[:, None] / torch.where(z > 0, z, torch.ones_like(z))[None, :]
    values = torch.stack((dx, dz), dim=-1)
    spatial = torch.linalg.solve(network.local_basis, values.permute(1, 0, 2).reshape(physics.n, -1))
    spatial = spatial.reshape(physics.n, len(q), 2).permute(1, 0, 2)
    coefficients = torch.linalg.solve(temporal, spatial.reshape(len(q), -1)).reshape_as(network.local_coefficients)
    network.local_coefficients.copy_(coefficients)
    torch.testing.assert_close(network(q, physics), target, rtol=1e-9, atol=1e-12)


def build(config, recorded_observations=None):
    physics, law, provenance = load_inputs(config, config["device"])
    network = network_from_config(physics, law, config)
    source = ROOT / config["initial_checkpoint"]
    saved = torch.load(source, map_location=config["device"], weights_only=True)
    for key in ("operator_sha256", "law_sha256", "law_values"):
        assert saved["provenance"][key] == provenance[key], key
    for key in ("width", "depth", "fourier_frequencies"):
        assert saved["config"][key] == config[key], key
    assert saved["config"].get("envelope_power", 1.0) == config["envelope_power"]
    network.load_initial_state(saved["network"], reset_local=config.get("initialize_local_from_branch", False))
    if recorded_observations is None:
        data = observations(config, physics)
    else:
        # Replay the checkpoint's recorded objective, including historical runs.
        data = copy.deepcopy(recorded_observations)
        for key, digest in data["source_hashes"].items():
            path = FRAME_EXCLUSIONS_PATH if key == "frame_exclusions_path" else (ROOT / config[key]).resolve()
            if sha256(path) != digest:
                raise ValueError(f"Recorded observation source changed: {key}")
    initial_fields = ROOT / config["initial_fields"]
    fields = np.load(initial_fields)
    if "initial_controls" in config:
        q = np.asarray(config["initial_controls"], dtype=float)
        if q.shape != (len(data["fractions"]),) or not np.all(np.diff(np.r_[0, q]) > 0):
            raise ValueError("Initial work coordinates must cover every ordered observation state")
        provenance["loading_coordinate_initialization"] = "Explicit work coordinates from the corrected neural loading branch."
    elif not data["profiles"] and "control_delta0" in saved:
        q = np.array([saved["control_delta0"]])
        selected = np.flatnonzero(np.isclose(fields["control"], q[0], atol=1e-14, rtol=0))
        assert len(selected) == 1, "Peak checkpoint is absent from its recorded neural fields"
        with torch.no_grad():
            neural_field = network(physics.f.new_tensor(q), physics).cpu().numpy()[0]
        np.testing.assert_allclose(neural_field, fields["slip_mm"][selected[0]], atol=1e-12, rtol=1e-9)
        provenance["peak_only_initialization"] = "Verified individual neural state from a local stability bracket; no curve interpolation or gauge target."
    else:
        peak = int(fields["load_kN"].argmax())
        loads, controls = fields["load_kN"][:peak+1], fields["control"][:peak+1]
        if not (np.diff(loads) > 0).all() or min(data["fractions"]) * loads[-1] < loads[0]:
            raise ValueError("Neural initialization does not cover all measured load fractions monotonically")
        q = np.interp(np.array(data["fractions"]) * loads[-1], loads, controls)
    if config.get("initialize_local_from_branch", False):
        initialize_local_from_branch(network, physics, q, ROOT / config["initial_control_branch"])
        provenance["neural_field_initialization"] = "Localized coefficients interpolate the saved neural branch; no extra residual targets enter the inverse."
    model = InverseModel(physics, law, network, data, q, config["physics_weight"])
    slip_scale = float(config.get("initial_slip_scale", 1.0))
    if not np.isfinite(slip_scale) or slip_scale <= 0:
        raise ValueError("Initial slip scale must be finite and positive")
    if slip_scale != 1.0:
        # Move the rigid work component and both law slip scales together.
        # Preserve the elastic correction and the common objective normalization.
        with torch.no_grad():
            law.log_parameters[1:3].add_(np.log(slip_scale))
            network.control_scale.mul_(slip_scale)
        provenance["initial_slip_scale"] = slip_scale
    provenance.update(initial_checkpoint_sha256=sha256(source), initial_fields_sha256=sha256(initial_fields),
                      observation_source_hashes=data["source_hashes"],
                      use_of_historical_results="Constitutive coefficients initialize trainable parameters; neural fields initialize coordinates. No conventional solution targets or comparisons.")
    if config.get("resume_inverse_checkpoint"):
        checkpoint_path = ROOT / config["resume_inverse_checkpoint"]
        checkpoint = torch.load(checkpoint_path, map_location=config["device"], weights_only=True)
        if checkpoint["config"].get("cutoff_smoothing_fraction", 0.0) != config.get("cutoff_smoothing_fraction", 0.0):
            raise ValueError("Cannot resume an inverse with a changed cutoff law")
        for key in ("operator_sha256", "law_sha256", "law_values"):
            assert checkpoint["provenance"][key] == provenance[key], key
        prior_data = json.loads((checkpoint_path.parent / "observations.json").read_text())
        assert prior_data == data, "Observation or reference convention changed"
        missing, unexpected = model.load_state_dict(checkpoint["model"], strict=False)
        assert not unexpected and all(name.startswith("physics.") for name in missing)
        provenance["resume_inverse_checkpoint_sha256"] = sha256(checkpoint_path)
        provenance["continuation"] = "Joint weights and constitutive coordinates with a fresh LM damping schedule."
    return model, data, provenance


class ParameterVector:
    def __init__(self, model):
        self.model = model
        self.layout = [(name, p.shape, p.numel()) for name, p in model.named_parameters()]

    def pack(self):
        return torch.cat([p.detach().flatten() for p in self.model.parameters()])

    def unpack(self, vector):
        result, offset = {}, 0
        for name, shape, count in self.layout:
            result[name] = vector[offset:offset+count].view(shape)
            offset += count
        return result

    def main(self, vector):
        return functional_call(self.model, self.unpack(vector), ("main",))

    def critical(self, vector):
        return functional_call(self.model, self.unpack(vector), ("critical",))

    @torch.no_grad()
    def install(self, vector):
        values = self.unpack(vector)
        for name, p in self.model.named_parameters():
            p.copy_(values[name])


def run(config_path):
    config = json.loads(config_path.read_text())
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(config.get("cpu_threads", 4))
    torch.manual_seed(config["seed"])
    model, data, provenance = build(config)
    out = ROOT / config["output_directory"]
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    snapshot = dict(config, source_sha256=sha256(Path(__file__)),
                    physics_source_sha256=sha256(Path(__file__).with_name("interface_pinn.py")))
    (out / "run_config.json").write_text(json.dumps(snapshot, indent=2) + "\n")
    (out / "observations.json").write_text(json.dumps(data, indent=2) + "\n")
    for filename in ("interface_pinn_inverse.py", "interface_pinn.py"):
        (out / filename).write_bytes(Path(__file__).with_name(filename).read_bytes())
    mapper = ParameterVector(model)
    vector = mapper.pack()
    initial = model.diagnostics()
    start = time.perf_counter()
    history = []
    damping = config["initial_damping"]
    stop = "Configured iteration budget reached"
    for iteration in range(config["lm_iterations"]):
        main, critical = mapper.main(vector).detach(), mapper.critical(vector).detach()
        residual = torch.cat((main, critical[None]))
        loss = float(residual @ residual)
        mapper.install(vector)
        row = dict(model.diagnostics(), iteration=iteration, loss=loss, damping=damping,
                   seconds=time.perf_counter() - start)
        history.append(row)
        with (out / "training.jsonl").open("a") as handle:
            handle.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)
        jac_main = jacrev(mapper.main, chunk_size=config["jacobian_chunk"])(vector).detach()
        jac_critical = jacrev(mapper.critical)(vector).detach()[None]
        jacobian = torch.cat((jac_main, jac_critical))
        del jac_main, jac_critical
        gram = jacobian @ jacobian.T
        if iteration == 0:
            damping *= float(gram.diagonal().max())
        identity = torch.eye(len(residual), device=vector.device)
        accepted = False
        for _ in range(14):
            update = -jacobian.T @ torch.linalg.solve(gram + damping * identity, residual)
            candidate = vector + update
            with torch.no_grad():
                rmain, rcritical = mapper.main(candidate), mapper.critical(candidate)
                candidate_loss = float(rmain @ rmain + rcritical.square())
            if np.isfinite(candidate_loss) and candidate_loss < loss:
                vector = candidate.detach()
                damping = max(damping / 3, 1e-16)
                accepted = True
                break
            damping *= 10
        del jacobian, gram, identity
        if not accepted:
            stop = "No decreasing joint neural-parameter step at tested damping values"
            break
    mapper.install(vector)
    final = model.diagnostics()
    targets_met = (final["force_relative_max"] < config["force_tolerance"]
                   and final["load_fraction_error_max"] < config["fraction_tolerance"]
                   and final["peak_tangent_relative"] < config["critical_tolerance"])
    metrics = {"kind": "joint_interface_pinn_inverse_pilot", "status": "physics_targets_met" if targets_met else "needs_refinement",
               "initial": initial, "final": final, "stop_reason": stop, "provenance": provenance,
               "elapsed_seconds": time.perf_counter() - start,
               "trainable_parameter_count": len(vector), "trainable_law_coordinates": 5,
               "scope": "One orientation and one geometry, identification-set observations only. No uniqueness, uncertainty, global-optimum or held-out validation claim."}
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    checkpoint_state = {key: value for key, value in model.state_dict().items() if not key.startswith("physics.")}
    torch.save({"model": checkpoint_state, "config": snapshot, "provenance": provenance}, out / "checkpoint.pt")
    with torch.no_grad():
        q, d, load, rf, gauges, _ = model.quantities()
    np.savez_compressed(out / "fields.npz", control=q.cpu().numpy(), slip_mm=d.cpu().numpy(),
                        load_kN=load.cpu().numpy(), force_residual_N=rf.cpu().numpy(), gauge_um=gauges.cpu().numpy(),
                        fractions=np.array(data["fractions"]))
    print(json.dumps({k: v for k, v in metrics.items() if k != "provenance"}, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    run(parser.parse_args().config)
