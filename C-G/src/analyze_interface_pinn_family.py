"""Replay a shared-law neural inverse and check each fitted physical branch."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.func import grad

from analyze_interface_pinn_inverse import inspect_model
from interface_pinn import sha256, validate_input_provenance
from interface_pinn_family import CaseCoordinates, build_family, physics_targets_met


def inspect_optimizer(models, fixed_law=False):
    coordinates = [CaseCoordinates(model) for model in models]
    pairs = [mapper.split(mapper.mapper.pack()) for mapper in coordinates]
    shared = pairs[0][0]
    sizes = [len(local) for _, local in pairs]
    vector = torch.cat((shared, *(local for _, local in pairs)))
    def objective(value):
        offset = 5
        result = value.new_zeros(())
        for mapper, size in zip(coordinates, sizes):
            residual = mapper.residual(value[:5], value[offset:offset+size])
            result = result + residual.square().sum()
            offset += size
        return result
    gradient = grad(objective)(vector).detach()
    shared_gradient = gradient[:5].tolist()
    if fixed_law:
        gradient[:5] = 0
    norm = gradient.norm()
    direction = -gradient / norm.clamp(min=1e-30)
    initial = float(objective(vector).detach())
    trials = []
    with torch.no_grad():
        for step in (1e-4, 1e-6, 1e-8, 1e-10, 1e-12):
            plus, minus = float(objective(vector + step * direction)), float(objective(vector - step * direction))
            trials.append({"parameter_step_norm": step, "loss_change_along_negative_gradient": plus - initial,
                           "central_directional_derivative": (plus - minus) / (2 * step),
                           "AD_directional_derivative": -float(norm)})
        margins = []
        for model in models:
            _, d, _, _, _, _ = model.quantities()
            n = model.physics.n
            slip = torch.sqrt(d[:, :n].square() + d[:, n:].square())
            _, d0, dc, _, _ = model.law.physical()
            margins.append({"closest_relative_peak_slip_margin": float(((slip - d0) / d0).abs().min()),
                            "closest_relative_cutoff_margin": float(((slip - dc) / dc).abs().min())})
    return {"loss": initial, "gradient_norm": float(norm), "law_held_fixed": fixed_law,
            "shared_log_law_gradient": shared_gradient,
            "negative_gradient_trials": trials, "constitutive_knot_margins": margins}


def analyze(directory, optimizer=False):
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(4)
    directory = directory.resolve()
    config = json.loads((directory / "run_config.json").read_text())
    config["device"] = "cpu"
    saved = torch.load(directory / "checkpoint.pt", map_location="cpu", weights_only=True)
    models, _, _, _, data, provenance, cases = build_family(config, recorded_observations=saved["data"])
    assert saved["data"] == data == json.loads((directory / "observations.json").read_text())
    assert len(saved["provenance"]) == len(provenance)
    for current, recorded in zip(provenance, saved["provenance"]):
        validate_input_provenance(current, recorded)
    assert len(saved["models"]) == len(models)
    metrics = json.loads((directory / "metrics.json").read_text())
    results = []
    for model, state, case, recorded in zip(models, saved["models"], cases, metrics["final"]):
        missing, unexpected = model.load_state_dict(state, strict=False)
        assert not unexpected and all(name.startswith("physics.") for name in missing)
        torch.testing.assert_close(model.law.log_parameters, saved["shared_log_law"], rtol=0, atol=0)
        depth = case["notch_depth_mm"]
        result = inspect_model(model, np.load(directory / f"fields_a{depth}.npz"))
        result["notch_depth_mm"] = depth
        difference = abs(result["diagnostics"]["peak_tangent_min_N_per_mm"] - recorded["peak_tangent_min_N_per_mm"])
        result["recorded_peak_tangent_min_N_per_mm"] = recorded["peak_tangent_min_N_per_mm"]
        result["normalized_peak_stiffness_replay_difference"] = difference / float(model.stiffness_scale)
        result["peak_stiffness_replay_within_target_tolerance"] = result["normalized_peak_stiffness_replay_difference"] < config["critical_tolerance"]
        results.append(result)
    output = {"checkpoint_replay": "pass", "exact_shared_law_across_cases": True,
              "peak_stiffness_replay_all_cases": all(r["peak_stiffness_replay_within_target_tolerance"] for r in results),
              "physics_targets_met": physics_targets_met([r["diagnostics"] for r in results], config),
              "positive_fitted_prepeak_tangents_all_cases": all(r["positive_tangent_at_fitted_prepeak_states"] is True
                  for r in results if r["fitted_prepeak_state_count"] > 0),
              "cases_without_fitted_prepeak_states_mm": [r["notch_depth_mm"] for r in results if r["fitted_prepeak_state_count"] == 0],
              "positive_controlled_tangents_all_cases": all(r["positive_controlled_tangent_at_fitted_states"] for r in results),
              "cases": results, "analysis_source_sha256": sha256(Path(__file__)),
              "checkpoint_sha256": sha256(directory / "checkpoint.pt")}
    (directory / "inverse_check.json").write_text(json.dumps(output, indent=2) + "\n")
    if optimizer:
        diagnostic = inspect_optimizer(models, fixed_law=metrics.get("optimized_law_coordinate_count", 5) == 0)
        diagnostic["checkpoint_sha256"] = output["checkpoint_sha256"]
        diagnostic["analysis_source_sha256"] = output["analysis_source_sha256"]
        (directory / "optimizer_check.json").write_text(json.dumps(diagnostic, indent=2) + "\n")
        print(json.dumps(diagnostic, indent=2))
    compact = {k: v for k, v in output.items() if k != "cases"}
    compact["case_diagnostics"] = [dict(r["diagnostics"], notch_depth_mm=r["notch_depth_mm"]) for r in results]
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    parser.add_argument("--optimizer", action="store_true")
    args = parser.parse_args()
    analyze(args.run_directory, optimizer=args.optimizer)
