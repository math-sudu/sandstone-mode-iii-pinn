"""Refine a physical stability crossing using neural equilibrium corrections."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from analyze_interface_pinn import physical_tangent
from interface_pinn import ROOT, evaluate, load_inputs, network_from_config, sha256
from interface_pinn_continuation import NeuralState, correct


def run(config_path):
    config = json.loads(config_path.read_text())
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(config.get("cpu_threads", 4))
    torch.manual_seed(config["seed"])
    physics, law, provenance = load_inputs(config, config["device"])
    network = network_from_config(physics, law, config)
    state = NeuralState(network, physics, law)
    vectors = []
    provenance["bracket_checkpoints"] = []
    for source in config["bracket_checkpoints"]:
        path = ROOT / source
        saved = torch.load(path, map_location=config["device"], weights_only=True)
        for key in ("operator_sha256", "law_sha256", "law_values"):
            assert saved["provenance"][key] == provenance[key], key
        for key in ("width", "depth", "fourier_frequencies", "envelope_power"):
            assert saved["config"][key] == config[key], key
        network.load_state_dict(saved["network"])
        vectors.append(state.pack(saved["control_delta0"]))
        provenance["bracket_checkpoints"].append({"path": source, "sha256": sha256(path)})
    left, right = vectors
    assert left[-1] < right[-1]
    out = ROOT / config["output_directory"]
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    snapshot = dict(config, source_sha256=sha256(Path(__file__)),
                    continuation_source_sha256=sha256(Path(__file__).with_name("interface_pinn_continuation.py")),
                    physics_source_sha256=sha256(Path(__file__).with_name("interface_pinn.py")))
    (out / "run_config.json").write_text(json.dumps(snapshot, indent=2) + "\n")
    for filename in ("refine_interface_pinn_limit.py", "interface_pinn_continuation.py", "interface_pinn.py", "analyze_interface_pinn.py"):
        (out / filename).write_bytes(Path(__file__).with_name(filename).read_bytes())
    basis = torch.linalg.qr(physics.f[:, None], mode="complete").Q[:, 1:]
    rows, fields = [], []
    start = time.perf_counter()

    def inspect(v):
        d, load, rf, _ = state.physical(v)
        tangent = physical_tangent(physics, law, d.detach())
        ev, eigenvectors = torch.linalg.eigh(tangent)
        constrained = torch.linalg.eigvalsh(basis.T @ tangent @ basis)
        return dict(state.diagnostics(v), tangent_min_N_per_mm=float(ev[0]),
                    second_tangent_eigenvalue_N_per_mm=float(ev[1]),
                    controlled_tangent_min_N_per_mm=float(constrained[0]),
                    load_mode_cosine=float(abs(eigenvectors[:, 0] @ physics.f) / physics.f.norm()),
                    weak_mode_force_defect_N=float((eigenvectors[:, 0] @ rf).detach()))

    def save(v, row):
        state.install(v)
        result = evaluate(network, physics, law, v[-1:])
        row = dict(row, index=len(rows), displacement_relative=float(result["displacement_relative"][0]),
                   hard_control_error_mm=float(result["control_error_mm"][0]))
        torch.save({"network": network.state_dict(), "law": law.state_dict(), "config": snapshot,
                    "provenance": provenance, "control_delta0": float(v[-1])}, out / f"state_{len(rows):04d}.pt")
        rows.append(row)
        fields.append(result["slip_mm"][0])
        (out / "states.json").write_text(json.dumps(rows, indent=2) + "\n")
        np.savez_compressed(out / "fields.npz", slip_mm=np.stack(fields),
                            control=np.array([r["control_delta0"] for r in rows]),
                            load_kN=np.array([r["load_kN"] for r in rows]),
                            force_relative=np.array([r["force_relative"] for r in rows]),
                            xz=physics.xz.cpu().numpy())
        print(json.dumps(dict(event="state_accepted", **row)), flush=True)

    def log(row):
        row = dict(row, state=len(rows), seconds=time.perf_counter() - start)
        with (out / "training.jsonl").open("a") as handle:
            handle.write(json.dumps(row) + "\n")
        if row["iteration"] % 50 == 0:
            print(json.dumps(row), flush=True)

    linfo, rinfo = inspect(left), inspect(right)
    assert linfo["tangent_min_N_per_mm"] > 0 > rinfo["tangent_min_N_per_mm"]
    for v, row in ((left, linfo), (right, rinfo)):
        assert row["force_relative"] < config["force_tolerance"]
        assert row["controlled_tangent_min_N_per_mm"] > 0
        save(v, row)
    failure = None
    for iteration in range(config["maximum_bisections"]):
        if float(right[-1] - left[-1]) < config["bracket_width_delta0"]:
            break
        midpoint = float((left[-1] + right[-1]) / 2)
        candidate, ok, info = correct(state, left, config, log, fixed_control=midpoint)
        if not ok:
            failure = "Neural midpoint correction did not meet equilibrium/control tolerances"
            state.install(candidate)
            torch.save({"network": network.state_dict(), "law": law.state_dict(), "config": snapshot,
                        "provenance": provenance, "control_delta0": float(candidate[-1]), "accepted": False},
                       out / "failed_candidate.pt")
            break
        row = inspect(candidate)
        # Work control must remain regular for this scalar bracket refinement.
        if row["controlled_tangent_min_N_per_mm"] <= 0:
            failure = "The fixed-work subspace lost positive stiffness; a different path coordinate is required"
            break
        save(candidate, dict(row, correction_iterations=info["iteration"]))
        if row["tangent_min_N_per_mm"] > 0:
            left, linfo = candidate, row
        else:
            right, rinfo = candidate, row
    report = {"kind": "neural_stability_crossing_refinement", "provenance": provenance,
              "status": "needs_refinement" if failure else "bracket_refined",
              "failure": failure, "left": linfo, "right": rinfo,
              "bracket_width_delta0": float(right[-1] - left[-1]),
              "accepted_states": len(rows), "elapsed_seconds": time.perf_counter() - start,
              "interpretation": "A local stability crossing bracket on neural equilibria; its force residual and sampling limits remain. This alone does not certify the first crossing on the entire loading branch."}
    if not failure and report["bracket_width_delta0"] >= config["bracket_width_delta0"]:
        report["status"] = "needs_refinement"
        report["failure"] = "Requested bracket width not reached within configured iterations"
    (out / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "provenance"}, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    run(parser.parse_args().config)
