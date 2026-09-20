"""Inspect saved neural states and the physical tangent without a reference solve."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from interface_pinn import ROOT, evaluate, load_inputs, network_from_config, sha256


def physical_tangent(physics, law, d):
    """Exact radial-law tangent on the unconstrained interface coordinates."""
    _, d0, _, _, _ = law.physical()
    sreg = max(float(d0.detach()) * 1e-6, 1e-12)
    n = physics.n
    dx, dz = d[:n], d[n:]
    slip = torch.sqrt(dx.square() + dz.square() + 1e-60)
    sr = slip.clamp(min=sreg).detach().requires_grad_(True)
    with torch.enable_grad():
        traction = law.tau(sr)
        slope = torch.autograd.grad(traction.sum(), sr)[0].detach()
    secant = traction.detach() / sr.detach()
    slope = torch.where(slip > sreg, slope, secant)
    ex, ez = dx / slip, dz / slip
    txx = physics.area * (secant + (slope - secant) * ex.square())
    tzz = physics.area * (secant + (slope - secant) * ez.square())
    txz = physics.area * (slope - secant) * ex * ez
    tangent = torch.cat((torch.cat((torch.diag(txx), torch.diag(txz)), dim=1),
                         torch.cat((torch.diag(txz), torch.diag(tzz)), dim=1)), dim=0)
    tangent = physics.K + tangent[physics.active][:, physics.active]
    return (tangent + tangent.T) / 2


def analyze(run_directory):
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(4)
    run_directory = run_directory.resolve()
    config = json.loads((run_directory / "run_config.json").read_text())
    physics, law, provenance = load_inputs(config, "cpu")
    saved = torch.load(run_directory / "checkpoint.pt", map_location="cpu", weights_only=True)
    for key in ("operator_sha256", "law_sha256", "law_values"):
        if saved["provenance"][key] != provenance[key]:
            raise ValueError(f"Input provenance changed: {key}")
    network = network_from_config(physics, law, config)
    network.load_state_dict(saved["network"])
    fields = np.load(run_directory / "fields.npz")
    reproduced = evaluate(network, physics, law, torch.tensor(fields["control"]))
    max_field_difference = float(np.max(abs(reproduced["slip_mm"] - fields["slip_mm"])))
    assert np.allclose(reproduced["slip_mm"], fields["slip_mm"], atol=1e-12, rtol=1e-9)
    assert np.allclose(reproduced["load_kN"], fields["load_kN"], atol=1e-9, rtol=1e-9)
    # The physical tangent is K + dt/dd, not the network-parameter Hessian.
    orthogonal = torch.linalg.qr(physics.f[:, None], mode="complete").Q[:, 1:]
    peak = int(np.argmax(fields["load_kN"]))
    selected = sorted(set(np.linspace(0, len(fields["control"]) - 1, 13, dtype=int).tolist()
                          + list(range(max(0, peak - 2), min(len(fields["control"]), peak + 3)))))
    rows = []
    for idx in selected:
        d = torch.tensor(fields["slip_mm"][idx])
        tangent = physical_tangent(physics, law, d)
        eigenvalues = torch.linalg.eigvalsh(tangent)
        constrained_values = torch.linalg.eigvalsh(orthogonal.T @ tangent @ orthogonal)
        rows.append({"index": idx, "control_delta0": float(fields["control"][idx]),
                     "load_kN": float(fields["load_kN"][idx]),
                     "force_relative": float(fields["force_relative"][idx]),
                     "tangent_min_N_per_mm": float(eigenvalues[0]),
                     "controlled_tangent_min_N_per_mm": float(constrained_values[0]),
                     "numerical_eigenvalue_tolerance": float(eigenvalues.abs().max()) * 1e-9})
    before = [row for row in rows if row["index"] < peak]
    after = [row for row in rows if row["index"] > peak]
    result = {
        "kind": "neural_state_physical_tangent_check", "analysis_source_sha256": sha256(Path(__file__)),
        "checkpoint_sha256": sha256(run_directory / "checkpoint.pt"),
        "checkpoint_replay": "pass", "maximum_replay_displacement_difference_mm": max_field_difference,
        "physical_tangent_sampling": rows,
        "positive_tangent_at_sampled_premaximum_states": bool(before) and all(
            r["tangent_min_N_per_mm"] >= -r["numerical_eigenvalue_tolerance"] for r in before),
        "negative_tangent_after_sampled_maximum": any(
            r["tangent_min_N_per_mm"] < -r["numerical_eigenvalue_tolerance"] for r in after),
        "controlled_tangent_nonnegative_at_all_sampled_states": all(
            r["controlled_tangent_min_N_per_mm"] >= -r["numerical_eigenvalue_tolerance"] for r in rows),
        "interpretation": "Sampled physical stability diagnostic at approximate neural equilibria; not a certified limit point or solution-error estimate.",
    }
    (run_directory / "path_check.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "physical_tangent_sampling"}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    analyze(parser.parse_args().run_directory)
