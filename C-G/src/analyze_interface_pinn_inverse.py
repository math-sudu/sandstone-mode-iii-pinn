"""Replay joint neural inverses and inspect physical stiffness at fitted states."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from analyze_interface_pinn import physical_tangent
from interface_pinn import sha256
from interface_pinn_inverse import build


def inspect_model(model, fields):
    with torch.no_grad():
        q, d, load, rf, gauge, _ = model.quantities()
    replay = {"control": q, "slip_mm": d, "load_kN": load,
              "force_residual_N": rf, "gauge_um": gauge}
    differences = {}
    for name, value in replay.items():
        array = value.cpu().numpy()
        differences[name] = float(np.max(abs(array - fields[name])))
        np.testing.assert_allclose(array, fields[name], rtol=1e-8, atol=1e-9)
    np.testing.assert_array_equal(model.data["fractions"], fields["fractions"])
    orthogonal = torch.linalg.qr(model.physics.f[:, None], mode="complete").Q[:, 1:]
    rows = []
    for i, field in enumerate(d):
        tangent = physical_tangent(model.physics, model.law, field).detach()
        eigenvalues = torch.linalg.eigvalsh(tangent)
        controlled = torch.linalg.eigvalsh(orthogonal.T @ tangent @ orthogonal)
        rows.append({"index": i, "fraction": model.data["fractions"][i],
                     "load_kN": float(load[i]), "control_delta0": float(q[i]),
                     "tangent_min_N_per_mm": float(eigenvalues[0]),
                     "tangent_second_N_per_mm": float(eigenvalues[1]),
                     "controlled_tangent_min_N_per_mm": float(controlled[0]),
                     "numerical_eigenvalue_tolerance": float(eigenvalues.abs().max()) * 1e-9})
    diagnostics = model.diagnostics()
    np.testing.assert_allclose(rows[-1]["tangent_min_N_per_mm"],
                               diagnostics["peak_tangent_min_N_per_mm"], atol=1e-6, rtol=1e-6)
    return {"checkpoint_replay": "pass", "maximum_replay_differences": differences,
            "diagnostics": diagnostics, "physical_tangent_at_fitted_states": rows,
            "fitted_prepeak_state_count": len(rows) - 1,
            "positive_tangent_at_fitted_prepeak_states": all(
                r["tangent_min_N_per_mm"] > r["numerical_eigenvalue_tolerance"] for r in rows[:-1]) if rows[:-1] else None,
            "positive_controlled_tangent_at_fitted_states": all(
                r["controlled_tangent_min_N_per_mm"] > r["numerical_eigenvalue_tolerance"] for r in rows),
            "scope": "Replay and independent physical-tangent formula at fitted observation states; no off-grid accuracy, unique identification or first-instability certification."}


def analyze(directory):
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(4)
    directory = directory.resolve()
    config = json.loads((directory / "run_config.json").read_text())
    config["device"] = "cpu"
    checkpoint = torch.load(directory / "checkpoint.pt", map_location="cpu", weights_only=True)
    recorded_observations = json.loads((directory / "observations.json").read_text())
    model, data, provenance = build(config, recorded_observations=recorded_observations)
    from interface_pinn import validate_input_provenance
    validate_input_provenance(provenance, checkpoint["provenance"])
    assert json.loads((directory / "observations.json").read_text()) == data
    missing, unexpected = model.load_state_dict(checkpoint["model"], strict=False)
    assert not unexpected and all(name.startswith("physics.") for name in missing)
    result = inspect_model(model, np.load(directory / "fields.npz"))
    result.update(analysis_source_sha256=sha256(Path(__file__)),
                  checkpoint_sha256=sha256(directory / "checkpoint.pt"))
    (directory / "inverse_check.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    analyze(parser.parse_args().run_directory)
