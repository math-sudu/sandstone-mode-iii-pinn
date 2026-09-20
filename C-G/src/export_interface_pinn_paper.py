"""Export manuscript tables from saved PINN states without fitting or retraining."""
import csv
import json
from pathlib import Path

import numpy as np
from scipy.integrate import quad

from interface_pinn import ROOT, sha256
from cohesive_law import theta_curve as traction

RUNS = {
    0: "b0_family_seed17_c2_initialized_inverse",
    45: "b45_family_seed17_reference",
    90: "b90_family_seed17_inverse",
}


def main():
    families, hashes, gauge_rows = [], {}, []
    for beta, name in RUNS.items():
        directory = ROOT / "experiments/interface_pinn/runs" / name
        metrics, config, observations, replay = [json.loads((directory / f).read_text())
            for f in ("metrics.json", "run_config.json", "observations.json", "inverse_check.json")]
        assert replay["checkpoint_replay"] == "pass"
        assert replay["exact_shared_law_across_cases"]
        assert replay["checkpoint_sha256"] == sha256(directory / "checkpoint.pt")
        for filename in ("metrics.json", "run_config.json", "observations.json", "inverse_check.json"):
            path = directory / filename
            hashes[path.relative_to(ROOT).as_posix()] = sha256(path)
        split_path, peak_path = [(ROOT / config[k]).resolve() for k in ("split_path", "peak_path")]
        for path, key in ((split_path, "split_path"), (peak_path, "peak_path")):
            assert all(d["source_hashes"][key] == sha256(path) for d in observations)
            hashes[str(path)] = sha256(path)
        split = json.loads(split_path.read_text())
        with peak_path.open(newline="") as handle:
            peaks = {r["specimen"]: float(r["peak_force_kN"]) for r in csv.DictReader(handle)}
        law = metrics["final"][0]["law_values"]
        assert all(row["law_values"] == law for row in metrics["final"])
        smoothing = config.get("cutoff_smoothing_fraction", 0.0)
        energy = sum(quad(lambda s: float(traction(s, law, smoothing)), lo, hi, epsabs=1e-12)[0]
                     for lo, hi in ((0, law["delta_0_mm"]), (law["delta_0_mm"], law["delta_c_mm"])))
        cases = []
        for depth, result, data in zip(metrics["geometry_depths_mm"], metrics["final"], observations):
            path = directory / f"fields_a{depth}.npz"
            hashes[path.relative_to(ROOT).as_posix()] = sha256(path)
            with np.load(path) as fields:
                np.testing.assert_allclose(fields["load_kN"][-1], result["peak_load_kN"], rtol=1e-10)
                peak_sse = float(np.sum(((result["peak_load_kN"] - np.array(data["peak_values_kN"]))
                                        / data["sigma_peak_kN"])**2))
                profiles, gauge_sse = [], 0.0
                for profile in data["profiles"]:
                    idx, ref, station = profile["level_indices"], profile["reference_index"], profile["station_index"]
                    pred = fields["gauge_um"][idx, station] - fields["gauge_um"][ref, station]
                    measured = np.array(profile["measured_um"])
                    sign = 1 if np.sum((pred - measured)**2) <= np.sum((-pred - measured)**2) else -1
                    signed = sign * pred
                    gauge_sse += float(np.sum(((signed - measured) / data["sigma_gauge_um"])**2))
                    fractions = np.array(data["fractions"])[idx].tolist()
                    profiles.append(dict(specimen=profile["specimen"], fractions=fractions,
                        measured_um=measured.tolist(), predicted_um=signed.tolist(), mounting_sign=sign))
                    for fraction, y, p in zip(fractions, measured, signed):
                        gauge_rows.append([profile["specimen"], beta, depth, fraction, float(y), float(p),
                                           float((p-y) / data["sigma_gauge_um"]), sign])
            np.testing.assert_allclose(peak_sse + gauge_sse, result["data_sum_squares"], rtol=1e-8)
            assert set(data["peak_specimens"]) <= set(split["identification_set"])
            heldout_id = f"{beta}-{depth}-3"
            assert heldout_id in split["held_out_set"]
            measured_peak = peaks[heldout_id]
            cases.append(dict(depth_mm=depth, **result, peak_data_sse=peak_sse, gauge_data_sse=gauge_sse,
                identification_peak_kN=data["peak_values_kN"], sigma_peak_kN=data["sigma_peak_kN"],
                profiles=profiles, heldout_specimen=heldout_id, heldout_measured_kN=measured_peak,
                heldout_signed_error_percent=100*(result["peak_load_kN"] / measured_peak - 1)))
        families.append(dict(beta_deg=beta, run=name, law=law, cutoff_smoothing_fraction=smoothing,
            energy_MPa_mm=energy, cases=cases,
            data_sse=sum(c["data_sum_squares"] for c in cases),
            heldout_mare_percent=float(np.mean([abs(c["heldout_signed_error_percent"]) for c in cases]))))
    payload = dict(kind="saved_pinn_manuscript_results", families=families, input_hashes=hashes,
        exporter_sha256=sha256(Path(__file__)),
        scope="Current joint PINN endpoints and measured replicates. No retraining or alternative solver evaluation. "
              "Energy is a derived fitted-law area, not an independently identified quantity. "
              "Historical numerical target flags are preserved in source runs and do not filter this export. "
              "Replicate-3 peaks enter no fitted observation or peak-weight estimate; valid replicate-3 gauges "
              "contributed to the preceding pooled 8.9 um gauge scale. The assessment is conditional on that "
              "fixed scale and the historical constitutive initialization.")
    output = ROOT / "experiments/interface_pinn/paper_results.json"
    output.write_text(json.dumps(payload, indent=2) + "\n")
    with output.with_name("paper_gauge_predictions.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["specimen", "beta_deg", "a_mm", "load_fraction", "measured_um",
                         "pinn_increment_um", "standardized_residual", "mounting_sign"])
        writer.writerows(gauge_rows)
    table_dir = ROOT / "paper/tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    law_rows, fit_rows, heldout_rows = [], [], []
    for f in families:
        b, law = f["beta_deg"], f["law"]
        law_rows.append(f"{b} & {law['tau_p_MPa']:.4f} & {law['delta_0_mm']:.6f} & "
                        f"{law['delta_c_mm']:.6f} & {law['p_r']:.4f} & {law['p_s']:.4f} & {f['energy_MPa_mm']:.5f}" + r" \\")
        for c in f["cases"]:
            fit_rows.append(f"{b} & {c['depth_mm']} & {len(c['control_delta0'])} & {c['peak_load_kN']:.4f} & "
                            f"{100*c['force_relative_max']:.4f} & {c['load_fraction_error_max']:.2e} & "
                            f"{c['peak_tangent_relative']:.2e} & {c['peak_data_sse']:.2f} & {c['gauge_data_sse']:.2f}" + r" \\")
            heldout_rows.append(f"{c['heldout_specimen']} & {c['peak_load_kN']:.4f} & "
                                f"{c['heldout_measured_kN']:.3f} & {c['heldout_signed_error_percent']:+.2f}" + r" \\")
    formats = {
        "laws": ("rrrrrrr", r"$\beta$ (deg) & $\tau_p$ (MPa) & $\delta_0$ (mm) & $\delta_c$ (mm) & $p_r$ & $p_s$ & $G_{\mathrm{IIIc}}$ (kJ/m$^2$)"),
        "fit": ("rrrrrrrrr", r"$\beta$ & $a$ (mm) & $n_q$ & $P_n$ (kN) & $\epsilon_f$ (\%) & $\epsilon_\ell$ & $\epsilon_k$ & $\sum(R^P)^2$ & $\sum(R^g)^2$"),
        "heldout": ("lrrr", r"Specimen & PINN peak (kN) & Measured peak (kN) & Signed error (\%)"),
    }
    for name, rows in (("laws", law_rows), ("fit", fit_rows), ("heldout", heldout_rows)):
        columns, header = formats[name]
        (table_dir / f"interface_pinn_{name}.tex").write_text(
            "% Generated by src/export_interface_pinn_paper.py from saved neural states.\n"
            + r"\begin{tabular}{" + columns + "}\n" + r"\toprule" + "\n"
            + header + r" \\" + "\n" + r"\midrule" + "\n" + "\n".join(rows)
            + "\n" + r"\bottomrule" + "\n" + r"\end{tabular}" + "\n")
    print(json.dumps({f["beta_deg"]: {k: f[k] for k in ("energy_MPa_mm", "data_sse", "heldout_mare_percent")}
                      for f in families}, indent=2))


if __name__ == "__main__":
    main()
