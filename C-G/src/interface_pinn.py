"""Displacement-controlled neural solve of the existing condensed interface physics.

The network represents both tangential jumps on every interface pair. A hard
constraint fixes the load-conjugate displacement. Load is a differentiable
readout of global equilibrium, not an independently fitted curve. Training uses
equilibrium residuals only in the fixed-law pilot. No conventional solver or
saved displacement/force solution is used as a training target.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
import time
import types
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_input_provenance(current: dict, recorded: dict) -> None:
    """Allow relocated source paths only when the corresponding bytes match."""
    for key, value in current.items():
        if key in ("operator_path", "law_path"):
            digest = key.replace("_path", "_sha256")
            if current.get(digest) and current[digest] == recorded.get(digest):
                continue
        if recorded.get(key) != value:
            raise ValueError(f"Checkpoint input changed: {key}")


class OperatorReader(pickle.Unpickler):
    """Read the trusted local operator record without importing its solver."""

    def find_class(self, module, name):
        if (module, name) == ("baxi_pinn.anbd3d_cohesive", "InterfaceOperators"):
            return types.SimpleNamespace
        return super().find_class(module, name)


class CohesiveLaw(nn.Module):
    """Five admissible coordinates; fixed in the initial state-solver pilot."""

    def __init__(self, values: dict, trainable: bool = False, cutoff_smoothing_fraction: float = 0.0):
        super().__init__()
        if not 0 <= cutoff_smoothing_fraction < 1:
            raise ValueError("Cutoff smoothing must occupy less than the full softening interval")
        self.cutoff_smoothing_fraction = cutoff_smoothing_fraction
        p = [values["tau_p_MPa"], values["delta_0_mm"],
             values["delta_c_mm"] - values["delta_0_mm"], values["p_r"], values["p_s"]]
        self.log_parameters = nn.Parameter(torch.tensor(p).log(), requires_grad=trainable)

    def physical(self):
        tp, d0, tail, pr, ps = self.log_parameters.exp().unbind()
        return tp, d0, d0 + tail, pr, ps

    def smooth_softening(self, slip):
        """Compact C2 closure, with an analytic slip derivative.

        The quintic switch changes only the final eta fraction of the softening
        interval. Its cubic zero makes the cutoff C2 for every positive p_s;
        both factors remain nonnegative and monotone toward peak traction.
        """
        tp, d0, dc, _, ps = self.physical()
        q = (dc - slip) / (dc - d0)
        active = (q > 0) & (q < 1)
        safe_q = torch.where(active, q, torch.ones_like(q))
        base = tp * torch.exp(ps * (safe_q.log() + 1 - safe_q))
        t = (q / self.cutoff_smoothing_fraction).clamp(0, 1)
        switch = t.pow(3) * (10 - 15 * t + 6 * t.square())
        switch_slope = -30 * t.square() * (1 - t).square() / (
            self.cutoff_smoothing_fraction * (dc - d0))
        slope = base * (ps * (1 - safe_q.reciprocal()) / (dc - d0) * switch + switch_slope)
        return base * switch, torch.where(active, slope, torch.zeros_like(slope))

    def tau(self, slip):
        tp, d0, dc, pr, ps = self.physical()
        r = (slip / d0).clamp(min=1e-15, max=1.0)
        q = ((dc - slip) / (dc - d0)).clamp(min=1e-15, max=1.0)
        rise = tp * torch.exp(pr * (r.log() + 1.0 - r))
        soft = tp * torch.exp(ps * (q.log() + 1.0 - q))
        if self.cutoff_smoothing_fraction:
            soft, _ = self.smooth_softening(slip)
        return torch.where((slip > 0) & (slip < dc),
                           torch.where(slip <= d0, rise, soft), torch.zeros_like(slip))

    def force(self, d, area):
        n = area.numel()
        dx, dz = d[..., :n], d[..., n:]
        # The same small-slip secant regularisation as the existing formulation.
        _, d0, _, _, _ = self.physical()
        sreg = torch.maximum(d0 * 1e-6, d0.new_tensor(1e-12))
        slip = torch.sqrt(dx.square() + dz.square() + 1e-60)
        sr = torch.maximum(slip, sreg)
        secant = area * self.tau(sr) / sr
        return torch.cat((secant * dx, secant * dz), dim=-1)

    def tangent(self, d, area):
        """Differentiable displacement tangent, including constitutive gradients."""
        n = area.numel()
        dx, dz = d[:n], d[n:]
        _, d0, dc, pr, ps = self.physical()
        sreg = torch.maximum(d0 * 1e-6, d0.new_tensor(1e-12))
        slip = torch.sqrt(dx.square() + dz.square() + 1e-60)
        sr = torch.maximum(slip, sreg)
        tau = self.tau(sr)
        rise = tau * pr * (sr.reciprocal() - d0.reciprocal())
        remaining = (dc - sr).clamp(min=1e-30)
        soft = tau * ps * ((dc - d0).reciprocal() - remaining.reciprocal())
        slope = torch.where(sr <= d0, rise, soft)
        differentiable_branch = (sr > 0) & (sr < dc) & ((dc - sr) / (dc - d0) > 1e-15)
        slope = torch.where(differentiable_branch, slope, torch.zeros_like(slope))
        if self.cutoff_smoothing_fraction:
            _, soft_slope = self.smooth_softening(sr)
            slope = torch.where(sr > d0, soft_slope, slope)
        secant = tau / sr
        slope = torch.where(slip > sreg, slope, secant)
        ex, ez = dx / slip, dz / slip
        txx = area * (secant + (slope - secant) * ex.square())
        tzz = area * (secant + (slope - secant) * ez.square())
        txz = area * (slope - secant) * ex * ez
        return torch.cat((torch.cat((torch.diag(txx), torch.diag(txz)), dim=1),
                          torch.cat((torch.diag(txz), torch.diag(tzz)), dim=1)), dim=0)


class InterfacePhysics(nn.Module):
    def __init__(self, record):
        super().__init__()
        self.n = int(record.n_pairs)
        c_full = np.asarray(record.C_tt, dtype=np.float64)
        # Exactly zero rows are displacement constraints, not small modes to drop.
        active = np.any(c_full != 0, axis=1)
        if not np.all(record.g_t[~active] == 0):
            raise ValueError("Constrained rows have nonzero loading")
        c = c_full[np.ix_(active, active)]
        if not np.allclose(c, c.T, atol=1e-14, rtol=1e-10):
            raise ValueError("Operator is not symmetric")
        c = (c + c.T) / 2
        inv_c = np.linalg.solve(c, np.eye(len(c)))
        k = inv_c - np.diag(record.kref_force[active])
        f = np.linalg.solve(c, record.g_t[active])
        # Relative rigid translation along x supplies a nonzero global load row.
        direction = np.r_[np.ones(self.n), np.zeros(self.n)][active]
        direction *= np.sign(direction @ f)
        if abs(direction @ f) < 1e-8:
            raise ValueError("Control direction is orthogonal to loading")
        self.gauge_keys = sorted(record.gauge_g)
        self.operator_checks = {
            "n_pairs": self.n, "n_active_dofs": int(active.sum()),
            "constrained_dofs": np.flatnonzero(~active).tolist(),
            "relative_rigid_mode_residual": float(np.linalg.norm(k @ direction)
                                                    / np.linalg.norm(inv_c @ direction)),
        }
        entries = {
            "active": torch.tensor(active), "xz": torch.tensor(record.xz),
            "area": torch.tensor(record.area), "C": torch.tensor(c),
            "K": torch.tensor(k), "f": torch.tensor(f),
            "direction": torch.tensor(direction), "chol_C": torch.tensor(np.linalg.cholesky(c)),
            "g_full": torch.tensor(record.g_t), "C_full": torch.tensor(c_full),
            "kref": torch.tensor(record.kref_force),
            "gauge_g": torch.tensor([record.gauge_g[key] for key in self.gauge_keys]),
            "gauge_W": torch.tensor(np.stack([record.gauge_W[key] for key in self.gauge_keys])),
        }
        for name, value in entries.items():
            self.register_buffer(name, value)

    def expand(self, active_d):
        d = active_d.new_zeros((*active_d.shape[:-1], 2 * self.n))
        d[..., self.active] = active_d
        return d

    def readout(self, d, law):
        t = law.force(d, self.area)
        da = d[..., self.active]
        internal = da @ self.K.T + t[..., self.active]
        load = (internal @ self.direction) / (self.direction @ self.f)
        force_residual = internal - load[..., None] * self.f
        correction = d * self.kref - t
        gauge_um = (load[..., None] * self.gauge_g + correction @ self.gauge_W.T) * 1000
        displacement_residual = d - load[..., None] * self.g_full - correction @ self.C_full.T
        return load, force_residual, gauge_um, displacement_residual


class InterfaceNetwork(nn.Module):
    def __init__(self, physics, law, width=48, depth=3, fourier_frequencies=(), envelope_power=1.0,
                 local_rbf=False, local_control_centers=()):
        super().__init__()
        self.fourier_frequencies = tuple(fourier_frequencies)
        self.envelope_power = float(envelope_power)
        layers = [nn.Linear(3 + 4 * len(self.fourier_frequencies), width), nn.Tanh()]
        for _ in range(depth - 1):
            layers.extend([nn.Linear(width, width), nn.Tanh()])
        layers.append(nn.Linear(width, 2))
        self.net = nn.Sequential(*layers)
        nn.init.normal_(self.net[-1].weight, std=1e-3)
        nn.init.zeros_(self.net[-1].bias)
        tp, d0, _, _, _ = law.physical()
        self.register_buffer("control_scale", d0.detach().clone())
        self.register_buffer("correction_scale", (tp / 6000).detach().clone())
        xz = physics.xz.clone()
        xz[:, 0] /= xz[:, 0].abs().max()
        xz[:, 1] /= xz[:, 1].max()
        self.register_buffer("coordinates", xz)
        self.local_coefficients = None
        self.local_control_log_scale = None
        if local_rbf:
            distance = torch.cdist(physics.xz, physics.xz)
            nearest = distance.clone()
            nearest.diagonal().fill_(float("inf"))
            widths = 0.5 * nearest.min(dim=0).values
            if (widths <= 0).any():
                raise ValueError("Local radial basis centers must be distinct")
            self.register_buffer("local_basis", torch.exp(-0.5 * (distance / widths[None, :]).square()),
                                 persistent=False)
            count = 3
            if len(local_control_centers):
                centers = torch.log1p(physics.f.new_tensor(local_control_centers))
                if (centers <= 0).any() or (torch.diff(centers) <= 0).any():
                    raise ValueError("Local load centers must be strictly increasing and positive")
                if len(centers) > 1:
                    distance = (centers[:, None] - centers[None, :]).abs()
                    distance.diagonal().fill_(float("inf"))
                    widths = 0.5 * distance.min(dim=0).values
                else:
                    widths = torch.full_like(centers, 0.5)
                self.register_buffer("local_load_centers", centers)
                self.register_buffer("local_load_widths", widths)
                self.local_control_log_scale = nn.Parameter(torch.zeros(()))
                count = len(centers)
            self.local_coefficients = nn.Parameter(torch.zeros(count, physics.n, 2))

    def load_initial_state(self, state, reset_local=False):
        """A zero local enrichment may initialize from an existing global MLP."""
        if reset_local:
            state = {key: value for key, value in state.items() if not key.startswith("local_")}
        missing, unexpected = self.load_state_dict(state, strict=False)
        allowed = {"local_coefficients"} if self.local_coefficients is not None else set()
        if self.local_control_log_scale is not None:
            allowed.update(("local_control_log_scale", "local_load_centers", "local_load_widths"))
        if unexpected or set(missing) - allowed:
            raise ValueError(f"Incompatible neural initializer: missing={missing}, unexpected={unexpected}")

    def local_load_features(self, control):
        if self.local_control_log_scale is not None:
            z = torch.log1p(control / self.local_control_log_scale.exp())
            return torch.exp(-0.5 * ((z[:, None] - self.local_load_centers) / self.local_load_widths).square())
        z = torch.log1p(control)
        return torch.stack((torch.ones_like(z), z, z.square()), dim=1)

    def forward(self, control, physics):
        count, n = control.numel(), physics.n
        coords = self.coordinates.expand(count, n, 2)
        q = control.reshape(count, 1, 1).expand(count, n, 1)
        features = [coords, torch.log1p(q)]
        for frequency in self.fourier_frequencies:
            phase = torch.pi * frequency * coords
            features.extend((phase.sin(), phase.cos()))
        values = self.net(torch.cat(features, dim=-1))
        if self.local_coefficients is not None:
            load_features = self.local_load_features(control)
            local = torch.einsum("bk,knc->bnc", load_features, self.local_coefficients)
            values = values + torch.einsum("ij,bjc->bic", self.local_basis, local)
        dx = values[..., 0]
        dz = values[..., 1] * coords[..., 1]  # exact z=0 symmetry constraint
        correction = torch.cat((dx, dz), dim=-1)[..., physics.active]
        envelope = control[:, None].pow(self.envelope_power)
        correction = correction * self.correction_scale * envelope / (1 + envelope)
        # Hard load-conjugate displacement: f^T d / (f^T direction) = q * d0.
        projected = (correction @ physics.f) / (physics.direction @ physics.f)
        correction = correction - projected[:, None] * physics.direction
        d = self.control_scale * control[:, None] * physics.direction + correction
        return physics.expand(d)


def network_from_config(physics, law, config):
    return InterfaceNetwork(physics, law, config["width"], config["depth"],
                            config.get("fourier_frequencies", ()), config.get("envelope_power", 1.0),
                            config.get("local_rbf", False), config.get("local_control_centers", ())).to(physics.f.device)


def load_inputs(config, device):
    operator_path = (ROOT / config["operator_path"]).resolve()
    law_path = (ROOT / config["law_path"]).resolve()
    with operator_path.open("rb") as handle:
        record = OperatorReader(handle).load()
    values = json.loads(law_path.read_text())["by_beta"][str(config["beta"])]["theta_hat"]
    physics = InterfacePhysics(record).to(device)
    law = CohesiveLaw(values, cutoff_smoothing_fraction=config.get("cutoff_smoothing_fraction", 0.0)).to(device)
    provenance = {
        "operator_path": str(operator_path), "operator_sha256": sha256(operator_path),
        "law_path": str(law_path), "law_sha256": sha256(law_path), "law_values": values,
        "use_of_historical_results": "Fixed constitutive coefficients only; no solution targets or comparisons.",
    }
    if config.get("fixed_family_checkpoint"):
        source = ROOT / config["fixed_family_checkpoint"]
        saved = torch.load(source, map_location=device, weights_only=True)
        if (saved["config"]["beta"] != config["beta"] or
                saved["config"].get("cutoff_smoothing_fraction", 0.0) != law.cutoff_smoothing_fraction):
            raise ValueError("Fixed family law must match the orientation and cutoff closure")
        index = next(i for i, case in enumerate(saved["config"]["cases"])
                     if case["notch_depth_mm"] == config["notch_depth_mm"])
        for key in ("operator_sha256", "law_sha256", "law_values"):
            if saved["provenance"][index][key] != provenance[key]:
                raise ValueError(f"Fixed family input changed: {key}")
        with torch.no_grad():
            law.log_parameters.copy_(saved["shared_log_law"])
        provenance.update(fixed_family_checkpoint_sha256=sha256(source),
                          fixed_control_scale_mm=float(saved["models"][index]["network.control_scale"]),
                          fixed_law_values=[float(value) for value in law.physical()])
    return physics, law, provenance


def loss_terms(network, physics, law, control, normalization="power"):
    d = network(control, physics)
    load, residual, _, _ = physics.readout(d, law)
    tp, _, _, pr, _ = law.physical()
    load_scale = tp.detach() * physics.area.sum() / abs(physics.direction @ physics.f)
    # C-metric is positive definite on active rows. No mode is truncated.
    energy_scale = load_scale.square() * (physics.f @ physics.C @ physics.f)
    # Relative load-level normalization avoids hiding the early loading branch.
    if normalization == "traction":
        level_scale = (law.tau(control * network.control_scale) / tp).detach().clamp(min=1e-8)
    elif normalization == "power":
        level_scale = control.clamp(min=0.02).pow(pr.detach()).clamp(max=1)
    else:
        raise ValueError(f"Unknown residual normalization: {normalization}")
    white = residual @ physics.chol_C / energy_scale.sqrt() / level_scale[:, None]
    loss = white.square().sum(dim=1).mean()
    return loss, load


@torch.no_grad()
def evaluate(network, physics, law, control):
    d = network(control, physics)
    load, residual, gauge, rdisp = physics.readout(d, law)
    t = law.force(d, physics.area)[..., physics.active]
    applied = load[:, None] * physics.f
    force_scale = torch.maximum(torch.linalg.vector_norm(applied, dim=1),
                                torch.linalg.vector_norm(t, dim=1)).clamp(min=1e-15)
    relative = torch.linalg.vector_norm(residual, dim=1) / force_scale
    disp_scale = torch.linalg.vector_norm(d, dim=1).clamp(min=1e-15)
    control_error = abs((d[..., physics.active] @ physics.f) / (physics.direction @ physics.f)
                        - control * network.control_scale)
    return {
        "control": control.cpu().numpy(), "load_kN": load.cpu().numpy(),
        "force_relative": relative.cpu().numpy(),
        "displacement_relative": (torch.linalg.vector_norm(rdisp, dim=1) / disp_scale).cpu().numpy(),
        "control_error_mm": control_error.cpu().numpy(), "gauge_um": gauge.cpu().numpy(),
        "slip_mm": d.cpu().numpy(),
    }


def run(config_path: Path):
    config = json.loads(config_path.read_text())
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(config.get("cpu_threads", 4))
    torch.manual_seed(config["seed"])
    device = torch.device(config["device"])
    physics, law, provenance = load_inputs(config, device)
    network = network_from_config(physics, law, config)
    out = ROOT / config["output_directory"]
    if (out / "run_config.json").exists():
        raise FileExistsError(f"Preserve the existing run; choose a fresh output directory: {out}")
    if config.get("initial_checkpoint"):
        checkpoint_path = (ROOT / config["initial_checkpoint"]).resolve()
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        for key in ("operator_sha256", "law_sha256", "law_values"):
            if checkpoint["provenance"][key] != provenance[key]:
                raise ValueError(f"Checkpoint input changed: {key}")
        for key in ("beta", "notch_depth_mm", "width", "depth"):
            if checkpoint["config"][key] != config[key]:
                raise ValueError(f"Checkpoint architecture or case changed: {key}")
        for key, default in (("fourier_frequencies", []), ("envelope_power", 1.0)):
            if checkpoint["config"].get(key, default) != config.get(key, default):
                raise ValueError(f"Checkpoint representation changed: {key}")
        network.load_initial_state(checkpoint["network"])
        provenance["initial_checkpoint"] = str(checkpoint_path)
        provenance["initial_checkpoint_sha256"] = sha256(checkpoint_path)
        provenance["continuation"] = "Continue network weights with a fresh optimizer; not an exact optimizer resume."
    out.mkdir(parents=True, exist_ok=True)
    config_snapshot = dict(config)
    config_snapshot["source_sha256"] = sha256(Path(__file__))
    (out / "run_config.json").write_text(json.dumps(config_snapshot, indent=2) + "\n")
    qmax = config["max_control_delta0"]
    control = torch.tensor(np.unique(np.r_[np.geomspace(0.005, 0.1, 6),
                                            np.linspace(0.1, qmax, config["training_levels"])]), device=device)
    heldout = torch.linspace(0.0067, qmax - 0.0023, config["evaluation_levels"], device=device)
    start = time.perf_counter()
    history = []
    best = float("inf")
    best_state = None

    def record(phase, step, value):
        nonlocal best, best_state
        value = float(value)
        if not np.isfinite(value):
            raise FloatingPointError("Non-finite physics loss")
        if value < best:
            best = value
            best_state = {k: v.detach().cpu().clone() for k, v in network.state_dict().items()}
        row = {"phase": phase, "step": step, "loss": value, "seconds": time.perf_counter() - start,
               "stage_max_control_delta0": stage_max}
        history.append(row)
        if step % config["log_every"] == 0:
            check = evaluate(network, physics, law, control[::3])
            row["training_probe_force_relative_max"] = float(check["force_relative"].max())
            print(json.dumps(row), flush=True)

    initial = evaluate(network, physics, law, heldout)
    maxima = config.get("curriculum_maxima", [qmax])
    if maxima[-1] != qmax or maxima != sorted(set(maxima)):
        raise ValueError("Curriculum must increase to the full configured loading interval")
    for stage_max in maxima:
        control = torch.tensor(np.unique(np.r_[np.geomspace(0.005, 0.1, 6),
                         np.linspace(0.1, stage_max, config["training_levels"])]), device=device)
        # Losses over different loading intervals cannot rank the same checkpoint.
        best, best_state = float("inf"), None
        adam = torch.optim.Adam(network.parameters(), lr=config["learning_rate"])
        for step in range(config["adam_steps"]):
            adam.zero_grad()
            loss, _ = loss_terms(network, physics, law, control, config.get("normalization", "power"))
            loss.backward()
            if step % 20 == 0:
                record("adam", step, loss.detach())
            adam.step()
        lbfgs = torch.optim.LBFGS(network.parameters(), max_iter=20, history_size=50,
                                 tolerance_grad=1e-12, tolerance_change=1e-14,
                                 line_search_fn="strong_wolfe")
        for block in range(config["lbfgs_blocks"]):
            def closure():
                lbfgs.zero_grad()
                value, _ = loss_terms(network, physics, law, control, config.get("normalization", "power"))
                value.backward()
                return value
            lbfgs.step(closure)
            value, _ = loss_terms(network, physics, law, control, config.get("normalization", "power"))
            record("lbfgs", (block + 1) * 20, value.detach())
        network.load_state_dict(best_state)
    network.load_state_dict(best_state)
    result = evaluate(network, physics, law, heldout)
    train_result = evaluate(network, physics, law, control)
    peak = int(np.argmax(result["load_kN"]))
    interior_peak = 0 < peak < len(heldout) - 1
    # This is a pilot residual target, not a certified solution-error bound.
    residual_pass = bool(result["force_relative"].max() < config["pilot_residual_target"])
    metrics = {
        "kind": "condensed_interface_pinn_fixed_law_pilot", "beta": config["beta"],
        "notch_depth_mm": config["notch_depth_mm"], "provenance": provenance,
        "operator_checks": physics.operator_checks, "best_physics_loss": best,
        "initial_force_relative_max": float(initial["force_relative"].max()),
        "training_force_relative_max": float(train_result["force_relative"].max()),
        "evaluation_force_relative_max": float(result["force_relative"].max()),
        "evaluation_force_relative_median": float(np.median(result["force_relative"])),
        "evaluation_displacement_relative_max": float(result["displacement_relative"].max()),
        "hard_control_error_mm_max": float(result["control_error_mm"].max()),
        "sampled_max_load_kN": float(result["load_kN"][peak]),
        "sampled_max_control_delta0": float(result["control"][peak]),
        "interior_load_maximum": interior_peak,
        "pilot_residual_target": config["pilot_residual_target"],
        "pilot_residual_target_met": residual_pass,
        "status": "state_pilot_pass" if residual_pass else "state_pilot_needs_refinement",
        "elapsed_seconds": time.perf_counter() - start,
        "network_parameter_count": sum(p.numel() for p in network.parameters()),
        "training_levels": control.cpu().tolist(), "evaluation_levels": heldout.cpu().tolist(),
        "runtime": {"torch": torch.__version__, "device": str(device),
                    "dtype": str(torch.get_default_dtype())},
        "scope": "Fixed law, one geometry. No inverse fit, no certified limit point, no uncertainty study.",
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    (out / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    torch.save({"network": network.state_dict(), "law": law.state_dict(),
                "config": config_snapshot, "provenance": provenance}, out / "checkpoint.pt")
    np.savez_compressed(out / "fields.npz", **result, xz=physics.xz.cpu().numpy())
    with (out / "response.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["control_delta0", "load_kN", "force_relative", "displacement_relative"]
                        + [f"gauge_{key}_um" for key in physics.gauge_keys])
        for j in range(len(heldout)):
            writer.writerow([result[k][j] for k in ["control", "load_kN", "force_relative", "displacement_relative"]]
                            + result["gauge_um"][j].tolist())
    print(json.dumps({k: v for k, v in metrics.items() if k not in
                      {"provenance", "training_levels", "evaluation_levels", "operator_checks"}}, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    run(parser.parse_args().config)
