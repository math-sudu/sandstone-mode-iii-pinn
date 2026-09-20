"""Eq. (cohesive_law) of the manuscript: single-peak traction-separation form.

tau = tau_p r^{p_r} e^{p_r(1-r)} on the rising branch (r = delta/delta_0),
tau = tau_p q^{p_s} e^{p_s(1-q)} on the softening branch
(q = (delta_c-delta)/(delta_c-delta_0)), zero traction beyond delta_c.

Shared by the figure scripts that reconstruct law curves from committed
parameter sets (make_f3_twin_recovery.py, make_f4_identified_laws.py).
"""
from __future__ import annotations

import numpy as np


def law(delta: np.ndarray, tau_p: float, d0: float, dc: float,
        p_r: float, p_s: float) -> np.ndarray:
    delta = np.asarray(delta, dtype=float)
    tau = np.zeros_like(delta)
    rise = (delta >= 0) & (delta <= d0)
    r = np.divide(delta, d0, out=np.zeros_like(delta), where=d0 > 0)
    tau[rise] = tau_p * r[rise] ** p_r * np.exp(p_r * (1.0 - r[rise]))
    soft = (delta > d0) & (delta <= dc)
    q = np.divide(dc - delta, dc - d0, out=np.zeros_like(delta),
                  where=(dc - d0) > 0)
    tau[soft] = tau_p * q[soft] ** p_s * np.exp(p_s * (1.0 - q[soft]))
    return tau


def theta_curve(delta: np.ndarray, th: dict, cutoff_smoothing_fraction: float = 0.0) -> np.ndarray:
    """Evaluate the law from a parameter dict using the artifact key names."""
    traction = law(delta, th["tau_p_MPa"], th["delta_0_mm"], th["delta_c_mm"],
                   th["p_r"], th["p_s"])
    if cutoff_smoothing_fraction:
        t = np.clip((th["delta_c_mm"] - np.asarray(delta)) /
                    (cutoff_smoothing_fraction * (th["delta_c_mm"] - th["delta_0_mm"])), 0, 1)
        traction *= t**3 * (10 - 15*t + 6*t**2)
    return traction
