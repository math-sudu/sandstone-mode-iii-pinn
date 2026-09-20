"""Verified physics constants + the LEFM Mode-III SIF map for the ANBD geometry.

All values are read from results/data/e1_inputs/notch_plane_geometry.json (ground truth)
and the locked claim contract. Nothing here is fitted; these are fixed by the experiment.

Unit convention (documented, used consistently everywhere)
----------------------------------------------------------
- Force  P : kN
- Length   : mm internally for geometry (R, t, a); converted to m only inside the SIF.
- Stress / traction tau_III : MPa  (1 kN / mm^2 = 1000 MPa, handled explicitly below)
- Slip / separation delta_III : mm  (matches DIC CSD_W after um->mm)
- SIF K_III : MPa*sqrt(m)
- Shear modulus G : MPa  (= GPa * 1000)
- Cohesive energy G_IIIc : MPa*mm = kJ/m^2

SIF (Bahrami 2022 ANBD form, verified in the geometry JSON eq:sif_anbd):
    K_III = P / (R * sqrt(2*pi*t)) * Y_III(a/t)
With P in kN, R and t in m, the prefactor P/(R*sqrt(2*pi*t)) has units
kN/m^2 = kPa*... -> to land in MPa*sqrt(m) we use: K [MPa*sqrt(m)] =
    1e-3 * P[kN] / (R[m]*sqrt(2*pi*t[m])) * Y_III ,  (1e-3 converts kPa*sqrt(m)->MPa*sqrt(m)).
This reproduces the source kiiic_results.csv scale (verified numerically in physics tests).

Friction correction (K3 target, eq:kiiic_friction):
    K_IIIc_hat = K_IIIc * (1 - mu * |K_I*/K_III*|)
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

from .config import project_root

GEOM_JSON_REL = "results/data/e1_inputs/notch_plane_geometry.json"

# a/t key -> string used in the geometry JSON Y_III map.
A_OVER_T_BY_A_MM = {4: "0.32", 6: "0.48", 8: "0.64"}


@dataclass(frozen=True)
class Physics:
    E_GPa: float
    nu: float
    UCS_MPa: float
    R_mm: float
    t_mm: float
    alpha_deg: float
    notch_width_mm: float
    Y_III_by_aot: Dict[str, float]  # "0.32" -> 1.185, ...
    source_mu: float
    # Optional bedding-orientation-dependent geometry factor (TI sensitivity study):
    # {"0": {"0.32": Y, ...}, "45": {...}, "90": {...}}. None (the default) keeps the
    # bedding-independent Y_III_by_aot lookup and leaves every existing run unchanged.
    Y_III_beta_by_aot: Dict[str, Dict[str, float]] | None = None

    # ---- derived ----
    @property
    def G_MPa(self) -> float:
        """Shear modulus G = E/(2(1+nu)) in MPa."""
        return (self.E_GPa * 1000.0) / (2.0 * (1.0 + self.nu))

    @property
    def G_GPa(self) -> float:
        return self.E_GPa / (2.0 * (1.0 + self.nu))

    def y_III(self, a_mm: int | float, beta: int | None = None) -> float:
        """Geometry factor Y_III(a/t), optionally per bedding orientation beta.

        With Y_III_beta_by_aot set AND beta given, the per-beta TI value is returned
        (missing beta keys fail loudly). Otherwise the bedding-independent lookup.
        """
        key = A_OVER_T_BY_A_MM[int(round(a_mm))]
        if beta is not None and self.Y_III_beta_by_aot is not None:
            return float(self.Y_III_beta_by_aot[str(int(beta))][key])
        return float(self.Y_III_by_aot[key])

    def k_III(self, P_kN, a_mm, beta: int | None = None) -> float:
        """K_III [MPa*sqrt(m)] from load P [kN] for notch depth a [mm].

        Works for python floats or torch tensors (uses math via duck-typing on P).
        beta only matters when a per-beta Y_III override is loaded (TI sensitivity).
        """
        R_m = self.R_mm * 1e-3
        t_m = self.t_mm * 1e-3
        pref = 1e-3 / (R_m * math.sqrt(2.0 * math.pi * t_m))  # kN -> MPa*sqrt(m) factor folded in
        return P_kN * pref * self.y_III(a_mm, beta=beta)

    def sif_prefactor(self, a_mm, beta: int | None = None) -> float:
        """The geometry+unit constant c so that K_III = c * P_kN."""
        R_m = self.R_mm * 1e-3
        t_m = self.t_mm * 1e-3
        pref = 1e-3 / (R_m * math.sqrt(2.0 * math.pi * t_m))
        return pref * self.y_III(a_mm, beta=beta)


def load_physics(
    geom_json_path: str | Path | None = None,
    y_III_beta_by_aot: Dict[str, Dict[str, float]] | None = None,
) -> Physics:
    p = Path(geom_json_path) if geom_json_path else (project_root() / GEOM_JSON_REL)
    with open(p, "r", encoding="utf-8") as f:
        g = json.load(f)
    sg = g["specimen_geometry"]
    mat = g["material"]
    yiii = g["frictionless_FE_Y_III"]["by_a_over_t"]
    src_mu = float(g["source_friction_correction"]["source_mu"])
    return Physics(
        E_GPa=float(mat["E_GPa"]),
        nu=float(mat["nu"]),
        UCS_MPa=float(mat["UCS_MPa"]),
        R_mm=float(sg["disc_diameter_D_mm"]) / 2.0,
        t_mm=float(sg["half_thickness_t_mm"]),
        alpha_deg=float(sg["notch_inclination_alpha_deg"]),
        notch_width_mm=float(sg["notch_width_mm"]),
        Y_III_by_aot={str(k): float(v) for k, v in yiii.items()},
        source_mu=src_mu,
        Y_III_beta_by_aot=(
            {str(b): {str(k): float(v) for k, v in m.items()} for b, m in y_III_beta_by_aot.items()}
            if y_III_beta_by_aot is not None
            else None
        ),
    )
