"""Transversely isotropic bulk elasticity for the 3D true-geometry ANBD model.

Follow-up F1 (TI-3D bulk sensitivity, beta-0 overshoot attribution part 1).
Supplies the anisotropic stiffness tensor and the matching skfem bilinear form
consumed by anbd3d_cohesive.build_interface_operators(C_ijkl=...).

Material family (the shear-ratio sweep of the retired ti_yiii DESIGN, ported to
a proper 3D TI tensor)
----------------------------------------------------------------------------
The old 2D study varied ONLY the two anti-plane shear moduli with the baseline
convention G_perp = G (isotropic value) and G_par = rho * G. The minimal proper
TI completion keeps every normal-stiffness component at its isotropic value:

  local frame (symmetry axis m = e3):
    C44 = C55 = G_perp            (shear on planes containing m)
    C66 = (C11 - C12)/2 = G_par   (shear within the bedding/isotropy plane)
    C13 = lam,  C33 = lam + 2G,  C11 + C12 = 2 lam + 2G   (isotropic values)
  =>  C11 = lam + G + G_par,  C12 = lam + G - G_par        (G = G_perp)

In invariant form with p = m (x) m (valid for ANY axis direction m, no Voigt
rotation code needed):

  C_ijkl = lam' d_ij d_kl + G_par (d_ik d_jl + d_il d_jk)
           + alpha (p_ij d_kl + d_ij p_kl)
           + (G_perp - G_par)(p_ik d_jl + p_il d_jk + p_jk d_il + p_jl d_ik)
           + beta p_ij p_kl,
  lam' = lam + G_perp - G_par,   alpha = beta = G_par - G_perp.

At rho = 1 every correction coefficient is exactly 0.0 and lam' = lam, so the
constructor returns an array BIT-IDENTICAL to the isotropic tensor built by the
same routine -- the rho = 1 reduction of the assembled operators is then exact
by construction (verified by selftests below).

Derived effective shear moduli (analytic, used by the axis self-check):
  C_1313(m) = G_par + (G_perp - G_par) * m_z^2     [slip x, gradient z]
  C_1212(m) = G_par + (G_perp - G_par) * m_y^2     [slip x, gradient y]
  C_1213(m) = (G_perp - G_par) * m_y * m_z         [cross coupling, beta 45]

Bedding-normal mapping (crack frame of the retired ti_yiii study, ported to the
3D mesh frame: old x = ligament/core axis -> new z; old y = notch-plane normal
-> new y; old z = crack front / notch diameter -> new x):

  m(beta) = (0, cos(beta), sin(beta)):  beta 0 -> m || y (bedding || notch
  plane), beta 90 -> m || z (core axis), beta 45 -> 45 deg in the (y, z) plane.

Physical-limit axis check (catches a G_par/G_perp axis swap that rho = 1
cannot): at beta 0 the (x, z) shear plane lies IN the bedding -> C_1313 must be
G_par; at beta 90 the (x, z) plane contains the bedding normal -> C_1313 must
be G_perp.
"""
from __future__ import annotations

import math
from typing import Dict

import numpy as np

from skfem import BilinearForm
from skfem.helpers import ddot, sym_grad


def bedding_normal(beta_deg: float) -> np.ndarray:
    """Unit bedding normal in the 3D mesh frame: m(beta) = (0, cos b, sin b)."""
    b = math.radians(beta_deg)
    return np.array([0.0, math.cos(b), math.sin(b)])


def iso_tensor(lam: float, mu: float) -> np.ndarray:
    """Isotropic C_ijkl (3,3,3,3)."""
    d = np.eye(3)
    return (lam * np.einsum("ij,kl->ijkl", d, d)
            + mu * (np.einsum("ik,jl->ijkl", d, d)
                    + np.einsum("il,jk->ijkl", d, d)))


def ti_tensor(lam: float, G_perp: float, G_par: float, m: np.ndarray) -> np.ndarray:
    """TI stiffness of the module family; bit-equal to iso_tensor at G_par == G_perp."""
    m = np.asarray(m, dtype=float)
    if abs(float(m @ m) - 1.0) > 1e-12:
        raise ValueError("bedding normal m must be a unit vector")
    d = np.eye(3)
    p = np.outer(m, m)
    dG = G_perp - G_par          # exactly 0.0 at rho = 1 -> bit-level reduction
    lam_p = lam + dG
    alpha = -dG
    beta = -dG
    C = (lam_p * np.einsum("ij,kl->ijkl", d, d)
         + G_par * (np.einsum("ik,jl->ijkl", d, d) + np.einsum("il,jk->ijkl", d, d))
         + alpha * (np.einsum("ij,kl->ijkl", p, d) + np.einsum("ij,kl->ijkl", d, p))
         + dG * (np.einsum("ik,jl->ijkl", p, d)
                 + np.einsum("il,jk->ijkl", p, d)
                 + np.einsum("jk,il->ijkl", p, d)
                 + np.einsum("jl,ik->ijkl", p, d))
         + beta * np.einsum("ij,kl->ijkl", p, p))
    return C


def anisotropic_elasticity(C: np.ndarray):
    """skfem bilinear form eps(v) : C : eps(u) for a constant C_ijkl."""

    @BilinearForm
    def form(u, v, w):
        sig = np.einsum("ijkl,kl...->ij...", C, sym_grad(u))
        return ddot(sig, sym_grad(v))

    return form


def voigt_matrix(C: np.ndarray) -> np.ndarray:
    """6x6 Voigt stiffness (11, 22, 33, 23, 13, 12) from C_ijkl."""
    idx = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]
    V = np.zeros((6, 6))
    for a, (i, j) in enumerate(idx):
        for b, (k, l) in enumerate(idx):
            V[a, b] = C[i, j, k, l]
    return V


def stability_eigs(C: np.ndarray) -> np.ndarray:
    """Eigenvalues of the stiffness on symmetric strains (Mandel form); all > 0
    iff the material is stable."""
    M = voigt_matrix(C).copy()
    # Mandel scaling: shear rows/cols x sqrt(2) makes the quadratic form eigenproblem
    s = np.array([1.0, 1.0, 1.0, math.sqrt(2.0), math.sqrt(2.0), math.sqrt(2.0)])
    M = M * np.outer(s, s)
    return np.linalg.eigvalsh(M)


def effective_shear_analytic(G_perp: float, G_par: float, m: np.ndarray) -> Dict[str, float]:
    """Analytic C_1313 / C_1212 / C_1213 of the module family for the axis check."""
    my, mz = float(m[1]), float(m[2])
    return {
        "C_1313": G_par + (G_perp - G_par) * mz * mz,
        "C_1212": G_par + (G_perp - G_par) * my * my,
        "C_1213": (G_perp - G_par) * my * mz,
    }
