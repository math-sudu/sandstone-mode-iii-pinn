"""Cohesive Mode-III interface solver on the condensed 3D true-geometry operators.

Method core of the rebuilt identification (user-approved M1-M4 plan). The bulk is
the probe-verified 3D elastic ANBD model (anbd3d.py; verification chain in
results/runs/anbd3d_probe/probe.json); the nonlinearity lives ONLY on the ligament
interface, condensed to an exact dense influence system so a forward solve costs
milliseconds and the identification/bootstrap loops stay cheap.

Formulation (exact, no soft constraints)
----------------------------------------
Split-ligament mesh: the plane y = 0 carries duplicate node pairs. Bulk stiffness
K_b (two half-bodies) is regularised by a REFERENCE interface stiffness into
    K_ref = K_b + B^T D_ref B,      D_ref = diag(k_n A_p on the normal comp,
                                               k_t A_p on the two tangential comps),
which is factorised once. With t(delta) the actual interface nodal force
(cohesive tangential + normal penalty), the equilibrium K_b u = f - B^T t(delta)
rewrites EXACTLY as K_ref u = f + B^T (D_ref B u - t), so with the normal law
CHOSEN equal to its reference entry (t_n = k_n A delta_n) the normal terms cancel
and the condensed unknown is the tangential jump vector d = B_t u:

    d = P g_t + C_tt (k_t A d - t_t(d)),      C_tt = B_t K_ref^{-1} B_t^T,
    gauge = P g_w + W (k_t A d - t_t(d)),     W = G_row K_ref^{-1} B_t^T.

Newton with an analytic tangent and a backtracking line search. The cohesive law
is the paper's single-peak effective slip-weakening form (Eq.(5)), colinear with
the tangential slip: t_t = A tau(|d|) d/|d|.

Normal behaviour: the ligament stays CLOSED (coplanar shear failure, source C1;
front-averaged K_I ~ 0 at alpha = 10 deg per the probe and Bahrami2022 Table 1),
enforced by the stiff normal penalty k_n. sigma_n on the interface is recovered
post hoc for the (pre-declared) sigma_n-coupled friction escalation variant.

Peak load: under load control the specimen fails at the limit point of the
P -> equilibrium family. The solver scans P upward with warm starts (tracking the
stable rising branch), brackets the first non-convergence/turnover, and bisects.

Units: mm, N internally at the interface (tractions MPa x areas mm^2); P in kN at
the API. Gauge values in um. Deterministic; no RNG.

Physics gates (run via --gates):
  R0 bonded limit: a near-rigid law reproduces the bonded probe gauge response;
  R1 LEFM limit: a brittle law fails at K_max(P_peak) ~ sqrt(2 G G_c), and
     P_peak(a) ratios follow the inverse K-per-kN ratios across notch depths;
  R2 net-section limit: a plateau law fails at P ~ 4 R (t-a) tau_p / cos(alpha).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu

from skfem import Basis, ElementTetP1, ElementVector, asm
from skfem.models.elasticity import lame_parameters, linear_elasticity

from .anbd3d import (
    AnbdMesh,
    GAUGE_DXN,
    GAUGE_MIN_PTS,
    GAUGE_YN_MAX,
    GAUGE_YN_MIN,
    assemble_diametral_load,
    build_anbd_mesh,
)
from .physics import Physics, load_physics

K_NORMAL_MPA_MM = 5.0e4   # normal closure penalty (sigma_n ~ MPa -> delta_n ~ 1e-4 mm)
K_TANG_REF_MPA_MM = 6.0e3  # reference tangential stiffness (conditioning only; exact)


# ------------------------------------------------------------------------------------
# Cohesive law (paper Eq.(5); adapted from forward_fullfield.LawIII, which is retired
# with its wrong-geometry module -- the law itself is geometry-free)
# ------------------------------------------------------------------------------------


@dataclass(frozen=True)
class LawIII:
    """Single-peak effective slip-weakening law.

    tau_p [MPa] peak resistance; delta_0 [mm] slip at peak; delta_c [mm] critical
    slip (traction-free beyond); p_r, p_s rising/softening exponents.
        r = d/delta_0 <= 1:  tau = tau_p r^{p_r} e^{p_r(1-r)}
        q = (delta_c-d)/(delta_c-delta_0): tau = tau_p q^{p_s} e^{p_s(1-q)}
    """

    tau_p: float
    delta_0: float
    delta_c: float
    p_r: float = 1.0
    p_s: float = 1.0

    def tau(self, d: np.ndarray) -> np.ndarray:
        d = np.asarray(d, dtype=float)
        out = np.zeros_like(d)
        d0, dc = self.delta_0, self.delta_c
        rising = (d > 0.0) & (d <= d0)
        soft = (d > d0) & (d < dc)
        r = np.clip(d[rising] / d0, 1e-300, None)
        out[rising] = self.tau_p * r ** self.p_r * np.exp(self.p_r * (1.0 - r))
        q = np.clip((dc - d[soft]) / (dc - d0), 1e-300, None)
        out[soft] = self.tau_p * q ** self.p_s * np.exp(self.p_s * (1.0 - q))
        return out

    def dtau(self, d: np.ndarray) -> np.ndarray:
        d = np.asarray(d, dtype=float)
        out = np.zeros_like(d)
        d0, dc = self.delta_0, self.delta_c
        rising = (d >= 0.0) & (d <= d0)
        soft = (d > d0) & (d < dc)
        r = np.clip(d[rising] / d0, 1e-300, None)
        out[rising] = (self.tau_p * self.p_r * r ** (self.p_r - 1.0)
                       * np.exp(self.p_r * (1.0 - r)) * (1.0 - r) / d0)
        q = np.clip((dc - d[soft]) / (dc - d0), 1e-300, None)
        out[soft] = (self.tau_p * self.p_s * q ** (self.p_s - 1.0)
                     * np.exp(self.p_s * (1.0 - q)) * (1.0 - q) * (-1.0 / (dc - d0)))
        return out

    def secant(self, s: np.ndarray, s_reg: float) -> np.ndarray:
        """tau(s)/s with the small-slip regularisation tau'(s_reg) as the s->0 limit."""
        s = np.asarray(s, dtype=float)
        sr = np.maximum(s, s_reg)
        return self.tau(sr) / sr

    def g_c(self) -> float:
        """Fracture energy integral of the law [MPa*mm] (numeric, fine grid)."""
        d = np.linspace(0.0, self.delta_c, 20001)
        return float(np.trapezoid(self.tau(d), d))


# ------------------------------------------------------------------------------------
# Condensed interface operators (built once per notch depth)
# ------------------------------------------------------------------------------------


def _gauge_row(md: AnbdMesh, basis: Basis, station_mm: float, comp: int) -> np.ndarray:
    """Strip-gauge difference row (mean over A strip minus mean over B strip)."""
    p = md.mesh.p
    nd = basis.nodal_dofs
    face = np.abs(p[2] - md.t_mm) < 1e-9
    ins = np.abs(p[0] - station_mm) < GAUGE_DXN
    A = face & ins & (p[1] > GAUGE_YN_MIN) & (p[1] < GAUGE_YN_MAX)
    B = face & ins & (p[1] < -GAUGE_YN_MIN) & (p[1] > -GAUGE_YN_MAX)
    if int(A.sum()) < GAUGE_MIN_PTS or int(B.sum()) < GAUGE_MIN_PTS:
        raise RuntimeError(f"gauge strip too sparse at station {station_mm}")
    row = np.zeros(basis.N)
    row[nd[comp, A]] = 1.0 / int(A.sum())
    row[nd[comp, B]] = -1.0 / int(B.sum())
    return row


@dataclass
class InterfaceOperators:
    """Exact condensed operators for one notch depth (see module docstring)."""

    a_mm: float
    scale: float
    n_pairs: int
    xz: np.ndarray            # (n_p, 2) interface pair coordinates
    area: np.ndarray          # (n_p,) tributary areas [mm^2]
    g_t: np.ndarray           # (2 n_p,) tangential jump per kN (bonded ref system)
    C_tt: np.ndarray          # (2 n_p, 2 n_p) influence matrix [mm/N]
    kref_force: np.ndarray    # (2 n_p,) D_ref tangential diagonal [N/mm]
    gauge_g: Dict[str, float]         # station -> per-kN gauge of the ref system [mm/kN]
    gauge_W: Dict[str, np.ndarray]    # station -> (2 n_p,) correction row [mm/N]
    cod_g: Dict[str, float]
    cod_W: Dict[str, np.ndarray]
    checks: Dict[str, float]
    # sigma_n recovery rows (build with sigma_n_rows=True; follow-up F3). None by
    # default and absent from pre-F3 pickles -- consume via getattr(ops, "g_n", None).
    g_n: Optional[np.ndarray] = None      # (n_p,) normal jump per kN (ref system)
    C_nt: Optional[np.ndarray] = None     # (n_p, 2 n_p) normal-jump influence [mm/N]
    k_n_used: Optional[float] = None      # normal penalty the build used [MPa/mm]
    # dense end-face W readout rows (build with surface_rows=True; S0.2 field
    # channel). Same influence-row construction as gauge_W, evaluated at every
    # node of the notched end face z = t: W(node) = P surf_g + surf_W q, in mm
    # (x1000 for um), with q = D_ref,t d - t_t(d) as in _pack. None by default
    # and absent from pre-S0.2 pickles -- consume via getattr(ops, "surf_W", None).
    surf_xy: Optional[np.ndarray] = None  # (n_s, 2) face-node (x, y) [mm]
    surf_g: Optional[np.ndarray] = None   # (n_s,) face u_z per kN (ref system)
    surf_W: Optional[np.ndarray] = None   # (n_s, 2 n_p) correction rows [mm/N]

    def tangential(self, vec_pairs_x: np.ndarray, vec_pairs_z: np.ndarray) -> np.ndarray:
        return np.concatenate([vec_pairs_x, vec_pairs_z])


def build_interface_operators(a_mm: float, scale: float = 1.0,
                              physics: Optional[Physics] = None,
                              stations: Tuple[float, ...] = (15.7, -15.7),
                              k_n: float = K_NORMAL_MPA_MM,
                              k_t: float = K_TANG_REF_MPA_MM,
                              C_ijkl: Optional[np.ndarray] = None,
                              mesh_kwargs: Optional[Dict] = None,
                              sigma_n_rows: bool = False,
                              station_map: Optional[Dict[str, float]] = None,
                              surface_rows: bool = False
                              ) -> InterfaceOperators:
    """C_ijkl (3,3,3,3), when given, replaces the isotropic bulk stiffness
    (TI sensitivity, follow-up F1); everything downstream is unchanged.
    mesh_kwargs (kerf_w_mm, corner_refine; follow-up F2) passes through to
    build_anbd_mesh -- the kerf variant carves a real slot so no slit split
    exists, while the ligament interface discretisation is shared with the
    sharp mesh. sigma_n_rows=True additionally stores the normal-jump
    recovery rows g_n and C_nt (projections of the SAME backsolve columns,
    so sigma_n = k_n * (P g_n + C_nt q_t) is exact for the condensed state;
    follow-up F3 sigma_n-coupled friction); the default False leaves the
    operators bit-identical to the committed identification's.
    station_map {label: signed x-position mm}, when given, replaces the
    nominal `stations` tuple: the gauge rows are keyed by the dict labels
    (e.g. specimen ids) and evaluated at the dict positions, so distinct
    sub-0.1 mm stations do not collide under the "%+.1f" nominal labelling
    (per-specimen ext_xn_mm gauges, Phase 0.7 R4). station_map=None keeps the
    committed +/-15.7 mm labelled behaviour bit-identical.
    surface_rows=True additionally stores the dense end-face W readout rows
    surf_xy/surf_g/surf_W (u_z of every node on the notched face z = t,
    projections of the SAME per-kN solve and backsolve columns the gauge rows
    are built from, so a strip average of the dense readout reproduces the
    gauge rows exactly; S0.2 full-field channel); the default False leaves
    the operators bit-identical to the committed identification's."""
    ph = physics if physics is not None else load_physics()
    md = build_anbd_mesh(a_mm, ph.R_mm, ph.t_mm, scale=scale, split_ligament=True,
                        **(mesh_kwargs or {}))
    basis = Basis(md.mesh, ElementVector(ElementTetP1()))
    if C_ijkl is None:
        E = ph.E_GPa * 1000.0
        lam, mu = lame_parameters(E, ph.nu)
        Kb = asm(linear_elasticity(lam, mu), basis)
    else:
        from .ti_elasticity import anisotropic_elasticity
        Kb = asm(anisotropic_elasticity(C_ijkl), basis)

    pairs = md.lig_pairs
    area = md.lig_area
    n_p = len(pairs)
    nd = basis.nodal_dofs
    N = basis.N

    # jump operator rows: [x-jumps (n_p), z-jumps (n_p), y-jumps (n_p)]
    def _B_comp(comp: int) -> sp.csr_matrix:
        rows = np.repeat(np.arange(n_p), 2)
        cols = np.column_stack([nd[comp, pairs[:, 1]], nd[comp, pairs[:, 0]]]).ravel()
        vals = np.tile([1.0, -1.0], n_p)
        return sp.csr_matrix((vals, (rows, cols)), shape=(n_p, N))

    Bx, Bz, By = _B_comp(0), _B_comp(2), _B_comp(1)
    B_all = sp.vstack([Bx, Bz, By]).tocsr()
    d_ref = np.concatenate([k_t * area, k_t * area, k_n * area])  # [N/mm]
    K_ref = (Kb + B_all.T @ sp.diags(d_ref) @ B_all).tocsc()

    # Dirichlet: u_z on z=0 (both bodies) + 3 pins for the common rigid modes
    p = md.mesh.p
    z0 = np.where(np.abs(p[2]) < 1e-9)[0]
    dofs = [nd[2, z0]]
    i_origin = z0[int(np.argmin(p[0, z0] ** 2 + p[1, z0] ** 2))]
    dofs.append(nd[0:2, i_origin].ravel())
    i_far = z0[int(np.argmax(np.abs(p[1, z0])))]
    dofs.append(nd[0:1, i_far].ravel())
    D = np.unique(np.concatenate(dofs))
    I = np.setdiff1d(np.arange(N), D)

    lu = splu(K_ref[I][:, I].tocsc())

    def ksolve(rhs: np.ndarray) -> np.ndarray:
        u = np.zeros(N)
        u[I] = lu.solve(rhs[I])
        return u

    f1, load_checks = assemble_diametral_load(md, basis, ph.alpha_deg, P_kN=1.0)
    u_f = ksolve(f1)
    B_t = sp.vstack([Bx, Bz]).tocsr()
    g_t = B_t @ u_f
    g_n = (By @ u_f) if sigma_n_rows else None

    # dense end-face W readout node set (opt-in, S0.2 field channel): u_z at
    # every node of the notched face z = t -- the same face the strip gauges
    # live on -- read from the same per-kN solve / backsolve columns below
    if surface_rows:
        surf_nodes = np.where(np.abs(p[2] - md.t_mm) < 1e-9)[0]
        surf_dofs = nd[2, surf_nodes]
        surf_xy = p[0:2, surf_nodes].T.copy()
        surf_g_v = u_f[surf_dofs].copy()
    else:
        surf_dofs = surf_xy = surf_g_v = None

    # gauge rows (CSD_W on u_z, COD on u_y). Default: the nominal +/-15.7 mm
    # stations labelled "%+.1f" (bit-identical to the committed identification).
    # A station_map {label: signed x mm} places each gauge at its own recorded
    # position under a collision-free label (per-specimen ext_xn_mm, R4).
    station_items = (list(station_map.items()) if station_map is not None
                     else [(f"{s:+.1f}", float(s)) for s in stations])
    g_rows_csd = {k: _gauge_row(md, basis, xn, comp=2) for k, xn in station_items}
    g_rows_cod = {k: _gauge_row(md, basis, xn, comp=1) for k, xn in station_items}
    gauge_g = {k: float(r @ u_f) for k, r in g_rows_csd.items()}
    cod_g = {k: float(r @ u_f) for k, r in g_rows_cod.items()}

    # influence columns: one backsolve per tangential interface dof
    n_t = 2 * n_p
    C = np.zeros((n_t, n_t))
    Cn = np.zeros((n_p, n_t)) if sigma_n_rows else None
    S = np.zeros((len(surf_nodes), n_t)) if surface_rows else None
    Wc = {k: np.zeros(n_t) for k in g_rows_csd}
    Wd = {k: np.zeros(n_t) for k in g_rows_cod}
    Bt_T = B_t.T.tocsc()
    for j in range(n_t):
        col = ksolve(np.asarray(Bt_T[:, j].todense()).ravel())
        C[:, j] = B_t @ col
        if sigma_n_rows:
            Cn[:, j] = By @ col
        if surface_rows:
            S[:, j] = col[surf_dofs]
        for k in g_rows_csd:
            Wc[k][j] = float(g_rows_csd[k] @ col)
            Wd[k][j] = float(g_rows_cod[k] @ col)

    kref_force = np.concatenate([k_t * area, k_t * area])
    checks = dict(load_checks)
    checks["n_nodes"] = md.stats["n_nodes"]
    checks["C_symmetry_rel"] = float(
        np.max(np.abs(C - C.T)) / (np.max(np.abs(C)) + 1e-30)
    )
    return InterfaceOperators(
        a_mm=float(a_mm), scale=float(scale), n_pairs=n_p, xz=md.lig_xz.copy(),
        area=area.copy(), g_t=g_t, C_tt=C, kref_force=kref_force,
        gauge_g=gauge_g, gauge_W=Wc, cod_g=cod_g, cod_W=Wd, checks=checks,
        g_n=g_n, C_nt=Cn, k_n_used=(float(k_n) if sigma_n_rows else None),
        surf_xy=surf_xy, surf_g=surf_g_v, surf_W=S,
    )


# ------------------------------------------------------------------------------------
# Interface Newton solver
# ------------------------------------------------------------------------------------


@dataclass
class InterfaceSolution:
    P_kN: float
    d_t: np.ndarray            # (2 n_p,) tangential jumps [mm]
    slip: np.ndarray           # (n_p,) |slip| [mm]
    converged: bool
    n_iter: int
    gauge_um: Dict[str, float]
    cod_um: Dict[str, float]
    max_slip: float


class CohesiveInterfaceSolver:
    """Newton solver on the condensed tangential-interface system."""

    def __init__(self, ops: InterfaceOperators, law: LawIII, s_reg_frac: float = 1e-6,
                 tangent_floor_frac: float = 3e-2):
        self.ops = ops
        self.law = law
        self.s_reg = max(s_reg_frac * law.delta_0, 1e-12)
        self.tangent_floor_frac = float(tangent_floor_frac)

    # interface force vector t_t(d) [N] and its (2x2-block) tangent [N/mm]
    def _force(self, d: np.ndarray) -> np.ndarray:
        n_p = self.ops.n_pairs
        dx, dz = d[:n_p], d[n_p:]
        s = np.hypot(dx, dz)
        sec = self.law.secant(s, self.s_reg)  # tau/s [MPa/mm]
        return np.concatenate([self.ops.area * sec * dx, self.ops.area * sec * dz])

    def _force_and_tangent(self, d: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Interface force (exact) and a FLOORED tangent for the Newton matrix.

        When the tangential stiffness vanishes over the WHOLE interface (a
        p_r > 1 law at zero slip, or a fully exhausted interface), the exact
        Newton matrix I - C (D_ref - T) is singular along the two half-bodies'
        relative rigid slide modes (identity: C D_ref r = r for those modes).
        The diagonal tangent blocks are therefore floored at a small fraction
        of the reference stiffness -- an inexact-Newton modification of J only;
        the residual/force stays exact, so converged solutions are unbiased.
        """
        n_p = self.ops.n_pairs
        dx, dz = d[:n_p], d[n_p:]
        s = np.hypot(dx, dz)
        sr = np.maximum(s, self.s_reg)
        sec = self.law.tau(sr) / sr           # tau/s
        dt = self.law.dtau(sr)                # tau'
        t = np.concatenate([self.ops.area * sec * dx, self.ops.area * sec * dz])
        # tangent blocks: A [ sec I + (dt - sec) e e^T ], e = d/s
        ex, ez = dx / sr, dz / sr
        c = (dt - sec)
        A = self.ops.area
        # floor on the LAW's own stiffness scale (tau_p/delta_c): large enough to
        # de-singularise the relative-rigid-slide modes (with the step cap doing
        # the heavy lifting), small enough not to distort Newton directions for
        # soft interfaces (a k_ref-scaled floor broke the net-section gate)
        floor = self.tangent_floor_frac * self.law.tau_p / self.law.delta_c
        Txx = A * np.maximum(sec + c * ex * ex, floor)
        Tzz = A * np.maximum(sec + c * ez * ez, floor)
        Txz = A * (c * ex * ez)
        return t, np.concatenate([Txx, Tzz, Txz])  # packed blocks

    def _residual(self, d: np.ndarray, P_kN: float) -> np.ndarray:
        q = self.ops.kref_force * d - self._force(d)
        return d - P_kN * self.ops.g_t - self.ops.C_tt @ q

    def solve(self, P_kN: float, d0: Optional[np.ndarray] = None,
              max_iter: int = 60, tol: float = 3e-8) -> InterfaceSolution:
        n_p = self.ops.n_pairs
        n_t = 2 * n_p
        d = np.zeros(n_t) if d0 is None else d0.copy()
        scale_ref = max(abs(P_kN) * float(np.linalg.norm(self.ops.g_t)), 1e-14)
        F = self._residual(d, P_kN)
        res = np.linalg.norm(F) / scale_ref
        it = 0
        for it in range(1, max_iter + 1):
            if res < tol:
                break
            t, blocks = self._force_and_tangent(d)
            Txx, Tzz, Txz = blocks[:n_p], blocks[n_p:2 * n_p], blocks[2 * n_p:]
            # J = I - C (diag(kref) - T). M = diag(kref)-T has <=2 nonzeros per
            # column (the diagonal and the x<->z partner), so C@M is two column
            # scalings of C -- O(n^2), not a dense matmul.
            m_diag = np.concatenate([self.ops.kref_force[:n_p] - Txx,
                                     self.ops.kref_force[n_p:] - Tzz])
            m_off = np.concatenate([-Txz, -Txz])  # M[j+/-n_p, j]
            C = self.ops.C_tt
            CM = C * m_diag[None, :]
            CM[:, :n_p] += C[:, n_p:] * m_off[None, :n_p]
            CM[:, n_p:] += C[:, :n_p] * m_off[None, n_p:]
            J = -CM
            J[np.arange(n_t), np.arange(n_t)] += 1.0
            try:
                step = np.linalg.solve(J, -F)
            except np.linalg.LinAlgError:
                return self._pack(d, P_kN, False, it)
            if not np.all(np.isfinite(step)):
                return self._pack(d, P_kN, False, it)
            # physical step cap: one critical slip per iteration beyond the
            # current state (kills the relative-rigid-slide blow-up direction)
            cap = self.law.delta_c + float(np.max(np.abs(d))) if d.size else 1.0
            mx = float(np.max(np.abs(step)))
            if mx > cap:
                step *= cap / mx
            alpha, ok = 1.0, False
            for _ in range(25):
                d_try = d + alpha * step
                F_try = self._residual(d_try, P_kN)
                r_try = np.linalg.norm(F_try) / scale_ref
                if np.isfinite(r_try) and r_try < (1.0 - 1e-4 * alpha) * res:
                    d, F, res, ok = d_try, F_try, r_try, True
                    break
                alpha *= 0.5
            if not ok:
                return self._pack(d, P_kN, res < 1e-6, it)
        return self._pack(d, P_kN, res < 1e-6, it)

    def _pack(self, d: np.ndarray, P_kN: float, conv: bool, it: int) -> InterfaceSolution:
        n_p = self.ops.n_pairs
        q = self.ops.kref_force * d - self._force(d)
        gauge = {k: (P_kN * self.ops.gauge_g[k] + float(self.ops.gauge_W[k] @ q)) * 1000.0
                 for k in self.ops.gauge_g}
        cod = {k: (P_kN * self.ops.cod_g[k] + float(self.ops.cod_W[k] @ q)) * 1000.0
               for k in self.ops.cod_g}
        slip = np.hypot(d[:n_p], d[n_p:])
        return InterfaceSolution(P_kN=float(P_kN), d_t=d, slip=slip, converged=bool(conv),
                                 n_iter=it, gauge_um=gauge, cod_um=cod,
                                 max_slip=float(slip.max() if len(slip) else 0.0))

    # -- load schedules ---------------------------------------------------------

    def curve(self, P_levels: np.ndarray, n_sub: int = 1,
              n_ramp: int = 8) -> List[InterfaceSolution]:
        """Warm-started continuation through a load schedule (rising branch).

        The first level is reached through an n_ramp cold-start ramp; subsequent
        (nearby) levels use n_sub warm sub-steps each.
        """
        out: List[InterfaceSolution] = []
        d = None
        P_prev = 0.0
        for P in np.asarray(P_levels, dtype=float):
            sub = np.linspace(P_prev, P, n_sub + 1)[1:] if d is not None else \
                np.linspace(0.0, P, n_ramp + 1)[1:]
            sol = None
            for Ps in sub:
                sol = self.solve(float(Ps), d0=d)
                if not sol.converged:
                    break
                d = sol.d_t
            out.append(sol)
            if sol is None or not sol.converged:
                break
            P_prev = P
        return out

    def peak_load(self, P_lo: float = 0.05, P_hi: float = 30.0, n_scan: int = 60,
                  tol_kN: Optional[float] = None) -> Dict[str, float]:
        """Limit-point failure load under load control (scan + bisection)."""
        Ps = np.linspace(P_lo, P_hi, n_scan)
        d = None
        prev = None  # (P, max_slip, d)
        bracket = None
        for P in Ps:
            sol = self.solve(float(P), d0=d)
            rising = sol.converged and np.isfinite(sol.max_slip) and (
                prev is None or sol.max_slip >= prev[1] - 1e-12
            )
            if not rising:
                bracket = (prev[0] if prev else P_lo, float(P))
                break
            prev = (float(P), sol.max_slip, sol.d_t.copy())
            d = sol.d_t
        if bracket is None:
            return {"P_peak_kN": float("nan"), "found": False,
                    "max_slip_at_top": prev[1] if prev else float("nan"),
                    "P_hi_scanned": float(P_hi)}
        if prev is None:
            # the FIRST scan point already failed: no equilibrium was ever seen,
            # so there is no bracketed limit point -- report failure, never a
            # fake peak at P_lo
            return {"P_peak_kN": float("nan"), "found": False,
                    "first_scan_point_failed": True, "P_lo": float(P_lo)}
        lo, hi = bracket
        if tol_kN is None:
            tol_kN = max(5e-3, 5e-3 * hi)  # 0.5% of the bracket top
        d_lo = prev[2] if prev else None
        s_lo = prev[1] if prev else 0.0
        while hi - lo > tol_kN:
            mid = 0.5 * (lo + hi)
            sol = self.solve(mid, d0=d_lo)
            if sol.converged and sol.max_slip >= s_lo - 1e-12:
                lo, s_lo, d_lo = mid, sol.max_slip, sol.d_t.copy()
            else:
                hi = mid
        return {"P_peak_kN": float(lo), "found": True, "max_slip_at_peak_mm": float(s_lo)}


# ------------------------------------------------------------------------------------
# sigma_n-coupled friction variant (follow-up F3, Phase 0.7 R3)
# ------------------------------------------------------------------------------------


class FrictionAugmentedLaw:
    """Cohesive law c(delta) plus a mobilised Coulomb term at frozen normal pressure.

        tau_tot(s) = c(s) + mu * p_n * m(s),      m(s) = min(s / delta_0, 1).

    p_n >= 0 [MPa] is the recovered compressive normal pressure per interface pair,
    held FIXED within one staggered pass (SigmaNCoupledSolver updates it between
    passes). The Coulomb resistance mobilises over the law's own peak-slip scale
    delta_0 (no new parameter; the stick state is regularised on the same scale the
    cohesion rises on) and PERSISTS beyond delta_c -- residual friction on the
    exhausted surface. Colinear with the slip direction: the monotonic-slip form,
    consistent with the rising-branch warm-started continuation. mu = 0 reproduces
    the base law bit-identically (the +0.0 additive term is exact).
    """

    def __init__(self, base: LawIII, mu: float, p_n: np.ndarray):
        self.base = base
        self.mu = float(mu)
        self.p_n = np.asarray(p_n, dtype=float)
        # attributes the Newton solver consumes (scales only)
        self.tau_p = base.tau_p
        self.delta_0 = base.delta_0
        self.delta_c = base.delta_c
        self.p_r = base.p_r
        self.p_s = base.p_s

    def _m(self, d: np.ndarray) -> np.ndarray:
        return np.minimum(np.asarray(d, dtype=float) / self.base.delta_0, 1.0)

    def tau(self, d: np.ndarray) -> np.ndarray:
        return self.base.tau(d) + self.mu * self.p_n * self._m(d)

    def dtau(self, d: np.ndarray) -> np.ndarray:
        d = np.asarray(d, dtype=float)
        dm = np.where(d < self.base.delta_0, 1.0 / self.base.delta_0, 0.0)
        return self.base.dtau(d) + self.mu * self.p_n * dm

    def secant(self, s: np.ndarray, s_reg: float) -> np.ndarray:
        s = np.asarray(s, dtype=float)
        sr = np.maximum(s, s_reg)
        return self.tau(sr) / sr

    def g_c(self) -> float:
        """Cohesive fracture energy of the BASE law only [MPa*mm]: the Coulomb
        term is pressure-supplied dissipation, not a material fracture energy."""
        return self.base.g_c()


class SigmaNCoupledSolver(CohesiveInterfaceSolver):
    """F3 escalation variant: tau(delta) = c(delta) + mu * p_n(x, z; P, a) * m(delta).

    sigma_n is recovered from the SAME condensed elastic state as the tangential
    system (the normal-penalty jump traction):

        sigma_n = k_n * delta_n,   delta_n = P g_n + C_nt (D_ref,t d - t_t(d)),

    tension positive (ligament pairs are ordered minus->plus in y, so delta_n > 0
    is opening); only compression feeds the Coulomb term, p_n = max(-sigma_n, 0).
    The coupled fixed point is solved by damped staggered iteration: freeze p_n ->
    tangential Newton (exact residual) -> recover sigma_n -> relaxed update. A
    solution is reported converged ONLY when both the tangential residual and the
    p_n fixed point (max |update| < outer_tol_MPa) are met, so converged solutions
    are exact for the coupled law. Warm-starts p_n across solve() calls (peak scan
    and curve continuation); outer_exhausted counts fixed-point non-convergences
    (diagnostic -- a bracket decided by one would need investigation, not trust).

    peak_load is overridden for mu > 0: the staggered acceptance tolerance leaves
    a ~1e-5..1e-3 RELATIVE jitter on max_slip along the warm-started continuation,
    which the parent's 1e-12-absolute rising test can misread as a turnover
    (observed: a 0.9% peak truncation at mu=0.1, a=6). Every suspected turnover is
    therefore re-verified by an INDEPENDENT cold-ramp solve before it may close
    the bracket; mu = 0 passes through to the parent bit-identically.
    """

    def __init__(self, ops: InterfaceOperators, law: LawIII, mu: float,
                 s_reg_frac: float = 1e-6, tangent_floor_frac: float = 3e-2,
                 outer_max: int = 60, outer_tol_MPa: float = 1e-4,
                 relax: float = 0.7):
        if getattr(ops, "g_n", None) is None or getattr(ops, "C_nt", None) is None:
            raise ValueError("ops lack sigma_n recovery rows: rebuild with "
                             "build_interface_operators(..., sigma_n_rows=True)")
        super().__init__(ops, law, s_reg_frac=s_reg_frac,
                         tangent_floor_frac=tangent_floor_frac)
        self.base_law = law
        self.mu = float(mu)
        self.outer_max = int(outer_max)
        self.outer_tol = float(outer_tol_MPa)
        self.relax = float(relax)
        self._p_warm: Optional[np.ndarray] = None
        self.last_p_n: Optional[np.ndarray] = None
        self.last_outer: Dict[str, float] = {}
        self.outer_exhausted = 0

    def sigma_n(self, d: np.ndarray, P_kN: float) -> np.ndarray:
        """Normal traction [MPa] recovered from the condensed state (tension +).

        Uses the CURRENT self.law for t_t(d) -- call with the law that produced d.
        """
        q = self.ops.kref_force * d - self._force(d)
        return self.ops.k_n_used * (P_kN * self.ops.g_n + self.ops.C_nt @ q)

    def solve(self, P_kN: float, d0: Optional[np.ndarray] = None,
              max_iter: int = 60, tol: float = 3e-8) -> InterfaceSolution:
        n_p = self.ops.n_pairs
        p_n = (np.zeros(n_p) if (self.mu == 0.0 or self._p_warm is None)
               else self._p_warm.copy())
        d = d0
        sol = None
        dp = None
        for outer in range(1, self.outer_max + 1):
            self.law = FrictionAugmentedLaw(self.base_law, self.mu, p_n)
            sol = super().solve(P_kN, d0=d, max_iter=max_iter, tol=tol)
            if not sol.converged:
                self.last_outer = {"n_outer": outer, "dp_MPa": dp}
                return sol
            sig = self.sigma_n(sol.d_t, P_kN)
            p_new = np.maximum(-sig, 0.0)
            dp = float(np.max(np.abs(p_new - p_n))) if n_p else 0.0
            if self.mu == 0.0 or dp < self.outer_tol:
                self._p_warm = p_n
                self.last_p_n = p_n
                self.last_outer = {"n_outer": outer, "dp_MPa": dp}
                return sol
            p_n = p_n + self.relax * (p_new - p_n)
            d = sol.d_t
        self.outer_exhausted += 1
        self.last_outer = {"n_outer": self.outer_max, "dp_MPa": dp}
        return InterfaceSolution(P_kN=sol.P_kN, d_t=sol.d_t, slip=sol.slip,
                                 converged=False, n_iter=sol.n_iter,
                                 gauge_um=sol.gauge_um, cod_um=sol.cod_um,
                                 max_slip=sol.max_slip)

    # -- verified peak finder (mu > 0) ------------------------------------------

    def _cold_verify(self, P: float) -> Optional[InterfaceSolution]:
        """Independent cold-ramp solve at P (fresh warm state); None if it fails."""
        sv = SigmaNCoupledSolver(self.ops, self.base_law, self.mu,
                                 outer_max=self.outer_max,
                                 outer_tol_MPa=self.outer_tol, relax=self.relax)
        sols = sv.curve([float(P)])
        sol = sols[-1] if sols else None
        if sol is not None and sol.converged:
            self._p_warm = sv._p_warm  # adopt the verified state as the new warm
            return sol
        return None

    @staticmethod
    def _rising(sol: InterfaceSolution, ref: Optional[float]) -> bool:
        """Rising-branch test with a relative slack for the staggered jitter."""
        return bool(sol.converged and np.isfinite(sol.max_slip) and (
            ref is None or sol.max_slip >= ref * (1.0 - 1e-6) - 1e-12))

    def peak_load(self, P_lo: float = 0.05, P_hi: float = 30.0, n_scan: int = 60,
                  tol_kN: Optional[float] = None) -> Dict[str, float]:
        if self.mu == 0.0:
            return super().peak_load(P_lo=P_lo, P_hi=P_hi, n_scan=n_scan,
                                     tol_kN=tol_kN)
        Ps = np.linspace(P_lo, P_hi, n_scan)
        d = None
        prev = None  # (P, max_slip, d)
        bracket = None
        for P in Ps:
            sol = self.solve(float(P), d0=d)
            ok = self._rising(sol, prev[1] if prev else None)
            if not ok:
                cold = self._cold_verify(float(P))
                if cold is not None and self._rising(cold, prev[1] if prev else None):
                    sol, ok = cold, True  # false turnover (warm-path artifact)
            if not ok:
                bracket = (prev[0] if prev else P_lo, float(P))
                break
            prev = (float(P), sol.max_slip, sol.d_t.copy())
            d = sol.d_t
        if bracket is None:
            return {"P_peak_kN": float("nan"), "found": False,
                    "max_slip_at_top": prev[1] if prev else float("nan"),
                    "P_hi_scanned": float(P_hi)}
        if prev is None:
            return {"P_peak_kN": float("nan"), "found": False,
                    "first_scan_point_failed": True, "P_lo": float(P_lo)}
        lo, hi = bracket
        if tol_kN is None:
            tol_kN = max(5e-3, 5e-3 * hi)
        d_lo = prev[2]
        s_lo = prev[1]
        while hi - lo > tol_kN:
            mid = 0.5 * (lo + hi)
            sol = self.solve(mid, d0=d_lo)
            ok = self._rising(sol, s_lo)
            if not ok:
                cold = self._cold_verify(mid)
                if cold is not None and self._rising(cold, s_lo):
                    sol, ok = cold, True
            if ok:
                lo, s_lo, d_lo = mid, sol.max_slip, sol.d_t.copy()
            else:
                hi = mid
        return {"P_peak_kN": float(lo), "found": True,
                "max_slip_at_peak_mm": float(s_lo)}

    def force_split(self, sol: InterfaceSolution) -> Dict[str, float]:
        """Cohesive vs Coulomb share of the interface shear force at a converged
        solution (uses last_p_n -- the frozen level the solution satisfies)."""
        s = np.maximum(sol.slip, self.s_reg)
        f_coh = float(np.sum(self.ops.area * self.base_law.tau(s)))
        m = np.minimum(s / self.base_law.delta_0, 1.0)
        f_fric = float(np.sum(self.ops.area * self.mu * self.last_p_n * m))
        tot = f_coh + f_fric
        return {"F_cohesive_N": f_coh, "F_friction_N": f_fric,
                "friction_share": (f_fric / tot) if tot > 0 else 0.0}


# ------------------------------------------------------------------------------------
# Physics gates
# ------------------------------------------------------------------------------------


def run_gates(scale: float = 1.0, verbose: bool = True) -> Dict[str, Dict]:
    """R0 bonded / R1 LEFM / R2 net-section gates on the a=6 (and R1: all-depth) ops."""
    import json as _json
    from pathlib import Path as _Path
    from .config import project_root as _root

    ph = load_physics()
    out: Dict[str, Dict] = {}
    ops6 = build_interface_operators(6.0, scale=scale)

    # R0: near-rigid law -> bonded-probe gauge response
    rigid = LawIII(tau_p=200.0, delta_0=2e-4, delta_c=4e-4, p_r=1.0, p_s=1.0)
    sol = CohesiveInterfaceSolver(ops6, rigid).solve(1.0)
    probe_path = _Path(_root()) / "results/runs/anbd3d_probe/probe.json"
    bonded = None
    if probe_path.exists():
        pj = _json.load(open(probe_path))
        bonded = pj["by_depth"]["6"]["gauge_per_kN"]["+15.7mm"]["csd_w_um_per_kN"]
    out["R0_bonded_limit"] = {
        "gauge_rigid_um_per_kN": sol.gauge_um["+15.7"],
        "gauge_bonded_probe_um_per_kN": bonded,
        "rel_diff": (sol.gauge_um["+15.7"] / bonded - 1.0) if bonded else None,
        "converged": sol.converged,
    }

    # R2: plateau law -> net-section limit  P = 4 R (t-a) tau_p / cos(alpha)
    tau_ns = 5.0
    plateau = LawIII(tau_p=tau_ns, delta_0=0.5, delta_c=50.0, p_r=1.0, p_s=1.0)
    pk = CohesiveInterfaceSolver(ops6, plateau).peak_load(P_lo=0.2, P_hi=8.0, n_scan=40)
    P_ns = 4.0 * ph.R_mm * (ph.t_mm - 6.0) * tau_ns / math.cos(
        math.radians(ph.alpha_deg)) / 1000.0
    out["R2_net_section"] = {"P_peak_kN": pk.get("P_peak_kN"),
                             "P_analytic_kN": P_ns,
                             "ratio": pk.get("P_peak_kN", float("nan")) / P_ns,
                             "found": pk.get("found")}

    # R1: brittle law -> LEFM limit across all three depths. delta_c chosen so the
    # cohesive length G*Gc/tau_p^2 ~ 0.4 mm stays resolvable by the near-front
    # interface spacing (~0.1-0.5 mm) while remaining << the ligament (4.5-8.5 mm).
    brittle = LawIII(tau_p=30.0, delta_0=2e-3, delta_c=4e-3, p_r=1.0, p_s=1.0)
    Gc = brittle.g_c()
    K_c = math.sqrt(2.0 * ph.G_MPa * Gc) * math.sqrt(1e-3)  # MPa*sqrt(m)
    r1 = {"G_c_MPa_mm": Gc, "K_c_MPa_sqrt_m": K_c, "by_a": {}}
    probe = _json.load(open(probe_path)) if probe_path.exists() else None
    for a in (4.0, 6.0, 8.0):
        ops = ops6 if a == 6.0 else build_interface_operators(a, scale=scale)
        pk = CohesiveInterfaceSolver(ops, brittle).peak_load(P_lo=0.2, P_hi=25.0,
                                                             n_scan=50)
        row = {"P_peak_kN": pk.get("P_peak_kN"), "found": pk.get("found")}
        if probe is not None and pk.get("found"):
            ksum = probe["by_depth"][str(int(a))]["K_summary"]
            k_per_kn_max = abs(ksum["K_III_max_MPa_sqrt_m_per_kN"])
            row["K_max_at_peak_MPa_sqrt_m"] = k_per_kn_max * pk["P_peak_kN"]
            row["K_max_over_Kc"] = row["K_max_at_peak_MPa_sqrt_m"] / K_c
        r1["by_a"][str(int(a))] = row
    if all(v.get("found") for v in r1["by_a"].values()):
        p4, p8 = r1["by_a"]["4"]["P_peak_kN"], r1["by_a"]["8"]["P_peak_kN"]
        r1["Ppeak_a4_over_a8"] = p4 / p8
        if probe is not None:
            k4 = abs(probe["by_depth"]["4"]["K_summary"]["K_III_max_MPa_sqrt_m_per_kN"])
            k8 = abs(probe["by_depth"]["8"]["K_summary"]["K_III_max_MPa_sqrt_m_per_kN"])
            r1["Kmax_a8_over_a4"] = k8 / k4
    out["R1_lefm_limit"] = r1

    if verbose:
        print("[R0] rigid-law gauge:", round(out["R0_bonded_limit"]["gauge_rigid_um_per_kN"], 4),
              "vs bonded probe", out["R0_bonded_limit"]["gauge_bonded_probe_um_per_kN"],
              "rel", out["R0_bonded_limit"]["rel_diff"])
        print("[R2] net-section: P_peak", out["R2_net_section"]["P_peak_kN"],
              "vs analytic", round(P_ns, 3), "ratio", round(out["R2_net_section"]["ratio"], 3))
        print("[R1] LEFM:", {a: {kk: (round(vv, 4) if isinstance(vv, float) else vv)
                                 for kk, vv in v.items()} for a, v in r1["by_a"].items()})
        if "Ppeak_a4_over_a8" in r1:
            print("[R1] P4/P8 =", round(r1["Ppeak_a4_over_a8"], 3),
                  "vs Kmax8/Kmax4 =", round(r1.get("Kmax_a8_over_a4", float("nan")), 3))
    return out


if __name__ == "__main__":
    import argparse
    import json as _json
    from .config import project_root as _pr
    ap = argparse.ArgumentParser()
    ap.add_argument("--gates", action="store_true")
    ap.add_argument("--scale", type=float, default=1.0)
    args = ap.parse_args()
    if args.gates:
        res = run_gates(scale=args.scale)
        outp = _pr() / "results" / "runs" / "rebuild_identify"
        outp.mkdir(parents=True, exist_ok=True)
        with open(outp / "solver_gates.json", "w", encoding="utf-8") as fh:
            _json.dump({"scale": args.scale, **res}, fh, indent=2, default=float)
        print(f"wrote {outp / 'solver_gates.json'}")
