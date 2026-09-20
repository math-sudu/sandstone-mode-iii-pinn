"""EXPERIMENTAL PROBE: 3D linear-elastic forward model of the true ANBD geometry.

Status
------
Feasibility probe for the method-rebuild decision (user-approved probe-first plan).
Nothing here is wired into the identification pipeline yet. The probe arbitrates:
  (P1) the geometry-factor backbone: recover Y_III(a/t) from an independent 3D FE
       solve of the TRUE axial-notch geometry and compare against the source FE
       values 1.185 / 1.582 / 2.079 (results/data/e1_inputs/notch_plane_geometry.json);
  (P2) the SIF mix on the front: K_I*/K_III* (config ki_over_kiii = 0.3,
       configs/identify.yaml line 35) and K_II smallness away from the corners
       (Bahrami2022 R001/full.md line 69);
  (P3) the DIC gauge forward map: the strip-gauge W-step across the notch trace on
       the notched end face (dic_extract.py strip_gauge recipe: A/B strips
       Yn in [1.5, 5] mm, |Xn - station| < 2 mm, CSD_W = mean W(A) - mean W(B)),
       its sign pattern at +/- stations and its per-kN magnitude against the
       measured elastic branch of csd_w_cod_curve.csv.

Geometry contract (verified against the sources)
------------------------------------------------
ANBD axially double-edge notched Brazilian disc (Bahrami2022, R001/full.md line 53:
"two axial notches of length a ... yields a ligament of length 2l = 2t - 2a between
the two notches"; F11/F12 specimen and setup figures; user-confirmed):
  * disc radius R = 25 mm, total thickness 2t = 25 mm (faces at z = +/-t);
  * each flat end face carries ONE notch slot along a full diameter; the two slots
    are coplanar (same axial plane y = 0); each is cut axially to depth a;
  * the uncut ligament is the central band |z| < t - a of the plane y = 0, spanning
    the full chord in x; the two crack fronts are the lines z = +/-(t - a), y = 0;
  * the disc is loaded in diametral compression, load line at alpha = 10 deg FROM
    THE NOTCH PLANE (R001/full.md line 53: "The angle between the loading direction
    and the notch plane, alpha"); contact azimuths +/-alpha from the +x axis;
  * notch faces carry no contact (R001/full.md line 55: the initial saw gap is much
    larger than the closing deformation), so the slit is traction-free; the sharp
    zero-width slit idealisation matches the source SIF convention (collapsed
    wedge-tip elements, R001/full.md line 55).
This is NOT the geometry of forward_fullfield.py (radial rim notches in the disc
plane); that module models a different specimen and is superseded by this probe.

Model reduction
---------------
The problem is symmetric under z -> -z (coplanar twin notches, z-uniform load):
u_x, u_y even, u_z odd. We mesh the half-thickness z in [0, t] with u_z = 0 on
z = 0 and the FRONT notched face at z = t. Isotropic elasticity (E = 14.3 GPa,
nu = 0.28 from the geometry JSON); TI bulk is a later sensitivity, not the probe.

Units: lengths mm, forces N (P given in kN is converted), stresses MPa,
displacements mm, K in MPa*sqrt(m) at the API boundary (internally MPa*sqrt(mm)).

Deterministic: structured graded point clouds, Delaunay in the upper half-disc
mirrored to the lower half (exact shared y=0 node line), prism-to-tet splitting by
the Dompierre smallest-vertex rule (conforming diagonals). No RNG anywhere.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.sparse.linalg import splu
from scipy.spatial import Delaunay

from skfem import Basis, ElementTetP1, ElementVector, FacetBasis, LinearForm, MeshTet, asm
from skfem.models.elasticity import lame_parameters, linear_elasticity

from .physics import Physics, load_physics

MM_SQRT_TO_M_SQRT = math.sqrt(1e-3)  # MPa*sqrt(mm) -> MPa*sqrt(m)

# DIC strip-gauge constants mirrored from dic_extract.py (do not drift)
GAUGE_YN_MIN = 1.5
GAUGE_YN_MAX = 5.0
GAUGE_DXN = 2.0
GAUGE_MIN_PTS = 3

# corner-refinement geometry (follow-up F2): two-level x refinement of the
# near-plane rows inside rim bands, plus extra z-layers hugging the front
CORNER_X_BANDS = (5.0, 2.5)   # mm from the rim; each band halves the x spacing
CORNER_Y_MAX = 3.7            # rows with y <= this get the x refinement
CORNER_Z_EXTRA = (0.05, 0.025)  # extra |z - front| layers on both sides


# ------------------------------------------------------------------------------------
# 2D disc triangulation with an exact node line on y = 0
# ------------------------------------------------------------------------------------


def _row_ys(scale: float, R: float = 25.0,
            keep_rows: Tuple[float, ...] = ()) -> np.ndarray:
    """Graded y >= 0 row ordinates: dense near the slit plane and the gauge band.

    The near-plane ladder is absolute (slit/gauge physics); the outer rows scale
    with the disc radius so the same builder serves benchmark geometries.
    keep_rows are inserted verbatim and survive coarsening (kerf wall lines).
    """
    near = np.array([0.0, 0.18, 0.42, 0.75, 1.2, 1.8, 2.6, 3.6, 5.0, 6.6, 8.6])
    outer = R * np.array([0.30, 0.44, 0.56, 0.70, 0.83, 0.936])
    base = np.unique(np.r_[near[near < 0.29 * R], outer])
    keep = np.asarray(keep_rows, dtype=float)
    if scale == 1.0:
        return np.unique(np.r_[base, keep]) if keep.size else base
    # refine: insert midpoints between consecutive rows (scale < 1 => finer)
    if scale < 1.0:
        mids = 0.5 * (base[:-1] + base[1:])
        return np.unique(np.r_[base, mids, keep])
    # coarsen: thin rows OUTSIDE the DIC gauge band; the band rows (1.5-5 mm plus
    # their immediate neighbours) are measurement geometry and are never dropped
    keep_band = base[(base >= 1.2) & (base <= 5.0)]
    rest = base[(base < 1.2) | (base > 5.0)]
    return np.unique(np.r_[rest[::2], keep_band, keep, base[-1]])


def _disc_points_upper(R: float, scale: float, keep_rows: Tuple[float, ...] = (),
                       corner_refine: bool = False) -> np.ndarray:
    """Deterministic upper-half-disc point cloud with rows on graded y-lines.

    Every row spans the full chord and INCLUDES its rim endpoints; a rim-cap arc
    covers azimuths above the last row. x-spacing grows away from the slit plane.
    corner_refine halves the x spacing of the near-plane rows twice inside the
    CORNER_X_BANDS rim bands (the crack-front corner zones sit at x = +/-R).
    """
    pts: List[Tuple[float, float]] = []
    ys = _row_ys(scale, R, keep_rows)
    ys = ys[ys < R - 0.8]
    for y in ys:
        xmax = math.sqrt(R * R - y * y)
        # x spacing: ~1.1 mm near y=0 growing to ~2.4 mm on far rows (scaled)
        h = (1.1 + 1.3 * (abs(y) / R)) * scale
        n = max(int(round(2 * xmax / h)), 6)
        xs = np.linspace(-xmax, xmax, n + 1)
        if corner_refine and y <= CORNER_Y_MAX:
            for band in CORNER_X_BANDS:
                mids = 0.5 * (xs[:-1] + xs[1:])
                xs = np.unique(np.r_[xs, mids[np.abs(mids) > R - band]])
        pts.extend((float(x), float(y)) for x in xs)
    # rim cap above the last row (azimuth window around 90 deg)
    y_top = ys[-1]
    phi_lo = math.asin(min(y_top / R, 1.0)) + 0.03
    n_arc = max(int(round((math.pi - 2 * phi_lo) * R / (1.6 * scale))), 8)
    for phi in np.linspace(phi_lo, math.pi - phi_lo, n_arc):
        pts.append((float(R * math.cos(phi)), float(R * math.sin(phi))))
    P = np.array(sorted(set(pts)), dtype=float)
    return P


def _tri_upper_mirrored(R: float, scale: float, keep_rows: Tuple[float, ...] = (),
                        corner_refine: bool = False,
                        wall_y: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Triangulate the upper half-disc and mirror to the lower half.

    Returns (points2d (N,2), tris (M,3)) covering the full disc with an exact,
    conforming node line on y = 0 and no element straddling it.

    wall_y (kerf variant) partitions the upper half at that row BEFORE
    triangulating: the two bands are Delaunay-triangulated independently and
    share the wall-row points, so no triangle straddles y = wall_y (a single
    global Delaunay puts sliver triangles across thin bands -- the empty-circle
    rule knows nothing about the wall). The seam is validated by the
    edge-manifold check in build_anbd_mesh.
    """
    P = _disc_points_upper(R, scale, keep_rows, corner_refine)
    if wall_y is None:
        tri_simplices = Delaunay(P).simplices
    else:
        inA = np.where(P[:, 1] <= wall_y + 1e-12)[0]
        inB = np.where(P[:, 1] >= wall_y - 1e-12)[0]
        TA = inA[Delaunay(P[inA]).simplices]
        TB = inB[Delaunay(P[inB]).simplices]
        tri_simplices = np.vstack([TA, TB])
    cen = P[tri_simplices].mean(axis=1)
    rad = np.hypot(cen[:, 0], cen[:, 1])
    v = P[tri_simplices]
    area = 0.5 * np.abs(
        (v[:, 1, 0] - v[:, 0, 0]) * (v[:, 2, 1] - v[:, 0, 1])
        - (v[:, 2, 0] - v[:, 0, 0]) * (v[:, 1, 1] - v[:, 0, 1])
    )
    keep = (rad <= R * (1 + 1e-9)) & (area > 1e-9)
    T_up = tri_simplices[keep]
    # mirror nodes with y > 0; y == 0 nodes are shared
    is_pos = P[:, 1] > 1e-12
    idx_pos = np.where(is_pos)[0]
    mirror_of = -np.ones(len(P), dtype=int)
    mirror_of[~is_pos] = np.where(~is_pos)[0]  # y=0 nodes map to themselves
    P_lo = P[idx_pos] * np.array([1.0, -1.0])
    mirror_of[idx_pos] = len(P) + np.arange(len(idx_pos))
    P_all = np.vstack([P, P_lo])
    T_lo = mirror_of[T_up]
    T_lo = T_lo[:, [0, 2, 1]]  # restore orientation after reflection
    return P_all, np.vstack([T_up, T_lo])


# ------------------------------------------------------------------------------------
# z-extrusion (graded toward the front line z = t - a) and prism -> tet splitting
# ------------------------------------------------------------------------------------


def _z_lines(t: float, a: float, scale: float, corner_refine: bool = False) -> np.ndarray:
    """Graded z-lines on [0, t]: coarse in the ligament, dense at the front z = t - a."""
    c = t - a
    lig = list(np.linspace(0.0, max(c - 1.4, 0.4), max(int(round(3 / scale)), 3), endpoint=False))
    band_lo = [c - 1.4, c - 0.9, c - 0.5, c - 0.24, c - 0.1]
    band_hi = [c + 0.1, c + 0.24, c + 0.5, c + 0.95, c + 1.7]
    if corner_refine:
        band_lo = band_lo + [c - dz for dz in CORNER_Z_EXTRA]
        band_hi = band_hi + [c + dz for dz in CORNER_Z_EXTRA]
    rest = list(np.linspace(c + 2.6, t, max(int(round((a - 2.6) / (1.6 * scale))) + 1, 3)))
    z = np.unique(np.round(np.r_[lig, band_lo, [c], band_hi, rest, [t]], 6))
    z = z[(z >= -1e-9) & (z <= t + 1e-9)]
    if scale < 1.0:  # fine level: split every layer once more
        z = np.unique(np.r_[z, 0.5 * (z[:-1] + z[1:])])
    return z


def _prisms_to_tets(tris: np.ndarray, n2d: int, nz: int) -> np.ndarray:
    """Split the prism stack into tets with the Dompierre smallest-vertex rule."""
    tets: List[Tuple[int, int, int, int]] = []
    for k in range(nz - 1):
        lo = k * n2d
        hi = (k + 1) * n2d
        for (i, j, l) in tris:
            b = (lo + i, lo + j, lo + l)
            u = (hi + i, hi + j, hi + l)
            # rotate so the smallest global id of the prism sits at position 0
            ids = b + u
            imin = int(np.argmin(ids))
            r = imin % 3
            b = (b[r], b[(r + 1) % 3], b[(r + 2) % 3])
            u = (u[r], u[(r + 1) % 3], u[(r + 2) % 3])
            if min(b[1], u[2]) < min(b[2], u[1]):
                tets.append((b[0], b[1], b[2], u[2]))
                tets.append((b[0], b[1], u[2], u[1]))
                tets.append((b[0], u[1], u[2], u[0]))
            else:
                tets.append((b[0], b[1], b[2], u[1]))
                tets.append((b[0], u[1], b[2], u[2]))
                tets.append((b[0], u[1], u[2], u[0]))
    return np.asarray(tets, dtype=np.int64)


def _orient_tets(p: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Ensure positive signed volume (swap two vertices where negative)."""
    v0 = p[t[:, 1]] - p[t[:, 0]]
    v1 = p[t[:, 2]] - p[t[:, 0]]
    v2 = p[t[:, 3]] - p[t[:, 0]]
    vol = np.einsum("ij,ij->i", np.cross(v0, v1), v2)
    neg = vol < 0
    t2 = t.copy()
    t2[neg, 2], t2[neg, 3] = t[neg, 3], t[neg, 2]
    return t2


# ------------------------------------------------------------------------------------
# The half-thickness ANBD mesh with the sharp slit (split nodes)
# ------------------------------------------------------------------------------------


@dataclass
class AnbdMesh:
    mesh: MeshTet
    a_mm: float
    R_mm: float
    t_mm: float
    n_nodes: int
    slit_pairs: np.ndarray      # (n_slit, 2) [minus-side original, plus-side duplicate]
    slit_xz: np.ndarray         # (n_slit, 2) (x, z) of each slit pair
    front_nodes: np.ndarray     # node ids on the front line z = t - a, y = 0
    stats: Dict[str, float]
    lig_pairs: Optional[np.ndarray] = None   # split_ligament=True: (n_lig, 2) pairs on z <= t-a
    lig_xz: Optional[np.ndarray] = None      # (n_lig, 2) (x, z)
    lig_area: Optional[np.ndarray] = None    # (n_lig,) tributary areas (mm^2)


def build_anbd_mesh(a_mm: float, R_mm: float = 25.0, t_mm: float = 12.5,
                    scale: float = 1.0, split_ligament: bool = False,
                    kerf_w_mm: Optional[float] = None,
                    corner_refine: bool = False) -> AnbdMesh:
    """Build the half-thickness ANBD tet mesh with a sharp slit on y=0, z>t-a.

    scale < 1 refines (both the 2D cloud and the z-ladder), > 1 coarsens.
    split_ligament=True additionally splits the LIGAMENT plane nodes (y=0,
    z <= t-a, front line included) into duplicate pairs so a cohesive interface
    can carry displacement jumps there; the default False keeps the bonded
    (probe / LEFM-null) mesh bit-identical to the original behaviour.

    kerf_w_mm replaces the sharp zero-width slit by a FINITE saw slot of that
    width: the walls y = +/-kerf_w_mm/2 are exact node rows, every tet whose
    centroid lies inside the slot (|y| < w/2, z > t-a) is removed, and no slit
    node splitting happens (the slot is a real cavity; the ligament interface
    below z = t-a is unchanged, so kerf and sharp meshes share the same
    interface discretisation). corner_refine adds two-level x refinement of
    the near-plane rows in the rim bands plus extra front z-layers (the
    crack-front corner zones at x = +/-R). Both default to the original
    bit-identical behaviour.
    """
    keep_rows = (0.5 * kerf_w_mm,) if kerf_w_mm else ()
    P2, T2 = _tri_upper_mirrored(R_mm, scale, keep_rows=keep_rows,
                                 corner_refine=corner_refine,
                                 wall_y=(0.5 * kerf_w_mm) if kerf_w_mm else None)
    if kerf_w_mm or corner_refine:
        # edge-manifold check: in the mirrored full-disc triangulation every
        # edge must bound exactly two triangles except on the rim (catches
        # collinear-point skips of qhull and any partition-seam mismatch)
        e = np.sort(np.r_[T2[:, [0, 1]], T2[:, [1, 2]], T2[:, [2, 0]]], axis=1)
        eu, cnt = np.unique(e, axis=0, return_counts=True)
        bmid = P2[eu[cnt == 1]].mean(axis=1)
        off_rim = np.hypot(bmid[:, 0], bmid[:, 1]) < R_mm - 0.9
        if off_rim.any() or (cnt > 2).any():
            raise RuntimeError(
                f"non-manifold 2D triangulation: {int(off_rim.sum())} interior "
                f"boundary edges, {int((cnt > 2).sum())} over-shared edges")
    zs = _z_lines(t_mm, a_mm, scale, corner_refine=corner_refine)
    n2d, nz = len(P2), len(zs)
    pts = np.zeros((n2d * nz, 3))
    for k, z in enumerate(zs):
        pts[k * n2d:(k + 1) * n2d, :2] = P2
        pts[k * n2d:(k + 1) * n2d, 2] = z
    tets = _prisms_to_tets(T2, n2d, nz)
    c = t_mm - a_mm

    # --- finite kerf: carve the slot out of the extruded stack
    if kerf_w_mm:
        hw = 0.5 * kerf_w_mm
        # conforming guard: no 2D triangle may straddle a wall line y = +/-hw
        # (the z-ladder cannot straddle z = c by construction); a straddling
        # element would leave a jagged wall after centroid-based deletion
        ytri = P2[T2][:, :, 1]
        for wall in (hw, -hw):
            strad = (ytri.min(axis=1) < wall - 1e-9) & (ytri.max(axis=1) > wall + 1e-9)
            if strad.any():
                raise RuntimeError(
                    f"kerf wall y={wall} straddled by {int(strad.sum())} triangles")
        cen = pts[tets].mean(axis=1)
        in_kerf = (np.abs(cen[:, 1]) < hw) & (cen[:, 2] > c)
        tets = tets[~in_kerf]
        keep_ids = np.unique(tets)
        remap_k = -np.ones(len(pts), dtype=np.int64)
        remap_k[keep_ids] = np.arange(len(keep_ids))
        pts = pts[keep_ids]
        tets = remap_k[tets]

    # --- split nodes on the interface plane y == 0:
    #     always the slit (z > t-a); with split_ligament also the ligament (z <= t-a)
    on_plane = np.abs(pts[:, 1]) < 1e-9
    above = pts[:, 2] > c + 1e-6
    split_mask = on_plane & above if not split_ligament else on_plane
    split_ids = np.where(split_mask)[0]
    is_slit_split = above[split_ids]
    n_old = len(pts)
    front = np.where(on_plane & (np.abs(pts[:, 2] - c) < 1e-6))[0]
    dup_of = {int(nid): n_old + k for k, nid in enumerate(split_ids)}
    pts = np.vstack([pts, pts[split_ids]])

    cen_y = pts[tets].mean(axis=1)[:, 1]
    plus_side = cen_y > 0
    if split_ids.size:
        # remap split-node references in plus-side elements to the duplicates
        remap = np.arange(len(pts), dtype=np.int64)
        for nid, dup in dup_of.items():
            remap[nid] = dup
        tp = tets[plus_side]
        tets[plus_side] = remap[tp]

    tets = _orient_tets(pts, tets)
    mesh = MeshTet(pts.T.copy(), tets.T.copy())

    all_pairs = np.array([[nid, dup_of[int(nid)]] for nid in split_ids], dtype=np.int64)
    all_xz = pts[split_ids][:, [0, 2]]
    slit_pairs = all_pairs[is_slit_split]
    slit_xz = all_xz[is_slit_split]
    lig_pairs = lig_xz = lig_area = None
    if split_ligament:
        lig_pairs = all_pairs[~is_slit_split]
        lig_xz = all_xz[~is_slit_split]
        # tributary areas on the structured (x, z) interface grid: half-gaps to the
        # neighbouring unique coordinates, edges clamped to the domain limits
        xs = np.unique(np.round(lig_xz[:, 0], 9))
        zl = np.unique(np.round(lig_xz[:, 1], 9))

        def _trib(vals: np.ndarray, lo: float, hi: float) -> Dict[float, float]:
            v = np.r_[lo, vals, hi]
            return {float(vals[i]): float(0.5 * (v[i + 2] - v[i])) for i in range(len(vals))}

        wx = _trib(xs, -R_mm, R_mm)
        wz = _trib(zl, 0.0, c)
        lig_area = np.array([
            wx[float(np.round(x, 9))] * wz[float(np.round(z, 9))] for x, z in lig_xz
        ])
    stats = {"n_nodes": float(len(pts)), "n_tets": float(len(tets)),
             "n_slit_pairs": float(len(slit_pairs)),
             "n_lig_pairs": float(0 if lig_pairs is None else len(lig_pairs)),
             "n2d": float(n2d), "nz": float(nz),
             "kerf_w_mm": float(kerf_w_mm or 0.0),
             "corner_refine": float(bool(corner_refine))}
    return AnbdMesh(mesh=mesh, a_mm=float(a_mm), R_mm=float(R_mm), t_mm=float(t_mm),
                    n_nodes=len(pts), slit_pairs=slit_pairs, slit_xz=slit_xz,
                    front_nodes=front, stats=stats, lig_pairs=lig_pairs,
                    lig_xz=lig_xz, lig_area=lig_area)


# ------------------------------------------------------------------------------------
# Diametral load assembly (shared by the probe model and the cohesive solver)
# ------------------------------------------------------------------------------------


def _rim_patch_facets(md: AnbdMesh, azimuth_deg: float, patch_half_deg: float) -> np.ndarray:
    m = md.mesh
    mid = m.p[:, m.facets].mean(axis=1)
    bnd = np.zeros(m.facets.shape[1], dtype=bool)
    bnd[m.boundary_facets()] = True
    rad = np.hypot(mid[0], mid[1])
    phi = np.degrees(np.arctan2(mid[1], mid[0]))
    dphi = (phi - azimuth_deg + 180.0) % 360.0 - 180.0
    sel = bnd & (rad > md.R_mm - 0.6) & (np.abs(dphi) < patch_half_deg + 1.0)
    return np.where(sel)[0]


def _patch_load_vector(md: AnbdMesh, basis: Basis, azimuth_deg: float,
                       patch_half_deg: float) -> np.ndarray:
    """Assemble the cosine-tapered inward radial traction on one rim patch."""
    fb = FacetBasis(md.mesh, basis.elem,
                    facets=_rim_patch_facets(md, azimuth_deg, patch_half_deg))
    az = math.radians(azimuth_deg)
    half = math.radians(patch_half_deg)

    @LinearForm
    def load(v, w):
        x, y = w.x[0], w.x[1]
        phi = np.arctan2(y, x)
        d = np.arctan2(np.sin(phi - az), np.cos(phi - az))
        taper = np.where(np.abs(d) < half, np.cos(0.5 * np.pi * d / half) ** 2, 0.0)
        r = np.sqrt(x ** 2 + y ** 2)
        tx, ty = -x / r, -y / r  # inward radial unit traction
        return taper * (tx * v[0] + ty * v[1])

    return asm(load, fb)


def assemble_diametral_load(md: AnbdMesh, basis: Basis, alpha_deg: float,
                            P_kN: float = 1.0, patch_half_deg: float = 4.0
                            ) -> Tuple[np.ndarray, Dict[str, float]]:
    """Self-equilibrated diametral load, total force P_kN on the FULL disc.

    The load line sits at alpha deg from the notch plane (+x axis): contact
    patches at azimuths alpha and alpha+180. The half-thickness model carries
    P/2. Forces in N (P_kN * 1000). Returns (load vector, checks).
    """
    f1 = _patch_load_vector(md, basis, alpha_deg, patch_half_deg)
    f2 = _patch_load_vector(md, basis, alpha_deg + 180.0, patch_half_deg)
    d = np.array([math.cos(math.radians(alpha_deg)), math.sin(math.radians(alpha_deg))])
    nd = basis.nodal_dofs
    rig = np.zeros(basis.N)
    rig[nd[0]] = d[0]
    rig[nd[1]] = d[1]
    F1 = float(f1 @ rig)  # net force of unit-amplitude patch along +d
    target = -(P_kN * 1000.0) / 2.0  # compressive (pushes toward the centre)
    s = target / F1
    f = s * (f1 + f2)
    checks = {"net_force_over_P": float(abs(f @ rig) / abs(target)),
              "patch_force_N": float(s * F1)}
    return f, checks


# ------------------------------------------------------------------------------------
# Forward model: assembly, load, solve, and the probe observables
# ------------------------------------------------------------------------------------


@dataclass
class LinearSolution:
    """Unit-load (P_total = 1 kN) linear solution and derived probe observables."""
    u: np.ndarray                 # (ndof,) displacements (mm) at P = 1 kN total
    P_kN: float
    gauge: Dict[str, Dict[str, float]]
    K_front: Dict[str, np.ndarray]   # per front station: x, K_I, K_II, K_III (MPa*sqrt(m))
    K_summary: Dict[str, float]
    ligament: Dict[str, np.ndarray]
    checks: Dict[str, float]


class Anbd3D:
    """3D linear-elastic ANBD forward model (half-thickness, sharp slit)."""

    def __init__(self, a_mm: float, physics: Optional[Physics] = None,
                 scale: float = 1.0, alpha_deg: Optional[float] = None,
                 patch_half_deg: float = 4.0, R_mm: Optional[float] = None,
                 t_mm: Optional[float] = None, E_MPa: Optional[float] = None,
                 nu: Optional[float] = None,
                 mesh_kwargs: Optional[Dict] = None):
        """Geometry/material overrides (R_mm, t_mm, E_MPa, nu) serve benchmark
        configurations (e.g. the Bahrami2022 published Table-1 case); defaults
        come from the project Physics (geometry JSON). mesh_kwargs passes
        variant options (kerf_w_mm, corner_refine) through to build_anbd_mesh."""
        self.ph = physics if physics is not None else load_physics()
        self.a = float(a_mm)
        self.R = float(R_mm) if R_mm is not None else self.ph.R_mm
        self.t = float(t_mm) if t_mm is not None else self.ph.t_mm
        self.alpha = float(alpha_deg if alpha_deg is not None else self.ph.alpha_deg)
        self.patch_half = float(patch_half_deg)
        self.E = float(E_MPa) if E_MPa is not None else self.ph.E_GPa * 1000.0
        self.nu = float(nu) if nu is not None else self.ph.nu
        self.G = self.E / (2.0 * (1.0 + self.nu))
        self.md = build_anbd_mesh(self.a, self.R, self.t, scale=scale,
                                  **(mesh_kwargs or {}))
        self.basis = Basis(self.md.mesh, ElementVector(ElementTetP1()))
        lam, mu = lame_parameters(self.E, self.nu)
        self.K = asm(linear_elasticity(lam, mu), self.basis)
        self._dirichlet = self._dirichlet_dofs()
        self._lu = None

    # -- constraints -------------------------------------------------------------

    def _dirichlet_dofs(self) -> np.ndarray:
        p = self.md.mesh.p
        nd = self.basis.nodal_dofs  # (3, n_nodes)
        z0 = np.where(np.abs(p[2]) < 1e-9)[0]
        dofs = [nd[2, z0]]  # u_z = 0 on the symmetry plane
        # pin remaining in-plane rigid modes at two z=0 nodes
        i_origin = int(np.argmin(p[0, z0] ** 2 + p[1, z0] ** 2))
        n_o = z0[i_origin]
        dofs.append(nd[0:2, n_o].ravel())
        i_far = int(np.argmax(np.abs(p[1, z0])))
        n_f = z0[i_far]
        dofs.append(nd[0:1, n_f].ravel())
        self._pin_nodes = (int(n_o), int(n_f))
        return np.unique(np.concatenate(dofs))

    # -- load --------------------------------------------------------------------

    def build_load(self, P_kN: float = 1.0) -> np.ndarray:
        f, checks = assemble_diametral_load(self.md, self.basis, self.alpha,
                                            P_kN, self.patch_half)
        self._load_checks = checks
        return f

    # -- solve ---------------------------------------------------------------------

    def solve(self, P_kN: float = 1.0) -> np.ndarray:
        from skfem import condense
        f = self.build_load(P_kN)
        if self._lu is None:
            Kc, fc, _, I = condense(self.K, f, D=self._dirichlet)
            self._cond_I = I
            self._lu = splu(Kc.tocsc())
            xI = self._lu.solve(fc)
        else:
            _, fc, _, _ = condense(self.K, f, D=self._dirichlet)
            xI = self._lu.solve(fc)
        u = np.zeros(self.basis.N)
        u[self._cond_I] = xI
        return u

    # -- observables -----------------------------------------------------------------

    def gauge_step(self, u: np.ndarray, station_mm: float) -> Dict[str, float]:
        """The dic_extract strip gauge on the notched face z = t (model side).

        A strip: y in (1.5, 5); B strip: y in (-5, -1.5); |x - station| < 2 mm;
        CSD_W = mean u_z(A) - mean u_z(B)   [mm -> reported um]
        COD   = mean u_y(A) - mean u_y(B)   [um]
        Face nodes only (z == t). Mirrors dic_extract.strip_gauge.
        """
        p = self.md.mesh.p
        nd = self.basis.nodal_dofs
        face = np.abs(p[2] - self.t) < 1e-9
        ins = np.abs(p[0] - station_mm) < GAUGE_DXN
        A = face & ins & (p[1] > GAUGE_YN_MIN) & (p[1] < GAUGE_YN_MAX)
        B = face & ins & (p[1] < -GAUGE_YN_MIN) & (p[1] > -GAUGE_YN_MAX)
        nA, nB = int(A.sum()), int(B.sum())
        out = {"station_mm": float(station_mm), "nA": nA, "nB": nB,
               "csd_w_um_per_kN": float("nan"), "cod_um_per_kN": float("nan"),
               "w_A_um": float("nan"), "w_B_um": float("nan")}
        if nA >= GAUGE_MIN_PTS and nB >= GAUGE_MIN_PTS:
            wA = float(u[nd[2, A]].mean()) * 1000.0
            wB = float(u[nd[2, B]].mean()) * 1000.0
            vA = float(u[nd[1, A]].mean()) * 1000.0
            vB = float(u[nd[1, B]].mean()) * 1000.0
            out.update(csd_w_um_per_kN=wA - wB, cod_um_per_kN=vA - vB, w_A_um=wA, w_B_um=wB)
        return out

    def w_antisym_index(self, u: np.ndarray, station_mm: float, nbin: int = 4) -> float:
        """Model-side emulation of dic_extract.w_antisymmetry on the face band."""
        p = self.md.mesh.p
        nd = self.basis.nodal_dofs
        face = np.abs(p[2] - self.t) < 1e-9
        ins = face & (np.abs(p[0] - station_mm) < GAUGE_DXN)
        sel = ins & (np.abs(p[1]) > GAUGE_YN_MIN) & (np.abs(p[1]) < GAUGE_YN_MAX)
        if int(sel.sum()) < 6:
            return float("nan")
        yy = p[1, sel]
        ww = u[nd[2, sel]] * 1000.0
        w0 = float(ww.mean())
        edges = np.linspace(GAUGE_YN_MIN, GAUGE_YN_MAX, nbin + 1)
        odd, even = [], []
        for i in range(nbin):
            mp = (yy >= edges[i]) & (yy < edges[i + 1])
            mm = (yy <= -edges[i]) & (yy > -edges[i + 1])
            if mp.sum() >= 1 and mm.sum() >= 1:
                wp = float(ww[mp].mean()) - w0
                wm = float(ww[mm].mean()) - w0
                odd.append(0.5 * (wp - wm))
                even.append(0.5 * (wp + wm))
        if not odd:
            return float("nan")
        o, e = np.asarray(odd), np.asarray(even)
        tot = float((o ** 2).sum() + (e ** 2).sum())
        return float((o ** 2).sum() / tot) if tot > 1e-15 else float("nan")

    def sif_along_front(self, u: np.ndarray, r_lo: float = 0.3, r_hi: float = 2.1,
                        bin_w: float = 1.6) -> Dict[str, np.ndarray]:
        """K_I, K_II, K_III along the front from the slit flank displacement jumps.

        Flank pairs sit at (x, y=0+/-, z = (t-a) + r). Anti-plane flank asymptote:
        jump_x(r) = (4 K_III / G) sqrt(r / 2pi); plane-strain modes:
        jump_y = (8 K_I / E') sqrt(r/2pi), jump_z = (8 K_II / E') sqrt(r/2pi),
        E' = E/(1-nu^2). Least-squares fit of jump = C sqrt(r) per x-bin. Signs:
        K_I < 0 reports interpenetration of the closed slit (no-contact model),
        matching the source convention (R001/full.md line 69). K in MPa*sqrt(m).
        """
        c = self.t - self.a
        nd = self.basis.nodal_dofs
        pr = self.md.slit_pairs
        xz = self.md.slit_xz
        r = xz[:, 1] - c
        use = (r > r_lo) & (r < r_hi)
        Ep = self.E / (1.0 - self.nu ** 2)
        xs_bins = np.arange(-self.R + 2.0, self.R - 2.0 + 1e-9, bin_w)
        out_x, kI, kII, kIII = [], [], [], []
        for xb in xs_bins:
            m = use & (np.abs(xz[:, 0] - xb) < bin_w / 2)
            if m.sum() < 3:
                continue
            rr = r[m]
            sq = np.sqrt(rr)
            den = float(sq @ sq)
            # jump = plus-side minus minus-side (pair[:,1] is the +y duplicate)
            jx = u[nd[0, pr[m, 1]]] - u[nd[0, pr[m, 0]]]
            jy = u[nd[1, pr[m, 1]]] - u[nd[1, pr[m, 0]]]
            jz = u[nd[2, pr[m, 1]]] - u[nd[2, pr[m, 0]]]
            Cx = float(sq @ jx) / den
            Cy = float(sq @ jy) / den
            Cz = float(sq @ jz) / den
            k3 = Cx * self.G * math.sqrt(2.0 * math.pi) / 4.0
            k1 = Cy * Ep * math.sqrt(2.0 * math.pi) / 8.0
            k2 = Cz * Ep * math.sqrt(2.0 * math.pi) / 8.0
            out_x.append(xb)
            kIII.append(k3 * MM_SQRT_TO_M_SQRT)
            kI.append(k1 * MM_SQRT_TO_M_SQRT)
            kII.append(k2 * MM_SQRT_TO_M_SQRT)
        return {"x_mm": np.asarray(out_x), "K_I": np.asarray(kI),
                "K_II": np.asarray(kII), "K_III": np.asarray(kIII)}

    def ligament_tractions(self, u: np.ndarray, band_mm: float = 0.5) -> Dict[str, np.ndarray]:
        """Ligament-plane traction maps from upper-side near-plane tets (P1 stress).

        Constant-strain stresses of tets whose centroid sits within band_mm above
        y = 0 and below the front (z < t-a), reported at the tet centroid (x, z):
        sigma_yy (normal), tau_yx (Mode-III driver), tau_yz.
        """
        m = self.md.mesh
        p, t = m.p, m.t
        cen = p[:, t].mean(axis=1)
        c = self.t - self.a
        sel = (cen[1] > 1e-6) & (cen[1] < band_mm) & (cen[2] < c - 0.05)
        idx = np.where(sel)[0]
        lam, mu = lame_parameters(self.E, self.nu)
        nd = self.basis.nodal_dofs
        sy, sx, sz, cx, cz = [], [], [], [], []
        for e in idx:
            nids = t[:, e]
            X = p[:, nids].T          # (4,3)
            U = np.c_[u[nd[0, nids]], u[nd[1, nids]], u[nd[2, nids]]]  # (4,3)
            M = np.c_[np.ones(4), X]  # linear shape coefficient system
            try:
                coef = np.linalg.solve(M, U)  # rows: const, d/dx, d/dy, d/dz
            except np.linalg.LinAlgError:
                continue
            gradu = coef[1:4].T        # (3,3): du_i/dx_j
            eps = 0.5 * (gradu + gradu.T)
            sig = lam * np.trace(eps) * np.eye(3) + 2 * mu * eps
            sy.append(sig[1, 1]); sx.append(sig[0, 1]); sz.append(sig[2, 1])
            cx.append(cen[0, e]); cz.append(cen[2, e])
        return {"x_mm": np.asarray(cx), "z_mm": np.asarray(cz),
                "sigma_yy_MPa_per_kN": np.asarray(sy),
                "tau_yx_MPa_per_kN": np.asarray(sx),
                "tau_yz_MPa_per_kN": np.asarray(sz)}

    # -- one-call probe solve ---------------------------------------------------------

    def probe_solution(self, stations_mm: Tuple[float, ...] = (15.7, -15.7),
                       P_kN: float = 1.0) -> LinearSolution:
        u = self.solve(P_kN)
        gauges = {f"{s:+.1f}mm": self.gauge_step(u, s) for s in stations_mm}
        for s in stations_mm:
            gauges[f"{s:+.1f}mm"]["w_antisym_index"] = self.w_antisym_index(u, s)
        Kf = self.sif_along_front(u)
        x, k3 = Kf["x_mm"], Kf["K_III"]
        core = np.abs(x) < 0.9 * self.R  # exclude the corner zones
        pref_over_Y = self.ph.sif_prefactor(self.a) / self.ph.y_III(self.a)
        k3_avg = float(np.mean(k3[core])) if core.any() else float("nan")
        k3_max = float(np.max(np.abs(k3[core]))) if core.any() else float("nan")
        k1_avg = float(np.mean(Kf["K_I"][core])) if core.any() else float("nan")
        k2_core = Kf["K_II"][core]
        k2_absavg = float(np.mean(np.abs(k2_core))) if core.any() else float("nan")
        ksum = {
            "K_III_avg_MPa_sqrt_m_per_kN": k3_avg / P_kN,
            "K_III_max_MPa_sqrt_m_per_kN": k3_max / P_kN,
            "K_I_avg_MPa_sqrt_m_per_kN": k1_avg / P_kN,
            "K_II_absavg_MPa_sqrt_m_per_kN": k2_absavg / P_kN,
            "Y_III_recovered_avg": k3_avg / (pref_over_Y * P_kN),
            "Y_III_recovered_max": k3_max / (pref_over_Y * P_kN),
            "Y_III_target": self.ph.y_III(self.a),
            "KI_over_KIII_avg": (k1_avg / k3_avg) if k3_avg else float("nan"),
            "KII_over_KIII_absavg": (k2_absavg / k3_avg) if k3_avg else float("nan"),
        }
        lig = self.ligament_tractions(u)
        checks = dict(self._load_checks)
        checks["mesh_n_nodes"] = self.md.stats["n_nodes"]
        checks["mesh_n_tets"] = self.md.stats["n_tets"]
        return LinearSolution(u=u, P_kN=P_kN, gauge=gauges, K_front=Kf,
                              K_summary=ksum, ligament=lig, checks=checks)


# ------------------------------------------------------------------------------------
# Self-tests (assembly / units / load-path zeroth-order correctness)
# ------------------------------------------------------------------------------------


def selftest_patch(E: float = 14300.0, nu: float = 0.28) -> Dict[str, float]:
    """Patch test on a slit-free coarse mesh: affine Dirichlet -> uniform stress."""
    md = build_anbd_mesh(a_mm=0.0, scale=2.0)  # a=0: no slit split (front at z=t)
    basis = Basis(md.mesh, ElementVector(ElementTetP1()))
    lam, mu = lame_parameters(E, nu)
    K = asm(linear_elasticity(lam, mu), basis)
    p = md.mesh.p
    A = np.array([[2e-4, 5e-5, 0.0], [5e-5, -1e-4, 3e-5], [0.0, 3e-5, 8e-5]])
    u_exact = (A @ p).T.ravel()
    u_full = np.zeros(basis.N)
    nd = basis.nodal_dofs
    for comp in range(3):
        u_full[nd[comp]] = (A @ p)[comp]
    bnodes = np.unique(md.mesh.facets[:, md.mesh.boundary_facets()])
    D = np.unique(nd[:, bnodes].ravel())
    from skfem import condense, solve as sk_solve
    f = np.zeros(basis.N)
    x = sk_solve(*condense(K, f, x=u_full, D=D))
    err = float(np.max(np.abs(x - u_full)) / (np.max(np.abs(u_full)) + 1e-30))
    return {"patch_rel_err": err}


def selftest_brazilian_center(scale: float = 1.6) -> Dict[str, float]:
    """Un-notched disc under the probe load: centre splitting stress vs 2D value.

    2D Brazilian: sigma_tension(centre) = P/(pi R B) transverse to the load line,
    sigma_compression(centre) = -3 P/(pi R B) along it (B = full thickness). The 3D
    mid-plane values deviate by some percent; the probe gate is order/sign fidelity.
    """
    model = Anbd3D(a_mm=0.0, scale=scale)
    u = model.solve(P_kN=1.0)
    # stress at tets nearest the centre (mid-plane z ~ 0)
    m = model.md.mesh
    cen = m.p[:, m.t].mean(axis=1)
    d2 = cen[0] ** 2 + cen[1] ** 2 + cen[2] ** 2
    idx = np.argsort(d2)[:24]
    lam, mu = lame_parameters(model.E, model.nu)
    nd = model.basis.nodal_dofs
    sigs = []
    for e in idx:
        nids = m.t[:, e]
        X = m.p[:, nids].T
        U = np.c_[u[nd[0, nids]], u[nd[1, nids]], u[nd[2, nids]]]
        coef = np.linalg.solve(np.c_[np.ones(4), X], U)
        gradu = coef[1:4].T
        eps = 0.5 * (gradu + gradu.T)
        sigs.append(lam * np.trace(eps) * np.eye(3) + 2 * mu * eps)
    S = np.mean(np.stack(sigs), axis=0)
    # rotate into the load frame (load direction d at alpha from +x)
    aL = math.radians(model.alpha)
    d = np.array([math.cos(aL), math.sin(aL), 0.0])
    n = np.array([-math.sin(aL), math.cos(aL), 0.0])
    s_load = float(d @ S @ d)
    s_perp = float(n @ S @ n)
    B = 2 * model.t
    ref = 1000.0 / (math.pi * model.R * B)  # MPa at P = 1 kN
    return {"sigma_perp_MPa": s_perp, "sigma_perp_over_2D": s_perp / ref,
            "sigma_load_MPa": s_load, "sigma_load_over_minus3x2D": s_load / (-3 * ref),
            "ref_2D_MPa": ref}


def selftest_prescribed_K(K0: float = 2.0, a_mm: float = 6.0, scale: float = 1.0,
                          mesh_kwargs: Optional[Dict] = None) -> Dict[str, float]:
    """End-to-end K_III-extraction validation on the true slit mesh (non-circular).

    The pure anti-plane crack asymptote u = (w(y, z), 0, 0) with
    w = (2 K0 / G) sqrt(r / 2pi) sin(theta/2) around the straight front z = t - a
    is an EXACT 3D elasticity solution (harmonic w, zero dilatation) whose crack
    flanks are traction-free. Prescribe it as Dirichlet data on the outer boundary
    (rim, z = 0, z = t) only, leave the slit faces natural, solve, and recover K0
    from the same flank fit used by the probe. Validates node splitting, assembly,
    the fit constants and the unit conversion, independent of the load path.
    mesh_kwargs (e.g. corner_refine) passes through to build_anbd_mesh; the
    asymptote stays an exact solution on any conforming slit mesh.
    """
    md = build_anbd_mesh(a_mm=a_mm, scale=scale, **(mesh_kwargs or {}))
    basis = Basis(md.mesh, ElementVector(ElementTetP1()))
    E, nu = 14300.0, 0.28
    G = E / (2.0 * (1.0 + nu))
    lam, mu = lame_parameters(E, nu)
    K = asm(linear_elasticity(lam, mu), basis)
    p = md.mesh.p
    c = md.t_mm - md.a_mm
    # exact field: front-local polars in the (y, z-c) plane; ahead = -z direction
    s = c - p[2]
    eta = p[1].copy()
    # split flank nodes carry eta = +/-0: give them their side's sign explicitly
    plus_ids = set(int(i) for i in md.slit_pairs[:, 1])
    minus_ids = set(int(i) for i in md.slit_pairs[:, 0])
    for nid in plus_ids:
        eta[nid] = +1e-12
    for nid in minus_ids:
        eta[nid] = -1e-12
    r = np.hypot(s, eta)
    th = np.arctan2(eta, s)
    w_exact = (2.0 * K0 / G) * np.sqrt(np.maximum(r, 0.0) / (2.0 * math.pi)) * np.sin(th / 2.0)
    # Dirichlet on the outer boundary only (rim + both z faces), slit faces natural
    bnodes = np.unique(md.mesh.facets[:, md.mesh.boundary_facets()])
    rad = np.hypot(p[0, bnodes], p[1, bnodes])
    on_rim = rad > md.R_mm - 1e-6
    on_zf = (np.abs(p[2, bnodes]) < 1e-9) | (np.abs(p[2, bnodes] - md.t_mm) < 1e-9)
    fix = bnodes[on_rim | on_zf]
    nd = basis.nodal_dofs
    x_full = np.zeros(basis.N)
    x_full[nd[0]] = w_exact
    D = np.unique(nd[:, fix].ravel())
    from skfem import condense, solve as sk_solve
    f = np.zeros(basis.N)
    u = sk_solve(*condense(K, f, x=x_full, D=D))
    # recover K along the front with the probe's fit
    model_like = Anbd3D.__new__(Anbd3D)  # reuse the fit without rebuilding a model
    model_like.md = md
    model_like.basis = basis
    model_like.G = G
    model_like.E = E
    model_like.nu = nu
    model_like.R = md.R_mm
    model_like.t = md.t_mm
    model_like.a = md.a_mm
    Kf = Anbd3D.sif_along_front(model_like, u)
    core = np.abs(Kf["x_mm"]) < 0.8 * md.R_mm
    k3 = Kf["K_III"][core] / MM_SQRT_TO_M_SQRT  # MPa*sqrt(mm), K0's unit
    return {
        "K0": K0,
        "K_recovered_mean": float(np.mean(k3)),
        "K_recovered_rel_err": float(np.mean(k3) / K0 - 1.0),
        "K_profile_cv": float(np.std(k3) / abs(np.mean(k3))),
    }


def benchmark_bahrami_table1(scale: float = 1.0, nu: float = 0.25) -> Dict[str, float]:
    """Independent published benchmark: Bahrami2022 Table 1 row (R001/full.md line 76).

    Geometry R = 42 mm, half-thickness t = 30 mm (their '2t = 60 mm' cores, line 88),
    a/t = 0.3, alpha = 10 deg, Bedretto granite E = 75 GPa (line 86; nu unreported
    there, 0.25 assumed -- K* is only weakly nu-dependent, run_probe adds +/-0.05).
    Published: K_I* avg = -0.01, K_III* avg = 1.70, K_III* max = 2.76, K_II/K_III
    avg = 0.13; normalization Eq.(2) line 63: K* = K * 2 pi R t / (F sqrt(pi a)).
    """
    R, t = 42.0, 30.0
    a = 0.3 * t
    model = Anbd3D(a_mm=a, scale=scale, R_mm=R, t_mm=t, E_MPa=75000.0, nu=nu,
                   alpha_deg=10.0)
    u = model.solve(P_kN=1.0)
    Kf = model.sif_along_front(u)
    core = np.abs(Kf["x_mm"]) < 0.9 * R
    F_N = 1000.0
    norm = 2.0 * math.pi * R * t / (F_N * math.sqrt(math.pi * a))  # 1/(MPa*sqrt(mm))
    k3 = Kf["K_III"][core] / MM_SQRT_TO_M_SQRT * norm
    k1 = Kf["K_I"][core] / MM_SQRT_TO_M_SQRT * norm
    k2 = Kf["K_II"][core] / MM_SQRT_TO_M_SQRT * norm
    return {
        "nu": nu,
        "K3_star_avg": float(np.mean(k3)),
        "K3_star_absavg": float(np.abs(np.mean(k3))),
        "K3_star_absmax": float(np.max(np.abs(k3))),
        "K1_star_avg": float(np.mean(k1)),
        "K1_over_K3_avg": float(np.mean(k1) / np.mean(k3)),
        "K2_over_K3_halffront": float(np.mean(np.abs(k2)) / abs(np.mean(k3))),
        "published_K3_star_avg": 1.70,
        "published_K3_star_max": 2.76,
        "published_K1_star_avg": -0.01,
        "published_K2_over_K3_avg": 0.13,
        "rel_err_K3_avg_vs_published": float(abs(np.mean(k3)) / 1.70 - 1.0),
        "n_nodes": int(model.md.stats["n_nodes"]),
    }


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--benchmark", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        r1 = selftest_patch()
        print(f"[patch] rel_err = {r1['patch_rel_err']:.3e}")
        r2 = selftest_brazilian_center()
        for k, v in r2.items():
            print(f"[brazilian] {k} = {v:.4f}")
        r3 = selftest_prescribed_K()
        for k, v in r3.items():
            print(f"[prescribed-K] {k} = {v:.4f}")
    if args.benchmark:
        rb = benchmark_bahrami_table1()
        for k, v in rb.items():
            print(f"[bahrami-t1] {k} = {v}")
