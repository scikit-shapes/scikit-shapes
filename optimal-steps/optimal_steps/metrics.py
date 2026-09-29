"""Surface metrics used in the paper (Table 1).

- Chamfer distance: mean of the two vertex-to-surface distances.
- HD95: max of the two 95th percentiles of the vertex-to-surface distances.
- Log-Jacobian variance: area-weighted variance of log(A_deformed / A_source)
  over the triangles, which measures how unevenly the deformation stretches
  and compresses the source mesh (0 = uniform scaling).

Distances are measured from the vertices of one mesh to the triangles of the
other (not vertex-to-vertex), so they do not depend on the sampling density.
"""

import numpy as np
import pyvista as pv


def _vertex_to_surface(points: np.ndarray, surface: pv.PolyData) -> np.ndarray:
    cloud = pv.PolyData(np.asarray(points, dtype=np.float64))
    return np.abs(cloud.compute_implicit_distance(surface)["implicit_distance"])


def surface_distances(mesh_a: pv.PolyData, mesh_b: pv.PolyData):
    """Distances from the vertices of A to B, and from the vertices of B to A."""
    return (
        _vertex_to_surface(mesh_a.points, mesh_b),
        _vertex_to_surface(mesh_b.points, mesh_a),
    )


def chamfer_hd95(mesh_a: pv.PolyData, mesh_b: pv.PolyData):
    d_ab, d_ba = surface_distances(mesh_a, mesh_b)
    chamfer = 0.5 * (d_ab.mean() + d_ba.mean())
    hd95 = max(np.percentile(d_ab, 95), np.percentile(d_ba, 95))
    return float(chamfer), float(hd95)


def log_jacobian_variance(deformed: pv.PolyData, source: pv.PolyData) -> float:
    """Area-weighted variance of the per-triangle log area ratio.

    `deformed` and `source` must share the same triangulation.
    """
    a = source.compute_cell_sizes(length=False, volume=False)["Area"]
    b = deformed.compute_cell_sizes(length=False, volume=False)["Area"]
    eps = 1e-12
    w = a / (a.sum() + eps)
    log_ratio = np.log((b + eps) / (a + eps))
    mean = np.sum(w * log_ratio)
    return float(np.sum(w * (log_ratio - mean) ** 2))


def evaluate_registration(
    deformed: pv.PolyData, target: pv.PolyData, source: pv.PolyData
) -> dict:
    """Chamfer / HD95 against the target, log-Jacobian variance against the source."""
    out = {}
    if deformed.n_faces_strict > 0 and target.n_faces_strict > 0:
        out["chamfer_mm"], out["hd95_mm"] = chamfer_hd95(deformed, target)
    if deformed.n_faces_strict > 0 and source.n_faces_strict > 0:
        out["log_jacobian_var"] = log_jacobian_variance(deformed, source)
    return out
