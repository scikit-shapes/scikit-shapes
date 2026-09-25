"""Mesh utilities: vertex areas, edge lengths, texture coordinates."""

import numpy as np
import pyvista as pv
import torch
from vtk.util.numpy_support import numpy_to_vtk


def add_xy_tcoords_vtk(m, flip_v=True):
    x = m.points[:, 0]
    y = m.points[:, 1]
    u = (x - x.min()) / (x.max() - x.min() + 1e-12)
    v = (y - y.min()) / (y.max() - y.min() + 1e-12)
    if flip_v:
        v = 1.0 - v
    uv = np.column_stack((u, v)).astype(np.float32)
    vtk_uv = numpy_to_vtk(np.ascontiguousarray(uv), deep=1)
    vtk_uv.SetName("TCoords")
    m.GetPointData().SetTCoords(vtk_uv)
    return m


def get_average_edge_length(
    mesh: pv.PolyData,
) -> float:
    edges = mesh.extract_all_edges()
    sizes = edges.compute_cell_sizes(length=True, area=False, volume=False)
    avg_length = np.mean(sizes.cell_data["Length"])
    return avg_length


def compute_vertex_areas(
    mesh: pv.PolyData,
) -> torch.Tensor:  # for normalizing the kernel
    n_points = mesh.n_points
    if mesh.n_cells == 0 or mesh.n_points == 0:
        return torch.ones(n_points, dtype=torch.float32)

    try:
        if not mesh.is_all_triangles:
            mesh = mesh.triangulate()
        if mesh.n_cells == 0:
            return torch.ones(n_points, dtype=torch.float32)

        sized = mesh.compute_cell_sizes(length=False, area=True, volume=False)
        face_areas = sized.cell_data["Area"]
        faces = mesh.faces.reshape(-1, 4)[:, 1:]

        vertex_areas = np.bincount(
            faces.flatten(), weights=np.repeat(face_areas, 3), minlength=n_points
        )
        vertex_areas = vertex_areas / 3.0
        return torch.from_numpy(vertex_areas.astype(np.float32))
    except Exception:
        return torch.ones(n_points, dtype=torch.float32)
