import torch
from pykeops.torch import LazyTensor
from ..utils import diagonal_ranges

from ..types import typecheck, Points, Optional, Triangles


@typecheck
def mesh_normals_areas(
    *,
    vertices: Points,
    triangles=Optional[Triangles],
    scale=[1.0],
    batch=None,
    normals=Optional[Points]
):
    """Returns a smooth field of normals, possibly at different scales.

    points, triangles or normals, scale(s)  ->      normals
    (N, 3),    (3, T) or (N,3),      (S,)   ->  (N, 3) or (N, S, 3)

    Simply put - if `triangles` are provided:
      1. Normals are first computed for every triangle using simple 3D geometry
         and are weighted according to surface area.
      2. The normal at any given vertex is then computed as the weighted average
         of the normals of all triangles in a neighborhood specified
         by Gaussian windows whose radii are given in the list of "scales".

    If `normals` are provided instead, we simply smooth the discrete vector
    field using Gaussian windows whose radii are given in the list of "scales".

    If more than one scale is provided, normal fields are computed in parallel
    and returned in a single 3D tensor.

    Args:
        vertices (Tensor): (N,3) coordinates of mesh vertices or 3D points.
        triangles (integer Tensor, optional): (3,T) mesh connectivity. Defaults to None.
        scale (list of floats, optional): (S,) radii of the Gaussian smoothing windows. Defaults to [1.].
        batch (integer Tensor, optional): batch vector, as in PyTorch_geometric. Defaults to None.
        normals (Tensor, optional): (N,3) raw normals vectors on the vertices. Defaults to None.

    Returns:
        (Tensor): (N,3) or (N,S,3) point normals.
        (Tensor): (N,) point areas, if triangles were provided.
    """

    # Single- or Multi-scale mode:
    if hasattr(scale, "__len__"):
        scales, single_scale = scale, False
    else:
        scales, single_scale = [scale], True
    scales = torch.Tensor(scales).type_as(vertices)  # (S,)

    # Compute the "raw" field of normals:
    if triangles is not None:
        # Vertices of all triangles in the mesh:
        A = vertices[triangles[0, :]]  # (N, 3)
        B = vertices[triangles[1, :]]  # (N, 3)
        C = vertices[triangles[2, :]]  # (N, 3)

        # Triangle centers and normals (length = surface area):
        centers = (A + B + C) / 3  # (N, 3)
        V = (B - A).cross(C - A)  # (N, 3)

        # Vertice areas:
        S = (V**2).sum(-1).sqrt() / 6  # (N,) 1/3 of a triangle area
        areas = torch.zeros(len(vertices)).type_as(vertices)  # (N,)
        areas.scatter_add_(0, triangles[0, :], S)  # Aggregate from "A's"
        areas.scatter_add_(0, triangles[1, :], S)  # Aggregate from "B's"
        areas.scatter_add_(0, triangles[2, :], S)  # Aggregate from "C's"

    else:  # Use "normals" instead
        areas = None
        V = normals
        centers = vertices

    # Normal of a vertex = average of all normals in a ball of size "scale":
    x_i = LazyTensor(vertices[:, None, :])  # (N, 1, 3)
    y_j = LazyTensor(centers[None, :, :])  # (1, M, 3)
    v_j = LazyTensor(V[None, :, :])  # (1, M, 3)
    s = LazyTensor(scales[None, None, :])  # (1, 1, S)

    D_ij = ((x_i - y_j) ** 2).sum(-1)  #  (N, M, 1)
    K_ij = (-D_ij / (2 * s**2)).exp()  # (N, M, S)

    # Support for heterogeneous batch processing:
    if batch is not None:
        batch_vertices = batch
        batch_centers = batch[triangles[0, :]] if triangles is not None else batch
        K_ij.ranges = diagonal_ranges(batch_vertices, batch_centers)

    if single_scale:
        U = (K_ij * v_j).sum(dim=1)  # (N, 3)
    else:
        U = (K_ij.tensorprod(v_j)).sum(dim=1)  # (N, S*3)
        U = U.view(-1, len(scales), 3)  # (N, S, 3)

    normals = F.normalize(U, p=2, dim=-1)  # (N, 3) or (N, S, 3)

    return normals, areas
