"""Point normals and tangent vectors."""

import torch
import torch.nn.functional as F
from pykeops.torch import LazyTensor

from ..input_validation import typecheck
from ..types import FloatTensor, Number, Points, Triangles
from ..utils import diagonal_ranges


@typecheck
def _point_normals(
    self,
    *,
    scale: Number | None = None,
    **kwargs,
) -> Points:
    if scale is None:
        if self.triangles is None:
            msg = "If no triangles are provided, you must specify a scale."
            raise ValueError(msg)

        tri_n = self.triangle_normals  # N.B.: magnitude = triangle area
        n = torch.zeros_like(self.points)
        # TODO: instead of distributing to the vertices equally, we should
        #       distribute according to the angles of the triangles.
        for k in range(3):
            # n[self.triangles[k, i]] += tri_n[i]
            n.scatter_reduce_(
                dim=0,
                index=self.triangles[k].view(-1, 1),
                src=tri_n,
                reduce="sum",
            )

    else:
        # Get a smooth field of normals via the Structure Tensor:
        local_cov = self.point_moments(
            order=2, scale=scale, central=True, **kwargs
        )
        local_QL = torch.linalg.eigh(local_cov)
        local_nuv = local_QL.eigenvectors  # (N, 3, 3)
        n = local_nuv[:, :, 0]

        # Orient the normals according to the triangles, if any:
        if self.triangles is not None:
            n_0 = self.point_normals(scale=None, **kwargs)
            assert n_0.shape == (self.n_points, 3)

        else:
            n_0 = n.mean(dim=0, keepdim=True)
            assert n_0.shape == (1, 3)

        n = (n_0 * n).sum(-1).sign().view(-1, 1) * n

    # Try to enforce some consistency...
    if False:
        n = F.normalize(n, p=2, dim=-1)
        # n_e = n[self.edges[0]] + n[self.edges[1]]
        # The backward of torch.index_select is much faster than that of
        # indexing:
        n_e = torch.index_select(n, 0, self.edges[0]) + torch.index_select(
            n, 0, self.edges[1]
        )
        assert n_e.shape == (len(self.edges[0]), 3)

        n_v = torch.zeros_like(n)
        n_v.scatter_reduce_(
            dim=0,
            index=self.edges.reshape(-1, 1),
            src=n_e.tile(2, 1),
            reduce="mean",
            include_self=False,
        )

        assert n_v.shape == (self.n_points, 3)
        # n_v = n_v - n
        n = (n_v * n).sum(-1).sign().view(-1, 1) * n

    n = F.normalize(n, p=2, dim=-1)
    assert n.shape == (self.n_points, 3)
    return n


@typecheck
def _point_frames(
    self,
    *,
    scale: Number | None = None,
    **kwargs,
) -> FloatTensor:
    N = self.n_points
    n = self.point_normals(scale=scale, **kwargs)  # (N, 3)

    # Complete the direct orthonormal basis [n,u,v]:
    uv = tangent_vectors(n)
    assert n.shape == (N, 3)
    assert uv.shape == (N, 2, 3)

    return torch.cat((n.view(N, 1, 3), uv), dim=1)


@typecheck
def smooth_normals(
    *,
    vertices: Points,
    triangles: Triangles | None = None,
    scale=None,
    batch=None,
    normals: Points | None = None,
) -> FloatTensor:
    """Smooth field of normals, possibly at different scales.

    points, triangles or normals, scale(s) -> normals
    (N, 3), (3, T) or (N,3), (S,) -> (N, 3) or (N, S, 3)

    Simply put - if `triangles` are provided:

    - Normals are first computed for every triangle using simple 3D geometry and are weighted according to surface area.
    - The normal at any given vertex is then computed as the weighted average of the normals of all triangles in a neighborhood specified by Gaussian windows whose radii are given in the list of "scales".

    If `normals` are provided instead, we simply smooth the discrete vector
    field using Gaussian windows whose radii are given in the list of "scales".

    If more than one scale is provided, normal fields are computed in parallel
    and returned in a single 3D tensor.

    Parameters
    ----------
    vertices
        (N,3) coordinates of mesh vertices or 3D points.
    triangles
        (3,T) mesh connectivity. Defaults to None.
    scale
        (S,) radii of the Gaussian smoothing windows.
    batch
        batch vector, as in PyTorch_geometric. Defaults to None.
    normals
        (N,3) raw normals vectors on the vertices. Defaults to None.

    Returns
    -------
    FloatTensor
        (N,3) or (N,S,3) point normals.

    """
    if scale is None:
        scale = [0.1]

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
        if False:
            S = (V**2).sum(-1).sqrt() / 6  # (N,) 1/3 of a triangle area
            areas = torch.zeros(len(vertices)).type_as(vertices)  # (N,)
            areas.scatter_add_(0, triangles[0, :], S)  # Aggregate from "A's"
            areas.scatter_add_(0, triangles[1, :], S)  # Aggregate from "B's"
            areas.scatter_add_(0, triangles[2, :], S)  # Aggregate from "C's"

    else:  # Use "normals" instead
        # areas = None
        V = normals
        centers = vertices

    # Normal of a vertex = average of all normals in a ball of size "scale":
    x_i = LazyTensor(vertices[:, None, :])  # (N, 1, 3)
    y_j = LazyTensor(centers[None, :, :])  # (1, M, 3)
    v_j = LazyTensor(V[None, :, :])  # (1, M, 3)
    s = LazyTensor(scales[None, None, :])  # (1, 1, S)

    D_ij = ((x_i - y_j) ** 2).sum(-1)  # (N, M, 1)
    K_ij = (-D_ij / (2 * s**2)).exp()  # (N, M, S)

    # Support for heterogeneous batch processing:
    if batch is not None:
        batch_vertices = batch
        batch_centers = (
            batch[triangles[0, :]] if triangles is not None else batch
        )
        K_ij.ranges = diagonal_ranges(batch_vertices, batch_centers)

    if single_scale:
        U = (K_ij * v_j).sum(dim=1)  # (N, 3)
    else:
        U = (K_ij.tensorprod(v_j)).sum(dim=1)  # (N, S*3)
        U = U.view(-1, len(scales), 3)  # (N, S, 3)

    return F.normalize(U, p=2, dim=-1)  # (N, 3) or (N, S, 3)


def tangent_vectors(normals) -> FloatTensor:
    """Compute tangent vectors to a normal vector field.

    Returns a pair of vector fields u and v to complete the orthonormal basis
    [n,u,v].

    normals -> uv
    (N, 3) or (N, S, 3) -> (N, 2, 3) or (N, S, 2, 3)

    This routine assumes that the 3D "normal" vectors are normalized.
    It is based on the 2017 paper from Pixar,
    "Building an orthonormal basis, revisited".

    Parameters
    ----------
    normals
        (N,3) or (N,S,3) normals `n_i`, i.e. unit-norm 3D vectors.

    Returns
    -------
    FloatTensor
        (N,2,3) or (N,S,2,3) unit vectors `u_i` and `v_i` to complete
        the tangent coordinate systems `[n_i,u_i,v_i]`.
    """
    x, y, z = normals[..., 0], normals[..., 1], normals[..., 2]
    s = (2 * (z >= 0)) - 1.0  # = z.sign(), but =1. if z=0.
    a = -1 / (s + z)
    b = x * y * a
    uv = torch.stack(
        (1 + s * x * x * a, s * b, -s * x, b, s + y * y * a, -y), dim=-1
    )
    return uv.view(uv.shape[:-1] + (2, 3))
