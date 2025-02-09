"""Point normals and tangent vectors."""

import torch
import torch.nn.functional as F
from pykeops.torch import LazyTensor

from ..input_validation import typecheck
from ..types import FloatTensor, Number, Points, TriangleNormals, Triangles
from ..utils import diagonal_ranges, scatter


@typecheck
def _triangle_area_normals(self) -> TriangleNormals:
    r"""The normals of the mesh triangles, weighted by their areas.

    For 3D triangles :math:`ABC`, this is the cross product
    :math:`\tfrac{1}{2} \overrightarrow{AB} \times \overrightarrow{AC}`.
    For 2D triangles, this is the 3D vector ``(0, 0, signed_area)``.

    Returns
    -------
    area_normals
        A ``(n_triangles, 3)`` tensor that contains the normal vector of each triangle,
        weighted by its area.

    Examples
    --------

    .. testcode::

        import skshapes as sks

        mesh_2D = sks.PolyData(
            points=[[0, 0], [1, 0], [1, 1]],
            triangles=[[0, 1, 2], [0, 2, 1], [0, 1, 1], [2, 2, 2]],
        )
        print(mesh_2D.triangle_area_normals)

    .. testoutput::

        tensor([[ 0.0000,  0.0000,  0.5000],
                [ 0.0000,  0.0000, -0.5000],
                [ 0.0000,  0.0000,  0.0000],
                [ 0.0000,  0.0000,  0.0000]])

    .. testcode::

        mesh_3D = sks.PolyData(
            points=[[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 2]],
            triangles=[[0, 1, 2], [0, 1, 3], [0, 2, 3]],
        )
        print(mesh_3D.triangle_area_normals)

    .. testoutput::

        tensor([[ 0.0000,  0.0000,  0.5000],
                [ 0.0000, -1.0000,  0.0000],
                [ 1.0000,  0.0000,  0.0000]])

    """
    ABC = self.triangle_points  # (n_triangles, 3, dim)
    A = ABC[:, 0, :]
    B = ABC[:, 1, :]
    C = ABC[:, 2, :]

    if self.dim == 2:
        # Add a zero z coordinate to the points to compute the
        # cross product
        A = torch.cat((A, torch.zeros_like(A[:, 0]).view(-1, 1)), dim=1)
        B = torch.cat((B, torch.zeros_like(B[:, 0]).view(-1, 1)), dim=1)
        C = torch.cat((C, torch.zeros_like(C[:, 0]).view(-1, 1)), dim=1)

    area_normals = torch.linalg.cross(B - A, C - A)
    assert area_normals.shape == (self.n_triangles, 3)
    return area_normals / 2


@typecheck
def _triangle_normals(self) -> TriangleNormals:
    """Unit-length normals associated to each triangle.

    Please note that if a triangle is degenerate (i.e. with zero area),
    the normal vector will be zero.

    Returns
    -------
    triangle_normals
        A ``(n_triangles, 3)`` tensor that contains the normal vector of each triangle.

    Examples
    --------

    .. testcode::

        import skshapes as sks

        mesh_2D = sks.PolyData(
            points=[[0, 0], [1, 0], [1, 1]],
            triangles=[[0, 1, 2], [0, 2, 1], [0, 1, 1], [2, 2, 2]],
        )
        print(mesh_2D.triangle_normals)

    .. testoutput::

        tensor([[ 0.,  0.,  1.],
                [ 0.,  0., -1.],
                [ 0.,  0.,  0.],
                [ 0.,  0.,  0.]])

    .. testcode::

        mesh_3D = sks.PolyData(
            points=[[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 4]],
            triangles=[[0, 1, 2], [0, 1, 3], [0, 2, 3], [0, 1, 1], [2, 2, 2]],
        )
        print(mesh_3D.triangle_normals)

    .. testoutput::

        tensor([[ 0.,  0.,  1.],
                [ 0., -1.,  0.],
                [ 1.,  0.,  0.],
                [ 0.,  0.,  0.],
                [ 0.,  0.,  0.]])

    """
    return torch.nn.functional.normalize(self.triangle_area_normals)


@typecheck
def _point_normals(
    self,
    **kwargs,
) -> Points:
    """Compute a smooth field of normals at the vertices of a mesh.

    Please refer to :ref:`this example <point_normals_example>` for an illustration
    of the difference between parameter values.

    Parameters
    ----------
    kwargs :
        These arguments will be passed to
        :meth:`~skshapes.Polydata.point_neighborhoods`
        in order to create a neighborhood structure.

    Returns
    -------
    Points
        A ``(n_points, 3)`` tensor of normal vectors at the vertices of the mesh.


    Examples
    --------

    .. testcode::

        import skshapes as sks

        mesh = sks.Sphere()
        raw_normals = mesh.point_normals()
        smooth_normals = mesh.point_normals(
            method="gaussian kernel",
            scale=0.3,
        )

        print(mesh.points.shape, raw_normals.shape, smooth_normals.shape)

    .. testoutput::

        torch.Size([842, 3]) torch.Size([842, 3]) torch.Size([842, 3])

    """
    if self.dim != 3:
        msg = "Currently, point normals are only supported in 3D."
        raise NotImplementedError(msg)

    # N.B.: self.point_neighborhoods is a cached method
    point_neighborhoods = self.point_neighborhoods(**kwargs)

    if self.is_triangle_mesh:
        # On a triangle mesh, "raw" point_normals are computed by aggregating
        # triangle normals onto vertices.

        # TODO: instead of distributing to the vertices equally, we should
        #       distribute according to the angles of the triangles,
        #       via the cotangent formula (see Keenan Crane's lecture notes
        #       on discrete differential geometry)
        raw_normals = sum(
            scatter(
                src=self.triangle_area_normals,  # N.B.: magnitude = triangle area,
                index=self.triangles[:, k],
                reduce="sum",
                min_length=self.n_points,
            )
            for k in range(3)
        )
        assert raw_normals.shape == self.points.shape
        raw_normals = F.normalize(raw_normals, p=2, dim=-1)
        assert raw_normals.shape == self.points.shape

        # N.B.: if point_neighborhoods is trivial, smooth_normals == raw_normals
        smooth_normals = point_neighborhoods.smooth(
            signal=raw_normals,
            input_type="function",
            output_type="function",
        )
        assert smooth_normals.shape == (self.n_points, 3)

    else:
        # if point_neighborhoods.is_trivial:
        #    msg = "If no triangles are provided, you must specify a non-trivial neighborhood structure with e.g. a positive keyword argument scale=..."
        #    raise ValueError(msg)

        # Get a smooth field of normals via the Structure Tensor:
        local_covariance = self.point_moments(**kwargs).covariances
        assert local_covariance.shape == (self.n_points, self.dim, self.dim)

        local_QL = torch.linalg.eigh(local_covariance)
        local_nuv = local_QL.eigenvectors  # (N, 3, 3)
        assert local_nuv.shape == (self.n_points, self.dim, self.dim)

        # The normal is the eigenvector associated to the smallest eigenvalue
        raw_normals = local_nuv[:, :, 0]

        # At this stage, smooth_normals is only defined up to a sign.
        # Arbitrarily, we decide to orient the normals to point locally outwards,
        # i.e. flip the normals if they point towards the local average:
        local_average = self.point_moments(**kwargs).means
        assert local_average.shape == self.points.shape
        local_direction = self.points - local_average

        # N.B.: sign(0) = 0, which is annoying, so we have to handle this
        #       by hand with booleans.
        assert local_direction.shape == (self.n_points, self.dim)
        no_flip = (
            ((local_direction * raw_normals).sum(-1) >= 0)
            .type_as(raw_normals)
            .view(-1, 1)
        )
        smooth_normals = (2 * no_flip - 1) * raw_normals
        assert smooth_normals.shape == (self.n_points, 3)

    smooth_normals = F.normalize(smooth_normals, p=2, dim=-1)
    assert smooth_normals.shape == (self.n_points, 3)
    return smooth_normals


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
