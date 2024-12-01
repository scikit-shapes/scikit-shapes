from functools import cached_property

import torch

from ..input_validation import typecheck
from ..types import (
    EdgeLengths,
    EdgeMidpoints,
    EdgePoints,
    TriangleAreas,
    TriangleCentroids,
    TrianglePoints,
)


@cached_property
@typecheck
def edge_points(self) -> EdgePoints:
    """The coordinates of edge vertices.

    Returns
    -------
    edge_points
        A ``(n_edges, 2, dim)`` Tensor (where the dimension of the ambient space
        ``dim`` is typically 2 or 3) that contains the coordinates of edge vertices,
        so that ``self.edge_points[e, i, :] == self.points[self.edges[e, i], :]``.


    Examples
    --------

    .. testcode::

        import skshapes as sks

        mesh = sks.PolyData(
            points=[[0, 0], [1, 0], [1, 1]],
            triangles=[[0, 1, 2]],
        )
        print(mesh.edges)

    .. testoutput::

        tensor([[0, 1],
                [0, 2],
                [1, 2]])

    .. testcode::

        print(mesh.edge_points)

    .. testoutput::

        tensor([[[0., 0.],
                 [1., 0.]],

                [[0., 0.],
                 [1., 1.]],

                [[1., 0.],
                 [1., 1.]]])

    """
    # Raise an error if edges are not defined
    if self.edges is None:
        msg = "Edges are not defined"
        raise AttributeError(msg)

    # Warning: Indexing efficiently in PyTorch is tricky.
    # This is, in part, due to the fact that the backward of an indexing operator
    # is a sum, and that the precise result of a sum with floating point numbers
    # depends on the order of the summation : (a + b) + c != a + (b + c).
    #
    # This means that 100% deterministic floating point operations cannot rely on
    # the fastest GPU kernels.
    #
    # As a consequence, there is a difference between:
    #
    # - points[indices], whose backward pass is 100% deterministic but slow.
    #
    # - torch.index_select(input=points, dim=0, index=indices),
    #   whose backward can be *much* faster on GPU for some values of indices
    #   (since it relies on scatter_add and does not try to perform sums in a
    #   specific sequential order), but may not be 100% deterministic
    #   due to floating point rounding uncertainties.
    #
    # Whenever relevant, we use torch.index_select instead of the basic indexing
    # operator.
    #
    # Equivalent to A = self.points[self.edges[:, 0]]
    A = torch.index_select(input=self.points, dim=0, index=self.edges[:, 0])
    # Equivalent to B = self.points[self.edges[:, 1]]
    B = torch.index_select(input=self.points, dim=0, index=self.edges[:, 1])
    assert A.shape == (self.n_edges, self.dim)
    assert B.shape == (self.n_edges, self.dim)

    edge_points = torch.stack((A, B), dim=1)
    assert edge_points.shape == (self.n_edges, 2, self.dim)
    return edge_points


@cached_property
@typecheck
def triangle_points(self) -> TrianglePoints:
    """The coordinates of triangle vertices.

    Returns
    -------
    triangle_points
        A ``(n_triangles, 3, dim)`` Tensor (where the dimension of the ambient space
        ``dim`` is typically 2 or 3) that contains the coordinates of triangle
        vertices, so that
        ``self.triangle_points[t, i, :] == self.points[self.triangle[t, i], :]``.


    Examples
    --------

    .. testcode::

        import skshapes as sks

        mesh = sks.PolyData(
            points=[[0, 0], [1, 0], [0, 1], [1, 1]],
            triangles=[[0, 1, 2], [1, 2, 3]],
        )
        print(mesh.triangle_points)

    .. testoutput::

        tensor([[[0., 0.],
                 [1., 0.],
                 [0., 1.]],

                [[1., 0.],
                 [0., 1.],
                 [1., 1.]]])

    """
    # Raise an error if triangles are not defined
    if self.triangles is None:
        msg = "Triangles are not defined"
        raise AttributeError(msg)

    # Warning: Indexing efficiently in PyTorch is tricky, see self.edge_points(...)
    # above.
    # Whenever relevant, we use torch.index_select instead of the basic indexing
    # operator.
    #
    # Equivalent to A = self.points[self.triangles[:, 0]]
    A = torch.index_select(
        input=self.points, dim=0, index=self.triangles[:, 0]
    )
    # Equivalent to B = self.points[self.triangles[:, 1]]
    B = torch.index_select(
        input=self.points, dim=0, index=self.triangles[:, 1]
    )
    # Equivalent to C = self.points[self.triangles[:, 1]]
    C = torch.index_select(
        input=self.points, dim=0, index=self.triangles[:, 2]
    )
    assert A.shape == (self.n_triangles, self.dim)
    assert B.shape == (self.n_triangles, self.dim)
    assert C.shape == (self.n_triangles, self.dim)

    triangle_points = torch.stack((A, B, C), dim=1)
    assert triangle_points.shape == (self.n_triangles, 3, self.dim)
    return triangle_points


@cached_property
@typecheck
def edge_midpoints(self) -> EdgeMidpoints:
    """The coordinates of points at the center of each edge.

    Returns
    -------
    edge_midpoints
        A ``(n_edges, dim)`` Tensor (where the dimension of the ambient space
        ``dim`` is typically 2 or 3) that contains the coordinates of edge midpoints.


    Examples
    --------

    .. testcode::

        import skshapes as sks

        mesh = sks.PolyData(
            points=[[0, 0], [1, 0], [1, 1]],
            triangles=[[0, 1, 2]],
        )
        print(mesh.edges)

    .. testoutput::

        tensor([[0, 1],
                [0, 2],
                [1, 2]])

    .. testcode::

        print(mesh.edge_midpoints)

    .. testoutput::

        tensor([[0.5000, 0.0000],
                [0.5000, 0.5000],
                [1.0000, 0.5000]])
    """
    midpoints = self.edge_points.mean(dim=1)
    assert midpoints.shape == (self.n_edges, self.dim)
    return midpoints


@cached_property
@typecheck
def edge_lengths(self) -> EdgeLengths:
    """The lengths of all edge segments.

    Returns
    -------
    edge_lengths
        A ``(n_edges,)`` Tensor that contains the lengths of all the edge segments.

    Examples
    --------

    .. testcode::

        import skshapes as sks

        mesh = sks.PolyData(
            points=[[0, 0], [1, 0], [1, 1]],
            triangles=[[0, 1, 2]],
        )
        print(mesh.edges)

    .. testoutput::

        tensor([[0, 1],
                [0, 2],
                [1, 2]])

    .. testcode::

        print(mesh.edge_lengths)

    .. testoutput::

        tensor([1.0000, 1.4142, 1.0000])

    """
    lengths = (self.edge_points[:, 0, :] - self.edge_points[:, 1, :]).norm(
        dim=1
    )
    assert lengths.shape == (self.n_edges,)
    return lengths


@cached_property
@typecheck
def triangle_centroids(self) -> TriangleCentroids:
    """The coordinates of points at the arithmetic center of each triangle.

    Returns
    -------
    triangle_centroids
        A ``(n_triangle, dim)`` Tensor (where the dimension of the ambient space
        ``dim`` is typically 2 or 3) that contains the coordinates of all
        triangle centers.


    Examples
    --------

    .. testcode::

        import skshapes as sks

        mesh = sks.PolyData(
            points=[[0, 0], [1, 0], [1, 1]],
            triangles=[[0, 1, 2], [0, 0, 1], [2, 2, 2]],
        )
        print(mesh.triangle_centroids)

    .. testoutput::

        tensor([[0.6667, 0.3333],
                [0.3333, 0.0000],
                [1.0000, 1.0000]])

    """
    midpoints = self.triangle_points.mean(dim=1)
    assert midpoints.shape == (self.n_triangles, self.dim)
    return midpoints


@cached_property
@typecheck
def triangle_areas(self) -> TriangleAreas:
    """Area of each triangle."""
    # Raise an error if triangles are not defined
    return self.triangle_area_normals.norm(dim=1) / 2
