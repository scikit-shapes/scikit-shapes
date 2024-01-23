"""Geometric properties of a triangular mesh.

This module provides functions to compute geometric properties of a triangle
mesh such as triangle areas, edge lengths, dihedral angles, etc.

In addition, it is possible to compute these properties for a set of triangle
meshes at the same time, if they have the same topology. This is useful when
we need to compute the same property along a sequence of deformation of a
triangle mesh.

All the function on this module are implemented in PyTorch. They take the
points and triangles of the mesh as input and return a tensor with the
computed property.

Arguments points can be either a tensor of shape:
- (d_points, dim) for a single mesh
- (n_meshes, n_poses, dim) for a sequence of poses of the same mesh
"""

from typing import Optional, Union

import torch

from ..input_validation import convert_inputs, typecheck
from ..types import (
    Float1dTensor,
    Float2dTensor,
    Float3dTensor,
    Points,
    PointsSequence,
    Triangles,
)
from .edge_topology import EdgeTopology


@convert_inputs
@typecheck
def triangle_normals(
    *,
    points: Union[Points, PointsSequence],
    triangles: Triangles,
    **kwargs,
) -> Union[Float2dTensor, Float3dTensor]:
    """Triangle normals of a triangular mesh.

    Parameters
    ----------
    points
        Points or sequence of points with shape (n_points, n_poses, dim).
    triangles
        Triangles of the mesh.

    Returns
    -------
        The normals of the triangles with shape (n_triangles, dim) for a single
        mesh or (n_meshes, n_triangles, dim) for a sequence of meshes.
    """
    A = points[triangles[:, 0]]
    B = points[triangles[:, 1]]
    C = points[triangles[:, 2]]

    dim = points.shape[1]
    if dim == 2:
        # Add a zero z coordinate to the points to compute the
        # cross product
        A = torch.cat((A, torch.zeros_like(A[:, 0]).view(-1, 1)), dim=1)
        B = torch.cat((B, torch.zeros_like(B[:, 0]).view(-1, 1)), dim=1)
        C = torch.cat((C, torch.zeros_like(C[:, 0]).view(-1, 1)), dim=1)

    return torch.cross(B - A, C - A)


@convert_inputs
@typecheck
def triangle_areas(
    *,
    points: Union[Points, PointsSequence],
    triangles: Triangles,
    **kwargs,
) -> Union[Float1dTensor, Float2dTensor]:
    """Areas of the triangles of a triangular mesh.

    Parameters
    ----------
    points
        Points or sequence of points with shape (n_points, n_poses, dim).
    triangles
        Triangles of the mesh.

    Returns
    -------
        The areas of the triangles with shape (n_triangles,) for a single
        mesh or (n_triangles, n_poses) for a sequence of meshes.
    """
    normals = triangle_normals(points=points, triangles=triangles)
    return 0.5 * torch.norm(normals, dim=-1)


# TODO: edge_areas and point_areas


@typecheck
def edge_lengths(
    *,
    points: Union[Points, PointsSequence],
    triangles: Triangles,
    edge_topology: Optional[EdgeTopology] = None,
) -> Union[Float1dTensor, Float2dTensor]:
    """Lengths of the edges of a triangular mesh.

    Parameters
    ----------
    points
        Points or sequence of points with shape (n_points, n_poses, dim).
    triangles
        Triangles of the mesh.
    edge_topology
        Edge topology of the mesh. If not provided, it will be computed from
        the triangles.

    Returns
    -------
        The lengths of the edges with shape (n_edges,) for a single
        mesh or (n_edges, n_poses) for a sequence of meshes.
    """
    if edge_topology is None:
        edge_topology = EdgeTopology(triangles)

    edges = edge_topology.edges
    return torch.norm(points[edges[:, 0]] - points[edges[:, 1]], dim=-1)


@typecheck
def triangle_centers(
    *,
    points: Union[Points, PointsSequence],
    triangles: Triangles,
    **kwargs,
) -> Union[Float2dTensor, Float3dTensor]:
    """Centers of the triangles of a triangular mesh.

    Parameters
    ----------
    points
        Points or sequence of points with shape (n_points, n_poses, dim).
    triangles
        Triangles of the mesh.

    Returns
    -------
        The centers of the triangles with shape (n_triangles, dim) for a single
        mesh or (n_triangles, n_poses, dim) for a sequence of meshes.
    """
    return (
        points[triangles[:, 0]]
        + points[triangles[:, 1]]
        + points[triangles[:, 2]]
    ) / 3


@typecheck
def edge_centers(
    points: Union[Points, PointsSequence],
    triangles: Triangles,
    edge_topology: Optional[EdgeTopology] = None,
) -> Union[Float2dTensor, Float3dTensor]:
    """Centers of the edges of a triangular mesh.

    Parameters
    ----------
    points
        Points or sequence of points with shape (n_points, n_poses, dim).
    triangles
        Triangles of the mesh.
    edge_topology
        Edge topology of the mesh. If not provided, it will be computed from
        the triangles.

    Returns
    -------
        _description_
    """
    if edge_topology is None:
        edge_topology = EdgeTopology(triangles)

    edges = edge_topology.edges
    return (points[edges[:, 0]] + points[edges[:, 1]]) / 2


@typecheck
def _get_geometry(
    points: Union[Points, PointsSequence],
    triangles: Triangles,
    edge_topology: Optional[EdgeTopology] = None,
) -> Union[
    tuple[Float2dTensor, Float2dTensor, Float2dTensor, Float2dTensor],
    tuple[Float3dTensor, Float3dTensor, Float3dTensor, Float3dTensor],
]:
    """Get the geometry of the edges of a triangular mesh.

    The geometry of an edge is defined by the four points Pi, Pj, Pk, Pl
    where Pi and Pj are the two extremities of the edge, Pk and Pl are
    the two points of the two adjacent triangles that are not Pi and Pj.

    Boundary edges are not considered.
    Edges with more than two adjacent triangles are not considered.

    Parameters
    ----------
    points
        Points of the mesh.
    triangles
        Triangles of the mesh.
    edge_topology
        Edge topology of the mesh.

    Returns
    -------
    Pi
        First extremity of the edges.
    Pj
        Second extremity of the edges.
    Pk
        Third point of the first adjacent triangle.
    Pl
        Third point of the second adjacent triangle.
    """
    if edge_topology is None:
        edge_topology = EdgeTopology(triangles)

    e = edge_topology.manifold_edges
    ef = edge_topology.manifold_adjacent_triangles
    ei = edge_topology.manifold_adjacent_points

    pi, pj = e[:, 0], e[:, 1]
    pk = triangles[ef[:, 0], ei[:, 0]]
    pl = triangles[ef[:, 1], ei[:, 1]]

    Pi = points[pi, :]
    Pj = points[pj, :]
    Pk = points[pk, :]
    Pl = points[pl, :]

    return Pi, Pj, Pk, Pl


@typecheck
def dihedral_angles(
    *,
    points: Union[Points, PointsSequence],
    triangles: Triangles,
    edge_topology: Optional[EdgeTopology] = None,
) -> Union[Float1dTensor, Float2dTensor]:
    """Dihedral angles of the edges of a triangular mesh.

    Parameters
    ----------
    points
        Points or sequence of points with shape (n_points, n_poses, dim).
    triangles
        Triangles of the mesh.
    edge_topology
        Edge topology of the mesh. If not provided, it will be computed from
        the triangles.

    Returns
    -------
        The dihedral angles of the edges with shape (n_edges,) for a single
        mesh or (n_edges, n_poses) for a sequence of meshes.
    """
    if edge_topology is None:
        edge_topology = EdgeTopology(triangles)

    Pi, Pj, Pk, Pl = _get_geometry(
        points=points,
        triangles=triangles,
        edge_topology=edge_topology,
    )

    nk = torch.cross(Pk - Pj, Pi - Pk, dim=-1)
    nk = nk / nk.norm(dim=-1).unsqueeze(-1)
    nl = torch.cross(Pl - Pi, Pj - Pl, dim=-1)
    nl = nl / nl.norm(dim=-1).unsqueeze(-1)

    cross_prod = torch.cross(nk, nl, dim=-1)
    edge_dir = Pj - Pi
    edge_dir = edge_dir / edge_dir.norm(dim=-1).unsqueeze(-1)

    aux = (nk * nl).sum(dim=-1)

    tmp = (cross_prod * edge_dir).sum(dim=-1)

    return torch.atan2(tmp, aux)
