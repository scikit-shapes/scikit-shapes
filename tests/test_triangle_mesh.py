"""Test the triangle mesh module."""

from math import sqrt

import skshapes as sks
import torch
from pyvista import examples
from skshapes.triangle_mesh import (
    EdgeTopology,
    dihedral_angles,
    edge_centers,
    edge_lengths,
    triangle_areas,
    triangle_centers,
    triangle_normals,
)
from skshapes.triangle_mesh.geometry import _get_geometry


def test_edge_topology():
    """Test the edge topology class."""
    hills = examples.load_random_hills()
    mesh = sks.PolyData(hills)

    triangles = mesh.triangles

    edge_topology = EdgeTopology(triangles)

    # Get manifold edges + adjacent triangles + adjacent points
    m_e = edge_topology.manifold_edges
    m_ef = edge_topology.manifold_adjacent_triangles
    m_ei = edge_topology.manifold_adjacent_points

    # Get boundary edges + adjacent triangles + adjacent points
    b_e = edge_topology.boundary_edges
    b_ef = edge_topology.boundary_adjacent_triangles
    b_ei = edge_topology.boundary_adjacent_points

    # Pick 2 random manifold edges
    for i in torch.randint(0, len(m_e), (2,)):
        edge = m_e[i]
        adjacent_triangles_0 = triangles[m_ef[i, 0]]
        opposed_point_0 = adjacent_triangles_0[m_ei[i, 0]]
        adjacent_triangles_1 = triangles[m_ef[i, 1]]
        opposed_point_1 = adjacent_triangles_1[m_ei[i, 1]]

        assert edge[0] in adjacent_triangles_0
        assert edge[1] in adjacent_triangles_0
        assert opposed_point_0 not in edge
        assert opposed_point_0 in adjacent_triangles_0

        assert edge[0] in adjacent_triangles_1
        assert edge[1] in adjacent_triangles_1
        assert opposed_point_1 not in edge
        assert opposed_point_1 in adjacent_triangles_1

    # Pick 2 random boundary edges
    for i in torch.randint(0, len(b_e), (2,)):
        edge = b_e[i]
        adjacent_triangle = triangles[b_ef[i]]
        opposed_point = adjacent_triangle[b_ei[i]]

        assert edge[0] in adjacent_triangle
        assert edge[1] in adjacent_triangle
        assert opposed_point not in edge
        assert opposed_point in adjacent_triangle


def test_functional_geometry():
    """Test some geometry/topology functions on a simple mesh."""
    # A simple 2D example to test the geometry functions on a simple mesh
    #
    #
    #  2 -- 3
    #  | \  |
    #  |  \ |
    #  0 -- 1

    points = torch.tensor(
        [
            [0, 0],
            [1, 0],
            [0, 1],
            [1, 1],
        ],
        dtype=sks.float_dtype,
    )

    triangles = torch.tensor(
        [
            [0, 1, 2],
            [1, 2, 3],
        ],
        dtype=sks.int_dtype,
    )

    expected_edges = torch.tensor(
        [
            [0, 1],
            [0, 2],
            [1, 2],
            [1, 3],
            [2, 3],
        ],
        dtype=sks.int_dtype,
    )

    expected_edges_centers = torch.tensor(
        [
            [0.5, 0],
            [0, 0.5],
            [0.5, 0.5],
            [1, 0.5],
            [0.5, 1],
        ],
        dtype=sks.float_dtype,
    )

    expected_edge_lengths = torch.tensor(
        [
            1,
            1,
            sqrt(2),
            1,
            1,
        ],
        dtype=sks.float_dtype,
    )

    expected_triangle_areas = torch.tensor([0.5, 0.5], dtype=sks.float_dtype)

    expected_triangles_centers = torch.tensor(
        [
            [1 / 3, 1 / 3],
            [2 / 3, 2 / 3],
        ],
        dtype=sks.float_dtype,
    )

    edge_topology = EdgeTopology(triangles)

    assert len(edge_topology.manifold_edges) == 1
    assert len(edge_topology.boundary_edges) == 4

    assert torch.allclose(edge_topology.edges, expected_edges)
    assert torch.allclose(
        edge_centers(points=points, triangles=triangles),
        expected_edges_centers,
    )
    assert torch.allclose(
        edge_lengths(points=points, triangles=triangles),
        expected_edge_lengths,
    )
    assert torch.allclose(
        triangle_areas(points=points, triangles=triangles),
        expected_triangle_areas,
    )
    assert torch.allclose(
        triangle_centers(points=points, triangles=triangles),
        expected_triangles_centers,
    )

    Pi, Pj, Pk, Pl = _get_geometry(points, triangles)
    assert torch.allclose(Pi[0], points[1])
    assert torch.allclose(Pj[0], points[2])
    assert torch.allclose(Pk[0], points[0])
    assert torch.allclose(Pl[0], points[3])


def _create_points_list_and_frame(n_frames, points):
    # Create the list of tensors with frame format
    n_points = points.shape[0]
    dim = points.shape[1]
    frames = torch.zeros((n_points, n_frames, dim), dtype=points.dtype)
    points_list = []
    for i in range(n_frames):
        random_displacement = torch.randn_like(points) * 0.1
        random_displacement = random_displacement.to(points.dtype)
        frames[:, i, :] = points + random_displacement
        points_list.append(points + random_displacement)

    return points_list, frames


def test_geometry():
    """Test the geometry functions with multiple frames."""
    mesh = sks.Sphere()

    points = mesh.points
    triangles = mesh.triangles
    edge_topology = EdgeTopology(triangles)

    # Make sure that geometry functions works with a list of tensors as input
    # points can be either of shape (n_points, dim)
    # or a list of tensors of shape (n_points, n_frames, dim)

    n_frames = 7
    points_list, frames = _create_points_list_and_frame(n_frames, points)

    for f, kwargs in (
        (triangle_normals, {}),
        (triangle_areas, {}),
        (triangle_centers, {}),
        (edge_lengths, {"edge_topology": edge_topology}),
        (edge_centers, {"edge_topology": edge_topology}),
        (dihedral_angles, {"edge_topology": edge_topology}),
    ):
        # Apply the function to the list of tensors with frame format
        f_frames = f(points=frames, triangles=triangles)

        for i in range(n_frames):
            f_points = f(
                points=points_list[i],
                triangles=triangles,
                **kwargs,
            )
            assert torch.allclose(f_frames[:, i], f_points)


def test_energy():
    """Test the energy function with multiple frames."""
    mesh = sks.PolyData(examples.load_random_hills())

    from skshapes.triangle_mesh import (
        bending_energy,
        membrane_energy,
        shell_energy,
    )

    points = mesh.points
    triangles = mesh.triangles

    n_frames = 7
    _, frames = _create_points_list_and_frame(n_frames, points)

    points_undef = frames
    points_def = frames + torch.randn_like(frames) * 0.1

    for energy_fn in (bending_energy, membrane_energy, shell_energy):
        energy_frames = energy_fn(
            points_undef=points_undef,
            points_def=points_def,
            triangles=triangles,
        )

        for i in range(n_frames):
            energy = energy_fn(
                points_undef=points_undef[:, i, :],
                points_def=points_def[:, i, :],
                triangles=triangles,
            )

            assert torch.allclose(energy, energy_frames[i])
