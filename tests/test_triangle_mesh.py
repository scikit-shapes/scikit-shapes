import skshapes as sks
from skshapes.triangle_mesh import EdgeTopology
from pyvista import examples
import torch


def test_edge_topology():
    hills = examples.load_random_hills()
    mesh = sks.PolyData(hills)

    points = mesh.points
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


from skshapes.triangle_mesh import (
    triangle_normals,
    triangle_areas,
    triangle_centers,
    edge_centers,
    edge_lengths,
    dihedral_angles,
)


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
    mesh = sks.Sphere()

    points = mesh.points
    triangles = mesh.triangles
    edge_topology = EdgeTopology(triangles)

    # Make sure that functions works with a list of tensors as input
    # points can be either of shape (n_points, dim)
    # or a list of tensors of shape (n_points, n_frames, dim)

    n_frames = 7
    points_list, frames = _create_points_list_and_frame(n_frames, points)

    for f in (
        triangle_normals,
        triangle_areas,
        triangle_centers,
        edge_lengths,
        edge_centers,
        dihedral_angles,
    ):
        # Apply the function to the list of tensors with frame format
        f_frames = f(points=frames, triangles=triangles)

        for i in range(n_frames):
            f_points = f(
                points=points_list[i],
                triangles=triangles,
                edge_topology=edge_topology,
            )
            assert torch.allclose(f_frames[:, i], f_points)


def test_energy():
    mesh = sks.PolyData(examples.load_random_hills())

    from skshapes.triangle_mesh import (
        bending_energy,
        membrane_energy,
        shell_energy,
    )

    points = mesh.points
    triangles = mesh.triangles
    edge_topology = EdgeTopology(triangles)

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
