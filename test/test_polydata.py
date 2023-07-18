import sys

sys.path.append(sys.path[0][:-4])
from skshapes.data import PolyData
import torch
import numpy as np
import pyvista


def _cube():
    points = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [1, 0, 1],
            [0, 0, 1],
            [0, 1, 0],
            [1, 1, 0],
            [1, 1, 1],
            [0, 1, 1],
        ],
        dtype=np.float64,
    )

    faces = np.array(
        [
            4,
            0,
            1,
            2,
            3,
            4,
            0,
            1,
            5,
            4,
            4,
            0,
            4,
            7,
            3,
            4,
            2,
            1,
            5,
            6,
            4,
            3,
            2,
            6,
            7,
            4,
            4,
            5,
            6,
            7,
        ]
    )

    return pyvista.PolyData(points, faces=faces)


def test_polydata_creation():
    # Shape with points and triangles
    points = torch.tensor([[0, 0, 0], [0, 0, 1], [0, 1, 0]], dtype=torch.float32)
    triangles = torch.tensor([[0], [1], [2]], dtype=torch.int64)
    triangle = PolyData(points=points, triangles=triangles)

    # edges are computed on the fly when the getter is called, and _edges remains None
    assert triangle.edges is not None
    assert triangle._triangles is not None
    assert triangle._edges is None
    assert triangle.n_triangles == 1
    assert triangle.n_edges == 3  # Should be 3 or not ?
    assert triangle.n_points == 3

    assert triangle.edge_centers is not None
    assert triangle.edge_lengths is not None
    assert triangle.triangle_areas is not None

    # /!\ Assinging edges manually will delete the triangles
    triangle.edges = triangle.edges
    assert triangle.edges is not None
    assert triangle._triangles is None
    assert triangle._edges is not None
    assert triangle.n_triangles == 0
    assert triangle.n_edges == 3
    assert triangle.n_points == 3

    assert triangle.edge_centers is not None
    assert triangle.edge_lengths is not None

    try:
        triangle.triangle_areas
    except ValueError:
        pass
    else:
        raise AssertionError("Assigning edges should delete triangles")


def test_interaction_with_pyvista():
    # Import/export from/to pyvista
    import pyvista
    from pyvista.examples import load_sphere

    mesh = load_sphere()
    n_points = mesh.n_points
    n_triangles = mesh.n_cells

    # Create a PolyData from a pyvista mesh
    polydata = PolyData(mesh)
    assert polydata.n_points == n_points
    assert polydata.n_triangles == n_triangles

    # back to pyvista, check that the mesh is the same
    mesh2 = polydata.to_pyvista()
    print(np.max(np.abs(mesh.points - mesh2.points)))
    assert np.allclose(mesh.points, mesh2.points)
    assert np.allclose(mesh.faces, mesh2.faces)

    # Open a quadratic mesh
    cube = _cube()
    # the cube is a polydata with 6 cells (quads) and 8 points
    assert not cube.is_all_triangles
    assert cube.n_cells == 6
    assert cube.n_points == 8
    # Create a PolyData from a pyvista mesh and check that the faces are converted to triangles
    polydata = PolyData(cube)
    assert polydata.n_points == 8
    assert polydata.n_triangles == 12
    # back to pyvista, check that the mesh is the same
    cube2 = polydata.to_pyvista()
    assert cube2.n_cells == 12
    assert cube2.n_points == 8
    assert cube2.is_all_triangles
    assert np.allclose(cube.points, cube2.points)


import skshapes as sks


def test_decimation():
    mesh = pyvista.Sphere().decimate(0.5)  # use pyvista to decimate
    polydata = sks.PolyData(pyvista.Sphere()).decimate(0.5)  # use skshapes to decimate

    # Check that the points are the same
    assert np.allclose(polydata.points.numpy(), mesh.points)


def test_mesh_cleaning():
    # Example of a mesh with duplicated points
    points = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 0],
            [1, 0, 0],
            [0, 0, 1],
        ],
        dtype=np.float64,
    )

    faces = np.array([3, 0, 1, 2, 3, 3, 4, 5])

    mesh = pyvista.PolyData(points, faces=faces)
    clean_mesh = mesh.clean()

    test = sks.PolyData(mesh)

    # Check that the mesh is cleaned when loaded by skshapes
    assert np.allclose(test.points.numpy(), clean_mesh.points)


def test_gpu():
    cube = sks.PolyData(_cube())
    cube_gpu = cube.to("cuda")

    assert cube_gpu.points.device == torch.Tensor().cuda().device
