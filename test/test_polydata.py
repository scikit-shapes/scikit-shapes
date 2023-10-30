import torch
import pyvista
from pyvista.core.pointset import PolyData as PyvistaPolyData
import numpy as np
import vedo
import skshapes as sks
import pytest
import os
from beartype.roar import BeartypeCallHintParamViolation


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
    # Shape initialized with points and triangles
    # dtype are automatically converted to float32 and int64 and numpy arrays
    # are converted to torch.Tensor
    points = torch.tensor(
        [[0, 0, 0], [0, 0, 1], [0, 1, 0]], dtype=torch.float64
    )
    triangles = torch.tensor([[0, 1, 2]], dtype=torch.int32).numpy()

    triangle = sks.PolyData(points=points, triangles=triangles)

    assert isinstance(triangle.points, torch.Tensor)
    assert triangle.points.dtype == sks.float_dtype
    assert isinstance(triangle.points, torch.Tensor)
    assert triangle.triangles.dtype == sks.int_dtype

    # edges are computed on the fly when the getter is called
    assert triangle._edges is None
    assert triangle.edges is not None
    assert triangle._edges is not None
    assert triangle._triangles is not None
    assert triangle.n_triangles == 1
    assert triangle.n_edges == 3
    assert triangle.n_points == 3
    assert triangle.is_triangle_mesh()

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
    assert triangle.is_triangle_mesh() is False

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
    from pyvista.examples import load_sphere

    mesh = load_sphere()
    n_points = mesh.n_points
    n_triangles = mesh.n_cells

    # Create a PolyData from a pyvista mesh
    polydata = sks.PolyData(mesh)
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
    # Create a PolyData from a pyvista mesh and check that the faces are
    # converted to triangles
    polydata = sks.PolyData(cube)
    assert polydata.n_points == 8
    assert polydata.n_triangles == 12
    # back to pyvista, check that the mesh is the same
    cube2 = polydata.to_pyvista()
    assert cube2.n_cells == 12
    assert cube2.n_points == 8
    assert cube2.is_all_triangles
    assert np.allclose(cube.points, cube2.points)


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


def test_point_data():
    mesh = sks.PolyData(pyvista.Sphere())

    # Add some point_data
    mesh.point_data["hessians"] = torch.rand(mesh.n_points, 3, 3, 4)
    mesh.point_data["normals"] = torch.rand(mesh.n_points, 3)
    mesh.point_data.append(torch.rand(mesh.n_points))

    try:
        mesh.point_data.append(torch.rand(mesh.n_points + 1))
    except ValueError:
        pass
    else:
        raise AssertionError(
            "Should have raised an error, the size of the tensor is not"
            + " correct"
        )

    # Check that the point_data are correctly copied
    copy = mesh.copy()
    assert torch.allclose(
        copy.point_data["hessians"], mesh.point_data["hessians"]
    )
    copy.point_data["hessians"] = torch.rand(
        *mesh["hessians"].shape
    )  # If the copy was not correct, this would also change the point_data of
    # the original mesh
    assert not torch.allclose(
        copy.point_data["hessians"], mesh.point_data["hessians"]
    )

    new_point_data = {
        "rotations": torch.rand(mesh.n_points, 3, 3),
        "colors": torch.rand(mesh.n_points, 3),
    }

    mesh.point_data = new_point_data  # Replace the point_data
    mesh.point_data.append(torch.rand(mesh.n_points, 2))  # Add a new feature
    assert list(mesh.point_data.keys()) == [
        "rotations",
        "colors",
        "attribute_0",
    ]  # Check the name of the point_data

    # Check that trying to set the point_data with a wrong type raises an error
    try:
        mesh.point_data = 4
    except BeartypeCallHintParamViolation:
        pass
    else:
        raise AssertionError(
            "Should have raised an error, the point_data should be a dict or"
            + "a sks.data.utils.DataAttributes object"
        )

    # Check that trying to set the point_data with an invalid dict (here the
    # size of the tensors is not correct) raises an error
    try:
        mesh.point_data = {
            "colors": torch.rand(mesh.n_points + 2, 3),
            "normals": torch.rand(mesh.n_points, 3),
        }
    except ValueError:
        pass
    else:
        raise AssertionError(
            "Should have raised an error, the size of the colors tensor is"
            + "not correct"
        )


def test_point_data2():
    # Load a pyvista.PolyData and add an attribute
    pv_mesh = pyvista.Sphere()
    # The attribute is matrix valued
    pv_mesh.point_data["curvature"] = np.random.rand(pv_mesh.n_points, 3)

    # Convert it to a skshapes.PolyData
    sks_mesh = sks.PolyData(pv_mesh)

    # Assert that the attribute curvature is correctly copied
    dtype = sks_mesh.point_data["curvature"].dtype
    assert torch.allclose(
        sks_mesh.point_data["curvature"],
        torch.from_numpy(pv_mesh.point_data["curvature"]).to(dtype),
    )

    # Assert that both device attributes for the mesh and the point_data are
    # cpu
    assert sks_mesh.device.type == "cpu"
    assert sks_mesh.point_data.device.type == "cpu"

    sks_mesh.point_data["color"] = torch.rand(sks_mesh.n_points, 3).cpu()
    # It is also possible to assign a numpy array, it will be automatically
    # converted to a torch.Tensor
    sks_mesh.point_data["normals"] = np.random.rand(sks_mesh.n_points, 3)

    back_to_pyvista = sks_mesh.to_pyvista()
    assert type(back_to_pyvista) is PyvistaPolyData

    # Assert that the point_data attributes are correctly copied
    assert "color" in back_to_pyvista.point_data.keys()
    assert np.allclose(
        back_to_pyvista.point_data["normals"],
        sks_mesh.point_data["normals"].numpy(),
    )

    back_to_vedo = sks_mesh.to_vedo()
    assert type(back_to_vedo) is vedo.Mesh
    assert np.allclose(
        back_to_vedo.pointdata["curvature"],
        sks_mesh.point_data["curvature"].numpy(),
    )

    # From vedo to sks
    sks_again = sks.PolyData(back_to_vedo)
    assert np.allclose(
        sks_again.point_data["curvature"].numpy(),
        pv_mesh.point_data["curvature"],
    )


def test_landmarks_creation():
    mesh1 = sks.Sphere()
    mesh2 = sks.Sphere()

    # Create landmarks with coo sparse tensor
    landmarks_indices = [0, 1, 2, 3]
    landmarks_values = 4 * [1.0]
    n_landmarks = len(landmarks_indices)
    n_points = mesh1.n_points
    landmarks = torch.sparse_coo_tensor(
        indices=[[0, 1, 2, 3], landmarks_indices],
        values=landmarks_values,
        size=(n_landmarks, n_points),
        device="cpu",
    )

    # Assert that initialize landmarks by vertex indices or with the
    # sparse tensor gives the same result
    mesh1.landmarks = landmarks
    mesh2.landmark_indices = landmarks_indices

    assert torch.allclose(mesh1.landmark_points, mesh2.landmark_points)
    assert torch.allclose(
        mesh1.landmark_indices, torch.tensor(landmarks_indices)
    )
    assert torch.allclose(
        mesh2.landmark_indices, torch.tensor(landmarks_indices)
    )

    assert mesh1.n_landmarks == 4
    mesh1.add_landmarks(8)
    assert mesh1.n_landmarks == 5
    assert mesh1.landmark_indices[-1] == 8

    mesh2.add_landmarks([8])

    mesh1 = sks.PolyData(mesh1.to_pyvista())

    assert torch.allclose(mesh1.landmark_points, mesh2.landmark_points)


def test_landmarks_conservation():
    # Create a mesh and add landmarks
    mesh = sks.PolyData(pyvista.Sphere())
    mesh.landmark_indices = [2, 45, 12, 125]

    # Check that the landmarks are preserved sks -> pyvista -> sks
    mesh_pv = mesh.to_pyvista()
    assert np.allclose(
        mesh_pv.field_data["landmark_points"], mesh.landmark_points.numpy()
    )
    mesh_back = sks.PolyData(mesh_pv)
    assert torch.allclose(mesh_back.landmark_points, mesh.landmark_points)

    # Check that the landmarks are preserved sks -> vedo -> sks
    mesh_vedo = mesh.to_vedo()
    assert np.allclose(
        mesh_vedo.metadata["landmark_points"], mesh.landmark_points.numpy()
    )
    mesh_back = sks.PolyData(mesh_vedo)
    assert torch.allclose(mesh_back.landmark_points, mesh.landmark_points)

    # Check that the landmarks are preserved after saving
    mesh.save("test.vtk")
    mesh_back = sks.PolyData("test.vtk")
    assert torch.allclose(mesh_back.landmark_points, mesh.landmark_points)
    os.remove("test.vtk")


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Cuda is required for this test"
)
def test_point_data_cuda():
    # Load a pyvista.PolyData and add an attribute
    pv_mesh = pyvista.Sphere()
    pv_mesh.point_data["curvature"] = np.random.rand(pv_mesh.n_points, 3)

    # Convert it to a skshapes.PolyData
    sks_mesh = sks.PolyData(pv_mesh)

    # Assert that the attribute curvature is correctly copied
    dtype = sks_mesh.point_data["curvature"].dtype
    assert torch.allclose(
        sks_mesh.point_data["curvature"],
        torch.from_numpy(pv_mesh.point_data["curvature"]).to(dtype),
    )

    # Assert that both device attributes for the mesh and the point_data are
    # cpu
    assert sks_mesh.device.type == "cpu"
    assert sks_mesh.point_data.device.type == "cpu"

    sks_mesh.point_data["color"] = torch.rand(sks_mesh.n_points, 3).cpu()
    # It is also possible to assign a numpy array, it will be automatically
    # converted to a torch.Tensor
    sks_mesh.point_data["normals"] = np.random.rand(sks_mesh.n_points, 3)

    # Assert that the point_data attributes are correctly formatted
    assert sks_mesh.point_data["color"].dtype == sks.float_dtype
    assert sks_mesh.point_data["normals"].dtype == sks.float_dtype

    # Move the mesh to cuda
    sks_mesh_cuda = sks_mesh.to("cuda")

    # Assert that both device attributes for the mesh and the point_data are
    # cuda
    assert sks_mesh_cuda.device.type == "cuda"
    assert sks_mesh_cuda.point_data.device.type == "cuda"

    # If a new attribute is added  to the mesh, it is automatically moved to
    # the same device as the mesh
    color = torch.rand(sks_mesh_cuda.n_points, 3).cpu()
    sks_mesh_cuda.point_data["color"] = color
    # It is also possible to assign a numpy array, it will be automatically
    # converted to a torch.Tensor
    normals = np.random.rand(sks_mesh_cuda.n_points, 3)
    sks_mesh_cuda.point_data["normals"] = normals

    assert sks_mesh_cuda.point_data["color"].device.type == "cuda"
    assert sks_mesh_cuda.point_data["normals"].device.type == "cuda"


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Cuda is required for this test"
)
def test_gpu():
    cube = sks.PolyData(_cube())
    cube_gpu = cube.to("cuda")

    assert cube_gpu.points.device == torch.Tensor().cuda().device
