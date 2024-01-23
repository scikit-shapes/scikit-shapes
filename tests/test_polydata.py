"""Tests for the PolyData class."""

from pathlib import Path

import numpy as np
import pytest
import pyvista
import skshapes as sks
import torch
import vedo
from pyvista.core.pointset import PolyData as PyvistaPolyData
from skshapes.errors import DeviceError, InputTypeError, ShapeError


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
    """Test the creation of a PolyData object."""
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

    # /!\ Assigning edges manually will delete the triangles
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

    with pytest.raises(AttributeError, match="Triangles are not defined"):
        triangle.triangle_areas  # noqa: B018 (useless expression)

    # Cannot change the number of points
    with pytest.raises(ShapeError):
        triangle.points = torch.rand(triangle.n_points + 1, 3)

    triangle_mesh = sks.Sphere()
    wireframe_mesh = sks.Circle()
    pointcloud_mesh = sks.PolyData(torch.rand(10, 3))

    for mesh in [triangle_mesh, wireframe_mesh, pointcloud_mesh]:
        vedo_mesh = mesh.to_vedo()
        pv_mesh = mesh.to_pyvista()

        vedo_back = sks.PolyData(vedo_mesh)
        pv_back = sks.PolyData(pv_mesh)

        assert torch.allclose(vedo_back.points, mesh.points)
        assert torch.allclose(pv_back.points, mesh.points)

        if mesh.n_edges > 0:
            assert torch.allclose(vedo_back.edges, mesh.edges)
            assert torch.allclose(pv_back.edges, mesh.edges)

        if mesh.n_triangles > 0:
            assert torch.allclose(vedo_back.triangles, mesh.triangles)
            assert torch.allclose(pv_back.triangles, mesh.triangles)

    mixed_mesh = wireframe_mesh.to_pyvista()
    # add a triangle to the wireframe mesh
    mixed_mesh.faces = np.concatenate([mixed_mesh.faces, [3, 0, 1, 2]])

    with pytest.raises(ValueError, match="both triangles and edges"):
        sks.PolyData(mixed_mesh)


def test_geometry_features():
    """Test some geometry features on a simple mesh."""
    square_points = torch.tensor(
        [[0, 0], [1, 0], [1, 1], [0, 1]], dtype=sks.float_dtype
    )
    square_edges = torch.tensor(
        [[0, 1], [1, 2], [2, 3], [3, 0]], dtype=sks.int_dtype
    )
    square = sks.PolyData(points=square_points, edges=square_edges)
    assert square.n_edges == 4

    with pytest.raises(AttributeError, match="Triangles are not defined"):
        square.triangle_centers  # noqa: B018 (useless expression)
    with pytest.raises(AttributeError, match="Triangles are not defined"):
        square.triangle_normals  # noqa: B018 (useless expression)

    assert torch.allclose(
        square.point_weights, torch.tensor([1, 1, 1, 1], dtype=sks.float_dtype)
    )

    assert torch.allclose(
        square.edge_lengths, torch.tensor([1, 1, 1, 1], dtype=sks.float_dtype)
    )

    assert torch.allclose(
        square.edge_centers,
        torch.tensor(
            [[0.5, 0], [1, 0.5], [0.5, 1], [0, 0.5]], dtype=sks.float_dtype
        ),
    )

    assert torch.allclose(
        square.mean_point, torch.tensor([0.5, 0.5], dtype=sks.float_dtype)
    )

    assert torch.allclose(
        square.standard_deviation,
        torch.tensor([0.5, 0.5], dtype=sks.float_dtype).sqrt(),
    )

    triangle_points = torch.tensor(
        [[0, 0], [1, 0], [0, 1]], dtype=sks.float_dtype
    )
    triangle_triangles = torch.tensor([[0, 1, 2]], dtype=sks.int_dtype)

    triangle = sks.PolyData(
        points=triangle_points, triangles=triangle_triangles
    )
    assert triangle.n_edges == 3
    assert triangle.n_triangles == 1
    assert torch.allclose(
        triangle.triangle_areas, torch.tensor([0.5], dtype=sks.float_dtype)
    )

    assert torch.allclose(
        triangle.triangle_centers,
        torch.tensor([[1 / 3, 1 / 3]], dtype=sks.float_dtype),
    )

    assert torch.allclose(
        triangle.triangle_normals,
        torch.tensor([[0, 0, 1]], dtype=sks.float_dtype),
    )

    pointcloud = sks.PolyData(triangle.points)

    for attribute in [
        "triangle_areas",
        "triangle_centers",
        "edge_lengths",
        "edge_centers",
    ]:
        with pytest.raises(AttributeError):
            getattr(pointcloud, attribute)


def test_polydata_creation_2d():
    """Test manually creating a 2d mesh + interaction with pv/vedo."""
    points = torch.tensor([[0, 0], [0, 1], [1, 0]], dtype=torch.float64)
    triangles = torch.tensor([[0, 1, 2]], dtype=torch.int32)

    flat_triangle = sks.PolyData(points=points, triangles=triangles)
    assert flat_triangle.dim == 2
    assert flat_triangle.n_triangles == 1
    assert flat_triangle.n_edges == 3

    # to_pyvista creates a z-coordinate equal to 0
    pv_mesh = flat_triangle.to_pyvista()
    assert pv_mesh.points.shape == (3, 3)
    assert np.allclose(pv_mesh.points[:, 2], 0)
    mesh_back = sks.PolyData(pv_mesh)
    assert mesh_back.dim == 2

    vedo_mesh = flat_triangle.to_vedo()
    assert vedo_mesh.points().shape == (3, 3)
    assert np.allclose(vedo_mesh.points()[:, 2], 0)
    mesh_back = sks.PolyData(vedo_mesh)
    assert mesh_back.dim == 2


def test_interaction_with_pyvista():
    """Test the interaction with pyvista."""
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

    # Importing a point cloud from pyvsita
    n = 100
    points = np.random.default_rng().random(size=(n, 3))
    polydata = pyvista.PolyData(points)
    mesh = sks.PolyData(polydata)
    assert mesh.n_points == n
    assert mesh.n_triangles == 0
    assert mesh.n_edges == 0


def test_mesh_cleaning():
    """Assert that .clean() is called when loading a mesh with skshapes."""
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
    """Test the PointData (signals) interface."""
    mesh = sks.PolyData(pyvista.Sphere())

    # Add some point_data
    mesh.point_data["hessians"] = torch.rand(mesh.n_points, 3, 3, 4)
    mesh.point_data["normals"] = torch.rand(mesh.n_points, 3)
    mesh.point_data.append(torch.rand(mesh.n_points))

    with pytest.raises(ShapeError):
        mesh.point_data.append(torch.rand(mesh.n_points + 1))

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
    for _key, _ in mesh.point_data.items():
        pass
    mesh.point_data.append(torch.rand(mesh.n_points, 2))  # Add a new feature
    for _key, _ in mesh.point_data.items():
        pass
    assert list(mesh.point_data.keys()) == [
        "rotations",
        "colors",
        "attribute_0",
    ]  # Check the name of the point_data

    # Check that trying to set the point_data with a wrong type raises an error
    with pytest.raises(InputTypeError):
        mesh.point_data = 4
    # Check that trying to set the point_data with an invalid dict (here the
    # size of the tensors is not correct) raises an error
    with pytest.raises(ShapeError):
        mesh.point_data = {
            "colors": torch.rand(mesh.n_points + 2, 3),
            "normals": torch.rand(mesh.n_points, 3),
        }

    dict = {
        "colors": torch.rand(mesh.n_points + 1, 3),
    }

    # Try to assign a dict with a wrong size
    with pytest.raises(ShapeError):
        mesh.point_data = dict

    # Try to get a point_data that does not exist
    with pytest.raises(KeyError):
        mesh.point_data["dummy"]

    # Same through the __getitem__ interface
    with pytest.raises(KeyError):
        mesh["dummy"]


def test_point_data2():
    """Test the PointData (signals) interface: higher dimension."""
    # Load a pyvista.PolyData and add an attribute
    rnd_generator = np.random.default_rng()
    pv_mesh = pyvista.Sphere()
    # The attribute has 4 dims
    pv_mesh.point_data["curvature"] = rnd_generator.random(
        size=(pv_mesh.n_points, 2)
    )
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
    sks_mesh.point_data["normals"] = rnd_generator.random(
        size=(sks_mesh.n_points, 3)
    )

    back_to_pyvista = sks_mesh.to_pyvista()
    assert type(back_to_pyvista) is PyvistaPolyData

    # Assert that the point_data attributes are correctly copied
    assert "color" in back_to_pyvista.point_data
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

    # Check that signal can be transferred back and forth with pyvista
    # and vedo if ndims is high
    sks_mesh["many_dims_signal"] = torch.rand(sks_mesh.n_points, 2, 2, 2, 2)

    to_pv = sks_mesh.to_pyvista()
    to_vedo = sks_mesh.to_vedo()

    back_from_pv = sks.PolyData(to_pv)
    back_from_vedo = sks.PolyData(to_vedo)

    assert torch.allclose(
        back_from_pv["many_dims_signal"], sks_mesh["many_dims_signal"]
    )
    assert torch.allclose(
        back_from_vedo["many_dims_signal"], sks_mesh["many_dims_signal"]
    )

    # Reset signals
    sks_mesh.point_data = None
    assert len(sks_mesh.point_data) == 0


def test_landmarks_creation():
    """Test the creation of landmarks as sparse tensors."""
    mesh1 = sks.Sphere()
    mesh2 = sks.Sphere()

    assert mesh1.n_landmarks == 0
    assert mesh1.landmark_points is None

    mesh1.add_landmarks(3)
    assert mesh1.n_landmarks == 1

    # Create landmarks with coo_sparse_tensor
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
    """Test landmarks conservation when converting to pyvista and vedo."""
    # Create a mesh and add landmarks
    mesh = sks.PolyData(pyvista.Sphere(), landmarks=[2, 45, 12, 125])

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
    Path.unlink(Path("test.vtk"))


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Cuda is required for this test"
)
def test_point_data_cuda():
    """Test the behavior of signals with respect to the device."""
    # Load a pyvista.PolyData and add an attribute
    rnd_generator = np.random.default_rng()
    pv_mesh = pyvista.Sphere()
    pv_mesh.point_data["curvature"] = rnd_generator.random(
        size=(pv_mesh.n_points, 3)
    )

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
    sks_mesh.point_data["normals"] = rnd_generator.random(
        size=(sks_mesh.n_points, 3)
    )

    # Assert that the point_data attributes are correctly formatted
    assert sks_mesh.point_data["color"].dtype == sks.float_dtype
    assert sks_mesh.point_data["normals"].dtype == sks.float_dtype

    # Move the mesh to cuda
    sks_mesh_cuda = sks_mesh.to("cuda")

    # Assert that both device attributes for the mesh and the point_data are
    # cuda
    assert sks_mesh_cuda.device.type == "cuda"
    assert sks_mesh_cuda.point_data.device.type == "cuda"

    # If a new attribute is added to the mesh, it is automatically moved to
    # the same device as the mesh
    color = torch.rand(sks_mesh_cuda.n_points, 3).cpu()
    sks_mesh_cuda.point_data["color"] = color
    # It is also possible to assign a numpy array, it will be automatically
    # converted to a torch.Tensor
    generator = np.random.default_rng()
    normals = generator.random(size=(sks_mesh_cuda.n_points, 3))
    sks_mesh_cuda.point_data["normals"] = normals

    assert sks_mesh_cuda.point_data["color"].device.type == "cuda"
    assert sks_mesh_cuda.point_data["normals"].device.type == "cuda"


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_polydata_signal_and_landmarks():
    """Test preservation of landmarks and signal when converting."""
    sphere = sks.Sphere()
    landmarks = torch.tensor([0, 1, 2, 3, 4, 5])
    signal = torch.rand(sphere.n_points, dtype=sks.float_dtype)

    sphere.landmark_indices = landmarks
    sphere["signal"] = signal

    sphere_pv = sphere.to_pyvista()

    sphere_back = sks.PolyData(sphere_pv)
    assert torch.allclose(sphere_back.landmark_indices, landmarks)
    assert torch.allclose(sphere_back["signal"], signal)

    # Add a point not connected to the mesh
    import numpy as np

    sphere_pv_notclean = sphere_pv.copy()
    sphere_pv_notclean.points = np.concatenate([sphere_pv.points, [[0, 0, 0]]])

    # Assert that as the mesh must be cleaned,
    # landmarks and signal are ignored
    sphere_back2 = sks.PolyData(sphere_pv_notclean)
    assert sphere_back2.landmark_indices is None
    assert len(sphere_back2.point_data) == 0


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Cuda is required for this test"
)
def test_gpu():
    """Test moving a polydata to the gpu."""
    cube = sks.PolyData(_cube())

    cube_gpu = cube.to("cuda")

    assert cube_gpu.points.device == torch.Tensor().cuda().device

    # Assert that copying a mesh that is on the gpu results in a mesh on the
    # gpu
    cube_gpu_copy = cube_gpu.copy()
    assert cube_gpu_copy.points.device == torch.Tensor().cuda().device


def test_control_points():
    """Test the interface of control points."""
    mesh = sks.Circle()
    grid = mesh.bounding_grid(N=5)

    mesh.control_points = grid
    # No copy is done
    assert mesh.control_points is grid
    assert grid.control_points is None

    # But if a copy of the mesh is done, the control points are copied
    mesh_copy = mesh.copy()
    assert mesh_copy.control_points is not grid


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Cuda is required for this test"
)
def test_control_points_device():
    """Test the behavior of the control points with respect to the device."""
    # mesh on cpu and control points on gpu -> should raise an error
    mesh = sks.Sphere().to("cpu")
    grid = mesh.bounding_grid(N=5)
    # grid = grid.to("cuda:0")
    grid.device = "cuda"
    with pytest.raises(DeviceError):
        mesh.control_points = grid

    # mesh on cpu and control points on cpu -> should not raise an error
    grid2 = grid.to("cpu")
    mesh.control_points = grid2
    loss = sks.L2Loss()
    assert loss(mesh.control_points, grid2) < 1e-10

    # copy
    mesh_gpu = mesh.to("cuda")
    assert mesh_gpu.points.device.type == "cuda"
    assert mesh_gpu.control_points.device.type == "cuda"

    mesh.device = "cuda"
    assert mesh.points.device.type == "cuda"
    assert mesh.control_points.device.type == "cuda"
