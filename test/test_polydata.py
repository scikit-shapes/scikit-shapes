# from skshapes.data import PolyData
import torch
import numpy as np
import pyvista
import vedo
import skshapes as sks


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
    triangle = sks.PolyData(points=points, triangles=triangles)

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
    # Create a PolyData from a pyvista mesh and check that the faces are converted to triangles
    polydata = sks.PolyData(cube)
    assert polydata.n_points == 8
    assert polydata.n_triangles == 12
    # back to pyvista, check that the mesh is the same
    cube2 = polydata.to_pyvista()
    assert cube2.n_cells == 12
    assert cube2.n_points == 8
    assert cube2.is_all_triangles
    assert np.allclose(cube.points, cube2.points)


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


def test_point_data():
    mesh = sks.PolyData(pyvista.Sphere())

    # Add some point_data
    mesh.point_data["hessians"] = torch.rand(mesh.n_points, 3, 3)
    mesh.point_data["normals"] = torch.rand(mesh.n_points, 3)
    mesh.point_data.append(torch.rand(mesh.n_points))

    try:
        mesh.point_data.append(torch.rand(mesh.n_points + 1))
    except:
        pass
    else:
        raise AssertionError(
            "Should have raised an error, the size of the tensor is not correct"
        )

    # Check that the point_data are correctly copied
    copy = mesh.copy()
    assert torch.allclose(copy.point_data["hessians"], mesh.point_data["hessians"])
    copy.point_data["hessians"] = torch.rand(
        mesh.n_points, 3, 3
    )  # If the copy was not correct, this would also change the point_data of the original mesh
    assert not torch.allclose(copy.point_data["hessians"], mesh.point_data["hessians"])

    new_point_data = {
        "rotations": torch.rand(mesh.n_points, 3, 3),
        "colors": torch.rand(mesh.n_points, 3),
    }

    mesh.point_data = new_point_data  # Replace the point_data
    mesh.point_data.append(torch.rand(mesh.n_points, 2))  # Add a new feature
    assert list(mesh.point_data.keys()) == [
        "rotations",
        "colors",
        "feature_0",
    ]  # Check the name of the point_data

    # Check that trying to set the point_data with a wrong type raises an error
    try:
        mesh.point_data = 4
    except:
        pass
    else:
        raise AssertionError(
            "Should have raised an error, the point_data should be a dict or a sks.data.utils.DataAttributes object"
        )

    # Check that trying to set the point_data with an invalid dict (here the size of the tensors is not correct) raises an error
    try:
        mesh.point_data = {
            "colors": torch.rand(mesh.n_points + 2, 3),
            "normals": torch.rand(mesh.n_points, 3),
        }
    except:
        pass
    else:
        raise AssertionError(
            "Should have raised an error, the size of the colors tensor is not correct"
        )


def test_point_data2():
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

    # Assert that both device attributes for the mesh and the point_data are cpu
    assert sks_mesh.device.type == "cpu"
    assert sks_mesh.point_data.device.type == "cpu"

    # Move the mesh to cuda
    sks_mesh_cuda = sks_mesh.to("cuda")

    # Assert that both device attributes for the mesh and the point_data are cuda
    assert sks_mesh_cuda.device.type == "cuda"
    assert sks_mesh_cuda.point_data.device.type == "cuda"

    # If a new attribute is added  to the mesh, it is automatically moved to the same device as the mesh
    sks_mesh_cuda.point_data["color"] = torch.rand(sks_mesh_cuda.n_points, 3).cpu()
    # It is also possible to assign a numpy array, it will be automatically converted to a torch.Tensor
    sks_mesh_cuda.point_data["normals"] = np.random.rand(sks_mesh_cuda.n_points, 3)

    assert sks_mesh_cuda.point_data["color"].device.type == "cuda"
    assert sks_mesh_cuda.point_data["normals"].device.type == "cuda"

    back_to_pyvista = sks_mesh_cuda.to_pyvista()
    assert type(back_to_pyvista) == pyvista.PolyData

    # Assert that the point_data attributes are correctly copied
    assert "color" in back_to_pyvista.point_data.keys()
    assert np.allclose(
        back_to_pyvista.point_data["normals"],
        sks_mesh_cuda.point_data["normals"].cpu().numpy(),
    )

    back_to_vedo = sks_mesh_cuda.to_vedo()
    assert type(back_to_vedo) == vedo.Mesh
    assert np.allclose(
        back_to_vedo.pointdata["curvature"], sks_mesh.point_data["curvature"].numpy()
    )

    # From vedo to sks
    sks_again = sks.PolyData(back_to_vedo)
    assert np.allclose(
        sks_again.point_data["curvature"].numpy(), pv_mesh.point_data["curvature"]
    )


def test_gpu():
    cube = sks.PolyData(_cube())
    cube_gpu = cube.to("cuda")

    assert cube_gpu.points.device == torch.Tensor().cuda().device
