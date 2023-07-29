import skshapes as sks
import pyvista
import torch
import pytest


def test_decimation_basic():
    ## Assert that the if we fit_transform the decimator from sks on a mesh, and then transform a copy of the mesh
    ## the result is the same

    sphere = sks.PolyData(pyvista.Sphere())
    sphere_copy = sks.PolyData(pyvista.Sphere())

    decimation = sks.Decimation(n_points=15, method="sks")

    # fit the decimator on the mesh
    decimated_sphere = decimation.fit_transform(sphere)
    # transform a copy of the mesh
    decimated_sphere_copy = decimation.transform(sphere_copy)

    # Assert that the result is the same
    assert torch.allclose(decimated_sphere.points, decimated_sphere_copy.points)
    assert torch.allclose(decimated_sphere.triangles, decimated_sphere_copy.triangles)

    ## Assert that the decimator with method="vtk" gives the same result as pyvista

    decimation = sks.Decimation(target_reduction=0.8, method="vtk")
    decimated_sphere_sks = decimation.fit_transform(sphere)
    decimated_sphere_vtk = sks.PolyData(
        sphere.to_pyvista().decimate(target_reduction=0.8)
    )

    assert torch.allclose(decimated_sphere_sks.points, decimated_sphere_vtk.points)
    assert torch.allclose(
        decimated_sphere_sks.triangles, decimated_sphere_vtk.triangles
    )


def test_decimation_landmarks():
    mesh = sks.PolyData(pyvista.Sphere())

    values = [1, 1, 0.3, 0.4, 0.3, 1]
    indices = [
        [0, 1, 2, 2, 2, 3],
        [4, 1, 2, 30, 0, 27],
    ]
    n_landmarks = 4
    n_points = mesh.n_points
    landmarks = torch.sparse_coo_tensor(
        indices=indices, values=values, size=(n_landmarks, n_points), device="cpu"
    )
    mesh.landmarks = landmarks

    decimator = sks.Decimation(target_reduction=0.9, method="sks")
    decimated_mesh = decimator.fit_transform(mesh)

    assert mesh.landmarks is not None
    assert decimated_mesh.landmarks is not None
    assert len(decimated_mesh.landmarks_3d) == len(
        mesh.landmarks_3d
    )  # assert that the number of landmarks is the same


def test_torch_sparse_tensor_repetitions():
    """Assert that the torch.sparse_coo_tensor can handle repetitions in the indices and sum the values"""
    import random

    a = random.random()
    b = random.random()

    M = torch.tensor([[0, a + b], [0, 0]])

    values = [a, b]
    indices = [
        [0, 0],
        [1, 1],
    ]
    landmarks = torch.sparse_coo_tensor(
        indices=indices, values=values, size=(2, 2), device="cpu"
    )

    assert torch.allclose(landmarks.to_dense(), M)


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Cuda is required for this test"
)
def test_decimation_gpu():
    # Assert that the decimation works with PolyData on the gpu
    sphere = sks.PolyData(pyvista.Sphere(), device="cuda")

    dec = sks.Decimation(n_points=15, method="sks")
    dec.fit(sphere)
    newsphere = dec.transform(sphere.to("cpu"))
    assert newsphere.points.device.type == "cpu"

    newsphere = dec.transform(sphere.to("cuda"))
    assert newsphere.points.device.type == "cuda"
