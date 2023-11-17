import skshapes as sks
import pyvista
import torch
import pytest


def test_decimation_basic():
    """Assert that the if we fit_transform the decimator from sks on a mesh,
    and then transform a copy of the mesh the result is the same"""

    sphere = sks.PolyData(pyvista.Sphere())
    sphere_copy = sks.PolyData(pyvista.Sphere())

    decimation = sks.Decimation(n_points=15)

    # fit the decimator on the mesh
    decimated_sphere = decimation.fit_transform(sphere)
    # transform a copy of the mesh
    decimated_sphere_copy = decimation.transform(sphere_copy)

    # Assert that the result is the same
    assert torch.allclose(
        decimated_sphere.points, decimated_sphere_copy.points
    )
    assert torch.allclose(
        decimated_sphere.triangles, decimated_sphere_copy.triangles
    )

    # Check that calling .decimate() on the mesh gives the same result

    decimated_sphere2 = sphere.decimate(n_points=15)

    assert torch.allclose(decimated_sphere.points, decimated_sphere2.points)
    assert torch.allclose(
        decimated_sphere.triangles, decimated_sphere2.triangles
    )

    # test with target_reduction
    decimation = sks.Decimation(target_reduction=0.9)
    decimated_sphere = decimation.fit_transform(sphere)
    decimated_sphere2 = sphere.decimate(target_reduction=0.9)

    assert torch.allclose(decimated_sphere.points, decimated_sphere2.points)
    assert torch.allclose(
        decimated_sphere.triangles, decimated_sphere2.triangles
    )

    # Some errors
    mesh = sks.Sphere()

    try:
        mesh.decimate(n_points=10, target_reduction=0.9)
    except ValueError:
        pass
    else:
        raise AssertionError(
            "Should have raised a ValueError as both"
            + " n_points and target_reduction are specified"
        )

    try:
        mesh.decimate()
    except ValueError:
        pass
    else:
        raise AssertionError(
            "Should have raised a ValueError as neither"
            + " n_points or target_reduction are specified"
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
        indices=indices,
        values=values,
        size=(n_landmarks, n_points),
        device="cpu",
    )
    mesh.landmarks = landmarks

    decimator = sks.Decimation(target_reduction=0.9)
    decimated_mesh = decimator.fit_transform(mesh)

    assert mesh.landmarks is not None
    assert decimated_mesh.landmarks is not None
    assert decimated_mesh.points.dtype == sks.float_dtype
    assert len(decimated_mesh.landmark_points) == len(
        mesh.landmark_points
    )  # assert that the number of landmarks is the same


def test_torch_sparse_tensor_repetitions():
    """Assert that the torch.sparse_coo_tensor can handle repetitions in the
    indices and sum the values"""
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
    sphere = sks.Sphere().to("cuda")

    dec = sks.Decimation(n_points=15)
    dec.fit(sphere)
    newsphere = dec.transform(sphere.to("cpu"))
    assert newsphere.points.device.type == "cpu"

    newsphere = dec.transform(sphere.to("cuda"))
    assert newsphere.points.device.type == "cuda"

    assert newsphere.points.dtype == sks.float_dtype
