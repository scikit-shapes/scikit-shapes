"""Tests for the decimation module."""

import pytest
import pyvista
import torch

import skshapes as sks
from skshapes.errors import InputStructureError


def test_indice_mapping_interface():
    """Test the interface of the indice mapping class."""
    sphere = sks.Sphere()
    d1 = sks.Decimation(n_points=10).fit(mesh=sphere)
    newsphere1, im1 = d1.transform(sphere, return_indice_mapping=True)

    d2 = sks.Decimation(n_points=10)
    newsphere2, im2 = d2.fit_transform(sphere, return_indice_mapping=True)

    assert torch.allclose(newsphere1.points, newsphere2.points)
    assert torch.allclose(im1, im2)


def test_mesh_decimation_n_points_strict():
    """Test that the n_points_strict argument works."""

    mesh = sks.Sphere()
    target_n_points = 10
    d = sks.Decimation(n_points=1)

    d.fit(mesh=mesh)

    decimated_mesh_strict = d.transform(
        mesh=mesh, n_points_strict=target_n_points
    )

    assert decimated_mesh_strict.n_points == target_n_points


def test_decimation_basic():
    """Test that fit + transform gives the same result as fit_transform."""
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

    target_reduction = 0.9

    # test with target_reduction
    decimation = sks.Decimation(target_reduction=target_reduction)
    decimated_sphere = decimation.fit_transform(sphere)
    decimated_sphere2 = sphere.decimate(target_reduction=target_reduction)

    assert torch.allclose(decimated_sphere.points, decimated_sphere2.points)
    assert torch.allclose(
        decimated_sphere.triangles, decimated_sphere2.triangles
    )

    # test with ratio (= 1 - target_reduction)
    decimated_sphere3 = decimation.transform(
        sphere, ratio=1 - target_reduction
    )
    assert torch.allclose(
        decimated_sphere3.points, decimated_sphere2.points
    )  # same points

    # Initialisation with ratio
    decimation = sks.Decimation(ratio=1 - target_reduction)
    decimated_sphere4 = decimation.fit_transform(sphere)
    assert torch.allclose(
        decimated_sphere4.points, decimated_sphere2.points
    )  # same points

    # Assert that the number of points is different when we use a different
    # ratio in the transform method
    decimated_sphere5 = decimation.transform(
        sphere,
        ratio=2 * (1 - target_reduction),
    )

    assert decimated_sphere5.n_points != decimated_sphere4.n_points

    # Some errors
    mesh = sks.Sphere()

    with pytest.raises(InputStructureError):
        # n_points and target_reduction are mutually exclusive
        mesh.decimate(n_points=10, target_reduction=0.9)

    with pytest.raises(InputStructureError):
        # at least one of n_points, n_target_reduction or ratio must be
        # specified
        mesh.decimate()


def test_decimation_landmarks():
    """Test decimation with landmarks."""
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
    """Test assign a value to a sparse tensor with repetitions (must sum)."""
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
    """Test decimation with PolyData on the gpu."""
    sphere = sks.Sphere().to("cuda")

    dec = sks.Decimation(n_points=15)
    dec.fit(sphere)
    newsphere = dec.transform(sphere.to("cpu"))
    assert newsphere.points.device.type == "cpu"

    newsphere = dec.transform(sphere.to("cuda"))
    assert newsphere.points.device.type == "cuda"

    assert newsphere.points.dtype == sks.float_dtype
