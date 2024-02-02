"""A bunch of tests for some expected errors."""

import pytest
import torch

import skshapes as sks
from skshapes.errors import NotFittedError, ShapeError


def test_errors_metrics():
    """Raise some errors for the metrics."""
    triangle_mesh = sks.Sphere().decimate(n_points=10)
    edges = triangle_mesh.edges
    points = triangle_mesh.points
    wireframe_mesh = sks.PolyData(points=points, edges=edges)
    pointcloud = sks.PolyData(points=points)

    target = pointcloud

    model_iso = sks.IntrinsicDeformation(
        n_steps=1,
        metric=sks.AsIsometricAsPossible(),
    )
    loss = sks.L2Loss()

    registration = sks.Registration(
        model=model_iso,
        loss=loss,
        gpu=False,
    )

    # registration.fit(source=triangle_mesh, target=target)
    # registration.fit(source=wireframe_mesh, target=target)
    with pytest.raises(
        AttributeError, match="This metric requires edges to be defined"
    ):
        registration.fit(source=pointcloud, target=target)

    model_shell = sks.IntrinsicDeformation(
        n_steps=1,
        metric=sks.ShellEnergyMetric(),
    )

    registration = sks.Registration(
        model=model_shell,
        loss=sks.L2Loss(),
        gpu=False,
    )

    # registration.fit(source=triangle_mesh, target=target)
    with pytest.raises(
        AttributeError, match="This metric requires triangles to be defined"
    ):
        registration.fit(source=wireframe_mesh, target=target)
    with pytest.raises(
        AttributeError, match="This metric requires triangles to be defined"
    ):
        registration.fit(source=pointcloud, target=target)


@pytest.mark.parametrize(
    "model",
    [
        sks.RigidMotion(),
        sks.ExtrinsicDeformation(),
        sks.IntrinsicDeformation(),
    ],
)
def tests_error_registration(model):
    """Raise some errors for the Registration class."""
    source = sks.Sphere()
    target = sks.Sphere()

    # wrong parameter shape
    initial_parameter = torch.rand(15, 15)

    registration = sks.Registration(
        model=model,
        loss=sks.L2Loss(),
    )

    with pytest.raises(ShapeError):
        registration.fit(
            source=source, target=target, initial_parameter=initial_parameter
        )


def test_errors_polydata():
    """Raise some errors for the PolyData class."""
    # Try to initialize a mesh with a complex tensor
    real = torch.tensor([1, 2], dtype=torch.float32)
    imag = torch.tensor([3, 4], dtype=torch.float32)
    z = torch.complex(real, imag)
    with pytest.raises(ValueError, match="Complex tensors are not supported"):
        polydata = sks.PolyData(z)

    # Polydata with 3 points
    point = torch.rand(3, 3)
    polydata = sks.PolyData(point)

    # try to assign an incorrect edge array
    edges = torch.tensor(
        [
            [0, 1],
            [1, 2],
            [10, 1],
        ]
    )

    with pytest.raises(IndexError):
        polydata.edges = edges

    # try to assign an incorrect triangle array
    triangles = torch.tensor(
        [
            [0, 1, 2],
            [1, 2, 3],
            [10, 1, 2],
        ]
    )

    with pytest.raises(IndexError):
        polydata.triangles = triangles


def test_errors_decimation():
    """Trigger some errors for the decimation class."""
    mesh_1 = sks.Sphere().decimate(target_reduction=0.9)
    mesh_2 = sks.Sphere().decimate(target_reduction=0.5)
    pointcloud = sks.PolyData(points=mesh_1.points)

    with pytest.raises(ValueError, match="only works on triangle meshes"):
        pointcloud.decimate(target_reduction=0.9)

    with pytest.raises(
        ValueError, match="n_points must be lower than mesh.n_points"
    ):
        mesh_1.decimate(n_points=100000)

    with pytest.raises(ValueError, match="n_points must be positive"):
        d = sks.Decimation(n_points=0)

    with pytest.raises(
        ValueError, match=r"target_reduction must be in the range \(0, 1\)"
    ):
        d = sks.Decimation(target_reduction=1.2)

    with pytest.raises(
        ValueError, match=r"ratio must be in the range \(0, 1\)"
    ):
        d = sks.Decimation(ratio=-0.2)

    d = sks.Decimation(ratio=0.5)

    with pytest.raises(NotFittedError):
        d.transform(mesh_1)

    d.fit(mesh_1)

    with pytest.raises(ValueError, match=r"n_points must be positive"):
        d.transform(mesh_1, n_points=-1)

    with pytest.raises(
        ValueError, match=r"target_reduction must be in the range \(0, 1\)"
    ):
        d.transform(mesh_1, target_reduction=1.2)

    with pytest.raises(
        ValueError, match=r"ratio must be in the range \(0, 1\)"
    ):
        d.transform(mesh_1, ratio=-0.2)

    with pytest.raises(
        ValueError, match=r"n_points must be lower than mesh.n_points"
    ):
        d.transform(mesh_1, n_points=100000)

    with pytest.raises(
        ValueError, match=r"mesh.n_points and mesh.triangles must be the same"
    ):
        d.transform(mesh_2, target_reduction=0.5)


@pytest.mark.parametrize(
    "property_name", ["collapses", "actual_reduction", "ref_mesh"]
)
def test_error_decimation_not_fitted(property_name):
    """Test that an error is raised if a property is accessed before fitting."""
    decimator = sks.Decimation(target_reduction=0.5)
    with pytest.raises(NotFittedError):
        getattr(decimator, property_name)
