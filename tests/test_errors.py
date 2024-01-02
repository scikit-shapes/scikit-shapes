"""A bunch of tests for some expected errors."""

import pytest
import torch
import skshapes as sks


def test_errors_metrics():
    triangle_mesh = sks.Sphere()
    edges = triangle_mesh.edges
    points = triangle_mesh.points
    wireframe_mesh = sks.PolyData(points=points, edges=edges)
    pointcloud = sks.PolyData(points=points)

    target = sks.Sphere()

    model_iso = sks.IntrinsicDeformation(
        n_steps=3,
        metric=sks.AsIsometricAsPossible(),
    )
    loss = sks.L2Loss()

    registration = sks.Registration(
        model=model_iso,
        loss=loss,
    )

    registration.fit(source=triangle_mesh, target=target)
    registration.fit(source=wireframe_mesh, target=target)
    with pytest.raises(ValueError):
        registration.fit(source=pointcloud, target=target)

    model_shell = sks.IntrinsicDeformation(
        n_steps=3,
        metric=sks.ShellEnergyMetric(),
    )

    registration = sks.Registration(
        model=model_shell,
        loss=sks.NearestNeighborsLoss(),
    )

    registration.fit(source=triangle_mesh, target=target)
    with pytest.raises(ValueError):
        registration.fit(source=wireframe_mesh, target=target)
    with pytest.raises(ValueError):
        registration.fit(source=pointcloud, target=target)


def tests_error_registration():
    source = sks.Sphere()
    target = sks.Sphere()

    # wrong parameter shape
    initial_parameter = torch.rand(15, 15)
    model = sks.RigidMotion()

    registration = sks.Registration(
        model=model,
        loss=sks.L2Loss(),
    )

    with pytest.raises(ValueError):
        registration.fit(
            source=source, target=target, initial_parameter=initial_parameter
        )


def test_errors_polydata():
    # Try to initialize a mesh with a complex tensor
    real = torch.tensor([1, 2], dtype=torch.float32)
    imag = torch.tensor([3, 4], dtype=torch.float32)
    z = torch.complex(real, imag)
    with pytest.raises(ValueError):
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

    with pytest.raises(ValueError):
        polydata.edges = edges

    # try to assign an incorrect triangle array
    triangles = torch.tensor(
        [
            [0, 1, 2],
            [1, 2, 3],
            [10, 1, 2],
        ]
    )

    with pytest.raises(ValueError):
        polydata.triangles = triangles
