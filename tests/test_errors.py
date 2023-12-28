"""A bunch of tests for some expected errors."""

import pytest
import torch
import skshapes as sks


def test_errors_():

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
        loss=loss,
    )

    registration.fit(source=triangle_mesh, target=target)
    with pytest.raises(ValueError):
        registration.fit(source=wireframe_mesh, target=target)
    with pytest.raises(ValueError):
        registration.fit(source=pointcloud, target=target)
