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
def test_error_registration(model):
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


def test_errors_dataset_attributes():
    """Raise some errors for the DatasetAttributes class."""
    attributes = sks.data.utils.DataAttributes(n=50, device="cpu")

    # add two attributes
    attributes.append(torch.rand(50, 3))
    attributes.append(torch.rand(50, 3))
    print(attributes)  # noqa: T201

    with pytest.raises(ValueError, match="should not be empty"):
        sks.data.utils.DataAttributes.from_dict({})

    with pytest.raises(
        ValueError, match="cannot change the number of elements"
    ):
        attributes.n = 51

    with pytest.raises(ValueError, match="cannot change the device"):
        attributes.device = "cuda"


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


def test_multiscale_not_implemented():
    mesh = sks.Sphere()
    ratios = [0.01, 0.5]
    M = sks.Multiscale(mesh, ratios=ratios)

    with pytest.raises(NotImplementedError):
        M.at(scale=0.2)

    with pytest.raises(NotImplementedError):
        sks.Multiscale(mesh, scales=[0.1, 0.5])

    with pytest.raises(NotImplementedError):
        M.append(scale=0.2)

    d = sks.Decimation(n_points=1)
    with pytest.raises(NotFittedError):
        # The decimation module is not fitted before passed to the Multiscale
        # object
        M = sks.Multiscale(mesh, ratios=ratios, decimation_module=d)

    with pytest.raises(
        NotImplementedError, match="Only triangle meshes are supported for now"
    ):
        M = sks.Multiscale(sks.Circle(n_points=4), ratios=ratios)

    M.at(ratio=0.5)["signal"] = torch.zeros(M.at(ratio=0.5).n_points)

    with pytest.raises(NotImplementedError):
        M.propagate(
            signal_name="signal",
            from_scale=0.5,
        )

    with pytest.raises(KeyError, match="unknown_signal not available"):
        M.propagate(
            signal_name="unknown_signal",
            from_ratio=0.5,
        )


def test_error_loss():
    with pytest.raises(NotImplementedError):
        sks.loss.baseloss.BaseLoss()

    with pytest.raises(ValueError, match="p must be positive"):
        sks.LpLoss(p=-1)
