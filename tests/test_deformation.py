"""Tests for the deformation modules."""

import pytest
import torch

import skshapes as sks
from skshapes.errors import DeviceError

deformation_models = [
    sks.IntrinsicDeformation,
    sks.ExtrinsicDeformation,
    sks.RigidMotion,
]

mesh_1 = sks.Sphere().decimate(target_reduction=0.99)
mesh_2 = mesh_1.copy()
mesh_2.points += 0.2


def test_deformation():
    """Compatibility of deformations modules wrt autograd."""
    for deformation_model in deformation_models:
        _test(deformation_model)


def _test(deformation_model):
    # Define a pair of shapes and a loss function
    shape = mesh_1
    target = mesh_2
    loss = sks.L2Loss()

    # Initialize the deformation model
    model = deformation_model()
    # Get an initial parameter
    p = model.inital_parameter(shape=shape)

    p.requires_grad_(True)

    morphed_shape = model.morph(shape=shape, parameter=p).morphed_shape
    L = loss(morphed_shape, target)

    L.backward()
    assert p.grad is not None

    if torch.cuda.is_available():
        p = p.cuda()
        with pytest.raises(DeviceError):
            model.morph(shape=shape, parameter=p).morphed_shape  # noqa: B018


def test_extrinsic_deformation():
    """Test for extrinsic deformation.

    More specifically, we test the backends: torchdiffeq and sks
    """
    source = mesh_1
    source.control_points = source.bounding_grid(N=3, offset=0.0)
    target = mesh_2

    assert not torch.allclose(source.points, target.points, rtol=5e-3)

    for control_points in [True, False]:
        n_steps = 2
        model_torchdiffeq = sks.ExtrinsicDeformation(
            backend="torchdiffeq",
            solver="euler",
            n_steps=n_steps,
            control_points=control_points,
        )

        model_sks = sks.ExtrinsicDeformation(
            backend="sks",
            integrator=sks.EulerIntegrator(),
            n_steps=n_steps,
            control_points=control_points,
        )

        # For both models, we register with only one iteration of gradient
        # descent we expect the same result (tolerance of 0.5%)
        optimizer = sks.SGD(lr=1.0)
        n_iter = 1
        loss = sks.L2Loss()
        gpu = False
        regularization = 0

        registration_torchdiffeq = sks.Registration(
            model=model_torchdiffeq,
            loss=loss,
            optimizer=optimizer,
            n_iter=n_iter,
            regularization_weight=regularization,
            gpu=gpu,
        )

        registration_sks = sks.Registration(
            model=model_sks,
            loss=loss,
            optimizer=optimizer,
            n_iter=n_iter,
            regularization_weight=regularization,
            gpu=gpu,
        )

        out_torchdiffeq = registration_torchdiffeq.fit_transform(
            source=source, target=target
        )
        out_sks = registration_sks.fit_transform(source=source, target=target)

        # Make sure that something happened, ie the points are not the same
        # after registration than before

        assert not torch.allclose(
            out_torchdiffeq.points,
            source.points,
            rtol=5e-3,
        )

        # Now, we check that the two backends give the same result
        assert torch.allclose(
            out_torchdiffeq.points,
            out_sks.points,
            rtol=5e-3,
        )


circle = sks.Circle(n_points=5)


@pytest.mark.parametrize("shape", [mesh_1, circle])
def test_rigid_motion_multiple_steps(shape):
    """Test for rigid motion with multiple steps (2D and 3D).

    For rigid motion, the n_steps parameter has no impact on the morphed shape.
    It is useful for creating an animation of the morphing process. In this
    test, we assess that the morphed shape is the same for n_steps=1 and
    n_steps=3, being accessed through morphed_shape or path[-1].
    """
    rigid_motion_1 = sks.RigidMotion(n_steps=1)
    rigid_motion_2 = sks.RigidMotion(n_steps=3)

    parameter = torch.rand_like(rigid_motion_1.inital_parameter(shape))

    out = rigid_motion_1.morph(shape=shape, parameter=parameter).morphed_shape
    out2 = rigid_motion_2.morph(
        shape=shape, parameter=parameter, return_path=True
    ).path[-1]

    assert torch.allclose(out.points, out2.points, rtol=1e-3)
