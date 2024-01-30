"""Tests for the deformation modules."""


import os

import pytest
import torch

import skshapes as sks
from skshapes.errors import DeviceError

deformation_models = [
    sks.IntrinsicDeformation,
    sks.ExtrinsicDeformation,
    sks.RigidMotion,
]


@pytest.mark.skip(reason="Fail on MacOS, disable for now")
def test_deformation():
    """Compatibility of deformations modules wrt autograd."""
    for deformation_model in deformation_models:
        _test(deformation_model)


def _test(deformation_model):
    # Define a pair of shapes and a loss function
    shape = sks.Sphere().decimate(target_reduction=0.99)
    target = shape.copy()
    target.points += 1
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


@pytest.mark.skipif(
    "data" not in os.listdir(),
    reason="Data folder is not present",
)
def test_extrinsic_deformation():
    """Test for extrinsic deformation.

    More specifically, we test the backends: torchdiffeq and sks
    """
    source = sks.PolyData("data/fingers/finger0.ply")
    source.control_points = source.bounding_grid(N=5)
    target = sks.PolyData("data/fingers/finger1.ply")

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
        optimizer = sks.SGD(lr=10.0)
        n_iter = 1
        loss = sks.L2Loss()
        gpu = False
        regularization = 1.0

        registration_torchdiffeq = sks.Registration(
            model=model_torchdiffeq,
            loss=loss,
            optimizer=optimizer,
            n_iter=n_iter,
            regularization=regularization,
            gpu=gpu,
        )

        registration_sks = sks.Registration(
            model=model_sks,
            loss=loss,
            optimizer=optimizer,
            n_iter=n_iter,
            regularization=regularization,
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
