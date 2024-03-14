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
