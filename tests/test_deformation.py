"""Tests for the deformation modules."""

import pytest
import torch

import skshapes as sks

deformation_models = [
    sks.IntrinsicDeformation(),
    sks.ExtrinsicDeformation(),
    sks.RigidMotion(),
]

sphere = sks.Sphere().resample(n_points=10)
circle = sks.Circle()


@pytest.mark.parametrize("model", deformation_models)
def test_deformation_polydata(model):
    """Test for deformation models with polydata shapes (2D and 3D)."""
    sks.validate_polydata_morphing_model(model, sphere)
    sks.validate_polydata_morphing_model(model, circle)


@pytest.mark.parametrize("shape", [sphere, circle])
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


@pytest.mark.parametrize("model", deformation_models)
def test_copy_model(model):

    # Copy the model
    model_copy = model.copy()
    # Assert that the number of steps is the same
    assert model_copy.n_steps == model.n_steps
    # Change the number of steps of the copy
    model_copy.n_steps = 2 * model.n_steps
    # Assert that the number of steps is different
    assert model_copy.n_steps != model.n_steps


def test_extrinsic_deformation_autograd():
    """Make sur that extrinsic deformation can be called with a parameter with requires_autograd"""

    model = sks.ExtrinsicDeformation(
        n_steps=2,
        kernel="gaussian",
        scale=0.5,
    )

    mesh = sks.Sphere().decimate(n_points=20)

    x = torch.rand(3)
    x.requires_grad = True
    parameter_shape = model.parameter_shape(shape=mesh)
    parameter = x.repeat(parameter_shape[0]).reshape(parameter_shape)

    morphed_mesh_1 = model.morph(
        shape=mesh,
        parameter=parameter,
    ).morphed_shape

    parameter_copy_nograd = parameter.detach().clone()
    parameter_copy_nograd.requires_grad = False

    morphed_mesh_2 = model.morph(
        shape=mesh,
        parameter=parameter_copy_nograd,
    ).morphed_shape

    assert torch.allclose(morphed_mesh_1.points, morphed_mesh_2.points)
