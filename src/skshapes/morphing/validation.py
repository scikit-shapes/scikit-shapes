"""Validation utilities for morphing module."""

import torch

from ..types import float_dtype
from .basemodel import BaseModel


def validate_polydata_morphing_model(model, shape):
    """Test a model with a PolyData shape.

    This function does basic tests on a model with a PolyData shape. In order
    to be used in a task such as `Registration`, the model must fulfill some
    requirements.

    It checks that the parameter tensor can be initialized automatically from the
    model and that the `morph` induces a transformation that is differentiable
    with respect to the parameter.

    Parameters
    ----------
    model
        The model to test.
    shape
        The shape on which to test the model.

    Examples
    --------

    import skshapes as sks
    shape_3d = sks.Sphere()
    shape_2d = sks.Circle()
    model = sks.RigidMotion()
    sks.validate_model_polydata(model, shape_3d)
    sks.validate_model_polydata(model, shape_2d)

    """

    # Check if model is an instance of BaseModel
    if not isinstance(model, BaseModel):
        error_msg = "model must be an instance of BaseModel"
        raise ValueError(error_msg)

    # Initialize parameter with zeros, inferring shape from model
    parameter = torch.zeros(
        size=model.parameter_shape(shape), dtype=float_dtype
    )
    parameter.requires_grad = True

    # Apply morphing and compute a dummy loss function
    output = model.morph(
        shape, parameter, return_path=True, return_regularization=True
    )
    loss = torch.sum(output.morphed_shape.points)
    loss.backward()

    # Check that the gradient has the same shape as the parameter
    # (torch will raise an error here if parameter grad is not defined)
    assert parameter.grad.shape == parameter.shape
