"""Validation utilities for morphing module."""

from math import prod
from typing import Any

import torch
import torch.random

from ..input_validation import typecheck
from ..types import float_dtype, shape_type
from .basemodel import BaseModel


@typecheck
def validate_polydata_morphing_model(model: Any, shape: shape_type) -> None:
    """Test a model with a PolyData shape.

    This function does basic tests on a model with a PolyData shape. In order
    to be used in a task such as `Registration`, the model must fulfill some
    requirements.

    It checks that the parameter tensor can be initialized automatically from the
    model and that the `morph` induces a transformation that is differentiable
    with respect to the parameter, even if the parameter is a non-leaf variable.

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
        error_msg = "The model must be an instance of BaseModel"
        raise ValueError(error_msg)

    if not hasattr(model, "morph"):
        error_msg = "The model must have a method morph"
        raise NotImplementedError(error_msg)

    if not hasattr(model, "parameter_shape"):
        error_msg = "The model must have a method parameter_shape"
        raise NotImplementedError(error_msg)

    # Initialize the parameter as a non leaf variable
    # (See PR https://github.com/scikit-shapes/scikit-shapes/pull/60)
    # x -> parameter -> morph() -> loss
    # and check that the gradient is actually propagated from loss
    # to x
    parameter_shape = list(model.parameter_shape(shape=shape))

    x = torch.rand(parameter_shape[0], dtype=float_dtype)
    x.requires_grad = True

    parameter = x.repeat(prod(parameter_shape[1:])).reshape(parameter_shape)

    # Apply morphing and compute a dummy loss function
    output = model.morph(
        shape, parameter, return_path=True, return_regularization=True
    )
    loss = torch.sum(output.morphed_shape.points)
    loss.backward()

    # Check that the gradient has the same shape as the leaf variable parameter
    # (torch will raise an error here if parameter grad is not defined)
    assert x.grad.shape == x.shape
