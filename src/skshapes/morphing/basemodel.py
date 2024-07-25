"""Base class for all deformation models."""

from __future__ import annotations

import torch

from ..input_validation import typecheck
from ..types import FloatTensor, float_dtype, shape_type


class BaseModel:
    """Base class for all deformation models."""

    @typecheck
    def inital_parameter(self, shape: shape_type) -> FloatTensor:
        """Return the initial parameters of the model.

        Parameters
        ----------
        shape
            the shape to morph

        Returns
        -------
        FloatTensor
            the initial parameters of the model

        """
        param_shape = self.parameter_shape(shape=shape)
        return torch.zeros(param_shape, dtype=float_dtype, device=shape.device)

    @typecheck
    def copy(self) -> BaseModel:
        """Return a copy of the model.

        Returns
        -------
        BaseModel
            a copy of the model

        """
        args_name = self.copy_features
        kwargs = {arg: getattr(self, arg, None) for arg in args_name}

        return self.__class__(**kwargs)
