"""Base class for all deformation models."""

from ..types import shape_type, float_dtype, FloatTensor
import torch


class BaseModel:
    """Base class for all deformation models."""

    def inital_parameter(self, shape: shape_type) -> FloatTensor:
        """Return the initial parameters of the model."""
        param_shape = self.parameter_shape(shape=shape)
        return torch.zeros(param_shape, dtype=float_dtype, device=shape.device)
