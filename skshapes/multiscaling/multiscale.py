"""This module contains the multiscale class."""
# TODO : add_points_data interface ? Deal with existing data ?
# TODO : landmark interface ?


# Signal management :
# maintain a dict of signals/policy
# when the multiscale is initialized, the list corresponds to the signals at the origin ratio
# when a ratio is added, the signals are propagated to the new ratio
# when at is called, the signal is propagated to the given ratio

from __future__ import annotations

from ..types import (
    typecheck,
    convert_inputs,
    Optional,
    FloatSequence,
    IntSequence,
    int_dtype,
    float_dtype,
    Int1dTensor,
    NumericalTensor,
    Number,
    polydata_type,
    shape_type,
)
from typing import List, Literal
from ..utils import scatter
import torch
from typing import Union
from .multiscale_triangle_mesh import MultiscaleTriangleMesh


class Multiscale:
    @typecheck
    def __new__(
        cls, shape: shape_type, **kwargs
    ) -> Union[MultiscaleTriangleMesh, None]:
        if hasattr(shape, "is_triangle_mesh") and shape.is_triangle_mesh:
            instance = super(Multiscale, cls).__new__(MultiscaleTriangleMesh)
            instance.__init__(shape=shape, **kwargs)
            return instance

        else:
            raise NotImplementedError("Only triangle meshes are supported for now")
