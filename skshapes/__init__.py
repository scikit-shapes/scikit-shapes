from .types import typecheck
from typing import Literal, Union
import torch

Float_dtype = Literal["float32", "float64"]

__all__ = [
    "morphing",
]


# from jaxtyping import Float32, Float64


# class PackageConfig:

#     @typecheck
#     def __init__(self) -> None:
#         self._float_dtype = "float32"
#         self.update_tensor_types()

#     @property
#     @typecheck
#     def float_dtype(self) -> Float_dtype:
#         return self._float_dtype

#     @float_dtype.setter
#     @typecheck
#     def float_dtype(self, dtype: Float_dtype) -> None:
#         self._float_dtype = dtype
#         self.update_tensor_types()

#     def update_tensor_types(self) -> None:
#         JaxFloat = _jax_float_type(self.float_dtype)
#         self.Float1dTensor = JaxFloat[torch.Tensor, "_"]
#         self.Float2dTensor = JaxFloat[torch.Tensor, "_ _"]
#         self.Float3dTensor = JaxFloat[torch.Tensor, "_ _ _"]
#         self.FloatTensor = JaxFloat[torch.Tensor, ...]


# #namedtuple type


# config = PackageConfig()

from .types import *

from .applications import *
from .data import *
from .features import *
from .loss import *
from .morphing import *
from .optimization import *
from .multiscaling import *

from .decimation import *

# from .preprocessing import *
from .tasks import *
