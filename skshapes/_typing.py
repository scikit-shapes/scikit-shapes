from beartype import beartype
from jaxtyping import jaxtyped, Float32, Int64
from typing import Optional, Union, TypeVar, Generic

import torch


def typecheck(func):
    return jaxtyped(beartype(func))


pointsType = Float32[torch.Tensor, "_ 3"]
edgesType = Int64[torch.Tensor, "2 _"]
trianglesType = Int64[torch.Tensor, "3 _"]
floatTensorArrayType = Float32[torch.Tensor, "_"]
landmarksType = Int64[torch.Tensor, "_"]

class PolyDataType:
    # Empty for the moment, will be useful if we want to rename our PolyData class without rewriting every annotation
    # And if later we want to make it possible to replace a PolyData by a string or a pyVista mesh
    pass