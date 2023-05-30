from beartype import beartype
from jaxtyping import jaxtyped, Float32, Int64
from typing import Optional, Union, TypeVar, Generic, List

import torch


def typecheck(func):
    return jaxtyped(beartype(func))


floatType = Float32


pointsType = floatType[torch.Tensor, "_ 3"]
edgesType = Int64[torch.Tensor, "2 _"]
trianglesType = Int64[torch.Tensor, "3 _"]
floatTensorArrayType = floatType[torch.Tensor, "_"]
float2dTensorType = floatType[torch.Tensor, "_ _"]
float3dTensorType = floatType[torch.Tensor, "_ _ _"]
landmarksType = Int64[torch.Tensor, "_"]
floatScalarType = floatType[torch.Tensor, ""]


class PolyDataType:
    # Empty for the moment, will be useful if we want to rename our PolyData class without rewriting every annotation
    # And if later we want to make it possible to replace a PolyData by a string or a pyVista mesh
    pass
