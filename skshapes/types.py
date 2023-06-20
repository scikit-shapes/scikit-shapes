from beartype import beartype
from jaxtyping import jaxtyped, Float32, Int64
from typing import Optional, Union, TypeVar, Generic, List, Tuple, NamedTuple
import torch


def typecheck(func):
    return jaxtyped(beartype(func))


# Type aliases
float = torch.float32
int = torch.int64
JaxFloat = Float32
JaxInt = Int64

# Numerical types
FloatTensorArray = JaxFloat[torch.Tensor, "_"]
Float1dTensor = JaxFloat[torch.Tensor, "_"]
Float2dTensor = JaxFloat[torch.Tensor, "_ _"]
Float3dTensor = JaxFloat[torch.Tensor, "_ _ _"]
FloatScalar = JaxFloat[torch.Tensor, ""]

# Specific numerical types
Points = JaxFloat[torch.Tensor, "_ 3"]
Edges = JaxInt[torch.Tensor, "2 _"]
Triangles = JaxInt[torch.Tensor, "3 _"]

# Jaxtyping does not provide annotation for sparse tensors
# Then we use the torch.Tensor type and checks are made at runtime
# with assert statements
Landmarks = Optional[torch.Tensor]


# Shape types
class Shape:
    pass


class PolyData(Shape):
    pass


class Image(Shape):
    pass


# Morphing types
class Loss:
    pass


class Morphing:
    pass


class Optimizer:
    pass


class MorphingOutput(NamedTuple):
    morphed_shape: Optional[Shape] = None
    regularization: Optional[FloatScalar] = None
    path: Optional[List[Shape]] = None
