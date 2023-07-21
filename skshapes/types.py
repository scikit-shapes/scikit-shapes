"""In this module, we define the basic types used in the library (those who are only relying on third-part libraries).
These types are used to annotate the functions and classes of the library.
Types that corresponds to classes are defined in the classes module to avoid circular import.

Ex: the generic type Shape is defned in skshapes.data, the generic type Loss is defined in skshapes.loss
"""

from beartype import beartype
from jaxtyping import jaxtyped, Float32, Int64, Float, Int
from typing import (
    Any,
    Optional,
    Union,
    TypeVar,
    Generic,
    List,
    Tuple,
    NamedTuple,
    Dict,
    TypeVar,
)
import torch
import numpy as np


def typecheck(func):
    return jaxtyped(beartype(func))


# Type aliases
Number = Union[int, float]
float_dtype = torch.float32
int_dtype = torch.int64
JaxFloat = Float32
JaxInt = Int64

# Numpy array types
FloatArray = Float[np.ndarray, "..."] # Any float format numpy array
IntArray = Int[np.ndarray, "..."] # Any int format numpy array
NumericalArray = Union[FloatArray, IntArray]

# Numerical types
FloatTensor = JaxFloat[torch.Tensor, "..."] # Only Float32 tensors are FloatTensors
IntTensor = JaxInt[torch.Tensor, "..."] # Only Int64 tensors are IntTensors
NumericalTensor = Union[FloatTensor, IntTensor]
FloatTensorArray = JaxFloat[torch.Tensor, "_"]
IntTensorArray = JaxInt[torch.Tensor, "_"]
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
try:
    from typing import Annotated  # Python >= 3.9
except ImportError:
    from typing_extensions import Annotated  # Python < 3.9

from beartype.vale import Is

Landmarks = Annotated[
    torch.Tensor, Is[lambda tensor: tensor.dtype == float_dtype and tensor.is_sparse]
]
