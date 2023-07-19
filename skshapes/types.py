from beartype import beartype
from jaxtyping import jaxtyped, Float32, Int64
from typing import Any, Optional, Union, TypeVar, Generic, List, Tuple, NamedTuple
import torch


def typecheck(func):
    return jaxtyped(beartype(func))


# Type aliases
Number = Union[int, float]

float_dtype = torch.float32
int_dtype = torch.int64
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
# TODO : use beartype validators to create a custom validator for landmarks
try:
    from typing import Annotated  # Python >= 3.9
except ImportError:
    from typing_extensions import Annotated  # Python < 3.9

from beartype.vale import Is


Landmarks = Annotated[
    torch.Tensor, Is[lambda tensor: tensor.dtype == float_dtype and tensor.is_sparse]
]


# Shape types
class ShapeType:
    pass


class PolyDataType(ShapeType):
    pass


class ImageType(ShapeType):
    pass


# Morphing types
class Loss:
    @typecheck
    def __add__(self, other: "Loss") -> "Loss":
        """Add two losses

        Args:
            other (Loss): the other loss to add

        Returns:
            Loss: a new loss which __call__ method is the sum of the two __call__ methods
        """

        class sum_of_losses(Loss):
            def __init__(self, loss1, loss2):
                self.loss1 = loss1
                self.loss2 = loss2

            def __call__(self, source: ShapeType, target: ShapeType) -> FloatScalar:
                return self.loss1.__call__(
                    source=source, target=target
                ) + self.loss2.__call__(source=source, target=target)

        loss1 = self
        loss2 = other

        return sum_of_losses(loss1=loss1, loss2=loss2)

    @typecheck
    def __rmul__(self, scalar: Number) -> "Loss":
        """Multiply a loss by a scalar

        Args:
            other (Number): the scalar to multiply the loss by

        Returns:
            Loss: a new loss which __call__ method is the product of the scalaer and the self.__call__ method
        """

        class newloss(Loss):
            def __init__(self, loss, scalar):
                self.loss = loss
                self.scalar = scalar

            def __call__(self, source: ShapeType, target: ShapeType) -> FloatScalar:
                return self.scalar * self.loss.__call__(source=source, target=target)

        loss = self
        return newloss(loss=loss, scalar=scalar)

    @typecheck
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        pass


class Morphing:
    pass


class Optimizer:
    pass


class MorphingOutput(NamedTuple):
    morphed_shape: Optional[ShapeType] = None
    regularization: Optional[FloatScalar] = None
    path: Optional[List[ShapeType]] = None
