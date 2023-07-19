from beartype import beartype
from jaxtyping import jaxtyped, Float32, Int64
from typing import Any, Optional, Union, TypeVar, Generic, List, Tuple, NamedTuple, Dict
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
FloatTensor = JaxFloat[torch.Tensor, "..."]
IntTensor = JaxInt[torch.Tensor, "..."]

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

class Features(dict):
    """This class is a dictionary that contains features associated to a set (e.g. a set of points, a set of triangles, etc.)
    When a feature is added to the dictionary, it is checked that the number of elements of the feature is the same as the number of elements of the set and
    it is passed to the device of the set.

    There are two ways to add a feature to the dictionary:
        - By using the __setitem__ method (e.g. A["feature"] = feature)
        - By using the append method (e.g. A.append(feature)) which will automatically name the feature "feature_{i}" where i is the minimum integer such that "feature_{i}" is not already in the dictionary

    Args:
        n (int): The number of elements of the set
        device (torch.device): The device on which the features should be stored
    """
    @typecheck
    def __init__(self,*,  n: int, device: Union[str, torch.device]) -> None:
        
        self._n = n
        self._device = device

    @typecheck
    def __getitem__(self,key):
        return dict.__getitem__(self,key)
    
    @typecheck
    def _check_value(self, value : Union[FloatTensor, IntTensor]) -> Union[FloatTensor, IntTensor]:

        assert value.shape[0] == self._n, f"Last dimension of the tensor should be {self._n}"
        if value.device != self._device:
            value = value.to(self._device)

        return value
    
    @typecheck
    def __setitem__(self, key: Any, value: Union[FloatTensor, IntTensor]) -> None:

        #assert that the 
        value = self._check_value(value)
        dict.__setitem__(self, key, value)

    @typecheck
    def append(self, value: Union[FloatTensor, IntTensor]) -> None:

        value = self._check_value(value)
        i = 0
        while f"feature_{i}" in self.keys():
            i += 1

        dict.__setitem__(self, f"feature_{i}", value)

    @typecheck
    def clone(self) -> "Features":

        clone = Features(n=self._n, device=self._device)
        for key, value in self.items():
            clone[key] = value.clone()
        return clone
    
    @typecheck
    def to(self, device: Union[str, torch.device]) -> "Features":
            
        clone = Features(n=self._n, device=device)
        for key, value in self.items():
            clone[key] = value.to(device)
        return clone
    
    @typecheck
    @classmethod
    def from_dict(cls, features: Dict[str, Union[FloatTensor, IntTensor]], device: Optional[Union[str, torch.device]] = None) -> "Features":
        """Create a Features object from a dictionary of features

        Args:
            features (Dict[str, Union[FloatTensor, IntTensor]]): The dictionary of features

        Returns:
            Features: The Features object
        """

        # Ensure that the number of elements of the features is the same
        n = list(features.values())[0].shape[0]
        for value in features.values():
            assert value.shape[0] == n, "The number of elements of the dictionnary should be the same to be converted into a Features object"

        if device is None:
            # Ensure that the features are on the same device
            device = list(features.values())[0].device
            for value in features.values():
                assert value.device == device, "The features should be on the same device to be converted into a Features object"

        output = cls(n=n, device=device)
        for key, value in features.items():
            output[key] = value
        
        return output

    @property
    @typecheck
    def n(self) -> int:
        return self._n
    
    @n.setter
    @typecheck
    def n(self, n: Any) -> None:
        raise ValueError("You cannot change the number of elements of the set after the creation of the Features object")

    @property
    @typecheck
    def device(self) -> Union[str, torch.device]:
        return self._device
    
    @device.setter
    @typecheck
    def device(self, device: Any) -> None:
        raise ValueError("You cannot change the device of the set after the creation of the Features object, use .to(device) to make a copy of the Features object on the new device")