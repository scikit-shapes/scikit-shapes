from ..types import typecheck

# from .polydata import PolyData
# import pyvista


# @typecheck
# def read(filename: str) -> PolyData:
#     mesh = pyvista.read(filename)
#     if type(mesh) == pyvista.PolyData:
#         return PolyData.from_pyvista(mesh)
#     else:
#         raise NotImplementedError("Images are not supported yet")


from ..types import Any, Optional, Union, FloatTensor, IntTensor, Dict
import torch


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
    def __init__(self, *, n: int, device: Union[str, torch.device]) -> None:
        self._n = n
        self._device = device

    @typecheck
    def __getitem__(self, key):
        return dict.__getitem__(self, key)

    @typecheck
    def _check_value(
        self, value: Union[FloatTensor, IntTensor]
    ) -> Union[FloatTensor, IntTensor]:
        assert (
            value.shape[0] == self._n
        ), f"Last dimension of the tensor should be {self._n}"
        if value.device != self._device:
            value = value.to(self._device)

        return value

    @typecheck
    def __setitem__(self, key: Any, value: Union[FloatTensor, IntTensor]) -> None:
        # assert that the
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
    def from_dict(
        cls,
        features: Dict[str, Union[FloatTensor, IntTensor]],
        device: Optional[Union[str, torch.device]] = None,
    ) -> "Features":
        """Create a Features object from a dictionary of features

        Args:
            features (Dict[str, Union[FloatTensor, IntTensor]]): The dictionary of features

        Returns:
            Features: The Features object
        """

        # Ensure that the number of elements of the features is the same
        n = list(features.values())[0].shape[0]
        for value in features.values():
            assert (
                value.shape[0] == n
            ), "The number of elements of the dictionnary should be the same to be converted into a Features object"

        if device is None:
            # Ensure that the features are on the same device
            device = list(features.values())[0].device
            for value in features.values():
                assert (
                    value.device == device
                ), "The features should be on the same device to be converted into a Features object"

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
        raise ValueError(
            "You cannot change the number of elements of the set after the creation of the Features object"
        )

    @property
    @typecheck
    def device(self) -> Union[str, torch.device]:
        return self._device

    @device.setter
    @typecheck
    def device(self, device: Any) -> None:
        raise ValueError(
            "You cannot change the device of the set after the creation of the Features object, use .to(device) to make a copy of the Features object on the new device"
        )
