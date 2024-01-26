"""Utility functions and classes for the data module."""

from __future__ import annotations

import functools
import weakref
from collections.abc import Callable
from functools import cached_property, lru_cache, partial, update_wrapper
from typing import Any, TypeVar

import numpy as np
import pyvista
import torch

from ..errors import DeviceError, ShapeError
from ..input_validation import convert_inputs, typecheck
from ..types import (
    FloatTensor,
    IntTensor,
    NumericalArray,
    NumericalTensor,
)


class DataAttributes(dict):
    """DataAttributes class.

    This class is a dictionary aimed to store attributes associated to a
    data structure (e.g. a set of points, a set of triangles, etc.)
    When a new attribute is added to the dictionary, it is checked that its
    size is compatible with the size of the data structure.

    The DataAttributes structure ensures that all the attributes are
    torch.Tensor and on the same device, doing the necessary conversions if
    needed.

    There are two ways to add an attribute to the dictionary:
        - With an explicit name, using the __setitem__ method (e.g.
            A["attribute"] = attribute)
        - Without an explicit name, using the append method (e.g.
            A.append(attribute)) which will automatically set "attribute_{i}"
            where i is the minimum integer such that "attribute_{i}" is not
            already in the dictionary
    """

    @typecheck
    def __init__(self, *, n: int, device: str | torch.device) -> None:
        """Class constructor.

        Parameters
        ----------
        n
            The number of elements of the set.
        device
            The device on which the attributes should be stored.
        """
        self._n = n
        device = torch.Tensor().to(device).device
        self._device = device

    @typecheck
    def __getitem__(self, key: Any) -> NumericalTensor:
        """Get an attribute from the DataAttributes object.

        Parameters
        ----------
        key
            The key of the attribute to get.

        Raises
        ------
        KeyError
            If the key is not in the DataAttributes object.

        Returns
        -------
        NumericalTensor
            The attribute.
        """
        return dict.__getitem__(self, key)

    def __repr__(self) -> str:
        """Representation of the DataAttributes object.

        Writes the attributes of the DataAttribute object, its shape and
        its dtype. Also writes the number of elements of the set and the
        device on which the attributes are stored.

        Returns
        -------
        str
            The representation of the DataAttributes object.
        """
        string = "DataAttributes Object with attributes:\n"
        for key, value in self.items():
            string += f"- {key}: {value.shape}, {value.dtype}\n"

        string += f"Number of elements: {self._n}\n"
        string += f"Device: {self._device}\n"
        return string

    @convert_inputs
    @typecheck
    def _check_value(self, value: NumericalTensor) -> NumericalTensor:
        if value.shape[0] != self._n:
            raise ShapeError(
                f"First dimension of the tensor should be {self._n}, got"
                + f"{value.shape[0]}"
            )

        if value.device != self._device:
            value = value.to(self._device)

        return value

    @convert_inputs
    @typecheck
    def __setitem__(
        self, key: Any, value: NumericalTensor | NumericalArray
    ) -> None:
        """Set an attribute of the DataAttributes object.

        Parameters
        ----------
        key
            The key of the attribute to set.
        value
            The value of the attribute to set.
        """
        value = self._check_value(value)
        dict.__setitem__(self, key, value)

    @convert_inputs
    @typecheck
    def append(self, value: FloatTensor | IntTensor) -> None:
        """Append an attribute to the DataAttributes object.

        The key of the attribute will be "attribute_{i}" where i is the minimum
        integer such that "attribute_{i}" is not already in the dictionary.

        Parameters
        ----------
        value
            The value of the attribute to append.
        """
        value = self._check_value(value)
        i = 0
        while f"attribute_{i}" in self.keys():
            i += 1

        dict.__setitem__(self, f"attribute_{i}", value)

    @typecheck
    def clone(self) -> DataAttributes:
        """Clone the DataAttributes object.

        Returns
        -------
        DataAttributes
            The cloned DataAttributes object.
        """
        clone = DataAttributes(n=self._n, device=self._device)
        for key, value in self.items():
            clone[key] = value.clone()
        return clone

    @typecheck
    def to(self, device: str | torch.device) -> DataAttributes:
        """Move the DataAttributes object on a given device.

        Parameters
        ----------
        device
            The device on which the object should be made.

        Returns
        -------
        DataAttributes
            The copy of the DataAttributes object on the new device.
        """
        clone = DataAttributes(n=self._n, device=device)
        for key, value in self.items():
            clone[key] = value.to(device)
        return clone

    @typecheck
    @classmethod
    def from_dict(
        cls,
        attributes: dict[Any, NumericalTensor | NumericalArray],
        device: str | torch.device | None = None,
    ) -> DataAttributes:
        """From dictionary constructor."""
        if len(attributes) == 0:
            raise ValueError(
                "The dictionary of attributes should not be empty to"
                + " initialize a DataAttributes object"
            )

        # Ensure that the number of elements of the attributes is the same
        n = next(iter(attributes.values())).shape[0]
        for value in attributes.values():
            if value.shape[0] != n:
                raise ShapeError(
                    "The number of rows of each value the dictionary must be"
                    + " the same to be convert to a DataAttributes object"
                )

        if device is None:
            # Ensure that the attributes are on the same device (if they are
            # torch.Tensor, unless they have no device attribute and we set
            # device to cpu)
            if hasattr(next(iter(attributes.values())), "device"):
                device = next(iter(attributes.values())).device
                for value in attributes.values():
                    if value.device != device:
                        raise DeviceError(
                            "The attributes should be on the same device to be"
                            + " converted into a DataAttributes object"
                        )
            else:
                device = torch.device("cpu")

        output = cls(n=n, device=device)
        for key, value in attributes.items():
            output[key] = value

        return output

    @classmethod
    def from_pyvista_datasetattributes(
        cls,
        attributes: pyvista.DataSetAttributes,
        device: str | torch.device | None = None,
    ) -> DataAttributes:
        """From pyvista.DataSetAttributes constructor."""
        # First, convert the pyvista.DataSetAttributes object to a dictionary
        dict_attributes = {}

        for key in attributes:
            if isinstance(attributes[key], np.ndarray):
                dict_attributes[key] = np.array(attributes[key])
            else:
                dict_attributes[key] = np.array(pyvista.wrap(attributes[key]))

        # return attributes

        # Then, convert the dictionary to a DataAttributes object with
        # from_dict
        return cls.from_dict(attributes=dict_attributes, device=device)

    @typecheck
    def to_numpy_dict(self) -> dict[Any, NumericalArray]:
        """Cast as dictionary of numpy arrays."""
        d = dict(self)
        for key, value in d.items():
            d[key] = value.detach().cpu().numpy()

        return d

    @property
    @typecheck
    def n(self) -> int:
        """Number of elements of the set getter."""
        return self._n

    @n.setter
    @typecheck
    def n(self, n: Any) -> None:  # noqa: ARG002
        """Setter for the number of elements.

        This setter is not meant to be used, as it would change the number of
        elements of the set after the creation of the DataAttributes object.

        Raises
        ------
        ValueError
            If this setter is called.
        """
        raise ValueError(
            "You cannot change the number of elements of the set after the"
            + " creation of the DataAttributes object"
        )

    @property
    @typecheck
    def device(self) -> str | torch.device:
        """Device getter."""
        return self._device

    @device.setter
    @typecheck
    def device(self, device: Any) -> None:  # noqa: ARG002
        """Device cannot be changed with the setter.

        If you want to change the device of the DataAttributes object, use
        .to(device) to make a copy of the DataAttributes object on the new
        device.

        Raises
        ------
        ValueError
            If this setter is called.
        """
        raise ValueError(
            "You cannot change the device of the set after the creation of the"
            + " DataAttributes object, use .to(device) to make a copy of the"
            + " DataAttributes object on the new device"
        )


def cached_method(*lru_args, **lru_kwargs):
    """Least-recently-used cache decorator for instance methods."""

    def decorator(func):
        @functools.wraps(func)
        def wrapped_func(self, *args, **kwargs):
            # We're storing the wrapped method inside the instance. If we had
            # a strong reference to self the instance would never die.
            self_weak = weakref.ref(self)

            @functools.wraps(func)
            @functools.lru_cache(*lru_args, **lru_kwargs)
            def cached_method(*args, **kwargs):
                return func(self_weak(), *args, **kwargs)

            setattr(self, func.__name__, cached_method)
            return cached_method(*args, **kwargs)

        return wrapped_func

    return decorator


T = TypeVar("T")


def instance_lru_cache(
    method: Callable[..., T] | None = None,
    *,
    maxsize: int | None = 128,
    typed: bool = False,
) -> Callable[..., T] | Callable[[Callable[..., T]], Callable[..., T]]:
    """Least-recently-used cache decorator for instance methods.

    The cache follows the lifetime of an object (it is stored on the object,
    not on the class) and can be used on unhashable objects. Wrapper around
    functools.lru_cache.

    If *maxsize* is set to None, the LRU features are disabled and the cache
    can grow without bound.

    If *typed* is True, arguments of different types will be cached separately.
    For example, f(3.0) and f(3) will be treated as distinct calls with
    distinct results.

    Arguments to the cached method (other than 'self') must be hashable.

    View the cache statistics named tuple (hits, misses, maxsize, currsize)
    with f.cache_info().  Clear the cache and statistics with f.cache_clear().
    Access the underlying function with f.__wrapped__.

    """

    def decorator(wrapped: Callable[..., T]) -> Callable[..., T]:
        def wrapper(self: object) -> Callable[..., T]:
            return lru_cache(maxsize=maxsize, typed=typed)(
                update_wrapper(partial(wrapped, self), wrapped)
            )

        return cached_property(wrapper)  # type: ignore[code]

    return decorator if method is None else decorator(method)


def cache_clear(self):
    """Reload all cached properties."""
    cls = self.__class__
    attrs = [
        a
        for a in dir(self)
        if isinstance(getattr(cls, a, cls), cached_property)
    ]
    for a in attrs:
        delattr(self, a)

    if hasattr(self, "cached_methods"):
        for a in self.cached_methods:
            getattr(self, a).cache_clear()
