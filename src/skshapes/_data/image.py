"""Image shape class."""
from __future__ import annotations

import torch

from ..input_validation import convert_inputs, typecheck
from ..types import NumericalTensor, image_type, JaxInt


class Image(image_type):
    """A D-dimensional grid, with a K-dimensional value associated to each of its voxels.

    An :class:`Image` object can be created from `torch.Tensors <https://pytorch.org/docs/stable/tensors.html#torch.Tensor>`_.

    Parameters
    ----------
    values
        The value of the image at each pixel/voxel.
        For "classical" images (with scalar values at each pixel), ``values`` should be a tensor of shape
        ``(X_1,...,X_D)``. For vector/matricial images (with multidimensional values at each pixel), the tensor should
        be of shape ``(X_1,...,X_D,V_1,...,V_K)`` such that ``values[x1,...,xD]`` gives the value of the image at voxel
        ``(x1,...,xD)``. In this case, the dimension of the image should be provided.
        Alternatively, an :class:`Image` can be directly provided.

    dim
        The dimension of the image. It should be only provided when ``image`` is multivalued and values is a torch
        Tensor.

    dtype
        The data type of the image values.
        If None it is inferred from the input.

    device
        The device on which the shape is stored (e.g. ``"cpu"`` or ``"cuda"``).
        If None it is inferred from the input.

    Examples
    --------

    .. testcode::

        import skshapes as sks

        image = sks.Image(values=torch.ones(size=(3, 5, 7)))
        print(image.shape)


    .. testoutput::

        (3, 5, 7)

    .. testcode::

        image = sks.Image(values=torch.ones(size=(3, 5, 7, 2, 2)), dim=3)
        print(image.shape)

    .. testoutput::

        (3, 5, 7)

    .. testcode::

        print(shape.dim)

    .. testoutput::

        3

    .. testcode::

        print(shape.values_shape)

    .. testoutput::

        (2, 2)

    .. testcode::

        print(shape.values_dim)

    .. testoutput::

        2

    """

    @convert_inputs
    @typecheck
    def __init__(
            self,
            values: torch.Tensor | Image,
            *,
            dim: int | None = None,
            dtype: torch.dtype | None = None,
            device: str | torch.device | None = None
    ):
        # Internally, the image values are stored in a flatten tensor. This make the dimension management easier, while
        # optimizing indices-related operations.

        if isinstance(values, Image):
            self._full_shape = values.full_shape
            self._dim = values.dim

            self._flat_values = values._flat_values
        else:
            if dim is None:
                self._full_shape = tuple(values.shape)
                self._dim = len(values.shape)
            else:
                self._full_shape = values.shape
                self._dim = dim

            self._flat_values = values.flatten(start_dim=0, end_dim=self._dim - 1)

        if dtype is None: dtype = values.dtype
        if device is None: device = values.device

        self._flat_values = self._flat_values.to(dtype=dtype, device=device)

    ##########################
    #### Image properties ####
    ##########################
    @property
    @typecheck
    def shape(self) -> tuple[int, ...]:
        return self._full_shape[:self.dim]

    @property
    @typecheck
    def values_shape(self) -> tuple[int, ...]:
        return self._full_shape[self.dim:]

    @property
    @typecheck
    def full_shape(self) -> tuple[int, ...]:
        return self._full_shape

    @property
    @typecheck
    def dim(self) -> int:
        return self._dim

    @property
    @typecheck
    def values_dim(self) -> int:
        return len(self.full_shape) - self.dim

    @property
    @typecheck
    def values(self) -> NumericalTensor:
        return self._flat_values.reshape(self.full_shape)

    @property
    @typecheck
    def dtype(self) -> torch.dtype:
        """Device getter."""
        return self._flat_values.dtype

    #########################
    #### Copy functions #####
    #########################
    @typecheck
    def copy(self) -> Image:
        """Copy the image.

        Returns
        -------
        Image
            The copy of the image.

        """
        return Image(values=self.values().clone())

    @typecheck
    def to(self, device: str | torch.device) -> Image:
        """Copy the shape onto a given device."""
        torch_device = torch.Tensor().to(device).device
        if self.device == torch_device:
            return self
        else:
            copy = self.copy()
            copy.device = device
            return copy

    ##############################
    #### Device getter/setter ####
    ##############################
    @property
    @typecheck
    def device(self) -> torch.device:
        """Device getter."""
        return self._flat_values.device

    @device.setter
    @typecheck
    def device(self, device: str | torch.device) -> None:
        """Device setter.

        Parameters
        ----------
        device
            The device on which the shape should be stored.
        """

        self._flat_values = self._flat_values.to(device=device)

    #################################
    #### Mathematical operations ####
    #################################
    def _apply_op(self, op: str, other: Image | torch.Tensor | JaxInt | None = None) -> Image:
        """
        Apply a generic tensor operation to the image.

        Parameters
        ----------
        op
            The name of the operation to apply. It must be a method of torch.Tensor.

        other
            If ``op`` is a binary operation, the other object to use as parameter.

        Returns
        -------
        Image
            The result of the operation applied.

        """

        if other is None:
            new_values = getattr(self.values, op)()
        else:
            if isinstance(other, Image):
                new_values = getattr(self.values, op)(other.values)
            else:
                new_values = getattr(self.values, op)(other)

        return Image(values=new_values, dim=self.dim)

    def __add__(self, other) -> Image:
        return self._apply_op('__add__', other)

    def __gt__(self, other) -> Image:
        return self._apply_op('__gt__', other)

    def __iadd__(self, other) -> Image:
        return self._apply_op('__iadd__', other)

    def __imul__(self, other) -> Image:
        return self._apply_op('__imul__', other)

    def __isub__(self, other) -> Image:
        return self._apply_op('__isub__', other)

    def __lt__(self, other) -> Image:
        return self._apply_op('__lt__', other)

    def __mul__(self, other) -> Image:
        return self._apply_op('__mul__', other)

    def __neg__(self) -> Image:
        return self._apply_op('__neg__')

    def __ne__(self, other) -> Image:
        return self._apply_op('__ne__', other)

    def __sub__(self, other) -> Image:
        return self._apply_op('__sub__', other)



    def unique(self) -> torch.Tensor:
        """Return the unique values pf the image.

        Returns
        -------
        torch.Tensor
            The Q unique values of the image, stored in a ``(Q,V_1,...,V_K)`` Tensor.
        """
        return self._flat_values.unique(dim=0)

    @typecheck
    def __repr__(self) -> str:
        """String representation of the image.

        Returns
        -------
        str
            A succinct description of the image data.
        """

        def suffix(dtype):
            return str(dtype).split(".")[-1].split("'")[0]

        repr_str = f"skshapes.{self.__class__.__name__} (0x{id(self):x} on {self.device}, {suffix(self.dtype)}), a {self.dim}D image."
        repr_str += f"- Image shape: {self.shape}, Values shape: {self.values_shape}"

        return repr_str
