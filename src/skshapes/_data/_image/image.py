"""Image shape class."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from ...input_validation import convert_inputs, typecheck
from ...types import Number, NumericalTensor
from ._indices_management import get_at_indices, set_at_indices
from .image_structure import ImageStructure

if TYPE_CHECKING:  # Handle sibling classes for type checking only
    from . import Mask, SparseImage


class Image(ImageStructure):
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
        import torch

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
        values: NumericalTensor,
        *,
        dim: int | None = None,
        dtype: torch.dtype | None = None,
        device: str | torch.device | None = None,
    ):
        # Internally, the image values are stored in a flatten tensor. This make the dimension management easier, while
        # optimizing indices-related operations.
        if dim is None:
            dim = len(values.shape)
        if dtype is None:
            dtype = values.dtype
        if device is None:
            device = values.device

        self._flat_values = values.flatten(start_dim=0, end_dim=dim - 1).to(
            dtype=dtype, device=device
        )

        shape, values_shape = values.shape[:dim], values.shape[dim:]
        super().__init__(
            shape=shape,
            values_shape=values_shape,
            dtype=self._flat_values.dtype,
            device=self._flat_values.device,
        )

    @classmethod
    def zeros(
        cls,
        shape: tuple[int, ...],
        *,
        values_shape: tuple[int, ...] | None = None,
        dtype: torch.dtype | None = None,
        device: str | torch.device | None = None,
    ) -> Image:
        if dtype is None:
            dtype = torch.float32

        if values_shape is None:
            return cls(values=torch.zeros(shape, dtype=dtype, device=device))
        else:
            return cls(
                values=torch.zeros(
                    shape + values_shape, dtype=dtype, device=device
                ),
                dim=len(shape),
            )

    ##########################
    #### Image properties ####
    ##########################

    @property
    @typecheck
    def values(self) -> NumericalTensor:
        """The image values expressed as a tensor."""
        return self._flat_values.reshape(self.full_shape)

    #########################
    #### Copy functions #####
    #########################

    @typecheck
    def copy(self) -> Image:
        return Image(values=self.values.clone())

    ##########################
    #### Image operations ####
    ##########################

    @typecheck
    def __getitem__(self, key: Mask) -> SparseImage:
        from . import SparseImage

        values = get_at_indices(
            flat_grid=self._flat_values,
            flat_indices=key._flat_indices,
            shape=self.shape,
        )
        return SparseImage(
            flat_indices=key._flat_indices, values=values, shape=self.shape
        )

    @typecheck
    def __setitem__(self, key: Mask, value: Number | NumericalTensor) -> None:
        if isinstance(value, Number):
            self._flat_values = set_at_indices(
                flat_grid=self._flat_values,
                flat_indices=key._flat_indices,
                new_values=value,
                shape=self.shape,
            )
        elif isinstance(value, ImageStructure):
            pass  # TODO implémenter

    @typecheck
    def unique(self) -> torch.Tensor:
        """Return the unique values pf the image.

        Returns
        -------
        torch.Tensor
            The Q unique values of the image, stored in a ``(Q,V_1,...,V_K)`` Tensor.
        """
        return (
            self._flat_values.unique()
            if len(self.values_shape) == 0
            else self._flat_values.unique(dim=0)
        )
