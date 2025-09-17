from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from ...input_validation import convert_inputs, one_and_only_one, typecheck
from ...types import IntTensor, Number, NumericalTensor
from ._indices_management import (
    flatten_indices,
    get_at_sparse_indices,
    indices_difference,
    indices_intersection,
    isin_sorted,
    masked_convolution,
    scatter_convolution,
    set_at_indices,
    sort_by_indices,
)
from .image_structure import ImageStructure, UniqueOutput

if TYPE_CHECKING:  # Handle sibling classes for type checking only
    from . import Image, Mask


class SparseImage(ImageStructure):
    """A D-dimensional grid, with a K-dimensional value associated to each of its voxels.


       An :class:`SparseImage` object only stores non-zero elements of the image, ensuring memory and time efficiency
       when working with images with many zeros.
       It can be created from a list of indices and a list of values, both provided as
       `torch.Tensors <https://pytorch.org/docs/stable/tensors.html#torch.Tensor>`_.

    Parameters
    ----------
    indices
        A tensor of shape ``(N, D)`` containing the voxel positions where the values are specified.

    values
        A tensor of shape ``(N,)`` or ``(N,V_1,...,V_K)`` containing the values at the specified positions.

    shape
        The shape of the full image.

    constant
        The value of the image at the positions not specified in ``indices``. It must be a number or a tensor of shape
        ``(V_1,...,V_K)``, depending on the values shape. By default, it is set to a tensor full of zeros.

    dtype
        The data type of the image values.
        If None it is inferred from ``values``.

    device
        The device on which the shape is stored (e.g. ``"cpu"`` or ``"cuda"``).
        If None it is inferred from the ``values``.

    Examples
    --------

    .. testcode::

        import skshapes as sks

        image = sks.SparseImage(
            indices=torch.tensor([[0, 1], [2, 1], [2, 2]]),
            values=torch.tensor([0.1, 0.25, 0.47]),
            shape=(3, 4),
        )
        print(image.values)

    .. testoutput::

        tensor([[0.0000, 0.1000, 0.0000, 0.0000],
                [0.0000, 0.0000, 0.0000, 0.0000],
                [0.0000, 0.2500, 0.4700, 0.0000]])

    .. testcode::

        import skshapes as sks

        image = sks.SparseImage(
            indices=torch.tensor([[0, 1], [2, 1], [2, 2]]),
            values=torch.tensor([0.1, 0.25, 0.47]),
            shape=(3, 4),
            constant=3.0,
        )
        print(image.values)

    .. testoutput::

        tensor([[3.0000, 0.1000, 3.0000, 3.0000],
                [3.0000, 3.0000, 3.0000, 3.0000],
                [3.0000, 0.2500, 0.4700, 3.0000]])

    """

    @one_and_only_one(["indices", "flat_indices"])
    @convert_inputs
    @typecheck
    def __init__(
        self,
        *,
        indices: IntTensor | None = None,
        flat_indices: IntTensor | None = None,
        values: NumericalTensor,
        shape: tuple[int, ...],
        constant: NumericalTensor | Number = 0,
        dtype: torch.dtype | None = None,
        device: str | torch.device | None = None,
    ) -> None:

        if dtype is None:
            dtype = values.dtype
        if device is None:
            device = values.device

        if flat_indices is None:
            flat_indices = flatten_indices(indices=indices, shape=shape)

        self._flat_values, self._flat_indices = sort_by_indices(
            flat_values=values.to(dtype=dtype, device=device),
            flat_indices=flat_indices.to(device=device),
            return_indices=True,
        )

        if isinstance(constant, Number):
            constant = torch.full(size=values.shape[1:], fill_value=constant)

        self._constant = constant.to(dtype=dtype, device=device)

        values_shape = values.shape[1:]
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
    ) -> SparseImage:
        if dtype is None:
            dtype = torch.float32

        flat_indices = torch.tensor([], device=device, dtype=torch.int64)
        values = torch.tensor([], device=device, dtype=dtype)
        if values_shape is not None:
            values = values.reshape((0, *values_shape))

        return cls(flat_indices=flat_indices, values=values, shape=shape)

    @classmethod
    def fill(
        cls,
        shape: tuple[int, ...],
        *,
        fill_value: NumericalTensor | Number,
        dtype: torch.dtype | None = None,
        device: str | torch.device | None = None,
    ) -> SparseImage:
        if dtype is None:
            dtype = torch.float32

        if not isinstance(fill_value, torch.Tensor):
            constant = torch.tensor(fill_value, dtype=dtype, device=device)
        else:
            constant = fill_value.to(dtype=dtype, device=device)

        flat_indices = torch.tensor([], device=device, dtype=torch.int64)
        values = torch.tensor([], device=device, dtype=dtype).reshape(
            (0, *constant.shape)
        )

        return cls(
            flat_indices=flat_indices,
            values=values,
            shape=shape,
            constant=constant,
        )

    ##########################
    #### Image properties ####
    ##########################

    @property
    @typecheck
    def values(self) -> NumericalTensor:
        """The image values expressed as a tensor."""

        dense_values = torch.zeros(
            size=self.full_shape, device=self.device, dtype=self.dtype
        )
        dense_values += self._constant.view(
            (1,) * len(self.shape) + self.values_shape
        )

        return set_at_indices(
            grid=dense_values,
            flat_indices=self._flat_indices,
            new_values=self._flat_values,
            shape=self.shape,
        )

    @one_and_only_one(["query_indices", "flat_query_indices"])
    @convert_inputs
    @typecheck
    def values_at(
        self,
        *,
        query_indices: IntTensor | None,
        flat_query_indices: IntTensor | None = None,
    ) -> NumericalTensor:
        """Get the values of the image at the indices specified in input."""

        return get_at_sparse_indices(
            flat_values=self._flat_values,
            flat_indices=self._flat_indices,
            constant=self._constant,
            query_indices=query_indices,
            flat_query_indices=flat_query_indices,
            shape=self.shape,
        )

    @property
    @typecheck
    def constant(self) -> torch.Tensor:
        return self._constant

    ############################
    #### Class conversions #####
    ############################

    @typecheck
    def to_dense(
        self,
        *,
        dtype: torch.dtype | None = None,
        device: str | torch.device | None = None,
    ) -> Image:
        from . import Image

        return Image(values=self.values(), dtype=dtype, device=device)

    #########################
    #### Copy functions #####
    #########################

    @typecheck
    def copy(self) -> SparseImage:
        return SparseImage(
            flat_indices=self._flat_indices.clone(),
            values=self._flat_values.clone(),
            shape=self.shape,
            constant=self._constant.clone(),
        )

    ###################################
    #### Value-based manipulations ####
    ###################################

    @typecheck
    def reshape_values(self, values_shape=tuple[int, ...]) -> SparseImage:
        return SparseImage(
            flat_indices=self._flat_indices,
            values=self._flat_values.reshape((-1, *values_shape)),
            shape=self.shape,
            constant=self._constant.reshape((-1, *values_shape)),
        )

    ##########################
    #### Image operations ####
    ##########################

    @typecheck
    def __getitem__(self, key: Mask) -> SparseImage:
        intersected_indices = indices_intersection(
            flat_indices1=self._flat_indices, flat_indices2=key._flat_indices
        )
        preserved_idx = isin_sorted(self._flat_indices, intersected_indices)

        new_indices = indices_difference(
            flat_indices1=key._flat_indices, flat_indices2=self._flat_indices
        )
        flat_indices = torch.cat(
            [self._flat_indices[preserved_idx], new_indices]
        )
        expanded_constant = self._constant.repeat(
            (new_indices.shape[0],) + (1,) * len(self._constant.shape)
        )
        flat_values = torch.cat(
            [self._flat_values[preserved_idx], expanded_constant]
        )

        return SparseImage(
            flat_indices=flat_indices, values=flat_values, shape=self.shape
        )

    @typecheck
    def __setitem__(self, key: Mask, value: Number | NumericalTensor) -> None:
        if value == self._constant:
            difference_indices = indices_difference(
                flat_indices1=self._flat_indices,
                flat_indices2=key._flat_indices,
            )
            preserved_idx = isin_sorted(self._flat_indices, difference_indices)

            self._flat_indices = self._flat_indices[preserved_idx]
            self._flat_values = self._flat_values[preserved_idx]
        else:
            overlapping_idx = isin_sorted(
                self._flat_indices, key._flat_indices
            )

            self._flat_indices = torch.cat(
                [self._flat_indices[~overlapping_idx], key._flat_indices]
            )
            self._flat_values = torch.cat(
                [
                    self._flat_values[~overlapping_idx],
                    torch.full(
                        size=(key.n_points,),
                        fill_value=value,
                        dtype=self.dtype,
                        device=self.device,
                    ),
                ]
            )  # TODO gérer cas multidim.

    @typecheck
    def unique(
        self, *, sorted: bool = True, return_counts: bool = False
    ) -> UniqueOutput:
        cat_vals = torch.cat([self._flat_values, self._constant.unsqueeze(0)])
        dim = None if len(self.values_shape) == 0 else 0

        if return_counts:
            output, revidx, counts = cat_vals.unique(
                dim=dim, sorted=sorted, return_counts=True, return_inverse=True
            )
            counts[revidx[0]] += self.numel - self._flat_indices.shape[0] - 1
        else:
            output, counts = cat_vals.unique(dim=dim, sorted=sorted), None

        return UniqueOutput(output=output, counts=counts)

    #######################################
    #### Neighborhood-based operations ####
    #######################################

    @typecheck
    def convolution(
        self,
        *,
        offsets: torch.Tensor,
        weights: torch.Tensor | None = None,
        clip: bool = True,
        kernel: str = "sum",
    ) -> SparseImage:
        # TODO gérer constante
        conv_indices, conv_values = scatter_convolution(
            flat_indices=self._flat_indices,
            flat_values=self._flat_values,
            offsets=offsets,
            weights=weights,
            shape=self.shape,
            clip=clip,
            kernel=kernel,
        )

        return SparseImage(
            flat_indices=conv_indices, values=conv_values, shape=self.shape
        )

    @typecheck
    def masked_convolution(
        self,
        *,
        mask: Mask,
        offsets: torch.Tensor,
        weights: torch.Tensor | None = None,
        kernel: str | callable = "sum",
    ) -> SparseImage | Mask:
        from . import Mask

        conv_values = masked_convolution(
            flat_mask_indices=mask._flat_indices,
            flat_indices=self._flat_indices,
            flat_values=self._flat_values,
            constant=self._constant,
            offsets=offsets,
            shape=self.shape,
            kernel=kernel,
            weights=weights,
        )
        if conv_values.dtype == torch.bool:
            return Mask(flat_indices=mask._flat_indices[conv_values])
        else:
            return SparseImage(
                flat_indices=mask._flat_indices,
                values=conv_values,
                shape=self.shape,
            )
