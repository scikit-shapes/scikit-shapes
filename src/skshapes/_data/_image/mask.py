from __future__ import annotations

from typing import TYPE_CHECKING, Literal, NamedTuple

import numpy as np
import scipy
import torch
from parso.python.tree import Number

from ...errors import ShapeError
from ...input_validation import convert_inputs, one_and_only_one, typecheck
from ...types import IntTensor
from ._indices_management import (
    check_sorted,
    expand_indices,
    flatten_indices,
    indices_difference,
    indices_intersection,
    indices_symmetric_difference,
    indices_union,
    masked_convolution,
    neighborhood_adjacency,
    scatter_boolean_convolution,
    set_at_indices,
)
from .grid_structure import GridStructure

if TYPE_CHECKING:  # Handle sibling classes for type checking only
    from . import Image, SparseImage


class ConnectedComponentsOutput(NamedTuple):
    output: SparseImage
    labels: torch.Tensor


class Mask(GridStructure):
    """A D-dimensional mask over a voxel grid.

    A :class:`Mask` object can be created from `torch.Tensors <https://pytorch.org/docs/stable/tensors.html#torch.Tensor>`_
    indicating indices of the voxels contained in the mask.
    The shape of the mask must also be provided.

    Masks are manipulated like sets of voxels. They are also used to select only a subset of the values of an image.

    Parameters
    ----------
    indices
        The indices of the voxels that are included in the mask.
        It should be a tensor of shape ``(N,D)`` where ``N`` is the number of voxels present in the mask.

    shape
        The shape of the original grid on which the mask is defined.

    device
        The device on which the shape is stored (e.g. ``"cpu"`` or ``"cuda"``).
        If None it is inferred from the input.
    """

    @one_and_only_one(["indices", "flat_indices"])
    @convert_inputs
    @typecheck
    def __init__(
        self,
        *,
        indices: IntTensor | None = None,
        flat_indices: IntTensor | None = None,
        shape: tuple[int, ...],
        device: str | torch.device | None = None,
    ) -> None:

        if flat_indices is None:
            flat_indices = flatten_indices(indices=indices, shape=shape)

        self._flat_indices = flat_indices.to(device=device)
        if not check_sorted(self._flat_indices):
            self._flat_indices = self._flat_indices.sort()[0]

        super().__init__(shape=shape, device=self._flat_indices.device)

    @typecheck
    def __repr__(self) -> str:
        """String representation of the image.

        Returns
        -------
        str
            A succinct description of the image data.
        """

        repr_str = f"skshapes.{self.__class__.__name__} (0x{id(self):x} on {self.device}), a {self.dim}D mask."
        repr_str += f" Mask shape: {self.shape}"
        return repr_str

    #########################
    #### Mask properties ####
    #########################

    @property
    @typecheck
    def n_points(self) -> int:
        """The number of voxels present in the mask."""
        return self._flat_indices.shape[0]

    @property
    @typecheck
    def indices(self) -> torch.Tensor:
        """The indices of the voxels that are included in the mask, expressed as a tensor."""
        return expand_indices(self._flat_indices, shape=self.shape)

    @property
    @typecheck
    def values(self) -> torch.Tensor:
        """The mask expressed as a tensor."""

        mask_tensor = torch.zeros(
            self.shape, dtype=torch.bool, device=self.device
        )
        return set_at_indices(
            grid=mask_tensor,
            flat_indices=self._flat_indices,
            new_values=True,
            shape=self.shape,
        )

    ############################
    #### Class conversions #####
    ############################

    @typecheck
    def to_image(
        self,
        *,
        dtype: torch.dtype | None = None,
        device: str | torch.device | None = None,
    ) -> Image:
        from . import Image

        if dtype is None:
            dtype = torch.int64
        if device is None:
            device = self.device

        return Image(values=self.values.to(dtype=dtype, device=device))

    @typecheck
    def to_sparseimage(
        self,
        *,
        dtype: torch.dtype | None = None,
        device: str | torch.device | None = None,
    ) -> SparseImage:
        from . import SparseImage

        if dtype is None:
            dtype = torch.int64
        if device is None:
            device = self.device

        values = torch.ones(
            size=(self._flat_indices.shape[0],), dtype=dtype, device=device
        )
        return SparseImage(
            flat_indices=self._flat_indices, values=values, device=device
        )

    #########################
    #### Copy functions #####
    #########################

    @typecheck
    def copy(self) -> Mask:
        return Mask(flat_indices=self._flat_indices.clone(), shape=self.shape)

    ########################
    #### Set operations ####
    ########################

    @typecheck
    def __and__(self, other: Mask) -> Mask:
        return self.intersection(other)

    @typecheck
    def __or__(self, other: Mask) -> Mask:
        return self.union(other)

    @typecheck
    def __xor__(self, other: Mask) -> Mask:
        return self.symmetric_difference(other)

    @typecheck
    def difference(self, other: Mask) -> Mask:
        if self.shape == other.shape:
            difference_indices = indices_difference(
                self._flat_indices, other._flat_indices
            )
            return Mask(flat_indices=difference_indices, shape=self.shape)
        else:
            msg = "Mask shapes must be the same when doing intersection."
            raise ShapeError(msg)

    @typecheck
    def intersection(self, other: Mask) -> Mask:
        if self.shape == other.shape:
            intersection_indices = indices_intersection(
                self._flat_indices, other._flat_indices
            )
            return Mask(flat_indices=intersection_indices, shape=self.shape)
        else:
            msg = "Mask shapes must be the same when doing intersection."
            raise ShapeError(msg)

    @typecheck
    def symmetric_difference(self, other: Mask) -> Mask:
        if self.shape == other.shape:
            symmetric_difference_indices = indices_symmetric_difference(
                self._flat_indices, other._flat_indices
            )
            return Mask(
                flat_indices=symmetric_difference_indices, shape=self.shape
            )
        else:
            msg = (
                "Mask shapes must be the same when doing symmetric difference."
            )
            raise ShapeError(msg)

    @typecheck
    def union(self, other: Mask) -> Mask:
        if self.shape == other.shape:
            union_indices = indices_union(
                self._flat_indices, other._flat_indices
            )
            return Mask(flat_indices=union_indices, shape=self.shape)
        else:
            msg = "Mask shapes must be the same when doing union."
            raise ShapeError(msg)

    #######################################
    #### Neighborhood-based operations ####
    #######################################

    @typecheck
    def _adjacency_csr_matrix(
        self, *, offsets: torch.Tensor
    ) -> scipy.sparse.csr_matrix:
        row, col = neighborhood_adjacency(
            flat_indices=self._flat_indices, offsets=offsets, shape=self.shape
        )
        data, row, col = (
            np.ones(shape=row.shape, dtype=bool),
            row.cpu().numpy(),
            col.cpu().numpy(),
        )

        return scipy.sparse.csr_matrix(
            (data, (row, col)), shape=(self.n_points, self.n_points)
        )

    @typecheck
    def connected_components(self, *, offsets: torch.Tensor) -> SparseImage:
        from . import SparseImage

        components = scipy.sparse.csgraph.connected_components(
            self._adjacency_csr_matrix(offsets=offsets)
        )[1]
        components = (
            torch.tensor(components, dtype=torch.int64, device=self.device) + 1
        )

        return SparseImage(
            flat_indices=self._flat_indices,
            values=components,
            shape=self.shape,
        )

    @typecheck
    def convolution(
        self, *, offsets: torch.Tensor, kernel: str = "any"
    ) -> Mask:
        conv_indices = scatter_boolean_convolution(
            flat_indices=self._flat_indices,
            offsets=offsets,
            shape=self.shape,
            kernel=kernel,
        )

        return Mask(flat_indices=conv_indices, shape=self.shape)

    @typecheck
    def masked_convolution(
        self,
        *,
        mask: Mask | None = None,
        offsets: torch.Tensor,
        weights: torch.Tensor | None = None,
        kernel: str | callable = "sum",
        outside_value: torch.Tensor | Number = 0,
    ) -> SparseImage | Mask:
        from . import SparseImage

        if mask is None:
            mask = self
        flat_values = torch.ones(
            size=(self._flat_indices.shape[0],),
            dtype=torch.bool,
            device=self.device,
        )

        conv_values = masked_convolution(
            flat_mask_indices=mask._flat_indices,
            flat_indices=self._flat_indices,
            flat_values=flat_values,
            offsets=offsets,
            shape=self.shape,
            kernel=kernel,
            weights=weights,
        )
        if conv_values.dtype == torch.bool:
            return Mask(
                flat_indices=mask._flat_indices[conv_values], shape=self.shape
            )
        else:
            return SparseImage(
                flat_indices=mask._flat_indices,
                values=conv_values,
                shape=self.shape,
                constant=outside_value,
            )

    ###########################
    #### Plotting utilities ###
    ###########################

    @typecheck
    def plot(
        self,
        backend: Literal["pyvista", "vedo"] = "pyvista",
        **kwargs,
    ) -> None:
        from . import Image

        Image(values=self.values.to(dtype=torch.int64)).plot(
            backend=backend, **kwargs
        )
