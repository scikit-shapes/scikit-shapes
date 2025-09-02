from __future__ import annotations

import torch

from ..errors import ShapeError
from ..images.indices import (
    _difference_indices,
    _expand_indices,
    _flatten_indices,
    _intersection_indices,
    _symmetric_difference_indices,
    _union_indices,
)
from ..input_validation import convert_inputs, typecheck


class Mask:
    """A D-dimensional mask over a voxel grid.

    A :class:`Mask` object can be created from `torch.Tensors <https://pytorch.org/docs/stable/tensors.html#torch.Tensor>`_
    indicating indices of the voxels contained in the mask.
    The shape of the mask must also be provided.

    """

    @convert_inputs
    @typecheck
    def __init__(
        self,
        indices: torch.Tensor,
        *,
        shape: tuple[int, ...],
        device: str | torch.device | None = None,
    ) -> None:
        self._shape = shape
        self._flat_indices = _flatten_indices(
            indices.to(device=device), shape=shape
        )

    #########################
    #### Mask properties ####
    #########################
    @property
    @typecheck
    def shape(self) -> tuple[int, ...]:
        return self._shape

    @property
    @typecheck
    def dim(self) -> int:
        return len(self.shape)

    @property
    @typecheck
    def n_points(self) -> int:
        return self._flat_indices.shape[0]

    @property
    @typecheck
    def numel(self) -> int:
        numel = 1
        for d in range(self.dim):
            numel *= self.shape[d]

        return numel

    @property
    @typecheck
    def indices(self) -> torch.Tensor:
        return _expand_indices(self._flat_indices, shape=self.shape)

    #########################
    #### Copy functions #####
    #########################
    @typecheck
    def copy(self) -> Mask:
        """Copy the mask.

        Returns
        -------
        Mask
            The copy of the mask.

        """
        return Mask(values=self._flat_indices.clone())

    @typecheck
    def to(self, device: str | torch.device) -> Mask:
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
        return self._flat_indices.device

    @device.setter
    @typecheck
    def device(self, device: str | torch.device) -> None:
        """Device setter.

        Parameters
        ----------
        device
            The device on which the shape should be stored.
        """

        self._flat_indices = self._flat_indices.to(device=device)

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
            difference_indices = _difference_indices(
                self._flat_indices, other._flat_indices
            )
            return Mask(indices=difference_indices, shape=self.shape)
        else:
            msg = "Mask shapes must be the same when doing intersection."
            raise ShapeError(msg)

    @typecheck
    def intersection(self, other: Mask) -> Mask:
        if self.shape == other.shape:
            intersection_indices = _intersection_indices(
                self._flat_indices, other._flat_indices
            )
            return Mask(indices=intersection_indices, shape=self.shape)
        else:
            msg = "Mask shapes must be the same when doing intersection."
            raise ShapeError(msg)

    @typecheck
    def symmetric_difference(self, other: Mask) -> Mask:
        if self.shape == other.shape:
            symmetric_difference_indices = _symmetric_difference_indices(
                self._flat_indices, other._flat_indices
            )
            return Mask(indices=symmetric_difference_indices, shape=self.shape)
        else:
            msg = (
                "Mask shapes must be the same when doing symmetric difference."
            )
            raise ShapeError(msg)

    @typecheck
    def union(self, other: Mask) -> Mask:
        if self.shape == other.shape:
            union_indices = _union_indices(
                self._flat_indices, other._flat_indices
            )
            return Mask(indices=union_indices, shape=self.shape)
        else:
            msg = "Mask shapes must be the same when doing union."
            raise ShapeError(msg)
