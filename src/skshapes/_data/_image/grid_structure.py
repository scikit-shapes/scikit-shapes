from __future__ import annotations

import torch

from ...input_validation import convert_inputs, typecheck


class GridStructure:
    @convert_inputs
    @typecheck
    def __init__(self, shape: tuple[int, ...], device: torch.device) -> None:
        self._shape = shape
        self._device = device

        self._dim = len(shape)
        numel = 1
        for d in range(self.dim):
            numel *= self.shape[d]
        self._numel = numel

    #########################
    #### Grid properties ####
    #########################

    @property
    @typecheck
    def shape(self) -> tuple[int, ...]:
        """The shape of the D-dimensional grid."""
        return self._shape

    @property
    @typecheck
    def dim(self) -> int:
        """The dimension D of the D-dimensional grid."""
        return self._dim

    @property
    @typecheck
    def numel(self) -> int:
        """The total number of voxels in the image."""
        return self._numel

    #########################
    #### Copy functions #####
    #########################

    @typecheck
    def copy(self) -> GridStructure:
        return GridStructure(shape=self._shape, device=self._device)

    @typecheck
    def to(self, device: str | torch.device) -> GridStructure:
        """Copy the instance onto a given device."""
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
        return self._device

    @device.setter
    @typecheck
    def device(self, device: str | torch.device) -> None:
        """Device setter.

        Parameters
        ----------
        device
            The device on which the shape should be stored.
        """
        for attr in dir(self):
            if attr.startswith("_"):
                attribute = getattr(self, attr)
                if isinstance(attribute, torch.Tensor):
                    setattr(self, attr, attribute.to(device))

        self._device = torch.Tensor().to(device).device
