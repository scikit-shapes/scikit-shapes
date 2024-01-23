"""Utility functions for the skshapes package."""

from typing import Literal, Optional

import torch

from .errors import DeviceError
from .input_validation import convert_inputs, typecheck
from .types import (
    Int1dTensor,
    Number,
    NumericalTensor,
)


def ranges_slices(batch):
    """Helper function for the diagonal ranges function."""
    Ns = batch.bincount()
    indices = Ns.cumsum(0)
    ranges = torch.cat((0 * indices[:1], indices))
    ranges = (
        torch.stack((ranges[:-1], ranges[1:]))
        .t()
        .int()
        .contiguous()
        .to(batch.device)
    )
    slices = (1 + torch.arange(len(Ns))).int().to(batch.device)

    return ranges, slices


def diagonal_ranges(batch_x=None, batch_y=None):
    """Encode the block-diagonal structure associated to a batch vector."""
    if batch_x is None and batch_y is None:
        return None  # No batch processing

    if batch_y is None:
        batch_y = batch_x  # "symmetric" case

    ranges_x, slices_x = ranges_slices(batch_x)
    ranges_y, slices_y = ranges_slices(batch_y)

    return ranges_x, slices_x, ranges_y, ranges_y, slices_y, ranges_x


src = torch.tensor([1, -1, 0.5, 0, 2, 1], dtype=torch.float32)
index = torch.tensor([0, 1, 2, 0, 2, 3])


@convert_inputs
@typecheck
def scatter(
    src: NumericalTensor,
    index: Int1dTensor,
    reduce: Literal["sum", "min", "max", "mean"] = "mean",
    min_length: Optional[int] = None,
    blank_value: Number = 0,
):
    """Scatter operation on a tensor.

    This function is a wrapper around the pytorch scatter function. Available
    reduce operations are "sum", "min", "max", "mean". It is compatible with
    automatic differentiation.

    Parameters
    ----------
    src
        The source tensor.
    index
        The indices of the elements to scatter.
    reduce
        The reduce operation to apply. Defaults to "mean".
    min_length
        The minimum length of the output tensor. If None it is set according to
        the highest index value.
    blank_value
        The value to set for the elements of the output tensor that are not
        referenced by the index tensor.
    """
    if src.device != index.device:
        msg = "The src and index tensors must be on the same device."
        raise DeviceError(msg)
    device = src.device

    if min_length is not None:
        if index.max() >= min_length:
            raise ValueError(
                "The min_length parameter must be greater than the maximum"
                + " index value."
            )

        length = min_length

    else:
        length = index.max() + 1

    # Pytorch syntax : "amin" instead of "min", "amax" instead of "max"
    if reduce == "min":
        reduce = "amin"
    elif reduce == "max":
        reduce = "amax"

    # src is a vector, we can use the scatter_reduce function

    if len(src.shape) > 1:
        index = index.clone()
        for _ in range(len(src.shape) - 1):
            index = index.unsqueeze(-1)

        index = index.expand_as(src)
        output = torch.full(
            (length, *(src[0].shape)), blank_value, dtype=src.dtype
        ).to(device)

    else:
        output = torch.full((length,), blank_value, dtype=src.dtype).to(device)

    # Scatter syntax for pytorch > 1.11
    return output.scatter_reduce(
        index=index, src=src, dim=0, reduce=reduce, include_self=False
    )
