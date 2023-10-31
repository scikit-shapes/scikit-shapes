import torch
from .types import NumericalTensor, Int1dTensor, convert_inputs, typecheck
from typing import Literal


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
    """Encodes the block-diagonal structure associated to a batch vector."""

    if batch_x is None and batch_y is None:
        return None  # No batch processing
    elif batch_y is None:
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
):
    """Scatter function

    This function is a wrapper around the pytorch scatter function. Available
    reduce operations are "sum", "min", "max", "mean".
    """
    if src.device != index.device:
        raise RuntimeError(
            "The src and index tensors must be on the same device."
        )
    device = src.device

    if len(torch.unique(index)) != int(index.max() + 1):
        raise RuntimeError(
            "The index vector should contain consecutive integers between 0"
            + " and max(index)."
        )

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
        output = torch.zeros(
            torch.max(index + 1), *(src[0].shape), dtype=src.dtype
        ).to(device)

    else:
        output = torch.zeros(torch.max(index + 1), dtype=src.dtype).to(device)

    # Scatter syntax for pytorch > 1.11
    output = output.scatter_reduce(
        index=index, src=src, dim=0, reduce=reduce, include_self=False
    )
    return output
