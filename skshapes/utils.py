import torch
from .types import NumericalTensor, Int1dTensor, convert_inputs, typecheck
from typing import Literal


def ranges_slices(batch):
    """Helper function for the diagonal ranges function."""
    Ns = batch.bincount()
    indices = Ns.cumsum(0)
    ranges = torch.cat((0 * indices[:1], indices))
    ranges = (
        torch.stack((ranges[:-1], ranges[1:])).t().int().contiguous().to(batch.device)
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
    if len(torch.unique(index)) != int(index.max() + 1):
        raise RuntimeError(
            "The index vector should contain consecutive integers between 0 and max(index)."
        )

    # Pytorch syntax : "amin" instead of "min", "amax" instead of "max"
    if reduce == "min":
        reduce = "amin"
    elif reduce == "max":
        reduce = "amax"

    if len(src.shape) == 1:
        # src is a vector, we can use the scatter_reduce function

        try:
            # Scatter syntax for pytorch >= 1.11
            output = torch.zeros(torch.max(index + 1), dtype=src.dtype)
            output = output.scatter_reduce(
                index=index, src=src, dim=0, reduce=reduce, include_self=False
            )
            return output

        except:
            try:
                # Scatter syntax for pytorch == 1.11
                output = torch.scatter_reduce(
                    input=src, index=index, dim=0, reduce=reduce
                )
                return output

            except:
                # Normally this should not happen, as skshapes requires pytorch >= 1.11
                raise RuntimeError(
                    f"Cannot define scatter operations, you are using pytorch {torch.__version__}, this should work for pytorch >= 1.11"
                )

    else:
        # we need to apply scatter on each column

        # Initialize the output with the correct shape
        len_output = torch.unique(index).shape[0]
        output_shape = (len_output,) + src.shape[1:]
        output = torch.zeros(output_shape).to(dtype=src.dtype)

        if len(src.shape) == 2:
            # src is a matrix
            for i in range(src.shape[1]):
                output[:, i] = scatter(src[:, i].view(-1), index, reduce)

        elif len(src.shape) == 3:
            # src is a 3d tensor
            for i in range(src.shape[1]):
                for j in range(src.shape[2]):
                    output[:, i, j] = scatter(src[:, i, j].view(-1), index, reduce)

        elif len(src.shape) >= 4:
            raise NotImplementedError(
                "Scatter reduction is not yet implemented for signals with dimension >= 2"
            )
            # Populate the output, dimension by dimension
            vector_shape = (-1,) + (1,) * (len(src.shape) - 1)
            from itertools import product

            for i in product(*[range(d) for d in src.shape[1:]]):
                print(i)
                # Only available in python 3.11+ : need to find an alternative
                # output[:, *i] = scatter(src[:, *i].view(-1), index, reduce).view(vector_shape)

        return output
