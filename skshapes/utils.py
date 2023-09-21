import torch


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


def scatter(src, index, reduce):
    """Scatter function

    This function is a wrapper around the pytorch scatter function. Available
    reduce operations are "sum", "min", "max", "mean".
    """

    # Pytorch syntax : "amin" instead of "min", "amax" instead of "max"
    if reduce == "min":
        reduce = "amin"
    elif reduce == "max":
        reduce = "amax"

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
            output = torch.scatter_reduce(input=src, index=index, dim=0, reduce=reduce)
            return output

        except:
            # Normally this should not happen, as skshapes requires pytorch >= 1.11
            raise RuntimeError(
                f"Cannot define scatter operations, you are using pytorch {torch.__version__}, this should work for pytorch >= 1.11"
            )
