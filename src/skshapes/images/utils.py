import math

import torch


def otsu_threshold(values: torch.tensor) -> float:
    """
    Computes the Otsu threshold of the values distribution (https://en.wikipedia.org/wiki/Otsu%27s_method).

    This is a copy of the code of sckikit-image, adapted for pytorch to improve performance:
    https://scikit-image.org/docs/stable/api/skimage.filters.html#skimage.filters.threshold_otsu

    Parameters
    ----------
    values
        The list of values to threshold.

    """
    counts, bin_edges = torch.histogram(values.to("cpu"), bins=256)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    weight1 = torch.cumsum(counts, dim=0)
    weight2 = torch.cumsum(counts.flip(dims=[0]), dim=0).flip(dims=[0])

    mean1 = torch.cumsum(counts * bin_centers, dim=0) / weight1
    mean2 = (
        torch.cumsum((counts * bin_centers).flip(dims=[0]), dim=0)
        / weight2.flip(dims=[0])
    ).flip(dims=[0])

    variance12 = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2

    idx = torch.argmax(variance12)
    return bin_centers[idx]


def eigenvalues(matrix: torch.Tensor) -> torch.Tensor:
    matrix = matrix.view(-1, 9)
    matrix_diag = matrix[:, [0, 4, 8]]
    matrix_upper = matrix[:, [1, 2, 5]]

    p1, q = (matrix_upper**2).sum(dim=1), matrix_diag.sum(
        dim=1, keepdims=True
    ) / 3
    p = torch.sqrt((((matrix_diag - q) ** 2).sum(dim=1) + 2 * p1) / 6)
    B_diag, B_upper = (matrix_diag - q), matrix_upper

    r = (
        B_diag.prod(dim=1)
        - (
            B_diag[:, 0] * B_upper[:, 2] ** 2
            + B_diag[:, 1] * B_upper[:, 1] ** 2
            + B_diag[:, 2] * B_upper[:, 0] ** 2
        )
        + 2 * B_upper.prod(dim=1)
    ) / (2 * p**3)
    phi = torch.where(
        r <= -1,
        torch.tensor(math.pi) / 3,
        torch.where(r >= 1, 0, torch.acos(r) / 3),
    )
    q = q.squeeze()

    eigs = torch.zeros((matrix.shape[0], 3), device=matrix.device)
    eigs[:, 0], eigs[:, 2] = q + 2 * p * torch.cos(phi), q + 2 * p * torch.cos(
        phi + 2 * torch.tensor(math.pi) / 3
    )
    eigs[:, 1] = 3 * q - eigs[:, 0] - eigs[:, 2]

    where_diag = torch.argwhere(p1 == 0)[:, 0]
    eigs[where_diag] = matrix_diag[where_diag]

    indices, sort_order = torch.arange(eigs.shape[0]).reshape(-1, 1).expand(
        eigs.shape[0], 3
    ), eigs.abs().argsort(dim=1)
    return eigs[indices, sort_order]
