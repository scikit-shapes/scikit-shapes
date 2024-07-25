"""Implicit quadrics."""

import numpy as np
import torch
from pykeops.torch import LazyTensor

from ..input_validation import typecheck
from ..types import Float1dTensor, Number, Points
from ..utils import diagonal_ranges

# from .normals import smooth_normals, tangent_vectors


def normalize_point_cloud(
    *,
    points: Points,
    weights: Float1dTensor,
):
    """Point cloud normalization for numerical stability."""
    assert points.shape[0] == weights.shape[0]

    # Center:
    mean_point = weights @ points
    assert mean_point.shape == (3,)
    points = points - mean_point  # (q, 3)

    # Rescale:
    sigma = (weights @ (points**2).sum(dim=1)).sqrt()
    assert sigma.shape == ()
    points = points / sigma  # (q, 3)

    return points, sigma, mean_point


@typecheck
def implicit_quadrics(
    *,
    points: Points,
    weights: Float1dTensor | None = None,
    scale: Number = 1.0,
    batch=None,
    reg: Number = 0.0001,
):
    """Fits an implicit quadric to each point of a point cloud.

    points, weights  ->      F
    (N, 3),   (N,)   ->  (N, 4, 4)

    The main reference for this function is the following paper:

    Gabriel Taubin, "Estimation of Planar Curves, Surfaces and Nonplanar Space
    Curves Defined by Implicit Equations, with Applications to Edge and Range
    Image Segmentation",
    IEEE Trans. PAMI, Vol. 13, 1991, pp1115-1138.
    http://mesh.brown.edu/taubin/pdfs/Taubin-pami91.pdf

    See also this StackOverflow answer for a full discussion:
    https://scicomp.stackexchange.com/questions/26105/fitting-implicit-surfaces-to-oriented-point-sets # noqa E501
    """
    # TODO: actually support batch processing...
    ranges = diagonal_ranges(batch)

    # Throughout, we stick to the notations of the 1991 paper by Taubin.
    # Number of points:
    q = points.shape[0]

    # Weights:
    if weights is None:
        weights = torch.ones_like(points[:, 0])  # (q,)

    # Normalize the weights to sum up to 1:
    weights = weights / weights.sum()

    # As explained in Section IX, we normalize the point cloud to ensure
    # numerical stability:
    points, sigma, mean_point = normalize_point_cloud(
        points=points, weights=weights
    )
    # points is now a (weighted) point cloud centered at the origin, with a
    # unit variance. sigma and mean_point are the scaling and translation
    # factors.

    scale = scale / sigma

    # Compute the features of order 0, 1 and 2: -------------------------------
    # X = [x^2, y^2, z^2, xy, xz, yz, x, y, z, 1]
    # (this is easier to implement in PyTorch than the order at the start of
    # Sec. VIII)
    x = points[:, 0:1]  # (q, 1)
    y = points[:, 1:2]  # (q, 1)
    z = points[:, 2:3]  # (q, 1)
    i = torch.ones_like(x)  # (q, 1)
    o = torch.zeros_like(x)  # (q, 1)

    X = torch.cat(
        [
            points**2,  # x^2, y^2, z^2
            x * y,  # xy
            x * z,  # xz
            y * z,  # yz
            points,  # x, y, z
            i,  # 1
        ],
        dim=1,
    )
    assert X.shape == (q, 10)

    # Compute the covariance matrices as a (q, 10, 10) tensor:
    XXt = X.view(q, 10, 1) * X.view(q, 1, 10)  # (q, 10, 10)
    assert XXt.shape == (q, 10, 10)

    # Compute the constraint matrices -----------------------------------------
    # Gradients wrt. x, y, z:
    DX_x = torch.cat([2 * x, o, o, y, z, o, i, o, o, o], dim=1)
    DX_y = torch.cat([o, 2 * y, o, x, o, z, o, i, o, o], dim=1)
    DX_z = torch.cat([o, o, 2 * z, o, x, y, o, o, i, o], dim=1)
    assert DX_x.shape == (q, 10)
    assert DX_y.shape == (q, 10)
    assert DX_z.shape == (q, 10)

    # Constraint matrices:
    DXDXt = (
        DX_x.view(q, 10, 1) * DX_x.view(q, 1, 10)
        + DX_y.view(q, 10, 1) * DX_y.view(q, 1, 10)
        + DX_z.view(q, 10, 1) * DX_z.view(q, 1, 10)
    )
    assert DXDXt.shape == (q, 10, 10)

    # Create the symbolic window matrix ---------------------------------------
    # Encode as symbolic tensors:
    # Points:
    x_i = LazyTensor(points.view(q, 1, 3) / (np.sqrt(2) * scale))
    x_j = LazyTensor(points.view(1, q, 3) / (np.sqrt(2) * scale))

    # Squared distance:
    d2_ij = ((x_j - x_i) ** 2).sum(-1)  # (q, q, 1)
    # Gaussian window exp(-||x_i - x_j||^2 / (2 * scale^2)):
    window_ij = (-d2_ij).exp()  # (q, q, 1)

    # Reduction - with batch support:
    window_ij.ranges = ranges
    assert window_ij.shape == (q, q)

    # Compute the weighted covariance matrices --------------------------------
    MD_i = (window_ij @ XXt.view(q, 10 * 10)).view(q, 10, 10)
    assert MD_i.shape == (q, 10, 10)

    ND_i = (window_ij @ DXDXt.view(q, 10 * 10)).view(q, 10, 10)
    assert ND_i.shape == (q, 10, 10)

    # Solve the generalized eigenvalue problem (Sec. VII) ---------------------
    # Add a small ridge regression:
    for k in range(10):
        ND_i[:, k, k] += reg

    eigenvalues, F = torch.lobpcg(MD_i, k=1, B=ND_i, largest=False)
    F = F[:, :, 0]  # We only care about one eigenvector
    assert eigenvalues.shape == (q, 1)
    assert F.shape == (q, 10), f"F.shape = {F.shape}"

    # For each point i, F[i, :] now contains the coefficients of the optimal
    # quadric for a window of scale "scale" centered at points[i, :].
    # a x^2 + b y^2 + c z^2 + d xy + e xz + f yz + g x + h y + i z + j = 0
    # We now need to convert these coefficients to the standard form of a
    # quadric as a 4x4 matrix Q_i:
    # Q_i = [[a, d/2, e/2, g/2],
    #        [d/2, b, f/2, h/2],
    #        [e/2, f/2, c, i/2],
    #        [g/2, h/2, i/2, j]]

    a = F[:, 0]  # (q, 1)
    b = F[:, 1]  # (q, 1)
    c = F[:, 2]  # (q, 1)
    d = F[:, 3]  # (q, 1)
    e = F[:, 4]  # (q, 1)
    f = F[:, 5]  # (q, 1)
    g = F[:, 6]  # (q, 1)
    h = F[:, 7]  # (q, 1)
    i = F[:, 8]  # (q, 1)
    j = F[:, 9]  # (q, 1)

    Q_i = torch.stack(
        [
            torch.stack([a, d / 2, e / 2, g / 2], dim=1),
            torch.stack([d / 2, b, f / 2, h / 2], dim=1),
            torch.stack([e / 2, f / 2, c, i / 2], dim=1),
            torch.stack([g / 2, h / 2, i / 2, j], dim=1),
        ],
        dim=1,
    )
    assert Q_i.shape == (q, 4, 4)

    if False:
        # Undo the normalization:
        Q_i = Q_i / scale**2
        Q_i[:, 3, :] = Q_i[:, 3, :] - Q_i[:, 3, :].mean(dim=1, keepdim=True)
        Q_i[:, :, 3] = Q_i[:, :, 3] - Q_i[:, :, 3].mean(dim=1, keepdim=True)
        Q_i[:, 3, 3] = Q_i[:, 3, 3] + 1

        # Undo the normalization:
        Q_i[:, 3, :] = Q_i[:, 3, :] * sigma
        Q_i[:, :, 3] = Q_i[:, :, 3] * sigma
        Q_i[:, 3, 3] = Q_i[:, 3, 3] * sigma**2

        # Undo the normalization:
        Q_i[:, 3, :] = Q_i[:, 3, :] + mean_point
        Q_i[:, :, 3] = Q_i[:, :, 3] + mean_point.view(q, 1, 3)
        Q_i[:, 3, 3] = Q_i[:, 3, 3] - mean_point @ mean_point

    return Q_i, mean_point, sigma
