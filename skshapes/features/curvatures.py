import torch
from pykeops.torch import LazyTensor

from ..utils import diagonal_ranges
from ..types import typecheck, Points, Optional, Triangles, Number

from .normals import smooth_normals, tangent_vectors


@typecheck
def smooth_curvatures(
    *,
    vertices: Points,
    triangles: Optional[Triangles] = None,
    scales=[1.0],
    batch=None,
    normals:Optional[Points] = None,
    reg: Number = 0.01
):
    """Returns a collection of mean (H) and Gauss (K) curvatures at different scales.

    points, faces, scales  ->  (H_1, K_1, ..., H_S, K_S)
    (N, 3), (3, N), (S,)   ->         (N, S*2)

    We rely on a very simple linear regression method, for all vertices:

      1. Estimate normals and surface areas.
      2. Compute a local tangent frame.
      3. In a pseudo-geodesic Gaussian neighborhood at scale s,
         compute the two (2, 2) covariance matrices PPt and PQt
         between the displacement vectors "P = x_i - x_j" and
         the normals "Q = n_i - n_j", projected on the local tangent plane.
      4. Up to the sign, the shape operator S at scale s is then approximated
         as  "S = (reg**2 * I_2 + PPt)^-1 @ PQt".
      5. The mean and Gauss curvatures are the trace and determinant of
         this (2, 2) matrix.

    As of today, this implementation does not weigh points by surface areas:
    this could make a sizeable difference if protein surfaces were not
    sub-sampled to ensure uniform sampling density.

    For convergence analysis, see for instance
    "Efficient curvature estimation for oriented point clouds",
    Cao, Li, Sun, Assadi, Zhang, 2019.

    Args:
        vertices (Tensor): (N,3) coordinates of the points or mesh vertices.
        triangles (integer Tensor, optional): (3,T) mesh connectivity. Defaults to None.
        scales (list of floats, optional): list of (S,) smoothing scales. Defaults to [1.].
        batch (integer Tensor, optional): batch vector, as in PyTorch_geometric. Defaults to None.
        normals (Tensor, optional): (N,3) field of "raw" unit normals. Defaults to None.
        reg (float, optional): small amount of Tikhonov/ridge regularization
            in the estimation of the shape operator. Defaults to .01.

    Returns:
        (Tensor): (N, S*2) tensor of mean and Gauss curvatures computed for
            every point at the required scales.
    """
    # Number of points, number of scales:
    N, S = vertices.shape[0], len(scales)
    ranges = diagonal_ranges(batch)

    # Compute the normals at different scales + vertice areas - (N, S, 3):
    normals_s = smooth_normals(
        vertices=vertices,
        triangles=triangles,
        normals=normals,
        scale=scales,
        batch=batch,
    )
    assert normals_s.shape == (N, S, 3)

    # Local tangent bases - (N, S, 2, 3):
    uv_s = tangent_vectors(normals_s)
    assert uv_s.shape == (N, S, 2, 3)

    features = []

    for s, scale in enumerate(scales):
        # Extract the relevant descriptors at the current scale:
        normals = normals_s[:, s, :].contiguous()  #  (N, 3)
        uv = uv_s[:, s, :, :].contiguous()  # (N, 2, 3)

        # Encode as symbolic tensors:
        # Points:
        x_i = LazyTensor(vertices.view(N, 1, 3))
        x_j = LazyTensor(vertices.view(1, N, 3))
        # Normals:
        n_i = LazyTensor(normals.view(N, 1, 3))
        n_j = LazyTensor(normals.view(1, N, 3))
        # Tangent bases:
        uv_i = LazyTensor(uv.view(N, 1, 6))

        # Pseudo-geodesic squared distance:
        d2_ij = ((x_j - x_i) ** 2).sum(-1) * ((2 - (n_i | n_j)) ** 2)  # (N, N, 1)
        # Gaussian window:
        window_ij = (-d2_ij / (2 * (scale**2))).exp()  # (N, N, 1)

        # Project on the tangent plane:
        P_ij = uv_i.matvecmult(x_j - x_i)  # (N, N, 2)
        Q_ij = uv_i.matvecmult(n_j - n_i)  # (N, N, 2)
        # Concatenate:
        PQ_ij = P_ij.concat(Q_ij)  # (N, N, 2+2)

        # Covariances, with a scale-dependent weight:
        PPt_PQt_ij = P_ij.tensorprod(PQ_ij)  # (N, N, 2*(2+2))
        PPt_PQt_ij = window_ij * PPt_PQt_ij  #  (N, N, 2*(2+2))

        # Reduction - with batch support:
        PPt_PQt_ij.ranges = ranges
        PPt_PQt = PPt_PQt_ij.sum(1)  # (N, 2*(2+2))

        # Reshape to get the two covariance matrices:
        PPt_PQt = PPt_PQt.view(N, 2, 2, 2)
        PPt, PQt = PPt_PQt[:, :, 0, :], PPt_PQt[:, :, 1, :]  # (N, 2, 2), (N, 2, 2)

        # Add a small ridge regression:
        PPt[:, 0, 0] += reg
        PPt[:, 1, 1] += reg

        # (minus) Shape operator, i.e. the differential of the Gauss map:
        # = (PPt^-1 @ PQt) : simple estimation through linear regression
        Sh = torch.linalg.solve(PPt, PQt)
        a, b, c, d = Sh[:, 0, 0], Sh[:, 0, 1], Sh[:, 1, 0], Sh[:, 1, 1]  # (N,)

        # Normalization
        mean_curvature = a + d
        gauss_curvature = a * d - b * c
        features += [mean_curvature.clamp(-1, 1), gauss_curvature.clamp(-1, 1)]

    features = torch.stack(features, dim=-1)
    assert features.shape == (N, S * 2)
    return features



@typecheck
def smooth_curvatures_2(
        *,
        points: Points,
        triangles:Optional[Triangles] = None,
        scale=1.0,
        batch=None,
        normals:Optional[Points] = None,
        reg: Number = 0.01
    ):
    # Number of points:
    N = points.shape[0]
    ranges = diagonal_ranges(batch)

    # Compute the normals at different scales + vertice areas - (N, S, 3):
    normals = smooth_normals(
        vertices=points,
        triangles=triangles,
        normals=normals,
        scale=scale,
        batch=batch,
    )
    assert normals.shape == (N, 3)

    # Local tangent bases - (N, 2, 3):
    uv = tangent_vectors(normals)
    assert uv.shape == (N, 2, 3)

    # Encode as symbolic tensors:
    # Points:
    x_i = LazyTensor(points.view(N, 1, 3))
    x_j = LazyTensor(points.view(1, N, 3))
    # Normals:
    n_i = LazyTensor(normals.view(N, 1, 3))
    # Tangent bases:
    uv_i = LazyTensor(uv.view(N, 1, 6))

    # Squared distance:
    d2_ij = ((x_j - x_i) ** 2).sum(-1)  # (N, N, 1)
    # Gaussian window:
    window_ij = (-d2_ij / (2 * (scale**2))).exp()  # (N, N, 1)

    # Project on the tangent plane:
    P_ij = uv_i.matvecmult(x_j - x_i)  # (N, N, 2)
    Q_ij = P_ij[0] * P_ij[1] # (N, N, 1)
    # Concatenate:
    R_ij = (P_ij ** 2).concat(Q_ij)  # (N, N, 2+1)

    N_ij = n_i.matvecmult(x_j - x_i)  # (N, N, 1)
    
    # Concatenate:
    R_N_ij = R_ij.concat(N_ij)  # (N, N, 4)

    # Covariances, with a scale-dependent weight:
    RRt_RNt_ij = R_ij.tensorprod(R_N_ij)  # (N, N, 3*(3+1))
    RRt_RNt_ij = window_ij * RRt_RNt_ij  #  (N, N, 3*(3+1))

    # Reduction - with batch support:
    RRt_RNt_ij.ranges = ranges
    RRt_RNt = RRt_RNt_ij.sum(1)  # (N, 3*(3+1))

    # Reshape to get the two covariance matrices:
    RRt_RNt = RRt_RNt.view(N, 3, 3+1)
    RRt, RNt = RRt_RNt[:, :, :3].contiguous(), RRt_RNt[:, :, 3].contiguous()  # (N, 3, 3), (N, 3)
    assert RRt.shape == (N, 3, 3)
    assert RNt.shape == (N, 3)

    # Add a small ridge regression:
    RRt[:, 0, 0] += reg
    RRt[:, 1, 1] += reg
    RRt[:, 2, 2] += reg

    # (RRt^-1 @ RNt) : simple estimation through linear regression
    acb = torch.linalg.solve(RRt, RNt)
    assert acb.shape == (N, 3)
    a, c, b = acb[:, 0], acb[:, 1], acb[:, 2]  # (N,)

    # Normalization
    mean_curvature = a + c
    gauss_curvature = a * c - b * b
    
    return {
        "mean": mean_curvature,
        "gauss": gauss_curvature,
    }