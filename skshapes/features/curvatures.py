import numpy as np
import torch
from pykeops.torch import LazyTensor

from ..utils import diagonal_ranges
from ..types import (
    typecheck,
    Points,
    Optional,
    Triangles,
    Number,
    Float1dTensor,
    Float2dTensor,
    FloatTensor,
    Tuple,
    Union,
)

from .normals import smooth_normals, tangent_vectors
from .structure_tensors import structure_tensors


@typecheck
def smooth_curvatures(
    *,
    vertices: Points,
    triangles: Optional[Triangles] = None,
    scales=[1.0],
    batch=None,
    normals: Optional[Points] = None,
    reg: Number = 0.01,
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
    triangles: Optional[Triangles] = None,
    scale=1.0,
    batch=None,
    normals: Optional[Points] = None,
    reg: Number = 0.01,
):
    # Number of points:
    N = points.shape[0]
    ranges = diagonal_ranges(batch)

    if False:
        # Compute the normals at different scales + vertice areas - (N, 3):
        normals = smooth_normals(
            vertices=points,
            triangles=triangles,
            normals=normals,
            scale=scale,
            batch=batch,
        )
        assert normals.shape == (N, 3)

    else:
        ST = structure_tensors(points=points, scale=scale / 3, ranges=ranges)
        # Perform an SVD decomposition:
        decomp = torch.linalg.eigh(ST)
        assert decomp.eigenvalues.shape == (N, 3)
        assert (decomp.eigenvalues[:, 1:] >= decomp.eigenvalues[:, :-1]).all()

        # Extract the eigenvectors:
        n = decomp.eigenvectors[:, :, 0].contiguous()  # (N, 3)

        if normals is None:
            normals = n
        else:
            # Align n with the normals
            normals = n * torch.sign((n * normals).sum(-1, keepdim=True))

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
    ones_ = LazyTensor(torch.ones(1, 1, 1, device=points.device, dtype=points.dtype))

    # Squared distance:
    d2_ij = ((x_j - x_i) ** 2).sum(-1)  # (N, N, 1)
    # Gaussian window:
    window_ij = (-d2_ij / (2 * (scale**2))).exp()  # (N, N, 1)

    # Project on the tangent plane:
    P_ij = uv_i.matvecmult(x_j - x_i)  # (N, N, 2)
    Q_ij = P_ij[0] * P_ij[1]  # (N, N, 1)
    # Concatenate:
    R_ij = (P_ij**2).concat(Q_ij)  # (N, N, 2+1)
    R_ij = R_ij.concat(ones_)  # (N, N, 2+1+1)

    N_ij = n_i.matvecmult(x_j - x_i)  # (N, N, 1)

    # Concatenate:
    R_N_ij = R_ij.concat(N_ij)  # (N, N, 4+1)

    # Covariances, with a scale-dependent weight:
    RRt_RNt_ij = R_ij.tensorprod(R_N_ij)  # (N, N, 4*(4+1))
    RRt_RNt_ij = window_ij * RRt_RNt_ij  #  (N, N, 4*(4+1))

    # Reduction - with batch support:
    RRt_RNt_ij.ranges = ranges
    RRt_RNt = RRt_RNt_ij.sum(1)  # (N, 4*(4+1))

    # Reshape to get the two covariance matrices:
    RRt_RNt = RRt_RNt.view(N, 4, 4 + 1)
    RRt, RNt = (
        RRt_RNt[:, :, :4].contiguous(),
        RRt_RNt[:, :, 4].contiguous(),
    )  # (N, 4, 4), (N, 4)
    assert RRt.shape == (N, 4, 4)
    assert RNt.shape == (N, 4)

    # Add a small ridge regression:
    for i in range(4):
        RRt[:, i, i] += reg

    # (RRt^-1 @ RNt) : simple estimation through linear regression
    acbo = torch.linalg.solve(RRt, RNt)
    assert acbo.shape == (N, 4)
    a, c, b = acbo[:, 0], acbo[:, 1], acbo[:, 2] / 2  # (N,)

    # Normalization
    mean_curvature = a + c
    gauss_curvature = a * c - b * b

    return {
        "mean": mean_curvature,
        "gauss": gauss_curvature,
    }


@typecheck
def point_quadratic_coefficients(
    self,
    *,
    scale: Optional[Number] = None,
    return_nuv: bool = False,
    **kwargs,
) -> Union[Float2dTensor, Tuple[Float2dTensor, dict]]:
    """Returns the point-wise principal curvatures."""

    # nuv are arranged row-wise!
    N = self.n_points
    nuv = self.point_frames(scale=scale, **kwargs)
    assert nuv.shape == (N, 3, 3)

    nuv = dict(
        n=nuv[:, 0, :].contiguous(),  # (N, 3)
        u=nuv[:, 1, :].contiguous(),  # (N, 3)
        v=nuv[:, 2, :].contiguous(),  # (N, 3)
    )
    for key, value in nuv.items():
        assert value.shape == (N, 3)

    # Recover the local moments of order 1, 2, 3, 4:
    def central_moments(*, order):
        return self.point_moments(
            order=order, scale=scale, central=True, dtype="double", **kwargs
        ).float()

    moms = [None] + [central_moments(order=k) for k in [1, 2, 3, 4]]

    def str_to_moment(s):
        if s == "":
            r = torch.ones_like(moms[1][:, 0])
            assert r.shape == (N,)
            return r

        if len(s) == 1:
            r = nuv[s]
            assert r.shape == (N, 3)

        elif len(s) == 2:
            a = nuv[s[0]]
            b = nuv[s[1]]
            r = a.view(N, 3, 1) * b.view(N, 1, 3)
            assert r.shape == (N, 3, 3)

        elif len(s) == 3:
            a = nuv[s[0]]
            b = nuv[s[1]]
            c = nuv[s[2]]
            r = a.view(N, 3, 1, 1) * b.view(N, 1, 3, 1) * c.view(N, 1, 1, 3)
            assert r.shape == (N, 3, 3, 3)

        elif len(s) == 4:
            a = nuv[s[0]]
            b = nuv[s[1]]
            c = nuv[s[2]]
            d = nuv[s[3]]
            r = (
                a.view(N, 3, 1, 1, 1)
                * b.view(N, 1, 3, 1, 1)
                * c.view(N, 1, 1, 3, 1)
                * d.view(N, 1, 1, 1, 3)
            )
            assert r.shape == (N, 3, 3, 3, 3)

        mom = moms[len(s)]
        assert r.shape == mom.shape
        res = (r.view(N, -1) * mom.view(N, -1)).sum(-1)
        assert res.shape == (N,)
        return res

    T = ["uu", "uv", "vv", "u", "v", ""]

    TT = [str_to_moment(pref + suf) for pref in T for suf in T]
    TN = [str_to_moment(pref + "n") for pref in T]

    TT = torch.stack(TT, dim=-1).view(N, len(T), len(T))  # (N, 6, 6)
    TN = torch.stack(TN, dim=-1).view(N, len(T))  # (N, 6)

    for i in range(len(T)):
        TT[:, i, i] += 1e-6

    # (TT^-1 @ TN) : simple estimation through linear regression
    coefs = torch.linalg.solve(TT, TN)
    assert coefs.shape == (N, len(T))

    if return_nuv:
        return coefs, nuv
    else:
        return coefs


@typecheck
def point_quadratic_fits(
    self,
    *,
    scale: Optional[Number] = None,
    **kwargs,
) -> FloatTensor:
    N = self.n_points

    # Local average:
    Xm = self.point_moments(order=1, scale=scale, central=False, **kwargs)
    assert Xm.shape == (N, 3)

    # Local quadratic coefficients in tangent space:
    coefs, nuv = self.point_quadratic_coefficients(
        scale=scale, return_nuv=True, **kwargs
    )
    assert coefs.shape == (N, 6)
    for key in ["n", "u", "v"]:
        assert nuv[key].shape == (N, 3)

    # First term: constant offset around the local average:
    offset = torch.Tensor(
        [
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 1.0],
        ],
        device=Xm.device,
    )
    term_1 = Xm.view(N, 3, 1, 1) * offset.view(1, 1, 3, 3)  # (N, 3, 3, 3)

    # Second term: linear term in tangent space, following u:
    tangent_1 = torch.Tensor(
        [
            [0, 0, 0.5],
            [0, 0, 0],
            [0.5, 0, 0],
        ],
        device=Xm.device,
    )
    term_2 = nuv["u"].view(N, 3, 1, 1) * tangent_1.view(1, 1, 3, 3)  # (N, 3, 3, 3)

    # Third term: linear term in tangent space, following v:
    tangent_2 = torch.Tensor(
        [
            [0, 0, 0],
            [0, 0, 0.5],
            [0, 0.5, 0],
        ],
        device=Xm.device,
    )
    term_3 = nuv["v"].view(N, 3, 1, 1) * tangent_2.view(1, 1, 3, 3)  # (N, 3, 3, 3)

    # Fourth term: quadratic term in tangent space, following n:
    UU = coefs[:, 0]
    UV = coefs[:, 1]
    VV = coefs[:, 2]
    U = coefs[:, 3]
    V = coefs[:, 4]
    O = coefs[:, 5]

    quadratic = torch.stack(
        [UU, UV / 2, U / 2, UV / 2, VV, V / 2, U / 2, V / 2, O], dim=-1
    ).view(N, 3, 3)

    term_4 = nuv["n"].view(N, 3, 1, 1) * quadratic.view(N, 1, 3, 3)  # (N, 3, 3, 3)

    # Sum:
    fit = term_1 + term_2 + term_3 + term_4
    assert fit.shape == (N, 3, 3, 3)
    return fit


@typecheck
def point_principal_curvatures(
    self,
    *,
    scale: Optional[Number] = None,
    **kwargs,
) -> Tuple[Float1dTensor, Float1dTensor]:
    """Returns the point-wise principal curvatures.

    We rely on the formulas detailed in Example 4.2 of
    Curvature formulas for implicit curves and surfaces, Goldman, 2005.
    """
    coefs = self.point_quadratic_coefficients(scale=scale, **kwargs)
    assert coefs.shape == (self.n_points, 6)

    a, b, c = 2 * coefs[:, 0], coefs[:, 1], 2 * coefs[:, 2]
    d, e = coefs[:, 3], coefs[:, 4]

    # Grad(f) = (d, e) and H(f) = [[a, b], [b, c]].
    denom = 1 + d**2 + e**2  # 1 + ||Grad(f)||^2
    gauss = a * c - b * b  # det(H(f))
    gauss = gauss / denom**2

    # Term 1: Grad(f)^T . H(f) . Grad(f)
    mean = d * d * a + 2 * d * e * b + e * e * c
    # Term 2: - (1 + ||Grad(f)||^2) * trace(H(f))
    mean = mean - denom * (a + c)
    mean = mean / denom ** (1.5)

    if self.triangles is None:
        # If we cannot orient the surface,
        # our convention is that the mean curvature is positive:
        mean = mean.abs()

    # delta = (trace ** 2 - 4 * det).relu().sqrt()
    delta = (mean**2 - 4 * gauss).relu().sqrt()
    kmax = (mean + delta) / 2
    kmin = (mean - delta) / 2

    assert kmax.shape == (self.n_points,)
    assert kmin.shape == (self.n_points,)
    return kmax, kmin


@typecheck
def point_shape_indices(self, **kwargs) -> Float1dTensor:
    """Returns the point-wise shape index, estimated at a given scale.

    For reference, see:
    "Surface shape and curvature scales", Koenderink and van Doorn, 1992.
    """
    kmax, kmin = self.point_principal_curvatures(**kwargs)
    si = (2 / np.pi) * torch.atan((kmax + kmin) / (kmax - kmin))

    if self.triangles is None:
        # If we cannot orient the surface, the shape index is only defined up to a sign:
        si = si.abs()

    return si


@typecheck
def point_curvedness(self, **kwargs) -> Float1dTensor:
    """Returns the point-wise curvedness, estimated at a given scale.

    For reference, see:
    "Surface shape and curvature scales", Koenderink and van Doorn, 1992.
    """
    kmax, kmin = self.point_principal_curvatures(**kwargs)
    return ((kmax**2 + kmin**2) / 2).sqrt()
