"""Moments for point clouds."""

from typing import Literal

import torch

from ..cache import add_cached_methods_to_sphinx, cache_methods_and_properties
from ..input_validation import typecheck
from ..types import (
    Double2dTensor,
    DoubleTensor,
    Float2dTensor,
    FloatTensor,
    Number,
    PointAnySignals,
    PointCovariances,
    PointDisplacements,
    PointMasses,
    Points,
    neighborhoods_type,
)


@add_cached_methods_to_sphinx
class Moments:
    @typecheck
    def __init__(
        self,
        points: Points,
        neighborhoods: neighborhoods_type,
        method: Literal["float32", "float64", "cosine"] = "cosine",
    ):
        assert points.shape[0] == neighborhoods.n_points
        assert points.device == neighborhoods.device
        assert points.dtype == neighborhoods.dtype

        self.points = points
        self.n_points = points.shape[0]
        self.dim = points.shape[1]
        self.neighborhoods = neighborhoods
        self.method = method

        # N.B.: Moments properties have no parameters, so there is no need
        #       to setup a cache_size
        cache_methods_and_properties(
            cls=Moments,
            instance=self,
            cache_size=None,
        )

    _cached_methods = ("tensors",)

    _cached_properties = (
        "masses",
        "means",
        "covariances",
        "covariance_eigenvalues",
        "covariance_eigenvectors",
        "covariance_eigenvalues_eigenvectors",
        "covariance_axes",
    )

    @typecheck
    def _masses(self) -> PointMasses:
        return self.tensors(order=0, central=False)

    @typecheck
    def _means(self) -> PointDisplacements:
        return self.tensors(order=1, central=False)

    @typecheck
    def _covariances(self) -> PointCovariances:
        return self.tensors(order=2, central=True)

    @typecheck
    def _covariance_eigenvalues_eigenvectors(self):
        covariances = self.covariances
        return torch.linalg.eigh(covariances)

    @typecheck
    def _covariance_eigenvalues(self) -> PointDisplacements:
        return self.covariance_eigenvalues_eigenvectors.eigenvalues

    @typecheck
    def _covariance_eigenvectors(self) -> PointCovariances:
        return self.covariance_eigenvalues_eigenvectors.eigenvectors

    @typecheck
    def _covariance_axes(self) -> PointCovariances:
        return self.covariance_eigenvectors * (
            self.covariance_eigenvalues.view(self.n_points, 1, self.dim) ** 0.5
        )

    @typecheck
    def _tensors(self, *, order: int, central: bool) -> PointAnySignals:
        """
        Examples
        --------

        .. testcode::

            import skshapes as sks

            # Create a 3-dimensional shape with 100 points
            shape = sks.Sphere().resample(n_points=100)
            # Consider the local moments at scale 0.5
            moments = shape.point_moments(scale=0.5)

            # Compute local masses
            mom_0 = moments.tensors(order=0, central=False)
            print(mom_0.shape, mom_0.dtype)

        .. testoutput::

            ...torch.Size([100]) torch.float32

        .. testcode::

            # Compute local average positions
            mom_1 = moments.tensors(order=1, central=False)
            print(mom_1.shape, mom_1.dtype)

        .. testoutput::

            ...torch.Size([100, 3]) torch.float32

        .. testcode::

            # Compute local covariance matrices
            mom_2 = moments.tensors(order=2, central=True)
            print(mom_2.shape, mom_2.dtype)

        .. testoutput::

            ...torch.Size([100, 3, 3]) torch.float32

        """
        if order == 0:
            masses = 1 / (self.neighborhoods.scaling**2)
            assert masses.shape == (self.n_points,)
            return masses

        elif order == 1:
            smooth_xyz = self.neighborhoods.smooth(self.points)
            assert smooth_xyz.shape == (self.n_points, self.dim)
            means = smooth_xyz / self.neighborhoods.smooth_1.view(
                self.n_points, 1
            )
            if central:
                means = means - self.points
            assert means.shape == (self.n_points, self.dim)
            return means

        elif order == 2:
            xyz = self.points
            N = self.n_points
            D = self.dim

            if central:
                xyz_ref = self.tensors(order=1, central=False)
                assert xyz_ref.shape == xyz.shape
            else:
                xyz_ref = xyz

            if self.method in ["float32", "float64"]:
                if self.method == "float32":
                    xyz = xyz.float()
                    xyz_ref = xyz_ref.float()
                elif self.method == "float64":
                    xyz = xyz.double()
                    xyz_ref = xyz_ref.double()
                    # TODO: neighborhoods -> fp64?

                # Terms of order 1:
                xyz_smooth = self.neighborhoods.smooth(xyz)
                assert xyz_smooth.shape == (N, D)

                # Terms of order 2:
                xyz2 = xyz.view(N, D, 1) * xyz.view(N, 1, D)

                xyz2_smooth = self.neighborhoods.smooth_symmetric(xyz2)
                assert xyz2_smooth.shape == (N, D, D)

                # TODO: simpler formula for central moments
                moments = (
                    xyz2_smooth
                    - xyz_smooth.view(N, D, 1) * xyz.view(N, 1, D)
                    - xyz_ref.view(N, D, 1) * xyz_smooth.view(N, 1, D)
                    + self.neighborhoods.smooth_1.view(N, 1, 1)
                    * xyz_ref.view(N, D, 1)
                    * xyz_ref.view(N, 1, D)
                )

                moments = moments.float()

            elif self.method == "cosine":
                omega = 2 * torch.pi / (100 * self.neighborhoods.scale)

                def trigonometric_features(xyz_):
                    XYZ = omega * xyz_

                    # N.B.: this is symmetric so we could cut sub-diagonal terms
                    dif_i = XYZ.view(N, D, 1) - XYZ.view(N, 1, D)
                    sum_i = XYZ.view(N, D, 1) + XYZ.view(N, 1, D)

                    assert dif_i.shape == (N, D, D)
                    assert sum_i.shape == (N, D, D)

                    trig_i = torch.stack(
                        [dif_i.cos(), dif_i.sin(), sum_i.cos(), sum_i.sin()],
                        dim=-1,
                    )
                    assert trig_i.shape == (N, D, D, 4)

                    return trig_i

                trig_i = trigonometric_features(xyz)

                # Channels 0, 2 and 3 are symmetric, but dif_i.sin() is skew_symmetric
                trig_j = self.neighborhoods.smooth_symmetric(
                    trig_i,
                    skew_symmetric_channels=[False, True, False, False],
                )
                assert trig_j.shape == (N, D, D, 4)

                trig_j = trig_j * torch.tensor([1, 1, -1, -1]).to(
                    trig_j.device
                )

                trig_ref = (
                    trigonometric_features(xyz_ref) if central else trig_i
                )

                moments = (0.5 / omega**2) * (trig_ref * trig_j).sum(-1)
                assert moments.shape == (N, D, D)

            assert moments.shape == (self.n_points, self.dim, self.dim)
            assert moments.device == self.points.device
            assert moments.dtype == torch.float32
            return moments
        else:
            msg = f"Moments of order {order} have not been implemented."
            raise NotImplementedError(msg)


@typecheck
def _point_moments(
    self,
    method: Literal["float32", "float64", "cosine"] = "cosine",
    **kwargs,
) -> Moments:
    return Moments(
        points=self.points,
        neighborhoods=self.point_neighborhoods(**kwargs),
        method=method,
    )


def symmetric_sum(a, b):
    """Symmetric terms that appear in the tensor expansion of (a+b)^n."""
    N = a.shape[0]
    D = a.shape[1]

    if a.shape == (N, D) and b.shape == (N, D, D):
        # Term a^1-b^2 that appears in the tensor expansion of (a+b)^3
        term_1 = a.view(N, D, 1, 1) * b.view(N, 1, D, D)
        term_2 = a.view(N, 1, D, 1) * b.view(N, D, 1, D)
        term_3 = a.view(N, 1, 1, D) * b.view(N, D, D, 1)
        res = term_1 + term_2 + term_3
        assert res.shape == (N, D, D, D)
        return res

    elif a.shape == (N, D) and b.shape == (N, D, D, D):
        # Term a^1-b^3 that appears in the tensor expansion of (a+b)^4
        term_1 = a.view(N, D, 1, 1, 1) * b.view(N, 1, D, D, D)
        term_2 = a.view(N, 1, D, 1, 1) * b.view(N, D, 1, D, D)
        term_3 = a.view(N, 1, 1, D, 1) * b.view(N, D, D, 1, D)
        term_4 = a.view(N, 1, 1, 1, D) * b.view(N, D, D, D, 1)
        res = term_1 + term_2 + term_3 + term_4
        assert res.shape == (N, D, D, D, D)
        return res

    elif a.shape == (N, D, D) and b.shape == (N, D, D):
        # Term a^2-b^2 that appears in the tensor expansion of (a+b)^4
        term_1 = a.view(N, D, D, 1, 1) * b.view(N, 1, 1, D, D)
        term_2 = a.view(N, D, 1, D, 1) * b.view(N, 1, D, 1, D)
        term_3 = a.view(N, D, 1, 1, D) * b.view(N, 1, D, D, 1)
        term_4 = a.view(N, 1, D, D, 1) * b.view(N, D, 1, 1, D)
        term_5 = a.view(N, 1, D, 1, D) * b.view(N, D, 1, D, 1)
        term_6 = a.view(N, 1, 1, D, D) * b.view(N, D, D, 1, 1)
        res = term_1 + term_2 + term_3 + term_4 + term_5 + term_6
        assert res.shape == (N, D, D, D, D)
        return res

    else:
        msg = f"Invalid shapes: {a.shape}, {b.shape}"
        raise ValueError(msg)


@typecheck
def _point_moments_old(
    self,
    *,
    order: int = 2,
    features: Float2dTensor | Double2dTensor | None = None,
    central: bool = False,
    rescale: bool = False,
    scale: Number | None = None,
    dtype: Literal["float", "double"] | None = None,
    **kwargs,
) -> FloatTensor | DoubleTensor:
    """Compute the local moments of a point cloud."""
    X = self.points if features is None else features

    if dtype == "float":
        X = X.float()
    elif dtype == "double":
        X = X.double()

    N = self.n_points
    D = X.shape[1]
    assert X.shape == (N, D)

    # We use a recursive formulation to best leverage our cache!
    def recursion(*, k: int):
        assert (k < order) or (central and k == order)
        return self.point_moments_old(
            order=k,
            features=features,
            central=False,
            rescale=False,
            scale=scale,
            dtype=dtype,
            **kwargs,
        )

    # Thanks to caching, Conv will only be used if central=False
    if not central:
        Conv = self.point_convolution(
            scale=scale,
            normalize=True,
            dtype=dtype,
            cutoff=1e-4,
            **kwargs,
        )
        assert Conv.shape == (N, N)
    else:
        Conv = None

    if order == 1:
        if not central:
            moments = Conv @ X  # (N, D)
        else:
            # Use recursion (-> cache) to compute Conv @ X
            moments = recursion(k=1)
            # Centering an order 1 moment -> 0
            moments = moments - moments

        assert moments.shape == (N, D)

    elif order == 2:
        if not central:
            XX = X.view(N, D, 1) * X.view(N, 1, D)  # (N, D, D)
            moments = (Conv @ XX.view(N, D * D)).view(N, D, D)  # (N, D, D)
        else:
            # Use recursion (-> cache) to compute Conv @ X and Conv @ XX
            Xm = recursion(k=1)  # (N, D)
            moments = recursion(k=2)  # (N, D, D)
            moments = moments - (
                Xm.view(N, D, 1) * Xm.view(N, 1, D)
            )  # (N, D, D)

        assert moments.shape == (N, D, D)

    elif order == 3:
        if not central:
            # X^3 as a (N, D, D, D) tensor
            XXX = X.view(N, D, 1, 1) * X.view(N, 1, D, 1) * X.view(N, 1, 1, D)
            moments = (Conv @ XXX.view(N, D * D * D)).view(
                N, D, D, D
            )  # (N, D, D, D)
        else:
            # Use recursion (cache) to compute Conv @ X, Conv @ XX, Conv @ X^3
            Xm = recursion(k=1)  # (N, D)
            mom_2 = recursion(k=2)  # (N, D, D)
            moments = recursion(k=3)  # (N, D, D, D)

            XmXmXm = (
                Xm.view(N, D, 1, 1) * Xm.view(N, 1, D, 1) * Xm.view(N, 1, 1, D)
            )

            moments = moments - symmetric_sum(Xm, mom_2) + 2 * XmXmXm

        assert moments.shape == (N, D, D, D)

    elif order == 4:
        if not central:
            # X^4 as a (N, D, D, D, D) tensor
            XXXX = (
                X.view(N, D, 1, 1, 1)
                * X.view(N, 1, D, 1, 1)
                * X.view(N, 1, 1, D, 1)
                * X.view(N, 1, 1, 1, D)
            )
            moments = (Conv @ XXXX.view(N, D * D * D * D)).view(N, D, D, D, D)
        else:
            # Use recursion (cache)
            # for Conv @ X, Conv @ XX, Conv @ X^3, Conv @ X^4
            Xm = recursion(k=1)
            mom_2 = recursion(k=2)
            mom_3 = recursion(k=3)
            moments = recursion(k=4)

            XmXm = Xm.view(N, D, 1) * Xm.view(N, 1, D)
            XmXmXmXm = (
                Xm.view(N, D, 1, 1, 1)
                * Xm.view(N, 1, D, 1, 1)
                * Xm.view(N, 1, 1, D, 1)
                * Xm.view(N, 1, 1, 1, D)
            )

            moments = (
                moments
                - symmetric_sum(Xm, mom_3)
                + symmetric_sum(XmXm, mom_2)
                - 3 * XmXmXmXm
            )

        assert moments.shape == (N, D, D, D, D)

    else:
        msg = f"Moments of order {order} have not been implemented."
        raise NotImplementedError(msg)

    if rescale:
        if scale is None:
            msg = "A finite scale must be provided if rescale is True"
            raise ValueError(msg)

        moments = moments / scale**order

    if dtype == "float":
        assert moments.dtype == torch.float32
    elif dtype == "double":
        assert moments.dtype == torch.float64

    return moments
