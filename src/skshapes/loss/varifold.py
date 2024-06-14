"""Nearest Neighbors loss for PolyData."""

from typing import Literal

from pykeops.torch import LazyTensor

from ..input_validation import typecheck
from ..types import Float1dTensor, FloatScalar, Points, polydata_type
from .baseloss import BaseLoss


@typecheck
def extract_geom(shape: polydata_type) -> tuple[Points, Points, Float1dTensor]:
    """Utility function to extract the geometry of a PolyData object."""

    centers = shape.triangle_centers
    normals = shape.triangle_normals
    areas = shape.triangle_areas
    normalized_normals = normals / normals.norm(dim=1)[:, None]

    return centers, normalized_normals, areas


def varifold_scalar(
    shape1: polydata_type,
    shape2: polydata_type,
    radial_kernel,
    sigma,
    zonal_kernel,
) -> FloatScalar:
    """Compute the varifold loss between two shapes.

    The formula implemented here is based on the paper "Elastic shape analysis
    of surfaces with second-order Sobolev metrics: a comprehensive numerical
    framework" (https://arxiv.org/abs/2204.04238), equation (4.4).

    Parameters
    ----------
    shape1
        the first shape
    shape2
        the second shape
    sigma
        the bandwidth of the Gaussian kernel

    """
    (source_centers, source_normals, source_volumes) = extract_geom(shape1)

    (target_centers, target_normals, target_volumes) = extract_geom(shape2)

    x_i = LazyTensor(source_centers[:, None, :])
    nx_i = LazyTensor(source_normals[:, None, :])
    if source_volumes.dim() == 1:
        source_volumes = source_volumes[:, None]
    V_x = LazyTensor(source_volumes[:, None])

    y_j = LazyTensor(target_centers[None, :, :])
    ny_j = LazyTensor(target_normals[None, :, :])
    if target_volumes.dim() == 1:
        target_volumes = target_volumes[:, None]
    V_y = LazyTensor(target_volumes[None, :])

    # Gaussian kernel
    if radial_kernel == "Gaussian":
        D_xy = ((x_i - y_j) ** 2).sum(-1)
        K_radial = (-D_xy / (sigma**2)).exp()
    else:
        msg = "Only Gaussian radial kernel is implemented do far"
        raise NotImplementedError(msg)

    # Cauchy-Binet kernel
    if zonal_kernel == "Cauchy-Binet":
        K_zonal = (nx_i * ny_j).sum(-1) ** 2
    else:
        msg = "Only Cauchy-Binet zonal kernel is implemented do far"
        raise NotImplementedError(msg)

    # product of volumes
    vol_product = V_x * V_y

    return (K_radial * K_zonal * vol_product).sum(dim=1).sum()


class VarifoldLoss(BaseLoss):
    """Varifold Loss.

    The formula implemented here is based on the paper "Elastic shape analysis
    of surfaces with second-order Sobolev metrics: a comprehensive numerical
    framework" (https://arxiv.org/abs/2204.04238), equation (4.4).

    Parameters
    ----------
    radial_kernel
        The radial kernel (between point positions)
    zonal_kernel
        The zonal kernel (between triangles normals)
    radial_bandwidth
        The bandwidth for the radial kernel
    """

    @typecheck
    def __init__(
        self,
        radial_kernel: Literal["Gaussian", "uniform"] = "Gaussian",
        zonal_kernel: Literal["Cauchy-Binet"] = "Cauchy-Binet",
        radial_bandwidth: float = 0.1,
    ) -> None:

        self.radial_kernel = radial_kernel
        self.zonal_kernel = zonal_kernel
        self.sigma = radial_bandwidth

    @typecheck
    def __call__(
        self, source: polydata_type, target: polydata_type
    ) -> FloatScalar:
        """Compute the loss.

        Parameters
        ----------
        source
            the source shape
        target
            the target shape

        Returns
        -------
        FloatScalar
            the loss
        """
        super().__call__(source=source, target=target)

        if source.dim != 3 or target.dim != 3:
            msg = f"Dimension of source and target must be 3 to compute varifold loss. Found {source.dim} for source, {target.dim} for target"
            raise ValueError(msg)

        if (source.triangles is None) or (target.triangles is None):
            msg = "Source and target must have triangles to compute varifold loss."
            raise ValueError(msg)

        kwargs = {
            "sigma": self.sigma,
            "radial_kernel": self.radial_kernel,
            "zonal_kernel": self.zonal_kernel,
        }

        a = varifold_scalar(source, source, **kwargs)
        b = varifold_scalar(target, target, **kwargs)
        c = varifold_scalar(source, target, **kwargs)

        return a + b - 2 * c
