"""Nearest Neighbors loss for PolyData."""

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
    sigma: float = 0.1,
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
    D_xy = ((x_i - y_j) ** 2).sum(-1)
    gaussian_kernel = (-D_xy / (sigma**2)).exp()

    # Cauchy-Binet kernel
    cb_kernel = (nx_i * ny_j).sum(-1) ** 2

    # product of volumes
    vol_product = V_x * V_y

    return (gaussian_kernel * cb_kernel * vol_product).sum(dim=1).sum()


def varifold_loss(shape1, shape2, sigma=0.1):

    a = varifold_scalar(shape1, shape1, sigma=sigma)
    b = varifold_scalar(shape2, shape2, sigma=sigma)
    c = varifold_scalar(shape1, shape2, sigma=sigma)

    return a + b - 2 * c


class VarifoldLoss(BaseLoss):
    """Varifold Loss.

    The formula implemented here is based on the paper "Elastic shape analysis
    of surfaces with second-order Sobolev metrics: a comprehensive numerical
    framework" (https://arxiv.org/abs/2204.04238), equation (4.4).

    Parameters
    ----------
    sigma
        the bandwidth of the Gaussian kernel
    """

    @typecheck
    def __init__(self, sigma=0.1) -> None:
        self.sigma = sigma

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

        a = varifold_scalar(source, source, sigma=self.sigma)
        b = varifold_scalar(target, target, sigma=self.sigma)
        c = varifold_scalar(source, target, sigma=self.sigma)

        return a + b - 2 * c
