"""Kernels used in the KernelDeformation class."""

from math import sqrt
from typing import Optional
from pykeops.torch import LazyTensor
from ..types import typecheck, Points, FloatScalar
from ..convolutions import LinearOperator


class Kernel:
    """Base class for kernels."""

    pass


class GaussianKernel(Kernel):
    """Gaussian kernel for spline models."""

    def __init__(self, sigma=0.1):
        """Initialize the kernel.

        Parameters
        ----------
        sigma
            Bandwidth parameter.
        """
        self.sigma = sigma

    def operator(
        self, q0: Points, q1: Optional[Points] = None
    ) -> LinearOperator:
        r"""Compute the operator $K_{q_0}^{q_1}$.

        The operator is a $(n_{q_0} x n_{q_1})$ matrix where $n_{q_0}$ is the
        number of points in $q_0$ and $n_{q_0}$ is the number of points in
        $q_1$.

        The (i, j) entry of the matrix is given by
        $K_{q_0}^{q_1}(q_0[i], q_1[j])$ where :

        $$ K_{q_0}^{q_1}(x, y) = - \exp(|| x - y ||^2) $$

        Parameters
        ----------
        q0
            The first set of points.
        q1
            The second set of points. If None, then q1 = q0.

        Returns
        -------
            The operator K_q0^q1.

        """
        if q1 is None:
            q1 = q0

        q0 = q0 / (sqrt(2) * self.sigma)
        q1 = q1 / (sqrt(2) * self.sigma)

        xi = LazyTensor(q0[:, None, :])
        yj = LazyTensor(q1[None, :, :])

        K = (-((xi - yj) ** 2)).sum(dim=2).exp()

        assert K.shape == (q0.shape[0], q1.shape[0])

        return LinearOperator(matrix=K)

    @typecheck
    def __call__(self, p: Points, q: Points) -> FloatScalar:
        """Compute the scalar product <p, K_q p>.

        Parameters
        ----------
        p
            The momentum.
        q
            The points.

        Returns
        -------
        FloatScalar
            The scalar product <p, K_q p>.

        """
        K = self.operator(q)

        Kqp = (
            K @ p
        )  # Matrix-vector product Kq.shape = NxN, shape.shape = Nx3 Kp
        return (p * Kqp).sum()  # Scalar product <p, Kqp>
