from ..types import typecheck, Points, FloatScalar
from pykeops.torch import LazyTensor


class Kernel:
    """All KernelDeformation's kernels should inherit from this class"""

    pass


class GaussianKernel(Kernel):
    """A Gaussian kernel for spline models"""

    def __init__(self, sigma=0.1):
        """Initialize the kernel

        Args:
            sigma (float, optional): bandwidth parameter. Defaults to 0.1.
        """
        self.sigma = sigma

    @typecheck
    def __call__(self, p: Points, q: Points) -> FloatScalar:
        """Compute the scalar product <p, K_q p> where K_q is the Gaussian
        kernel matrix with bandwidth sigma and points q.


        Args:
            p (Points): tensor of points (momentum)
            q (Points): tensor of points (position)

        Returns:
            FloatScalar: scalar product <p, K_q p>
        """
        # Compute the <p, K_q p>
        from math import sqrt

        q = q / (sqrt(2) * self.sigma)

        Kq = (
            (-((LazyTensor(q[:, None, :]) - LazyTensor(q[None, :, :])) ** 2))
            .sum(dim=2)
            .exp()
        )  # Symbolic matrix of kernel distances Kq.shape = NxN
        Kqp = (
            Kq @ p
        )  # Matrix-vector product Kq.shape = NxN, shape.shape = Nx3 Kp
        return (p * Kqp).sum()  # Scalar product <p, Kqp>
