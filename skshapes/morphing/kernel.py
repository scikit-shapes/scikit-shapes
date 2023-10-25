from __future__ import annotations

import torch
from .basemodel import BaseModel
from ..types import (
    typecheck,
    Points,
    FloatScalar,
    polydata_type,
    Number,
)

from beartype.typing import Tuple

import torch
from math import sqrt
from pykeops.torch import LazyTensor


class Kernel:
    """All cometrics used in spline models should inherit from this class"""

    pass


class Integrator:
    """All hamiltonian integrators used in spline models should inherit from this class"""

    pass


class GaussianKernel(Kernel):
    def __init__(self, sigma=0.1):
        self.sigma = sigma

    @typecheck
    def __call__(self, p: Points, q: Points) -> FloatScalar:
        # Compute the <p, K_q p>
        from math import sqrt

        q = q / (sqrt(2) * self.sigma)

        Kq = (
            (-((LazyTensor(q[:, None, :]) - LazyTensor(q[None, :, :])) ** 2))
            .sum(dim=2)
            .exp()
        )  # Symbolic matrix of kernel distances Kq.shape = NxN
        Kqp = Kq @ p  # Matrix-vector product Kq.shape = NxN, shape.shape = Nx3 Kp
        return (p * Kqp).sum()  # Scalar product <p, Kqp>


class EulerIntegrator(Integrator):
    def __init__(self) -> None:
        pass

    @typecheck
    def __call__(self, p: Points, q: Points, H, dt: Number) -> Tuple[Points, Points]:
        Gp, Gq = torch.autograd.grad(H(p, q), (p, q), create_graph=True)

        pdot, qdot = -Gq, Gp
        p = p + pdot * dt
        q = q + qdot * dt
        return p, q


from .utils import MorphingOutput
from beartype.typing import Literal


class KernelDeformation(BaseModel):
    @typecheck
    def __init__(
        self,
        n_steps: int = 1,
        integrator: Integrator = EulerIntegrator(),
        kernel: Kernel = GaussianKernel(),
        **kwargs,
    ) -> None:
        self.integrator = integrator
        self.cometric = kernel
        self.n_steps = n_steps

    @typecheck
    def morph(
        self,
        shape: polydata_type,
        parameter: Points,
        return_path: bool = False,
        return_regularization: bool = False,
    ) -> MorphingOutput:
        if parameter.device != shape.device:
            p = parameter.to(shape.device)
        else:
            p = parameter
        p.requires_grad = True

        q = shape.points.clone()
        q.requires_grad = True

        # Compute the regularization
        regularization = torch.tensor(0.0, device=shape.device)
        if return_regularization:
            regularization = self.cometric(parameter, q) / 2

        H = lambda p, q: self.cometric(p, q) / 2  # Hamiltonian
        dt = 1 / self.n_steps  # Time step

        if not return_path:
            path = None
        else:
            path = [shape.copy() for _ in range(self.n_steps + 1)]

        for i in range(self.n_steps):
            p, q = self.integrator(p, q, H, dt)
            if return_path:
                path[i + 1].points = q.clone()

        morphed_shape = shape.copy()
        morphed_shape.points = q

        return MorphingOutput(
            morphed_shape=morphed_shape, path=path, regularization=regularization
        )

    @typecheck
    def parameter_shape(self, shape: polydata_type) -> Tuple[int, int]:
        return shape.points.shape
