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

from typing import Tuple

import torch
from math import sqrt
from pykeops.torch import LazyTensor


class Cometric:
    """All cometrics used in spline models should inherit from this class"""

    pass


class Integrator:
    """All hamiltonian integrators used in spline models should inherit from this class"""

    pass


class GaussianCometric(Cometric):
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
from typing import Literal


class SplineDeformation(BaseModel):
    @typecheck
    def __init__(
        self,
        n_steps: int = 1,
        integrator: Integrator = EulerIntegrator(),
        cometric: Cometric = GaussianCometric(),
        **kwargs,
    ) -> None:
        self.integrator = integrator
        self.cometric = cometric
        self.n_steps = n_steps

    @typecheck
    def morph(
        self,
        shape: polydata_type,
        parameter: Points,
        return_path: bool = False,
        return_regularization: bool = False,
    ) -> MorphingOutput:
        # If intial_shape or target_shape are not defined, raise an error
        # TODO : maybe better to add a boolean attribute self.initialized ?

        path = []

        p = parameter.clone().detach()
        p.requires_grad = True
        q = shape.points.clone()
        q.requires_grad = True

        # Compute the regularization
        if return_regularization:
            regularization = self.cometric(parameter, q) / 2
        else:
            regularization = None

        q.requires_grad = True  # Needed to compute the gradient of H
        H = lambda p, q: self.cometric(p, q) / 2  # Hamiltonian
        dt = 1 / self.n_steps  # Time step

        if not return_path:
            path = None
        else:
            path = [shape.copy() for _ in range(self.n_steps + 1)]

        for i in range(self.n_steps):
            if self.integrator == "Euler":
                p, q = Euler_integrator(p, q, H, dt)

        morphed_shape = shape.copy()
        morphed_shape.points = q.clone().detach()

        return MorphingOutput(
            morphed_shape=morphed_shape, path=path, regularization=regularization
        )

    @typecheck
    def parameter_shape(self, shape: polydata_type) -> Tuple[int, int]:
        return shape.points.shape
