from ..data import Shape
from typing import NamedTuple, Optional
import torch
from ..types import typecheck, FloatScalar, Number, Points


class MorphingOutput(NamedTuple):
    morphed_shape: Optional[Shape] = None
    regularization: Optional[FloatScalar] = None
    path: Optional[list[Shape]] = None


class Integrator:
    """All hamiltonian integrators used in spline models should inherit from
    this class"""

    pass


class EulerIntegrator(Integrator):
    """A basic Euler integrator for Hamiltonian systems"""

    def __init__(self) -> None:
        """Initialize the integrator"""
        pass

    @typecheck
    def __call__(
        self, p: Points, q: Points, H, dt: Number
    ) -> tuple[Points, Points]:
        """Update the position and momentum using the Euler integrator

        Args:
            p (Points): momentum
            q (Points): position
            H (_type_): Hamiltonian
            dt (Number): time step

        Returns:
            tuple[Points, Points]: (momentum, position)
        """

        Gp, Gq = torch.autograd.grad(H(p, q), (p, q), create_graph=True)

        pdot, qdot = -Gq, Gp
        p = p + pdot * dt
        q = q + qdot * dt
        return p, q
