"""Utility functions for morphing.

It defines a named tuple MorphingOutput which is used to return the result of
the morphing algorithms.
It also defines the integrators used in the kernel deformation model.
"""

from ..data import Shape
from typing import NamedTuple, Optional
import torch
from ..types import typecheck, FloatScalar, Number, Points


class MorphingOutput(NamedTuple):
    """Named tuple containing the result of the morphing algorithms."""

    morphed_shape: Optional[Shape] = None
    regularization: Optional[FloatScalar] = None
    path: Optional[list[Shape]] = None


class Integrator:
    """Base class for integrators."""

    pass


class EulerIntegrator(Integrator):
    """A basic Euler integrator for Hamiltonian systems."""

    def __init__(self) -> None:
        """Initialize the integrator."""
        pass

    @typecheck
    def __call__(
        self, p: Points, q: Points, H, dt: Number
    ) -> tuple[Points, Points]:
        """Update the position and momentum using the Euler integrator.

        Parameters
        ----------
        p
            The momentum.
        q
            The position.
        H
            The Hamiltonian function.
        dt
            The time step.

        Returns
        -------
        tuple[Points, Points]
            (momentum, position)

        """
        Gp, Gq = torch.autograd.grad(H(p, q), (p, q), create_graph=True)

        pdot, qdot = -Gq, Gp
        p = p + pdot * dt
        q = q + qdot * dt
        return p, q
