"""This module implements the kernel deformation morphing algorithm.

This algorithm decribes the morphing as a deformation of the ambiant space.
The parameter is a vector field, which is referred to as the momentum. This
momentum is smoothed by a kernel, and the morphed shape is obtained by
integrating the momentum. The regularization is given by <p, K_q p> where
p is the momentum, q is the initial position of the shape and K_q is the kernel
at q.
"""

from __future__ import annotations

import torch
from .basemodel import BaseModel
from ..types import (
    typecheck,
    Points,
    polydata_type,
)
from .utils import MorphingOutput, Integrator, EulerIntegrator
from .kernels import Kernel, GaussianKernel


class KernelDeformation(BaseModel):
    """Kernel deformation morphing algorithm"""

    @typecheck
    def __init__(
        self,
        n_steps: int = 1,
        integrator: Integrator = EulerIntegrator(),
        kernel: Kernel = GaussianKernel(),
        **kwargs,
    ) -> None:
        """Initialize the model

        Args:
            n_steps (int, optional): Number of integration steps.
                Defaults to 1.
            integrator (Integrator, optional): Hamiltonian integrator.
                Defaults to EulerIntegrator().
            kernel (Kernel, optional): Kernel used to smooth the momentum.
                Defaults to GaussianKernel().
        """
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
        """Morph a shape using the kernel deformation algorithm

        Args:
            shape (polydata_type): the shape to morph
            parameter (Points): the momentum
            return_path (bool, optional): True if you want to have access to
                the morphing's sequence of polydatas. Defaults to False.
            return_regularization (bool, optional): True to have access to the
                regularization. Defaults to False.

        Returns:
            MorphingOutput: a named tuple containing the morphed shape, the
                regularization and the path if needed.
        """
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

        # Define the hamiltonian
        def H(p, q):
            return self.cometric(p, q) / 2

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
            morphed_shape=morphed_shape,
            path=path,
            regularization=regularization,
        )

    @typecheck
    def parameter_shape(self, shape: polydata_type) -> tuple[int, int]:
        """Return the shape of the parameter

        Args:
            shape (polydata_type): the shape to morph

        Returns:
            tuple[int, int]: the shape of the parameter
        """
        return shape.points.shape
