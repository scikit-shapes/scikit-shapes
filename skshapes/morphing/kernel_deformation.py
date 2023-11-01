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
        return shape.points.shape
