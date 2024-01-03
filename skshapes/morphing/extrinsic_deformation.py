"""Kernel deformation morphing algorithm.

This algorithm decribes the morphing as a deformation of the ambiant space.
The parameter is a vector field, which is referred to as the momentum. This
momentum is smoothed by a kernel, and the morphed shape is obtained by
integrating the momentum. The regularization is given by <p, K_q p> where
p is the momentum, q is the initial position of the shape and K_q is the kernel
at q.
"""

from __future__ import annotations

import torch
from torchdiffeq import odeint
from typing import Optional, Literal

from .basemodel import BaseModel
from ..types import (
    Points,
    polydata_type,
    MorphingOutput,
)
from ..input_validation import typecheck

from .utils import Integrator, EulerIntegrator
from .kernels import Kernel, GaussianKernel


class ExtrinsicDeformation(BaseModel):
    """Kernel deformation morphing algorithm."""

    @typecheck
    def __init__(
        self,
        n_steps: int = 1,
        integrator: Optional[Integrator] = None,
        kernel: Optional[Kernel] = None,
        control_points: bool = True,
        backend: Literal["sks", "torchdiffeq"] = "sks",
        **kwargs,
    ) -> None:
        """Class constructor.

        Parameters
        ----------
        n_steps
            Number of integration steps.
        integrator
            Hamiltonian integrator.
        kernel
            Kernel used to smooth the momentum.
        control_points
            If True, the control points are the control points of the shape
            (accessible with `shape.control_points`). If False, the control
            points are the points of the shape. If `shape.control_points` is
            None, the control points are the points of the shape.

        """
        if integrator is None:
            integrator = EulerIntegrator()
        if kernel is None:
            kernel = GaussianKernel()

        self.integrator = integrator
        self.cometric = kernel
        self.n_steps = n_steps
        self.control_points = control_points
        self.backend = backend

    @typecheck
    def morph(
        self,
        shape: polydata_type,
        parameter: Points,
        return_path: bool = False,
        return_regularization: bool = False,
    ) -> MorphingOutput:
        """Morph a shape using the kernel deformation algorithm.

        Parameters
        ----------
        shape
            The shape to morph.
        parameter
            The momentum.
        return_path
            True if you want to have access to the sequence of polydatas.
        return_regularization
            True to have access to the regularization.

        Returns
        -------
        MorphingOutput
            A named tuple containing the morphed shape, the regularization and
            the path if needed.
        """
        if parameter.device != shape.device:
            raise ValueError(
                "The shape and the parameter must be on the same device."
            )

        p = parameter
        p.requires_grad = True

        if self.control_points is False or shape.control_points is None:
            q = shape.points.clone()
            points = None
        else:
            points = shape.points.clone()
            q = shape.control_points.points.clone()
        q.requires_grad = True

        # Compute the regularization
        regularization = torch.tensor(0.0, device=shape.device)
        if return_regularization:
            regularization = self.cometric(parameter, q) / 2

        # Define the hamiltonian
        def H(p, q):
            K = self.cometric.operator(q, q)
            return torch.sum(p * (K @ p)) / 2

        if not return_path:
            path = None
        else:
            path = [shape.copy() for _ in range(self.n_steps + 1)]

        if self.backend == "torchdiffeq":

            def ode_func(t, y):
                if len(y) == 2:
                    p, q = y
                elif len(y) == 3:
                    p, q, pts = y

                Gp, Gq = torch.autograd.grad(
                    H(p, q), (p, q), create_graph=True
                )
                pdot, qdot = -Gq, Gp

                if len(y) == 2:
                    return pdot, qdot
                else:
                    Gp2 = self.cometric.operator(pts, q) @ p
                    ptsdot = Gp2
                    return pdot, qdot, ptsdot

            y_0 = (p, q) if points is None else (p, q, points)
            time = torch.linspace(0, 1, self.n_steps + 1).to(p.device)

            if points is None:
                (
                    path_p,
                    path_q,
                ) = odeint(func=ode_func, y0=y_0, t=time, method="rk4")
                path_pts = path_q  # points and control points are the same
            else:
                (
                    path_p,
                    path_q,
                    path_pts,
                ) = odeint(func=ode_func, y0=y_0, t=time, method="rk4")

            if return_path:
                for i in range(self.n_steps + 1):
                    path[i].points = path_pts[i]

            morphed_shape = shape.copy()
            morphed_shape.points = path_pts[-1]

        elif self.backend == "sks":
            dt = 1 / self.n_steps  # Time step

            if self.n_steps == 1:
                # If there is only one step, we can compute the transormation
                # directly without using the integrator as we do not need p_1
                K = self.cometric.operator(shape.points, q)
                points = shape.points + dt * (K @ p)
                if return_path:
                    path[1].points = points

            else:
                for i in range(self.n_steps):
                    if not self.control_points or shape.control_points is None:
                        # Update the points
                        # no control points -> q = shape.points
                        p, q = self.integrator(p, q, H, dt)
                        points = q.clone()
                    else:
                        # Update the points
                        K = self.cometric.operator(points, q)
                        points = points + dt * (K @ p)
                        p, q = self.integrator(p, q, H, dt)

                    if return_path:
                        path[i + 1].points = points

            morphed_shape = shape.copy()
            morphed_shape.points = points

        return MorphingOutput(
            morphed_shape=morphed_shape,
            path=path,
            regularization=regularization,
        )

    @typecheck
    def parameter_shape(self, shape: polydata_type) -> tuple[int, int]:
        """Return the shape of the parameter.

        Parameters
        ----------
        shape
            The shape to morph.

        Returns
        -------
        tuple[int, int]
            The shape of the parameter.
        """
        if not self.control_points or shape.control_points is None:
            return shape.points.shape
        else:
            return shape.control_points.points.shape
