"""Kernel deformation morphing algorithm.

This algorithm describes the morphing as a deformation of the ambiant space.
The parameter is a vector field, which is referred to as the momentum. This
momentum is smoothed by a kernel, and the morphed shape is obtained by
integrating the momentum. The regularization is given by <p, K_q p> where
p is the momentum, q is the initial position of the shape and K_q is the kernel
at q.
"""

from __future__ import annotations

import torch
from torchdiffeq import odeint as odeint
from typing import Optional, Literal, Union

from .basemodel import BaseModel
from ..types import (
    Points,
    polydata_type,
    MorphingOutput,
    FloatScalar,
)
from ..input_validation import typecheck
from ..errors import DeviceError

from .utils import Integrator, EulerIntegrator
from .kernels import Kernel, GaussianKernel


class ExtrinsicDeformation(BaseModel):
    """Kernel deformation morphing algorithm.

    Parameters
    ----------
    n_steps
        Number of integration steps.
    integrator
        Integrator used to integrate the momentum. If None, an Euler scheme
        is used. See `skshapes.morphing.utils.Integrator` for more
        information. Only used if `backend` is "sks".
    kernel
        Kernel used to smooth the momentum.
    control_points
        If True, the control points are the control points of the shape
        (accessible with `shape.control_points`). If False, the control
        points are the points of the shape. If `shape.control_points` is
        None, the control points are the points of the shape.
    backend
        The backend to use for the integration. Can be "sks" or
        "torchdiffeq".
    solver
        The solver to use for the integration if `backend` is
        "torchdiffeq". Can be "euler", "midpoint" or "rk4".
    """

    @typecheck
    def __init__(
        self,
        n_steps: int = 1,
        integrator: Optional[Integrator] = None,
        kernel: Optional[Kernel] = None,
        control_points: bool = True,
        backend: Literal["sks", "torchdiffeq"] = "sks",
        solver: Literal["euler", "midpoint", "rk4"] = "euler",
        **kwargs,
    ) -> None:
        """Class constructor."""
        if integrator is None:
            integrator = EulerIntegrator()
        if kernel is None:
            kernel = GaussianKernel()

        self.integrator = integrator
        self.cometric = kernel
        self.n_steps = n_steps
        self.control_points = control_points
        self.backend = backend
        self.solver = solver

        self.ode_module = ODEModule(self.ode_func)

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
            raise DeviceError(
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
            points.requires_grad = True
        q.requires_grad = True

        # Compute the regularization
        regularization = torch.tensor(0.0, device=shape.device)
        if return_regularization:
            regularization = self.cometric(parameter, q) / 2

        if not return_path:
            path = None
        else:
            path = [shape.copy() for _ in range(self.n_steps + 1)]

        dt = 1 / self.n_steps  # Time step
        if self.n_steps == 1:
            # If there is only one step, we can compute the transformation
            # directly without using the integrator as we do not need p_1
            K = self.cometric.operator(shape.points, q)
            points = shape.points + (K @ p)
            if return_path:
                path[1].points = points

        else:
            if self.backend == "torchdiffeq":
                y_0 = (p, q) if points is None else (p, q, points)
                time = torch.linspace(0, 1, self.n_steps + 1).to(p.device)

                if len(y_0) == 3:
                    (
                        path_p,
                        path_q,
                        path_pts,
                    ) = odeint(
                        func=self.ode_module,
                        y0=y_0,
                        t=time,
                        method=self.solver,
                    )
                    points = path_pts[-1].clone()
                    if return_path:
                        for i, m in enumerate(path):
                            m.points = path_pts[i].clone()

                if len(y_0) == 2:
                    (
                        path_p,
                        path_q,
                    ) = odeint(
                        func=self.ode_module,
                        y0=y_0,
                        t=time,
                        method=self.solver,
                    )

                    # Update the points
                    points = path_q[-1].clone()
                    if return_path:
                        for i, m in enumerate(path):
                            m.points = path_q[i].clone()

            elif self.backend == "sks":
                dt = 1 / self.n_steps  # Time step

                for i in range(self.n_steps):
                    if not self.control_points or shape.control_points is None:
                        # Update the points
                        # no control points -> q = shape.points
                        p, q = self.integrator(p, q, self.H, dt)
                        points = q.clone()
                    else:
                        # Update the points
                        K = self.cometric.operator(points, q)
                        points = points + dt * (K @ p)
                        p, q = self.integrator(p, q, self.H, dt)

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

    @typecheck
    def H(self, p: Points, q: Points) -> FloatScalar:
        """Hamiltonian function."""
        K = self.cometric.operator(q, q)
        return torch.sum(p * (K @ p)) / 2

    @typecheck
    def ode_func(self, t: float, y: type_y) -> type_y:
        """ODE function."""
        if len(y) == 2:
            p, q = y
        elif len(y) == 3:
            p, q, pts = y

        Gp, Gq = torch.autograd.grad(self.H(p, q), (p, q), create_graph=True)

        pdot = -Gq
        qdot = Gp

        if len(y) == 2:
            return pdot, qdot
        else:
            Gp2 = self.cometric.operator(pts, q) @ p
            ptsdot = Gp2
            return pdot, qdot, ptsdot


# Type for the ODE function, can be a couple of tensors (moment, points) or
# a triple (moment, control_points, points)
type_y = Union[tuple[Points, Points], tuple[Points, Points, Points]]


class ODEModule(torch.nn.Module):
    """Define the ODE function as a module for torchdiffeq.

    Wrap the ODE function in a torch.nn.Module module to be used with
    torchdiffeq.

    Parameters
    ----------
    func
        The function that defines the ODE, see th documentation of
        [torchdiffeq](https://github.com/rtqichen/torchdiffeq).
    """

    def __init__(self, func: callable) -> None:
        """Class constructor."""
        super().__init__()
        self.func = func

    def __call__(self, t: float, y: type_y) -> type_y:
        """Call the ODE function."""
        return self.func(t, y)
