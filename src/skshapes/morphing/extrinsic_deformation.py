"""Kernel deformation morphing algorithm.

This algorithm describes the morphing as a deformation of the ambiant space.
The parameter is a vector field, which is referred to as the momentum. This
momentum is smoothed by a kernel, and the morphed shape is obtained by
integrating the momentum. The regularization is given by <p, K_q p> where
p is the momentum, q is the initial position of the shape and K_q is the kernel
at q.
"""

from __future__ import annotations

from typing import Literal

import torch
from torchdiffeq import odeint

from ..convolutions import LinearOperator
from ..convolutions.squared_distances import squared_distances
from ..errors import DeviceError, ShapeError
from ..input_validation import typecheck
from ..types import (
    FloatScalar,
    MorphingOutput,
    Number,
    NumericalTensor,
    Points,
    polydata_type,
)
from .basemodel import BaseModel


class ExtrinsicDeformation(BaseModel):
    """Kernel deformation morphing algorithm.

    Parameters
    ----------
    n_steps
        Number of integration steps.
    kernel
        Type of kernel.
    scale
        Scale of the kernel.
    normalization
        Normalization of the kernel matrix.
    control_points
        If True, the control points are the control points of the shape
        (accessible with `shape.control_points`). If False, the control
        points are the points of the shape. If `shape.control_points` is
        None, the control points are the points of the shape.
    solver
        The solver to use for the integration. Can be "euler", "midpoint" or
        "rk4".
    """

    @typecheck
    def __init__(
        self,
        n_steps: int = 1,
        kernel: Literal["gaussian", "uniform"] = "gaussian",
        scale: Number = 0.1,
        normalization: Literal["rows", "columns", "both"] | None = None,
        control_points: bool = False,
        solver: Literal["euler", "midpoint", "rk4"] = "euler",
    ) -> None:
        """Class constructor."""
        self.n_steps = n_steps
        self.control_points = control_points
        self.solver = solver

        self.kernel = kernel
        self.scale = scale
        self.normalization = normalization

        self.ode_module = ODEModule(self.ode_func)

    @typecheck
    def morph(
        self,
        shape: polydata_type,
        parameter: Points,
        return_path: bool = False,
        return_regularization: bool = False,
        final_time: float = 1.0,
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
        final_time
            The final time of the integration. Default is 1.0, it can be set to
            a different value for extrapolation.

        Returns
        -------
        MorphingOutput
            A named tuple containing the morphed shape, the regularization and
            the path if needed.
        """
        if parameter.device != shape.device:
            msg = "The shape and the parameter must be on the same device."
            raise DeviceError(msg)

        if parameter.shape != self.parameter_shape(shape):
            msg = (
                "The shape of the parameter is not correct. "
                f"Expected {self.parameter_shape(shape)}, "
                f"got {parameter.shape}."
            )
            raise ShapeError(msg)

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
            regularization = self.H(p, q)

        if not return_path:
            path = None
        else:
            path = [shape.copy() for _ in range(self.n_steps + 1)]

        if self.n_steps == 1:
            # If there is only one step, we can compute the transformation
            # directly without using the integrator as we do not need p_1
            K = self.K(shape.points, q)
            points = shape.points + (K @ p)
            if return_path:
                path[1].points = points

        else:
            y_0 = (p, q) if points is None else (p, q, points)
            time = torch.linspace(0, final_time, self.n_steps + 1).to(p.device)

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

        morphed_shape = shape.copy()
        morphed_shape.points = points

        # If control points are used, save the final control points
        # as control points of the morphed shape
        if self.control_points and shape.control_points is not None:
            q = q + self.K(q, q) @ p if self.n_steps == 1 else path_q[-1]
            control_points = shape.control_points.copy()
            control_points.points = q.detach().clone()
            morphed_shape.control_points = control_points

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
        K = self.K(q, q)
        return torch.sum(p * (K @ p)) / 2

    @typecheck
    def ode_func(self, t: float, y: type_y) -> type_y:  # noqa: ARG002
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
            Gp2 = self.K(pts, q) @ p
            ptsdot = Gp2
            return pdot, qdot, ptsdot

    @typecheck
    def K(
        self,
        q0: Points,
        q1: Points | None = None,
        weights0: NumericalTensor | None = None,
        weights1: NumericalTensor | None = None,
    ) -> LinearOperator:

        if q1 is None:
            q1 = q0

        # Compute the kernel matrix
        if self.kernel == "gaussian":

            sqrt_2 = 1.41421356237
            q0 = q0 / (sqrt_2 * self.scale)
            q1 = q1 / (sqrt_2 * self.scale)

            K = squared_distances(
                points=q1,
                target_points=q0,
                kernel=lambda d2: (-d2).exp(),
            )

        elif self.kernel == "uniform":

            q0 = q0 / self.scale
            q1 = q1 / self.scale

            K = squared_distances(
                points=q1,
                target_points=q0,
                kernel=lambda d2: (d2 < 1),
            )

        # If applicable, normalize the kernel
        if self.normalization == "rows":
            sq = K.sum(dim=1).view(-1)
            weights0 = (
                weights0 if weights0 is not None else torch.ones_like(sq)
            )
            return LinearOperator(matrix=K, output_scaling=weights0 / sq)

        elif self.normalization == "columns":
            sq = K.sum(dim=0).view(-1)
            weights1 = (
                weights1 if weights1 is not None else torch.ones_like(sq)
            )
            return LinearOperator(matrix=K, input_scaling=weights1 / sq)

        elif self.normalization == "both":

            if K.shape[0] == K.shape[1]:
                m = (
                    weights0
                    if weights0 is not None
                    else torch.ones(
                        K.shape[0], dtype=q0.dtype, device=q0.device
                    )
                )
                sq = torch.ones(K.shape[1], dtype=q0.dtype, device=q0.device)

                for _i in range(5):
                    sq = (sq / ((K @ sq) * m)).sqrt()

                return LinearOperator(
                    input_scaling=sq, matrix=K, output_scaling=sq
                )

            else:
                error_message = "The 'both' normalization can only be used with square matrices."
                raise ValueError(error_message)

        return LinearOperator(matrix=K)


# Type for the ODE function, can be a couple of tensors (moment, points) or
# a triple (moment, control_points, points)
type_y = tuple[Points, Points] | tuple[Points, Points, Points]


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
