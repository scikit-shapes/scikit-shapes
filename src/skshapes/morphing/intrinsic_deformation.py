"""Vector field deformation model.

This module contains the implementation of the vector field deformation model.
This model is described by a sequence of speed vectors, which are summed
to obtain the sequence of points of the morphed shape. The morphing is
regularized by a Riemannian metric on the shape space.
"""

from typing import Literal

import torch

from ..errors import DeviceError
from ..input_validation import convert_inputs, typecheck
from ..types import (
    Edges,
    Float3dTensor,
    FloatScalar,
    MorphingOutput,
    Number,
    Points,
    Triangles,
    polydata_type,
)
from .basemodel import BaseModel


class IntrinsicDeformation(BaseModel):
    """Vector field deformation model.

    Parameters
    ----------
    n_steps
        Number of integration steps.
    metric
        Riemannian metric used to regularize the morphing.
    endpoints
        The endpoints of the morphing. If None, the endpoints are not fixed and
        the morphed shape is free to move. If not None, the morphed shape is
        constrained to be at the endpoints and the only free steps are the
        intermediate steps. Providing endpoints is useful to minimize the
        energy of the morphing while keeping the endpoints fixed.
    **kwargs
        Additional keyword arguments.
    """

    @typecheck
    def __init__(
        self,
        n_steps: int = 1,
        metric: Literal[
            "as_isometric_as_possible", "shell_energy", "H2"
        ] = "as_isometric_as_possible",
        endpoints: None | Points = None,
        **kwargs,
    ) -> None:

        self.n_steps = n_steps

        if metric == "as_isometric_as_possible":
            self.metric_kwargs = {}
            self.metric = as_isometric_as_possible

        elif metric == "shell_energy":
            self.metric_kwargs = {
                "bending_weight": kwargs.get("bending_weight", 0.001)
            }
            self.metric = shell_energy_metric

        elif metric == "H2":
            self.metric_kwargs = {}
            self.metric = H2_energy_metric

        if endpoints is not None:
            self.fixed_endpoints = True
            self.endpoints = endpoints
        else:
            self.fixed_endpoints = False

    @convert_inputs
    @typecheck
    def morph(
        self,
        shape: polydata_type,
        parameter: Float3dTensor,
        return_path: bool = False,
        return_regularization: bool = False,
    ) -> MorphingOutput:
        """Morph a shape.

        Parameters
        ----------
        shape
            The shape to morph.
        parameter
            Sequence of velocity vectors.
        return_path
            True if you want to have access to the morphing's sequence of
            polydatas.
        return_regularization
            True to have access to the regularization.

        Returns
        -------
        MorphingOutput
            A named tuple containing the morphed shape, the regularization and
            the path if requested.
        """
        if parameter.device != shape.device:
            msg = "The shape and the parameter must be on the same device."
            raise DeviceError(msg)

        if self.fixed_endpoints and self.endpoints.device != shape.device:
            self.endpoints = self.endpoints.detach().to(shape.device)

        if self.fixed_endpoints and shape.points.shape != self.endpoints.shape:
            msg = "The endpoints must have the same dimension as the shape to morph."
            raise ValueError(msg)

        assert parameter.shape == self.parameter_shape(shape)

        ##### First, we compute the sequence of morphed points #####

        n_points, d = shape.points.shape
        # Compute the cumulative sum of the velocity sequence

        cumulative_velocities = torch.concatenate(
            (
                torch.zeros(size=(n_points, 1, d), device=shape.device),
                torch.cumsum(parameter.to(shape.device), dim=1),
            ),
            dim=1,
        )
        # Compute the sequence of points by adding the cumulative sum of the
        # velocity sequence to the initial shape
        newpoints = (
            shape.points.repeat(self.n_free_steps + 1, 1)
            .reshape(self.n_free_steps + 1, n_points, d)
            .permute(1, 0, 2)
            + cumulative_velocities
        )

        # If we have fixed endpoints, we add the endpoints to the sequence
        if self.fixed_endpoints:
            last_velocity = self.endpoints - newpoints[:, -1, :]
            newpoints = torch.cat(
                (newpoints, self.endpoints.unsqueeze(1)), dim=1
            )

        velocities = (
            parameter
            if not self.fixed_endpoints
            else torch.cat((parameter, last_velocity.unsqueeze(1)), dim=1)
        )

        # Then, we compute the morphed shape + regularization/path if needed

        # Compute the morphed shape
        morphed_shape = shape.copy()
        morphed_shape.points = newpoints[:, -1]

        # Compute the regularization value if needed (0 by default)
        regularization = torch.tensor(0.0, device=shape.device)
        if return_regularization:
            regularization = self.metric(
                points_sequence=newpoints[:, :-1, :],
                velocities_sequence=velocities,
                edges=shape.edges,
                triangles=shape.triangles,
                **self.metric_kwargs,
            )

        # Compute the path if needed
        path = None
        if return_path:
            path = [shape.copy() for _ in range(self.n_steps + 1)]
            for i in range(self.n_steps + 1):
                path[i].points = newpoints[:, i, :]

        assert parameter.shape == self.parameter_shape(shape)

        # Finally, we return the NamedTuple containing this information
        return MorphingOutput(
            morphed_shape=morphed_shape,
            path=path,
            regularization=regularization,
        )

    @typecheck
    def parameter_shape(self, shape: polydata_type) -> tuple[int, int, int]:
        """Return the shape of the parameter.

        Parameters
        ----------
        shape
            The shape to morph.

        Returns
        -------
            The shape of the parameter.

        """
        n_points = shape.points.shape[0]
        dim = shape.dim
        return (n_points, self.n_free_steps, dim)

    @typecheck
    @property
    def n_free_steps(self) -> int:
        """Number of integration steps."""
        if self.fixed_endpoints:
            return self.n_steps - 1
        return self.n_steps


def as_isometric_as_possible(
    points_sequence: Float3dTensor,
    velocities_sequence: Float3dTensor,
    edges: Edges | None = None,
    triangles: Triangles | None = None,  # noqa: ARG001
) -> FloatScalar:
    """
    As isometric as possible metric.

    This metric penalizes the non-isometricity of the mesh.

    Compute the mean velocities' metric along the sequence of points.

    Parameters
    ----------
    points_sequence
        Sequence of points
    velocities_sequence
        Sequence of velocities.
    edges
        Edges.
    triangles
        Triangles.

    Raises
    ------
        AttributeError: This metric requires edges to be defined

    Returns
    -------
        FloatScalar: the mean velocities metric

    References
    ----------
    [KILIAN, Martin, MITRA, Niloy J., et POTTMANN, Helmut. Geometric
    modeling in shape space. In : ACM SIGGRAPH 2007 papers. 2007. p. 64-es.](https://dmg.tuwien.ac.at/geom/ig/publications/oldpub/2007/kilian-2007-gmss/paper_docs/shape_space_sig_07.pdf)

    """
    if edges is None:
        msg = "This metric requires edges to be defined"
        raise AttributeError(msg)

    next_points_sequence = points_sequence + velocities_sequence

    n_steps = points_sequence.shape[1]
    e0, e1 = edges[:, 0], edges[:, 1]
    a1 = (
        (velocities_sequence[e0] - velocities_sequence[e1])
        * (points_sequence[e0] - points_sequence[e1])
    ).sum(dim=2)

    a2 = (
        (velocities_sequence[e0] - velocities_sequence[e1])
        * (next_points_sequence[e0] - next_points_sequence[e1])
    ).sum(dim=2)

    scale = (points_sequence[e0] - points_sequence[e1]).norm(dim=2).mean()

    L2 = (
        ((velocities_sequence[e0] - velocities_sequence[e1]) ** 2)
        .sum(dim=2)
        .mean()
    )

    return torch.sum(a1**2 + a2**2 + 0.001 * L2) / (2 * n_steps * (scale**4))


def shell_energy_metric(
    points_sequence: Float3dTensor,
    velocities_sequence: Float3dTensor,
    edges: Edges | None = None,  # noqa: ARG001
    triangles: Triangles | None = None,
    bending_weight: Number = 0.001,
) -> FloatScalar:
    """Shell energy metric.

    This metric is the sum of a membrane contribution, measuring the local
    stretching and a bending contribution, measuring the change in curvature.
    The parameter weight controls the importance of the bending energy.

    Parameters
    ----------
    points_sequence
        Sequence of points
    velocities_sequence
        Sequence of velocities.
    edges
        Edges.
    triangles
        Triangles.
    bending_weight
        The weight of the bending energy.

    Raises
    ------
        AttributeError: This metric requires triangles to be defined

    Returns
    -------
        FloatScalar: the mean velocities metric

    References
    ----------
    [HEEREN, Behrend, RUMPF, Martin, SCHRÖDER, Peter, et al. Exploring the
    geometry of the space of shells. In : Computer Graphics Forum. 2014.
    p. 247-256.](https://ddg.math.uni-goettingen.de/pub/HeRuSc14.pdf)
    """
    if triangles is None:
        msg = "This metric requires triangles to be defined"
        raise AttributeError(msg)

    n_steps = points_sequence.shape[1]

    points_undef = points_sequence[:, 0:n_steps, :]
    points_def = points_undef + velocities_sequence

    from ..triangle_mesh import shell_energy

    return shell_energy(
        points_undef=points_undef,
        points_def=points_def,
        triangles=triangles,
        weight=bending_weight,
    ).mean()


def H2_energy_metric(
    points_sequence: Float3dTensor,
    velocities_sequence: Float3dTensor,  # noqa: ARG001
    edges: Edges | None = None,  # noqa: ARG001
    triangles: Triangles | None = None,
) -> FloatScalar:

    if triangles is None:
        msg = "This metric requires triangles to be defined"
        raise AttributeError(msg)

    from ..triangle_mesh import H2_energy

    return H2_energy(
        points_sequence=points_sequence,
        triangles=triangles,
    )
