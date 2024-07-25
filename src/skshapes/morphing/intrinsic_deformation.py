"""Vector field deformation model.

This module contains the implementation of the vector field deformation model.
This model is described by a sequence of speed vectors, which are summed
to obtain the sequence of points of the morphed shape. The morphing is
regularized by a Riemannian metric on the shape space.
"""

from collections.abc import Callable
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
    use_stiff_edges
        If the source PolyData has a `stiff_edges` property and this argument
        is `True`, the `stiff_edges` are passed to the metric. If the source
        PolyData has no `stiff_edges`, `edges are passed by default.`
    **kwargs
        Additional keyword arguments.
    """

    @typecheck
    def __init__(
        self,
        n_steps: int = 1,
        metric: (
            Literal["as_isometric_as_possible", "shell_energy"] | Callable
        ) = "as_isometric_as_possible",
        endpoints: None | Points = None,
        use_stiff_edges: bool = True,
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

        elif callable(metric):
            self.metric_kwargs = {}
            metric_validation(metric)
            self.metric = metric

        if endpoints is not None:
            self.fixed_endpoints = True
            self.endpoints = endpoints
        else:
            self.fixed_endpoints = False
            self.endpoints = None

        self.use_stiff_edges = use_stiff_edges

        self.copy_features = [
            "n_steps",
            "metric",
            "endpoints",
            "use_stiff_edges",
            "metric_kwargs",
        ]

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

        # Choose edges regarding the use_stiff_edges argument
        if self.use_stiff_edges and shape.stiff_edges is not None:
            edges = shape.stiff_edges

        else:
            edges = shape.edges

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
                edges=edges,
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
        """Number of integration steps.

        If the endpoints are fixed, the number of free steps is n_steps - 1 as
        the last step is fixed by the endpoints. Otherwise, the number of free
        steps is n_steps.
        """
        if self.fixed_endpoints:
            return self.n_steps - 1
        return self.n_steps


@typecheck
def metric_validation(metric: Callable) -> None:
    """Test the validity of a callable metric.

    Parameters
    ----------
    metric
        The metric to test.

    Raises
    -------
    ValueError
        If the metric has one of the following issues:
        - The metric does not have the expected arguments (points_sequence,
        velocities_sequence, edges, triangles).
        - The metric does not return a `torch.Tensor`.
        - The metric does not return a scalar.
        - The metric is not differentiable wrt the velocities.
    """
    from inspect import signature

    # test the arguments names
    fct_signature = signature(metric)

    expected_args = [
        "points_sequence",
        "velocities_sequence",
        "edges",
        "triangles",
    ]

    for arg in expected_args:
        if arg not in fct_signature.parameters:

            msg = (
                f"The metric must have the following arguments: "
                f"{', '.join(expected_args)}. The argument {arg} is missing."
            )
            raise ValueError(msg)

    # Create a set of random points, triangles, edges and velocities
    n_steps = 3
    n_points = 10
    dim = 3
    n_triangles = 12
    n_edges = 15
    points_sequence = torch.rand(n_points, n_steps, dim)
    triangles = torch.randint(0, n_points, (n_triangles, 3))
    edges = torch.randint(0, n_points, (n_edges, 2))
    velocities_sequence = torch.rand(n_points, n_steps, dim)
    velocities_sequence.requires_grad = (
        True  # We need to compute the gradient wrt the velocities
    )

    # Compute the metric
    a = metric(
        points_sequence=points_sequence,
        velocities_sequence=velocities_sequence,
        edges=edges,
        triangles=triangles,
    )

    # Assert that a is a scalar
    if not torch.is_tensor(a):
        msg = "The metric must return a tensor."
        raise ValueError(msg)

    if a.shape != ():
        msg = "The metric must return a scalar."
        raise ValueError(msg)

    # Try to compute the gradient wrt the velocities, it must not raise an error
    try:
        torch.autograd.grad(a, velocities_sequence)
    except RuntimeError as err:
        msg = "The metric must be differentiable wrt the velocities."
        raise ValueError(msg) from err


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

    n_points, n_steps, dim = points_sequence.shape
    assert velocities_sequence.shape == (n_points, n_steps, dim)

    next_points_sequence = points_sequence + velocities_sequence

    # e0 and e1 are (n_edges,)
    n_edges = edges.shape[0]
    assert edges.shape == (n_edges, 2)
    e0, e1 = edges[:, 0], edges[:, 1]

    # Implement Eqs. (4-6) in Kilian et al. 2007:
    # - velocities_sequence[i] is X_i
    # - points_sequence[i] is P_k[i]
    # - next_points_sequence[i] is P_(k+1)[i]

    # Compute the left-most term in Eq. (6), << X_i, X_i >>_{P_i}
    a1 = (
        (velocities_sequence[e0] - velocities_sequence[e1])  # X_p - X_q
        * (points_sequence[e0] - points_sequence[e1])  # p - q
    ).sum(
        dim=2
    ) ** 2  # Compute the dot product, and don't forget to square it
    assert a1.shape == (n_edges, n_steps)

    # Compute the right-most term in Eq. (6), << X_i, X_i >>_{P_(i+1)}
    a2 = (
        (velocities_sequence[e0] - velocities_sequence[e1])  # X_p - X_q
        * (
            next_points_sequence[e0] - next_points_sequence[e1]
        )  # p - q at the next step
    ).sum(
        dim=2
    ) ** 2  # Compute the dot product, and don't forget to square it
    assert a2.shape == (n_edges, n_steps)

    scale = 1  # (points_sequence[e0] - points_sequence[e1]).norm(dim=2).mean()

    # Compute Eq. (5)
    L2 = (velocities_sequence**2).sum(dim=2)
    assert L2.shape == (n_points, n_steps)
    # Currently, for L2, we use simple weights that sum up to 1 instead of the proper A_p
    # TODO: fix it

    return ((a1 + a2).sum() + 0.001 * L2.sum() / n_points) / (
        2 * n_steps * (scale**4)
    )


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
