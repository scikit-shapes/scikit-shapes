"""Vector field deformation model.

This module contains the implementation of the vector field deformation model.
This model is described by a sequence of speed vectors, which are summed
to obtain the sequence of points of the morphed shape. The morphing is
regularized by a Riemannian metric on the shape space.
"""

from typing import Optional

import torch

from ..errors import DeviceError
from ..input_validation import convert_inputs, typecheck
from ..types import (
    Float3dTensor,
    MorphingOutput,
    polydata_type,
)
from .basemodel import BaseModel
from .metrics import AsIsometricAsPossible, Metric


class IntrinsicDeformation(BaseModel):
    """Vector field deformation model."""

    @typecheck
    def __init__(
        self, n_steps: int = 1, metric: Optional[Metric] = None
    ) -> None:
        """Initialize the model.

        Parameters
        ----------
        n_steps
            Number of integration steps.
        metric
            Riemannian metric used to regularize the morphing.

        """
        if metric is None:
            metric = AsIsometricAsPossible()

        self.n_steps = n_steps
        self.metric = metric

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
            raise DeviceError(
                msg
            )

        assert parameter.shape == self.parameter_shape(shape)

        ##### First, we compute the sequence of morphed points #####

        n_points, d = shape.points.shape
        # Compute the cumulative sum of the velocity sequence

        cumvelocities = torch.concatenate(
            (
                torch.zeros(size=(n_points, 1, d), device=shape.device),
                torch.cumsum(parameter.to(shape.device), dim=1),
            ),
            dim=1,
        )
        # Compute the sequence of points by adding the cumulative sum of the
        # velocity sequence to the initial shape
        newpoints = (
            shape.points.repeat(self.n_steps + 1, 1)
            .reshape(self.n_steps + 1, n_points, d)
            .permute(1, 0, 2)
            + cumvelocities
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
                velocities_sequence=parameter,
                edges=shape.edges,
                triangles=shape.triangles,
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
        return (n_points, self.n_steps, dim)
