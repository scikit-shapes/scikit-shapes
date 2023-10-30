import torch

from .basemodel import BaseModel

from ..types import (
    typecheck,
    Float3dTensor,
    FloatScalar,
    Edges,
    polydata_type,
    convert_inputs,
)

from .utils import MorphingOutput


class ElasticMetric(BaseModel):
    @typecheck
    def __init__(self, n_steps: int = 1) -> None:
        self.n_steps = n_steps

    @convert_inputs
    @typecheck
    def morph(
        self,
        shape: polydata_type,
        parameter: Float3dTensor,
        return_path: bool = False,
        return_regularization: bool = False,
    ) -> MorphingOutput:
        if parameter.device != shape.device:
            parameter = parameter.to(shape.device)

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
        # Compute the sequence of points by adding the cumulative sum of the velocity sequence to the initial shape
        newpoints = (
            shape.points.repeat(self.n_steps + 1, 1)
            .reshape(self.n_steps + 1, n_points, d)
            .permute(1, 0, 2)
            + cumvelocities
        )

        ###### Then, we compute the morphed shape + regularization/path if needed #####

        # Compute the morphed shape
        morphed_shape = shape.copy()
        morphed_shape.points = newpoints[:, -1]

        # Compute the regularization value if needed (0 by default)
        regularization = torch.tensor(0.0, device=shape.device)
        if return_regularization:
            regularization = self.metric(
                newpoints[:, :-1, :], shape.edges, parameter
            ) / (2 * self.n_steps)

        # Compute the path if needed
        path = None
        if return_path:
            path = [shape.copy() for _ in range(self.n_steps + 1)]
            for i in range(self.n_steps + 1):
                path[i].points = newpoints[:, i, :]

        assert parameter.shape == self.parameter_shape(shape)

        ###### Finally, we return the NamedTuple containing this information #####
        return MorphingOutput(
            morphed_shape=morphed_shape,
            path=path,
            regularization=regularization,
        )

    @convert_inputs
    @typecheck
    def metric(
        self, points: Float3dTensor, edges: Edges, velocities: Float3dTensor
    ) -> FloatScalar:
        """Compute the sum of the norms of a sequence of speed vectors in Riemannian metric with respect to associated points.

        Args:
            velocity (float3dTensorType (n_points, n_steps, d) ): The sequence of speed vectors ).
            points (float3dTensorType (n_points, n_steps, d) ): The sequence of points.

        Returns:
            floatScalarType: The sum of the squared norm of the sequence of speed vectors in the Riemannian metric.

        """
        e0, e1 = edges[:, 0], edges[:, 1]

        a1 = (
            (velocities[e0] - velocities[e1]) * (points[e0] - points[e1])
        ).sum(dim=2)

        return torch.sum(a1**2)

    @typecheck
    def parameter_shape(self, shape: polydata_type) -> tuple[int, int, int]:
        n_points = shape.points.shape[0]
        return (n_points, self.n_steps, 3)
