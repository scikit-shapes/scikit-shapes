import torch
from .._typing import *


# class IntrinsicMetric:
#     @typecheck
#     def __init__(self, n_steps: int) -> None:
#         self.n_steps = n_steps

#     @typecheck
#     def morph(
#         self, parameter: float3dTensorType, return_path: bool = False
#     ) -> Union[float2dTensorType, float3dTensorType]:

#         N, D = self.source_points.shape

#         if return_path:

#             # Compute the cumulative sum of the velocity sequence
#             cumvelocities = torch.cat(
#                 (
#                     torch.zeros(size=(1, N, D), device=parameter.device),
#                     torch.cumsum(parameter, dim=0),
#                 )
#             )
#             # Compute the path by adding the cumulative sum of the velocity sequence to the initial shape
#             return (
#                 self.source_points.repeat(self.n_steps + 1, 1).reshape(
#                     self.n_steps + 1, N, D
#                 )
#                 + cumvelocities
#             )

#         else:
#             return self.source_points + torch.sum(parameter, dim=0)

#     @typecheck
#     def regularization(self, parameter: float3dTensorType) -> floatScalarType:
#         shape_sequence = self.morph(parameter, return_path=True)
#         reg = 0
#         for i in range(self.n_steps):
#             reg += self.metric(shape_sequence[i], parameter[i])
#         return reg / (2 * self.n_steps)

#     @property
#     @typecheck
#     def parameter_template(self) -> float3dTensorType:
#         return torch.zeros(
#             self.n_steps, *self.source_points.shape, device=self.source_points.device
#         )


# class ElasticMetric(IntrinsicMetric):
#     def __init__(self, **kwargs) -> None:
#         super().__init__(**kwargs)

#     @typecheck
#     def fit(self, *, source: PolyDataType):
#         assert hasattr(
#             source, "edges"
#         ), "The shape must have edges to use the as-isometric-as-possible metric"

#         self.edges_0 = source.edges[0]
#         self.edges_1 = source.edges[1]

#         self.source_points = source.points
#         return self

#     @typecheck
#     def metric(
#         self, points: pointsType, velocity: float2dTensorType
#     ) -> floatScalarType:
#         a1 = (
#             (velocity[self.edges_0] - velocity[self.edges_1])
#             * (points[self.edges_0] - points[self.edges_1])
#         ).sum(dim=1)
#         a2 = (
#             (velocity[self.edges_0] - velocity[self.edges_1])
#             * (points[self.edges_0] - points[self.edges_1])
#         ).sum(dim=1)

#         return torch.sum(a1 * a2)


# TODO : is it better to have an explicit regularization function or to have a regularization parameter in the morph function ?
class ElasticMetric:
    @typecheck
    def __init__(self, n_steps: int) -> None:

        self.n_steps = n_steps

    @typecheck
    def morph(
        self,
        shape: PolyDataType,
        parameter: float3dTensorType,
        return_path: bool = False,
        return_regularization: bool = False,
    ) -> MorphingOutput:

        ##### First, we compute the sequence of morphed points #####

        N, D = shape.points.shape
        # Compute the cumulative sum of the velocity sequence
        cumvelocities = torch.cat(
            (
                torch.zeros(size=(1, N, D), device=parameter.device),
                torch.cumsum(parameter, dim=0),
            )
        )
        # Compute the sequence of points by adding the cumulative sum of the velocity sequence to the initial shape
        newpoints = (
            shape.points.repeat(self.n_steps + 1, 1).reshape(self.n_steps + 1, N, D)
            + cumvelocities
        )

        ###### Then, we compute the morphed shape + regularization/path if needed #####

        # Compute the morphed shape
        morphed_shape = shape.copy()
        morphed_shape.points = newpoints[-1]

        # Compute the regularization value if needed (0 by default)
        regularization = torch.tensor(0.0, device=parameter.device)
        if return_regularization:
            regularization = self.metric(newpoints[:-1], shape.edges, parameter) / (
                2 * self.n_steps
            )

        # Compute the path if needed
        path = None
        if return_path:
            path = [shape.copy() for _ in range(self.n_steps + 1)]
            for i in range(self.n_steps + 1):
                path[i].points = newpoints[i]

        ###### Finally, we return the NamedTuple containing this information #####
        return MorphingOutput(
            morphed_shape=morphed_shape, path=path, regularization=regularization
        )

    @typecheck
    def metric(
        self, points: float3dTensorType, edges: edgesType, velocities: float3dTensorType
    ) -> floatScalarType:
        """Compute the sum of the norms of a sequence of speed vectors in Riemannian metric with respect to associated points.

        Args:
            velocity (float3dTensorType (n, N, d) ): The sequence of speed vectors ).
            points (float3dTensorType (n, N, d) ): The sequence of points.

        Returns:
            floatScalarType: The sum of the squared norm of the sequence of speed vectors in the Riemannian metric.

        """
        e0, e1 = edges

        a1 = (
            (velocities[:, e0, :] - velocities[:, e1, :])
            * (points[:, e0, :] - points[:, e1, :])
        ).sum(dim=2)

        return torch.sum(a1**2)

    @typecheck
    def parameter_shape(self, shape: PolyDataType) -> Tuple[int, int, int]:
        return (self.n_steps, *shape.points.shape)
