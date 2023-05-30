import torch
from .._typing import *


class IntrinsicMetric:
    @typecheck
    def __init__(self, n_steps: int) -> None:
        self.n_steps = n_steps

    @typecheck
    def morph(
        self, parameter: float3dTensorType, return_path: bool = False
    ):  # -> Union[float2dTensorType, float3dTensorType]:

        N, D = self.source_points.shape

        if return_path:

            # Compute the cumulative sum of the velocity sequence
            cumvelocities = torch.cat(
                (
                    torch.zeros(size=(1, N, D), device=parameter.device),
                    torch.cumsum(parameter, dim=0),
                )
            )
            # Compute the path by adding the cumulative sum of the velocity sequence to the initial shape
            return (
                self.source_points.repeat(self.n_steps + 1, 1).reshape(
                    self.n_steps + 1, N, D
                )
                + cumvelocities
            )

        else:
            return self.source_points + torch.sum(parameter, dim=0)

    @typecheck
    def regularization(self, parameter: float3dTensorType) -> floatScalarType:
        shape_sequence = self.morph(parameter, return_path=True)
        reg = 0
        for i in range(self.n_steps):
            reg += self.metric(shape_sequence[i], parameter[i])
        return reg / (2 * self.n_steps)

    @property
    @typecheck
    def parameter_template(self) -> float3dTensorType:
        return torch.zeros(
            self.n_steps, *self.source_points.shape, device=self.source_points.device
        )


class ElasticMetric(IntrinsicMetric):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    @typecheck
    def fit(self, *, source: PolyDataType):
        assert hasattr(
            source, "edges"
        ), "The shape must have edges to use the as-isometric-as-possible metric"

        self.edges_0 = source.edges[0]
        self.edges_1 = source.edges[1]

        self.source_points = source.points
        return self

    @typecheck
    def metric(
        self, points: pointsType, velocity: float2dTensorType
    ) -> floatScalarType:
        a1 = (
            (velocity[self.edges_0] - velocity[self.edges_1])
            * (points[self.edges_0] - points[self.edges_1])
        ).sum(dim=1)
        a2 = (
            (velocity[self.edges_0] - velocity[self.edges_1])
            * (points[self.edges_0] - points[self.edges_1])
        ).sum(dim=1)

        return torch.sum(a1 * a2)


class ElasticMetric2:
    @typecheck
    def __init__(self, n_steps: int) -> None:

        self.n_steps = n_steps

    @typecheck
    def morph(
        self,
        shape: PolyDataType,
        parameter: Union[float2dTensorType, float3dTensorType],
        return_path: bool = False,
    ) -> Union[PolyDataType, List[PolyDataType]]:

        N, D = shape.points.shape

        if return_path:

            if self.n_steps == 1:
                # In this case, we only need to return the initial shape and the final shape
                return [shape.copy(), self.morph(shape, parameter, return_path=False)]

            else:
                # Compute the cumulative sum of the velocity sequence
                cumvelocities = torch.cat(
                    (
                        torch.zeros(size=(1, N, D), device=parameter.device),
                        torch.cumsum(parameter, dim=0),
                    )
                )
                # Compute the path by adding the cumulative sum of the velocity sequence to the initial shape
                newpoints = (
                    shape.points.repeat(self.n_steps + 1, 1).reshape(
                        self.n_steps + 1, N, D
                    )
                    + cumvelocities
                )
                newshapes = [shape.copy() for _ in range(self.n_steps + 1)]
                for i in range(self.n_steps + 1):
                    newshapes[i].points = newpoints[i]

                return newshapes

        else:
            if self.n_steps == 1:
                newpoints = shape.points + parameter
            else:
                newpoints = shape.points + torch.sum(parameter, dim=0)

            newshape = shape.copy()
            newshape.points = newpoints
            return newshape

    @typecheck
    def metric(
        self, shape: PolyDataType, velocity: float2dTensorType
    ) -> floatScalarType:

        e0, e1 = shape.edges

        a1 = (
            (velocity[e0] - velocity[e1]) * (shape.points[e0] - shape.points[e1])
        ).sum(dim=1)
        a2 = (
            (velocity[e0] - velocity[e1]) * (shape.points[e0] - shape.points[e1])
        ).sum(dim=1)

        return torch.sum(a1 * a2)

    @typecheck
    def regularization(
        self,
        shape: PolyDataType,
        parameter: Union[float2dTensorType, float3dTensorType],
    ) -> floatScalarType:

        if self.n_steps == 1:
            return self.metric(shape, parameter)
        else:

            shape_sequence = self.morph(shape, parameter, return_path=True)
            reg = 0
            for i in range(self.n_steps):
                reg += self.metric(shape_sequence[i], parameter[i])
            return reg / (2 * self.n_steps)
