from ..types import (
    typecheck,
    FloatScalar,
    Edges,
    Triangles,
    Float3dTensor,
)
from typing import Optional
import torch


class Metric:
    pass


class ElasticMetric:
    def __init__(self) -> None:
        pass

    @typecheck
    def __call__(
        self,
        points_sequence: Float3dTensor,
        velocities_sequence: Float3dTensor,
        edges: Optional[Edges] = None,
        triangles: Optional[Triangles] = None,
    ) -> FloatScalar:
        if edges is None:
            raise TypeError("This metric requires edges to be defined")

        n_steps = points_sequence.shape[1]
        e0, e1 = edges[:, 0], edges[:, 1]
        a1 = (
            (velocities_sequence[e0] - velocities_sequence[e1])
            * (points_sequence[e0] - points_sequence[e1])
        ).sum(dim=2)

        return torch.sum(a1**2) / (2 * n_steps)
