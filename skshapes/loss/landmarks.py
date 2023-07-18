from typing import Any
import torch

from ..types import typecheck, Loss, PolyDataType, FloatScalar, Number


class LandmarkLoss(Loss):
    @typecheck
    def __init__(self, p: Number = 2) -> None:
        self.p = p

    @typecheck
    def __call__(self, source: PolyDataType, target: PolyDataType) -> FloatScalar:
        return torch.norm(source.landmarks_3d - target.landmarks_3d, p=self.p)
