from __future__ import annotations
from ..types import typecheck, Number, FloatScalar, Any, float_dtype

from typing import List
from ..data import Shape

import torch


class BaseLoss:
    @typecheck
    def __add__(self, other: BaseLoss) -> BaseLoss:
        """Add two losses

        Args:
            other (Loss): the other loss to add

        Returns:
            Loss: a new loss which __call__ method is the sum of the two __call__ methods
        """
        loss1 = self
        loss2 = other

        return SumLoss(loss1=loss1, loss2=loss2)

    @typecheck
    def __rmul__(self, scalar: Number) -> BaseLoss:
        """Multiply a loss by a scalar

        Args:
            other (Number): the scalar to multiply the loss by

        Returns:
            Loss: a new loss which __call__ method is the product of the scalaer and the self.__call__ method
        """
        loss = self
        return ProductLoss(loss=loss, scalar=scalar)

    @typecheck
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        pass


class EmptyLoss(BaseLoss):
    @typecheck
    def __init__(self) -> None:
        pass

    @typecheck
    def __call__(self, source: Shape, target: Shape) -> FloatScalar:
        assert source.device.type == target.device.type
        return torch.tensor(0.0, dtype=float_dtype)


class SumLoss(BaseLoss):
    @typecheck
    def __init__(
        self, loss1: BaseLoss = EmptyLoss(), loss2: BaseLoss = EmptyLoss()
    ) -> None:
        self.loss1 = loss1
        self.loss2 = loss2

    @typecheck
    def __call__(self, source: Shape, target: Shape) -> FloatScalar:
        return self.loss1.__call__(source=source, target=target) + self.loss2.__call__(
            source=source, target=target
        )


class ProductLoss(BaseLoss):
    @typecheck
    def __init__(self, loss: BaseLoss = EmptyLoss(), scalar: Number = 1.0) -> None:
        self.loss = loss
        self.scalar = scalar

    @typecheck
    def __call__(self, source: Shape, target: Shape) -> FloatScalar:
        return self.scalar * self.loss.__call__(source=source, target=target)
