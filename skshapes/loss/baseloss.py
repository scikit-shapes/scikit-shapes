"""Abstract classes for losses."""

from __future__ import annotations
from ..types import typecheck, Number, FloatScalar, Any, float_dtype, shape_type

from typing import List

import torch


class BaseLoss:
    """Base class for losses.

    This class defines the + and * operators for losses, and each loss should
    inherit from this class. The + operator returns a SumLoss object, and the
    * operator returns a ProductLoss object.

    This class is not meant to be used directly, if the constructor is called
    it raises an error.
    """

    @typecheck
    def __init__(self) -> None:
        """Constructor of the BaseLoss class.

        Raises:
            NotImplementedError: this class is abstract and should not be
            instantiated
        """
        raise NotImplementedError(
            "BaseLoss is an abstract class and should not be instantiated"
        )

    @typecheck
    def __add__(self, other: BaseLoss) -> BaseLoss:
        """Addition of two losses

        Args:
            self (Loss): the first loss to add
            other (Loss): the other loss to add to self

        Returns:
            SumLoss: a new loss which __call__ method is the sum of the two
            __call__ methods
        """

        loss1 = self
        loss2 = other

        return SumLoss(loss1=loss1, loss2=loss2)

    @typecheck
    def __rmul__(self, scalar: Number) -> BaseLoss:
        """(Right) multiplication of a loss by a scalar

        Args:
            scalar (Number): the scalar to multiply the loss by

        Returns:
            Loss: a new loss which __call__ method is the product of the
            scalar and the self.__call__ method
        """
        loss = self
        return ProductLoss(loss=loss, scalar=scalar)

    @typecheck
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        pass


class EmptyLoss(BaseLoss):
    """Empty loss, which always returns 0.

    This loss is useful to serve as a default value for losses which are not
    specified.
    """

    @typecheck
    def __init__(self) -> None:
        """Constructor of the EmptyLoss class."""
        pass

    @typecheck
    def __call__(self, source: shape_type, target: shape_type) -> FloatScalar:
        """__call__ method of the EmptyLoss class. Always returns 0."""
        assert source.device.type == target.device.type
        return torch.tensor(0.0, dtype=float_dtype)


class SumLoss(BaseLoss):
    """Abstract class for losses which are the sum of two losses.

    This class can be directly instantiated, but it is more convenient to use
    the + operator to add two losses, which returns a SumLoss object.

    Note that adding two losses that are not compatible (e.g. a loss for images
    and a loss for meshes) will not raise an error at the time of the addition.
    However it will raise an error when the __call__ method is used, thanks
    to the runtime type checker.
    """

    @typecheck
    def __init__(
        self, loss1: BaseLoss = EmptyLoss(), loss2: BaseLoss = EmptyLoss()
    ) -> None:
        """Constructor of the SumLoss class.

        It saves the two losses as attributes of the class.

        Args:
            loss1 (Loss, optional). Defaults to EmptyLoss().
            loss2 (Loss, optional). Defaults to EmptyLoss().
        """

        self.loss1 = loss1
        self.loss2 = loss2

    @typecheck
    def __call__(self, source: shape_type, target: shape_type) -> FloatScalar:
        """__call__ method of the SumLoss class.

        It returns the sum of the two losses.

        Args:
            source (shape_type): further restrictions on the source shape's type
                            can be imposed by the added losses
            target (shape_type): further restrictions on the target shape's type
                            can be imposed by the added losses

        Returns:
            FloatScalar: the sum of the two losses
        """
        return self.loss1.__call__(source=source, target=target) + self.loss2.__call__(
            source=source, target=target
        )


class ProductLoss(BaseLoss):
    """Abstract class for losses which are the product of a loss and a scalar.

    This class can be directly instantiated, but it is more convenient to use
    the * operator to multiply a loss by a scalar, which returns a ProductLoss
    object.
    """

    @typecheck
    def __init__(self, loss: BaseLoss = EmptyLoss(), scalar: Number = 1.0) -> None:
        """Constructor of the ProductLoss class.

        It saves the loss and the scalar as attributes of the class.

        Args:
            loss (Loss, optional). Defaults to EmptyLoss().
            scalar (Number, optional). Defaults to 1.0.
        """
        self.loss = loss
        self.scalar = scalar

    @typecheck
    def __call__(self, source: shape_type, target: shape_type) -> FloatScalar:
        return self.scalar * self.loss.__call__(source=source, target=target)
