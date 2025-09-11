from __future__ import annotations

from ..types import Callable, Generic, TypeVar

# Define type variables for the domain and codomain
In = TypeVar("In")
Out = TypeVar("Out")


class LinearOperator(Generic[In, Out]):
    def __init__(
        self,
        *,
        operator: Callable[[In], Out],
        transpose: Callable[[Out], In],
        n_points: int,
        n_features: int,
    ) -> None:
        self._operator = operator
        self._transpose = transpose
        self.n_points = n_points
        self.n_features = n_features

    def __matmul__(self, x: In) -> Out:
        assert x.shape == (self.n_points, self.n_features)
        out = self._operator(x)
        assert out.shape == (self.n_points, self.n_features)
        assert out.dtype == x.dtype
        assert out.device == x.device
        return out

    def t(self) -> LinearOperator[Out, In]:
        return LinearOperator(
            operator=self._transpose,
            transpose=self._operator,
            n_points=self.n_points,
            n_features=self.n_features,
        )
