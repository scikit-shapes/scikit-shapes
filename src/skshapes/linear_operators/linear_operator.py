from pydantic import BaseModel

from ..input_validation import typecheck
from ..types import Callable, Generic, Literal, TypeVar


@typecheck
class InverseParameters(BaseModel):
    solver: Literal["auto", "cg", "cholesky", "minres", "pyamg"] = "auto"


# Define type variables for the domain and codomain
In = TypeVar("In")
Out = TypeVar("Out")


class LinearOperator(Generic[In, Out]):
    def __init__(
        self,
        *,
        operator: Callable[[In], Out],
        n_points: int,
        n_features: int,
    ) -> None:
        self._operator = operator
        self.n_points = n_points
        self.n_features = n_features

    def __call__(self, x: In) -> Out:
        assert x.shape == (self.n_points, self.n_features)
        out = self._operator(x)
        assert out.shape == (self.n_points, self.n_features)
        return out
