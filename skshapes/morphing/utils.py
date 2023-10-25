from ..data import Shape
from typing import NamedTuple, Optional
from ..types import FloatScalar


class MorphingOutput(NamedTuple):
    morphed_shape: Optional[Shape] = None
    regularization: Optional[FloatScalar] = None
    path: Optional[list[Shape]] = None
