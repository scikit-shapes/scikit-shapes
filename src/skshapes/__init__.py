"""Scikit-Shapes: shape analysis in python."""

from .applications import *
from .convolutions import *
from .data import *
from .decimation import *
from .features import *
from .input_validation import *
from .loss import *
from .morphing import *
from .multiscaling import *
from .optimization import *
from .particles import *
from .tasks import *
from .triangle_mesh import *
from .types import *

__all__ = [
    "types",
    "features",
    "convolutions",
    "applications",
    "data",
    "triangle_mesh",
    "loss",
    "morphing",
    "optimization",
    "particles",
    "multiscaling",
    "decimation",
    "tasks",
    "input_validation",
]
