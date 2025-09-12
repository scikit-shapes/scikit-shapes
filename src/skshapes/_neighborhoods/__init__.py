"""Neighborhood structures for smoothing and Laplacians."""

from .graph_neighborhoods import GraphNeighborhoods
from .kernel_neighborhoods import KernelNeighborhoods
from .neighborhoods import Neighborhoods
from .old_neighborhoods import OldNeighborhoods
from .point_neighborhoods import _point_neighborhoods
from .spectrum import Spectrum
from .trivial_neighborhoods import TrivialNeighborhoods

__all__ = [
    "OldNeighborhoods",
    "Spectrum",
    "Neighborhoods",
    "_point_neighborhoods",
]
