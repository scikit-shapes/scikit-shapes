from .polydata import PolyData
from .image import Image

from .dataset import Dataset
from .utils import DataAttributes

from typing import Union

Shape = Union[PolyData, Image]


import pyvista


def Sphere():
    """Create a sphere."""
    return PolyData(pyvista.Sphere())
