from .polydata import PolyData
from .image import Image
from typing import Union
import pyvista

Shape = Union[PolyData, Image]


def Sphere():
    """Create a sphere."""
    return PolyData(pyvista.Sphere())
