from .polydata import PolyData
from .image import Image
from .baseshape import BaseShape


from .dataset import Dataset
from .utils import Features

from typing import Union

Shape = Union[Image, PolyData]
