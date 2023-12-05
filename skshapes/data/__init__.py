from .polydata import PolyData
from .image import Image
from typing import Union
from ..types import polydata_type
from ..input_validation import typecheck
from math import pi
import pyvista
import torch

Shape = Union[PolyData, Image]


@typecheck
def Sphere() -> polydata_type:
    """Create a sphere."""
    return PolyData(pyvista.Sphere())


@typecheck
def Circle(n_points: int = 20) -> polydata_type:
    """Return a circle.

    Parameters
        n_points (int, optional): Number of points. Defaults to 20.

    Returns:
        polydata_type: the circle
    """
    linspace = torch.linspace(0, 1, n_points)

    circle_xs = 0.5 + 0.5 * torch.cos(2 * pi * linspace)
    circle_ys = 0.5 + 0.5 * torch.sin(2 * pi * linspace)
    circle_points = torch.stack([circle_xs, circle_ys], dim=1)

    edges = torch.tensor(
        [[i, (i + 1) % n_points] for i in range(n_points - 1)]
    )
    return PolyData(points=circle_points, edges=edges)
