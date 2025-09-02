"""Polygonal data (point clouds, curves, surfaces), segmentation maps and images."""

from math import pi
from typing import Union

import pyvista
import torch

from ..input_validation import typecheck
from ..types import polydata_type
from .data_attributes import DataAttributes
from .image import Image
from .mask import Mask
from .polydata import PolyData

Shape = PolyData | Image


@typecheck
def Sphere() -> polydata_type:
    """Return a sphere."""

    sphere = PolyData(pyvista.Sphere())
    sphere.point_data.clear()  # Remove the normals
    return sphere


@typecheck
def Circle(n_points: int = 20) -> polydata_type:
    """Return a circle.

    Parameters
        n_points (int, optional): Number of points. Defaults to 20.

    Returns:
        polydata_type: the circle
    """
    linspace = torch.linspace(0, 1, n_points + 1)

    circle_xs = 0.5 + 0.5 * torch.cos(2 * pi * linspace[:n_points])
    circle_ys = 0.5 + 0.5 * torch.sin(2 * pi * linspace[:n_points])
    circle_points = torch.stack([circle_xs, circle_ys], dim=1)

    edges = torch.tensor([[i, (i + 1) % n_points] for i in range(n_points)])
    return PolyData(points=circle_points, edges=edges)
