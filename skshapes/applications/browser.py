"""Wrapper for vedo.applications.Browser."""

import vedo
from ..types import shape_type


class Browser:
    """Browser to visualize a sequence of shapes."""

    def __init__(self, shapes: list[shape_type]):
        """Class constructor.

        Parameters
        ----------
        shapes
            The shapes to visualize.
        """
        vedo_shapes = [shape.to_vedo() for shape in shapes]
        self.vedo_browser = vedo.applications.Browser(vedo_shapes)

    def show(self):
        """Show the browser."""
        self.vedo_browser.show()
