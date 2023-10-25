import vedo
from ..types import typecheck, shape_type
from beartype.typing import List


class Browser:
    def __init__(self, shapes: List[shape_type]):
        vedo_shapes = [shape.to_vedo() for shape in shapes]
        self.vedo_browser = vedo.applications.Browser(vedo_shapes)

    def show(self):
        self.vedo_browser.show()
