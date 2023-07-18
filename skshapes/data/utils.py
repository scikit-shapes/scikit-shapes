from ..types import typecheck, PolyDataType
from .polydata import PolyData
import pyvista


@typecheck
def read(filename: str) -> PolyDataType:
    mesh = pyvista.read(filename)
    if type(mesh) == pyvista.PolyData:
        return PolyData.from_pyvista(mesh)
    else:
        raise NotImplementedError("Images are not supported yet")
