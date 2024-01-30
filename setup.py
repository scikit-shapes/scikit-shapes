from pathlib import Path

import numpy as np
from Cython.Build import cythonize
from setuptools import Extension, setup


def find_source_extension(path: Path) -> Path:
    """Find the source file for a cython module.

    During python -m build, the ".pyx" is compiled to ".cpp" first, then the
    content of the package is moved to a temporary folder. In the temporary
    folder, the ".pyx" file is not present anymore but the ".cpp" is.

    This functions allows to adapt the extension source path according to the
    context.

    Parameters
    ----------
    path
        The path to the source file (".pyx").

    Raises
    ------
    ValueError
        If the source file is not a ".pyx".
    FileNotFoundError
        If the source file is not found, either as ".pyx", ".c" ".cpp".
    """
    if path.suffix != ".pyx":
        msg = f"{path} is not a cython source file (.pyx)"
        raise ValueError(msg)

    if path.exists():
        pass
    elif path.with_suffix(".cpp").exists():
        path = path.with_suffix(".cpp")
    elif path.with_suffix(".c").exists():
        path = path.with_suffix(".c")
    else:
        msg = f"{path} not found"
        raise FileNotFoundError(msg)

    return path


edge_extraction_source = Path(
    "src/skshapes/triangle_mesh/edges_extraction.pyx"
    )
edge_extraction_source = find_source_extension(edge_extraction_source)

# Construct the extension
edges_extraction = Extension(
    name="skshapes.triangle_mesh.edges_extraction",
    sources=[
        str(edge_extraction_source),
    ],
    include_dirs=[np.get_include(), "src/skshapes/triangle_mesh/"],
    define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
)

# Setup the package
setup(
    ext_modules=cythonize([edges_extraction]),
)
