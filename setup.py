import numpy as np
from Cython.Build import cythonize
from setuptools import Extension, setup

edges_extraction = Extension(
    name="skshapes.triangle_mesh.edges_extraction",
    sources=[
        "src/skshapes/triangle_mesh/edges_extraction.pyx",
    ],
    include_dirs=[np.get_include()],
    define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
)

setup(
    ext_modules=cythonize([edges_extraction]),
)
