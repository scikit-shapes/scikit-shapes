import numpy as np
from Cython.Build import cythonize
from setuptools import Extension, setup

edges_extraction = Extension(
    name="skshapes.triangle_mesh.edges_extraction",
    sources=[
        "skshapes/triangle_mesh/edges_extraction.pyx",
    ],
    include_dirs=[np.get_include()],
    define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
)


# dependencies = [
#     "numpy",
#     "torch>=1.11",
#     "torchdiffeq",
#     "pykeops",
#     "geomloss",
#     "jaxtyping",
#     "beartype>=0.16.1",
#     "pyvista",
#     "vedo",
#     "fast-simplification",
# ]

# submodules = [
#     "applications",
#     "convolutions",
#     "multiscaling",
#     "data",
#     "triangle_mesh",
#     "decimation",
#     "features",
#     "loss",
#     "morphing",
#     "optimization",
#     "tasks",
# ]

setup(
    # name="scikit-shapes",
    # version="1.0",
    # description="Shape Analysis in Python",
    # author="Scikit-Shapes Developers",
    # author_email="skshapes@gmail.com",
    # url="",
    # install_requires=dependencies,
    # packages=["skshapes"]
    # + ["skshapes." + submodule for submodule in submodules],
    ext_modules=cythonize([edges_extraction]),
)
