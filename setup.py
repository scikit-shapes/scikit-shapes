from setuptools import setup, Extension
import numpy as np
from Cython.Build import cythonize

extension = Extension(
    name="skshapes.data.edges_extraction",
    sources=[
        "skshapes/data/edges_extraction.pyx",
    ],
    include_dirs=[np.get_include()],
    define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
)


dependencies = [
    "numpy",
    "torch>=1.11",
    "pykeops",
    "geomloss",
    "jaxtyping",
    "beartype>=0.16.1",
    "pyvista",
    "vedo",
    "fast-simplification",
    # 'git+https://github.com/Louis-Pujol/fast-simplification.git',
]

submodules = [
    "applications",
    "convolutions",
    "multiscaling",
    "data",
    "decimation",
    "features",
    "loss",
    "morphing",
    "optimization",
    "tasks",
]

setup(
    name="scikit-shapes",
    version="1.0",
    description="Shape Analysis in Python",
    author="Scikit-Shapes Developers",
    author_email="skshapes@gmail.com",
    url="",
    install_requires=dependencies,
    packages=["skshapes"] + ["skshapes." + submodule for submodule in submodules],
    ext_modules=cythonize(extension),
)