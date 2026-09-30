.. _motivation:

Why Scikit-Shapes?
==================

**Purpose.** Scikit-Shapes is a Python library for statistical shape analysis. It provides tools for the analysis of shapes in 2D and 3D that may be encoded as point clouds, Gaussian splats, curves, surfaces or segmented images. It is especially suited to:

- **Registration** with both rigid and non-rigid models.
- **Atlas** construction from a collection of raw or annotated shapes.
- **Dimensionality reduction** for visualization and statistical analysis.

Scikit-Shapes provides fast and differentiable methods that fit well
in data processing pipelines based on `Scikit-Learn <https://scikit-learn.org/stable/>`_
or `PyTorch <https://pytorch.org/>`_.
We provide a transparent interface with `PyVista <https://docs.pyvista.org/>`_
and the `VTK <https://vtk.org/>`_ software suite for visualization.

Context
-------

**The geometer's dilemma.**
Scikit-Shapes intends to bring modern geometric methods to the Python ecosystem.
Historically, this has been a challenge: while linear algebra has always benefitted
from first-class support by the `SciPy <https://scipy.org/>`_ or
`PyTorch <https://pytorch.org/>`_ ecosystems, key numerical routines
for point cloud and mesh processing have long remained C++ exclusives.
This has slowed down progress in the field, with researchers having to choose between:

- Developing **C++** codebases, which can fully leverage modern hardware or data structures
  but are hard to install, maintain and extend.
- Developing **Python** codebases, which are readable and modular but orders of magnitude
  slower than their C++ counterparts.

**Solution.**
Fortunately, a collection of open source packages has just solved this issue.
Modern libraries such as `PyTorch <https://pytorch.org/>`_,
`KeOps <https://www.kernel-operations.io/>`_,
`PotPourri3D <https://github.com/nmwsharp/potpourri3d>`_ and
`Taichi <https://www.taichi-lang.org/>`_
now let geometers implement their methods in Python with near-optimal performance.
Scikit-Shapes is about leveraging these advances to finally bridge the gap
between "showcase" and "production" codebases in geometric data analysis.


Our core principles
-------------------

**Target audience.** Statistical shape analysis has major applications in:

- **Computational anatomy** and geometric morphometry, the study of bones, brains and other organs for medicine and biology.
- **Computer graphics**, the design of authorable assets for video games and movies.
- **Computer vision**, the analysis of environments and human poses for scene understanding.
- **Computer-aided design**, the creation of objects under constraints for engineering and architecture.

Scikit-shapes is meant to be used in these fields, both as a reference implementation
of standard methods and as a stepping stone for cutting-edge research projects.


**Design philosophy.** Scikit-Shapes is designed to be:

- **User-centric.** Our interface is structured around data processing steps
  instead of mathematical theories and concepts.
  We favor plain descriptive names over cryptic acronyms.
- **Modular.** Our methods are meant to be used as utility functions in larger codebases.
  For instance, our deformation models can all be used as layers in a neural network
  or combined with user-defined loss functions.
- **Fully tested and documented.** we provide extensive unit tests and tutorials
  to ensure that our code is both reliable and easy to use.
- **Fast.** While ease-of-use and modularity are our top priorities, we believe
  that keeping run-times comparable to those of optimized C++ implementations
  is a key requirement for credibility.
  Scikit-Shapes scales to high-resolution data and fully leverages
  the GPU if it is available.
- **Portable.** Our long-term goal is to have Scikit-Shapes run on mobile devices.
  As of 2025, this is still a stretch too far so we focus on
  supporting standard laptops and workstations.
  We keep our codebase modular and intend to provide a
  `WebGPU <https://www.w3.org/TR/webgpu/>`_ backend as early as reasonably possible.

Scikit-Shapes is named after
`Scikit-Image <https://scikit-image.org/>`_
and
`Scikit-Learn <https://scikit-learn.org/stable/>`_:
whenever possible, we follow their conventions and design principles.
While the infrastructure of the library is still under construction,
we hope to release a first usable version and open our code to contributions
by the end of 2025.
