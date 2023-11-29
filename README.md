# scikit-shapes
Shape processing in Python.

### Presentation
Scikit-shapes is a python package for the analysis of 2D and 3D shape data. It gathers tools for:

- Compute features for shapes such as curvature
- Preprocess shapes with downscaling or landmarks setting
- Register shapes with or without landmarks
- Population Analysis : distance matrices

### Philosophy

Scikit-shapes is thinked to be intuitive and user-friendly, while allowing 

- task-oriented : basic objects are high-level tasks, such as `Registration` or `Multiscaling`. The methematical details as loss function, deformation model, are defined as hyperparameters. (Mettre un diagramme pour registration)

- modular : workflows can be designed following a plug-and-play approach, allowing comparison accross different methods for the same task. In addition, it is possible to implement new modules such as deformation model or loss function and integrate them in existing pipelines.

- efficient : skshapes relies mostly on pyTorch and pyKeOps for computations. It allows to speed-up numerical intensive part of the analysis with parallelization on CPU or GPU.

```python
import skshapes as sks

shape1 = sks.read("data/shape1.vtk")
shape2 = sks.read("data/shape2.vtk")

registration = sks.Registration(

    model = sks.ExtrinsicDeformation(n_steps=5, kernel="gaussian", blur=0.5),
    loss = sks.NearestNeighborLoss(),
    gpu = True,
)

registration.fit(source=shape1, target=shape2)
transformed_shape = registration.transform(source=shape1)

```


### Interoperability and modularity

Scikit-shapes relies on other open-source software, our main dependencies are :
- PyTorch and KeOps : skshapes uses pytorch tensors as basic array structure and take benefits of the pytorch ecosystem to let the possibility to accelerate computation on GPU
- PyVista and Vedo : skshapes relies on PyVista for data loading and visualization, and on vedo for creating interactive visualization. Skshapes objects are exportable to vedo or pyvista through `.to_vedo()` and `.to_pyvista()` methods
- Jaxtyping and Beartype : scikit-shapes is a runtime type checked library. Types are documented with annotations and error are raised if a function is called with a wrong argument's type. This prevents silent errors due to common mistakes such as bad numerical type. Our runtime type checking engine is build around Beartype and annotations for numerical arrays are based on Jaxtyping

arguments of skshapes methods and functions are checked at runtime using beartype. This leads to clear error messages to diagnosis tensor format error like passing a tensor with the wrong dtype or the wrong number of dimension...


## Installation

### From source

To install scikit-shapes directly from the source code, first clone the repository. Then on a terminal, navigate to the scikit-shapes directory and run :

```bash
pip install .
```

If you also want to install the developpers tools, run the following 

```bash
pip install -e .[dev]
```

Then, you will have the possibility to interactively edit the code, run the linting tool and syntax checker, run the tests and build the documentation. From scikit-shapes directory, use the following commands :

```bash
# Lint with black
black .
# Check syntax
flake8 skshapes
# Run tests and show code coverage
pytest --cov-config=.coveragerc --cov=skshapes --cov-report=html test/
firefox htmlcov/index.html
# Build documentation
mkdocs serve
```


## Contributing

TBA