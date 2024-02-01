:construction_worker: scikit-shapes development is in pre-alpha stage, the [documentation](https://scikit-shapes.github.io/scikit-shapes/) may be incomplete, and the interface could be subject to important changes in a near future.

# Scikit-Shapes
Shape processing in Python

![](docs/animation.gif)

## Presentation

Scikit-shapes is a python package for the analysis of 2D and 3D shape data. It gathers tools for:

- Compute **features** for shapes such as curvature
- **Preprocess** shapes with downscaling or landmarks setting
- **Register** shapes with or without landmarks
- **Population Analysis** : distance matrices

## Philosophy

Scikit-shapes is thinked to be intuitive and user-friendly, our am is to provide a library that is
- **Task-oriented**: basic objects are high-level tasks, such as `Registration` or `Multiscaling`. The mathematical details as loss function, deformation model, are defined as hyperparameters.
- **Modular**: workflows can be designed following a plug-and-play approach, allowing comparison across different methods for the same task. In addition, it is possible to implement new modules such as deformation model or loss function and integrate them in existing pipelines.
- **Efficient**: skshapes relies mostly on pyTorch and pyKeOps for computations. It allows to speed-up numerical intensive part of the analysis with parallelization on CPU or GPU.

Here is a code snippet illustrating how a registration model is build by combining a loss function and a deformation model:

```python
import skshapes as sks

shape1 = sks.read("data/shape1.vtk")
shape2 = sks.read("data/shape2.vtk")

registration = sks.Registration(
    model=sks.ExtrinsicDeformation(n_steps=5, kernel="gaussian", blur=0.5),
    loss=sks.NearestNeighborLoss(),
    gpu=True,
)

registration.fit(source=shape1, target=shape2)
transformed_shape = registration.transform(source=shape1)
```


## Connection to other open-source projects

Scikit-shapes relies on other open-source software, our main dependencies are :
- [PyTorch](https://pytorch.org/) and [KeOps](https://www.kernel-operations.io/keops/index.html) : skshapes uses pytorch tensors as basic array structure and take benefits of the pytorch ecosystem to let the possibility to accelerate computations on GPU.
- [PyVista](https://docs.pyvista.org/version/stable/) and [Vedo](https://vedo.embl.es/) : skshapes relies on PyVista for data loading and visualization, and on vedo for creating interactive visualization. Skshapes objects are exportable to vedo or pyvista through `.to_vedo()` and `.to_pyvista()` methods.
- [Jaxtyping](https://github.com/google/jaxtyping) and [Beartype](https://beartype.readthedocs.io/en/latest/) : scikit-shapes is a runtime type checked library. Types are documented with annotations and error are raised if a function is called with a wrong argument's type. This prevents silent errors due to common mistakes such as bad numerical type. Our runtime type checking engine is build around Beartype and annotations for numerical arrays are based on Jaxtyping.

# Installation

For now, the recommended way to install scikit-shapes is directly from the main branch
```bash
pip install 'skshapes @ git+https://github.com/scikit-shapes/scikit-shapes@main'
```

If you consider contrinuting to the codebase, you can also install scikit-shapes locally from a clone of the repository. On a terminal, navigate to the scikit-shapes directory and run :

```bash
pip install .
```

Then you can :

-  run the pre-commit hooks:
```bash
pip install -e requirements_dev.txt
pre-commit install
pre-commit run --all-files
```

- run the tests:
```bash
pip install -e requirements_dev.txt
pytest
```
- build the documentation (and serve it locally)
```bash
pip install -r requirement_docs.txt
mkdocs serve
```

# Contributing

We warmly welcome all contribution, if you found a bug, a typo or want to contribute with a new feature, please open an [issue](https://github.com/scikit-shapes/scikit-shapes/issues).

You can also open a [discussion](https://github.com/scikit-shapes/scikit-shapes/discussions) if you have any question regarding the project.

For more information about contributing with new code, see the [dedicated section](https://scikit-shapes.github.io/scikit-shapes/contributing/) of the documentation.
