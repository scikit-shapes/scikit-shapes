Getting started
===============

Scikit-Shapes is a pure Python package, but some of its dependencies are only available for **Linux** and **macOS** systems.
If you are a **Windows** user, we advise you to use the official
[Windows Subsystem for Linux](https://learn.microsoft.com/en-us/windows/wsl/about):
it should run seamlessly.

:::{note}
We welcome user feedback!
If you run into issues with the installation process, please
open an issue on our [GitHub repository](https://github.com/scikit-shapes/scikit-shapes/issues).
:::


Using pip
---------

Scikit-Shapes is available on [PyPI](https://pypi.org/project/skshapes/)
as the `skshapes` package. You can install the latest release with:

```bash
pip install skshapes
```

:::{note}
The most likely source of installation problems is the
[KeOps](https://kernel-operations.io/keops/python/installation.html) dependency.
You may also want to install
[PyTorch](https://pytorch.org/) using a specific command, especially
if you do not have a GPU.
:::

:::{warning}
On Google Colab, you may run into conflicts between NumPy 1.x and 2.x.
To avoid this, use:

```bash
!pip uninstall -y numpy
!pip install skshapes
```
:::

From source
-----------

To install Scikit-Shapes from source, start by cloning the [GitHub repository](https://github.com/scikit-shapes/scikit-shapes). Then, on a terminal, navigate to the directory and run
```bash
pip install .
```

From source (developer mode)
----------------------------

To install Scikit-Shapes with the development environment, start by cloning the [GitHub repository](https://github.com/scikit-shapes/scikit-shapes). Then, on a terminal, navigate to the directory and run
```bash
pip install --editable .
pip install -r requirements_dev.txt
```
The `--editable` option links the package in `site-package` to the source code in the current working directory. Then, any local change to the source code reflects directly in the environment.

The development environment contains tools for linting, syntax checking and testing. Linting and syntax tests can be run by executing the command:
```bash
pre-commit run --all-files
```

You can run tests with:
```bash
pytest
```
This creates a coverage report in `htmlcov/`, that you can open in a web browser using:
```bash
firefox htmlcov/index.html
```

You can also install the necessary tools to build the documentation with:
```bash
pip install -r requirements_docs.txt
```

And render this website with:
```bash
cd doc/source
make clean
make html
```

To serve it locally:
```bash
cd doc/_build/html
python -m http.server
```
