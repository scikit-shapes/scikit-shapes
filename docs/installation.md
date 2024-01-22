# Installation

## With pip

TBA

## With docker

TBA

## From source

To install `scikit-shapes` directly from source, start by cloning the [GitHub repository](https://github.com/scikit-shapes/scikit-shapes). Then, on a terminal, navigate to the directory and run
```bash
pip install .
```

## From source (developers)

To install `scikit-shapes` with the development environment, start by cloning the [GitHub repository](https://github.com/scikit-shapes/scikit-shapes). Then, on a terminal, navigate to the directory and run
```bash
pip install --editable .[dev]
```
The `--editable` option links the package in `site-package` to the source code in the current working directory. Then, any local change to the source code reflects directly in the environment.

The development environment contains tools for :
- linting ([black](https://github.com/psf/black))
- syntax checking ([flake8](https://flake8.pycqa.org/en/latest/))
- testing ([pytest](https://docs.pytest.org/en/7.4.x/))
- building documentation ([mkdocs](https://www.mkdocs.org/))
as well as plugins for these.

In this environment, you can execute the following directly in the `scikit-shapes` folder
```bash
# Lint
black .
# Check syntax
flake8 skshapes
# Run tests and show code coverage
pytest --cov-config=.coveragerc --cov=skshapes --cov-report=html test/
firefox htmlcov/index.html
# Build documentation
mkdocs serve
```
