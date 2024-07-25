Installation
============

So far, scikit-shapes is only available for linux and macOS, if you are a windows user, you can consider using [wsl](https://learn.microsoft.com/en-us/windows/wsl/about)

With pip
--------

TBA

From source
-----------

To install `scikit-shapes` directly from source, start by cloning the [GitHub repository](https://github.com/scikit-shapes/scikit-shapes). Then, on a terminal, navigate to the directory and run
```bash
pip install .
```

From source (developers)
------------------------

To install `scikit-shapes` with the development environment, start by cloning the [GitHub repository](https://github.com/scikit-shapes/scikit-shapes). Then, on a terminal, navigate to the directory and run
```bash
pip install --editable .
pip install -r requirements_dev.txt
```
The `--editable` option links the package in `site-package` to the source code in the current working directory. Then, any local change to the source code reflects directly in the environment.

The development environment contains tools for linting, syntax checking and testing. Linting and syntax tests can be run by executing the command:
```bash
pre-commit run --all-files
```

Tests can be run with
```bash
pytest
```
a coverage report is created in `htmlcov/`, you can open it in a web browser using for example
```bash
firefox htmlcov/index.html
```

You can also install the necessary tools to build the documentation with:
```bash
pip install -r requirements_docs.txt
```

then, to build the documentation run:
```bash
sphinx-apidoc -o doc/source/api/ --module-first --force src/skshapes
sphinx-build -b html doc/source/ doc/_build/html
```

And to serve it locally:
```bash
cd doc/_build/html
python -m http.server
```
