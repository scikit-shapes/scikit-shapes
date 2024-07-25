Contributing
============

Scikit-shapes is open to external contributions. You can contribute with new features, bug fixing, typo fixing, documentation improvement or whatever you think could be improved.

The first step is to open an issue describing the problem you have or the modification you want to implement. The issue discussion will serve as a place to discuss about what to do next and to assign the work that has to be done.

Then, if modifications have to be done to the codebase, the following instructions are made to help you in the process of integrating new code to scikit-shapes the smoothest way as possible.

How to push new code to scikit-shapes
-------------------------------------

### Our continuous integration workflow

Scikit-shapes is designed to become community-driven. It is a necessity to have an infrastructure that ease the integration of external code. The project infrastrucure is largely influenced by the guides from scientific-python: https://learn.scientific-python.org/

All new code should be tested and documented and must pass our CI:

The pre-commit hooks enforce some style requirements
- lint and format the code with [ruff](https://docs.astral.sh/ruff/)
- fix common misspellings with [codespell](https://github.com/codespell-project/codespell)

Then, the unit and functional tests are run:
- run the tests with [pytest](https://docs.pytest.org/en/8.0.x/)
- create a code coverage report with [pytest-cov](https://pytest-cov.readthedocs.io/en/latest/)
- build the documentation with [sphinx](https://www.sphinx-doc.org/en/master/)

Contributing with new code
--------------------------

If you want to contribute to scikit-shapes, the first step is to fork the main repository and create a local branch in which you will push your modification.

Test your modification locally
------------------------------

When you have a clone of your scikit-shapes branch on your system and have made some modifications to the codebase, you may want to test your modification locally

### In a virtual environment

Install your local version of scikit-shapes. With the `--editable` argument, further changes to the codebase will modify also the `skshapes` package on your environment. Then you don't need to re-install the package every time you make a change.
```bash
pip install --editable .
```

Once your local version of scikit-shapes is installed, install the tests and documentation dependencies by running:
```bash
pip install -r requirements_dev.txt
pip install -r requirements_docs.txt
```

Then, you can run :

- the precommit hooks:

```bash
pre-commit install
pre-commit run --all-files
```

- the tests (and see code coverage report in your browser)

```bash
pytest
firefox htmlcov/index.html # replace firefox by your web browser
```
- the documentation building

```bash
# Build the documentation
sphinx-apidoc -o doc/source/api/ --module-first --force src/skshapes
sphinx-build -b html doc/source/ doc/_build/html
# Serve it locally
cd doc/_build/html
python -m http.server
```

### Using nox

Nox is a solution to build virtual environment on-the-fly and run a sequence of instruction (a session). We provide three nox sessions, defined on the file `noxfile.py`, they can be run with the following instructions:

```bash
pip install nox
nox -s tests
nox -s documentation
nox -s precommit
nox # Run all the sessions
```

Note that running nox sessions for scikit-shapes is very time-consuming, because the library and its dependencies must be reinstalled every time a nox session is started (and scikit-shapes depends on pytorch). We recommend to configure a virtual environment.

Open a pull request
-------------------

When you think your code is ready to be merged, open a pull request to the main branch. The pre-commit hooks and the tests will run and a coverage report will be issued. Your modifications need to be approved by a maintainer before being merged to the main repository. If needed, modifications can be done after the opening of the pull request, each new push to your local branch will trigger the checks again.
