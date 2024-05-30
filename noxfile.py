import nox

# If a package is not installed in the virtualenv, raise an error
# (default is False and the package is loaded from the system)
nox.options.error_on_external_run = True


# From https://github.com/facebookresearch/hydra/blob/main/noxfile.py
def install_cpu_torch(session: nox.Session) -> None:
    """
    Install the CPU version of pytorch.
    This is a much smaller download size than the normal version `torch`
    package hosted on pypi.
    The smaller download prevents our CI jobs from timing out.
    """
    session.install(
        "torch", "--extra-index-url", "https://download.pytorch.org/whl/cpu"
    )
    print_installed_package_version(session, "torch")


def print_installed_package_version(
    session: nox.Session, package_name: str
) -> None:
    pip_list: str = session.run("pip", "list", silent=True)
    for line in pip_list.split("\n"):
        if package_name in line:
            pass


@nox.session(python=["3.11"])
def tests(session: nox.Session) -> None:
    """Run the tests."""
    install_cpu_torch(session)
    session.install("-r", "requirements_dev.txt")
    session.install(".")
    session.run("pytest", *session.posargs)


@nox.session(python=["3.11"])
def precommit(session: nox.Session) -> None:
    """Run the pre-commit hooks."""
    session.install("-r", "requirements_dev.txt")
    session.run("pre-commit", "run", "--all-files")


@nox.session(python=["3.11"])
def documentation(session: nox.Session) -> None:
    """Run the pre-commit hooks."""
    install_cpu_torch(session)
    session.install("-r", "requirements_docs.txt")
    session.install(".")

    session.run(
        "sphinx-apidoc",
        "-o",
        "doc/source/api/",
        "--module-first",
        "--force",
        "src/skshapes",
    )
    session.run("sphinx-build", "-b", "html", "doc/source/", "doc/_build/html")
