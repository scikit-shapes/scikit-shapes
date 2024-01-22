import nox


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
            print(f"Installed {package_name} version: {line}")


# TODO the test does not work: it does not install cython modules
# @nox.session(python=["3.11"])
# def tests(session: nox.Session) -> None:
#     """
#     Run the unit and regular tests.
#     """
#     install_cpu_torch(session)
#     session.install(".[dev]")
#     session.run("pytest", *session.posargs)


@nox.session(python=["3.11"])
def lint(session: nox.Session) -> None:
    """
    Run the linter.
    """
    session.install("flake8", "flake8-black", "flake8-bugbear")
    session.run("flake8", "skshapes", "tests", "examples", "noxfile.py")
