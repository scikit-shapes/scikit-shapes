import argparse

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
    """
    Run the tests.
    """
    install_cpu_torch(session)
    session.install("-r", "requirements_dev.txt")
    session.install(".")
    session.run("pytest", *session.posargs)

@nox.session(reuse_venv=True, python=["3.11"])
def docs(session: nox.Session) -> None:
    """
    Build the docs. Pass "--serve" to serve. Pass "-b linkcheck" to check links.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--serve", action="store_true", help="Serve after building")
    parser.add_argument(
        "-b", dest="builder", default="html", help="Build target (default: html)"
    )
    args, posargs = parser.parse_known_args(session.posargs)

    if args.builder != "html" and args.serve:
        session.error("Must not specify non-HTML builder with --serve")

    extra_installs = ["sphinx-autobuild"] if args.serve else []

    session.install("-r", "requirements_sphinx.txt")
    session.install(".", *extra_installs)
    session.chdir("docs")

    if args.builder == "linkcheck":
        session.run(
            "sphinx-build", "-b", "linkcheck", ".", "_build/linkcheck", *posargs
        )
        return

    shared_args = (
        "-n",  # nitpicky mode
        "-T",  # full tracebacks
        f"-b={args.builder}",
        ".",
        f"_build/{args.builder}",
        *posargs,
    )

    if args.serve:
        session.run("sphinx-autobuild", *shared_args)
    else:
        session.run("sphinx-build", "--keep-going", *shared_args)
