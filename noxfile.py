"""Local development sessions for tests, coverage, linting, and type checks."""

import nox


PYTHON_VERSIONS = ["3.9", "3.10", "3.11", "3.12"]


def _install(session: nox.Session) -> None:
    session.install("-e", ".[all]")


@nox.session(python=PYTHON_VERSIONS)
def tests(session: nox.Session) -> None:
    _install(session)
    session.run("python", "-m", "pytest", *session.posargs)


@nox.session(python="3.11")
def coverage(session: nox.Session) -> None:
    _install(session)
    session.run(
        "python",
        "-m",
        "pytest",
        "--cov=pheat",
        "--cov-report=term-missing",
        "--cov-report=xml",
        *session.posargs,
    )


@nox.session(python="3.11")
def coverage_html(session: nox.Session) -> None:
    _install(session)
    session.run(
        "python",
        "-m",
        "pytest",
        "--cov=pheat",
        "--cov-report=term-missing",
        "--cov-report=xml",
        "--cov-report=html",
        *session.posargs,
    )


@nox.session(python="3.11")
def lint(session: nox.Session) -> None:
    _install(session)
    session.run("python", "-m", "ruff", "check", ".")


@nox.session(python="3.11")
def mypy(session: nox.Session) -> None:
    _install(session)
    session.run("python", "-m", "mypy", "src/pheat")
