"""Import sanity: the package loads with core dependencies only."""

from __future__ import annotations

import re

import ncpolopt


def test_version_string() -> None:
    """The version is a PEP 440-style string."""
    assert re.match(r"^\d+\.\d+\.\d+", ncpolopt.__version__)


def test_setup_logging_is_idempotent() -> None:
    """Calling setup_logging repeatedly must not stack handlers."""
    import logging

    root = logging.getLogger()
    # Detach the handler installed by earlier calls so the assertions below
    # start from a known state (pytest's own capture handlers stay in place).
    for handler in list(root.handlers):
        if getattr(handler, "name", None) == "ncpolopt":
            root.removeHandler(handler)
    ncpolopt.setup_logging()
    ncpolopt.setup_logging()
    ncpolopt.setup_logging()
    ours = [h for h in root.handlers if getattr(h, "name", None) == "ncpolopt"]
    assert len(ours) == 1


def test_import_does_not_load_solver_libraries() -> None:
    """The lazy-import invariant: no solver library after import.

    Runs in a subprocess so the measurement starts from a clean module
    table (the test process itself has everything loaded).
    """
    import subprocess
    import sys

    code = (
        "import sys, ncpolopt; "
        "leaked = [m for m in "
        "('cvxpy', 'mosek', 'picos', 'cvxopt', 'clarabel', 'chompack') "
        "if m in sys.modules]; "
        "sys.exit(len(leaked))"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True)
    assert result.returncode == 0, result.stderr.decode()
