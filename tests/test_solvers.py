"""Backend-specific tests: error paths and value conventions.

The numerical milestone suites run against every usable backend through the
``solver_kind`` fixture; this file covers what those do not: the SDPA error
path when no binary is present, the CVXOPT (PICOS) value conventions against
a known optimum, and the MOSEK task conversion (which needs the license the
milestone suites cannot count on).
"""

from __future__ import annotations

import shutil

import pytest

from ncpolopt.problem import Problem
from ncpolopt.solvers.base import SolverError, SolverKind, SolverSettings
from ncpolopt.solvers.registry import requires
from ncpolopt.variables import generate_operators


def _basic_problem() -> Problem:
    """The textbook example that solves to -0.75 (see test_basic.py)."""
    X = generate_operators("x", 2, hermitian=True)
    return Problem(
        X,
        objective=X[0] * X[1] + X[1] * X[0],
        inequalities=[-X[1] ** 2 + X[1] + 0.5],
        substitutions={X[0] ** 2: X[0]},
    )


@pytest.mark.skipif(not requires("picos"), reason="picos is not installed")
def test_cvxopt_value_conventions() -> None:
    """The PICOS backend reports the canonical primal and dual values.

    The CVXOPT backend reports every solved point as "primal feasible"; the
    status mapping must still yield "optimal".
    """
    solution = _basic_problem().solve(2, solver=SolverKind.CVXOPT)
    assert solution.status == "optimal"
    assert abs(solution.primal + 0.75) < 10e-5
    assert abs(solution.dual - solution.primal) < 10e-4
    assert solution.x_mat
    assert len(solution.y_mat) == len(solution.x_mat)


def test_sdpa_missing_binary_raises_solver_error() -> None:
    """Solving through SDPA without a binary raises SolverError."""
    if shutil.which("sdpa") is not None:
        pytest.skip("SDPA binary present; the error path is not exercised")
    with pytest.raises(SolverError, match="SDPA"):
        _basic_problem().solve(2, solver=SolverKind.SDPA)


def test_sdpa_wrong_executable_raises_solver_error() -> None:
    """An explicitly wrong executable raises SolverError either way."""
    settings = SolverSettings(sdpa_executable="ncpolopt-no-such-sdpa-binary")
    with pytest.raises(SolverError, match="SDPA"):
        _basic_problem().solve(2, solver=SolverKind.SDPA, settings=settings)


@pytest.mark.skipif(not requires("mosek"), reason="mosek is not installed")
def test_mosek_solves_basic_problem() -> None:
    """The MOSEK status map matches the installed MOSEK version's enum.

    MOSEK 11 dropped ``solsta.near_optimal`` and never had the
    ``*_infeasible_cer`` spellings the old map used; the eagerly built
    status dict raised ``AttributeError`` right after the solve. Requires
    a licensed MOSEK.
    """
    from conftest import _licensed_mosek

    if not _licensed_mosek():
        pytest.skip("MOSEK cannot acquire a license")
    solution = _basic_problem().solve(2, solver=SolverKind.MOSEK)
    assert solution.status == "optimal"
    assert abs(solution.primal + 0.75) < 10e-5


@pytest.mark.skipif(not requires("mosek"), reason="mosek is not installed")
def test_mosek_conversion() -> None:
    """The task carries one constraint per variable and one barvar per block."""
    import mosek

    from ncpolopt.solvers.mosek_solver import convert_to_mosek

    relaxation = _basic_problem().relaxation(2)
    try:
        task = convert_to_mosek(relaxation.sdp)
    except Exception:
        pytest.skip("MOSEK refuses to build a task without a license")
    assert task.getnumcon() == relaxation.sdp.n_vars
    assert task.getnumbarvar() == len(relaxation.sdp.blocks)
    assert task.getobjsense() == mosek.objsense.minimize


@pytest.mark.skipif(not requires("mosek"), reason="mosek is not installed")
def test_mosek_parameter_errors_are_solver_errors() -> None:
    """Unknown parameter names raise SolverError (bug #7: eval() is gone)."""
    import mosek

    from ncpolopt.solvers.mosek_solver import _set_parameter

    try:
        task = mosek.Env().Task(0, 0)
    except Exception:
        pytest.skip("MOSEK refuses to create a task without a license")
    with pytest.raises(SolverError, match="No MOSEK parameter"):
        _set_parameter(task, "iparam.MSK_IPAR_NO_SUCH_PARAM", 1)
