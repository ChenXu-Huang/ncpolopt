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
from ncpolopt.solvers.base import (
    SolverError,
    SolverKind,
    SolverSettings,
    UnsupportedSdpError,
)
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


def _offdiagonal_constant_problem() -> Problem:
    """A relaxation whose moment block carries an off-diagonal constant.

    The substitutions pin X*X, X*Y and Y*X to 1, so the (X, Y) entry of
    the moment matrix is the constant 1, stored once in the upper triangle.
    """
    X = generate_operators("X", 1, hermitian=True)[0]
    Y = generate_operators("Y", 1, hermitian=True)[0]
    return Problem(
        variables=[X, Y],
        objective=-X + Y * Y,
        substitutions={X * X: 1, X * Y: 1, Y * X: 1},
    )


def test_dual_value_pairs_offdiagonal_constants_fully(
    solver_kind: SolverKind,
) -> None:
    """Reported duals use the full symmetric constant matrix.

    Regression: dual_value paired Y with the stored upper triangle of A0
    only, halving every off-diagonal constant contribution (the dual of
    this problem came out at -1.0 instead of the primal value 0.0).
    """
    if solver_kind is SolverKind.CVXOPT:
        pytest.skip("numbers are not asserted on the cvxopt/PICOS path")
    solution = _offdiagonal_constant_problem().solve(1, solver=solver_kind)
    assert solution.status == "optimal"
    assert solution.primal == pytest.approx(0.0, abs=1e-5)
    assert solution.dual == pytest.approx(solution.primal, abs=1e-4)
    # Canonical stationarity pairing for X (objective coefficient -1):
    # -tr(Y @ A_X) = -c_X = 1; the upper-triangle-only pairing gave 0.5.
    X = generate_operators("X", 1, hermitian=True)[0]
    assert solution.dual_value(X) == pytest.approx(1.0, abs=1e-4)


def test_elimination_maps_duals_through_basis(solver_kind: SolverKind) -> None:
    """dual_value resolves pre-elimination variables through the basis.

    Regression: after removeequalities the monomial index still referred
    to the pre-elimination variable numbers, so dual_value paired the dual
    blocks with the wrong coefficient rows.
    """
    if solver_kind is SolverKind.CVXOPT:
        pytest.skip("numbers are not asserted on the cvxopt/PICOS path")
    from ncpolopt.variables import generate_variables

    x = generate_variables("x", 1)[0]
    problem = Problem([x], objective=x**2, equalities=[x - 0.4])
    reference = problem.solve(2, solver=solver_kind)
    eliminated = problem.solve(2, removeequalities=True, solver=solver_kind)
    assert eliminated.dual_value(x) == pytest.approx(
        reference.dual_value(x), abs=1e-5
    )
    assert eliminated.dual == pytest.approx(reference.dual, abs=1e-5)


@pytest.mark.skipif(not requires("picos"), reason="picos is not installed")
def test_picos_rejects_complex_sdps() -> None:
    """The PICOS backend refuses complex SDPs instead of mis-solving them.

    Regression: the complex path built non-Hermitian LMIs (the mirror
    entries were not conjugated) and crashed or mis-solved.
    """
    X = generate_operators("X", 1, hermitian=True)[0]
    problem = Problem([X], objective=X, complex_matrix=True)
    with pytest.raises(UnsupportedSdpError, match="complex"):
        problem.solve(1, solver=SolverKind.CVXOPT)


@pytest.mark.skipif(not requires("scs"), reason="scs is not installed")
def test_scs_selection_uses_scs() -> None:
    """solver='scs' drives SCS, not the CVXPY default preference order.

    Regression: the SCS kind was registered to the bare CVXPY backend,
    which picks CLARABEL whenever it is installed.
    """
    from typing import Any, cast

    solution = _basic_problem().solve(2, solver=SolverKind.SCS)
    assert solution.status == "optimal"
    assert abs(solution.primal + 0.75) < 1e-3
    raw = cast(Any, solution.result.raw)
    assert raw.solver_stats.solver_name == "SCS"


@pytest.mark.skipif(not requires("picos"), reason="picos is not installed")
def test_cvxopt_infeasible_returns_status() -> None:
    """An infeasible model yields a status, not a backend exception.

    Regression: P.solve() and the value reads leaked picos exceptions
    (SolutionFailure, NotValued) instead of reporting the outcome.
    """
    X = generate_operators("X", 1, hermitian=True)
    # x >= 1 and x <= -2 cannot hold at once.
    problem = Problem(X, objective=X[0], inequalities=[X[0] - 1, -X[0] - 2])
    solution = problem.solve(1, solver=SolverKind.CVXOPT)
    assert solution.status in {"infeasible", "unknown"}


def _fake_sdpa(tmp_path, out_text: str | None) -> str:
    """A fake SDPA binary: fails without output, or writes a canned .out."""
    import os
    import stat

    script = tmp_path / "sdpa-fake"
    if out_text is None:
        script.write_text("#!/bin/sh\nexit 1\n")
    else:
        script.write_text(
            "#!/bin/sh\n"
            'out=""\n'
            "while [ $# -gt 0 ]; do\n"
            '  if [ "$1" = "-o" ]; then out="$2"; fi\n'
            "  shift\n"
            "done\n"
            f"cat > \"$out\" <<'OUTEOF'\n{out_text}OUTEOF\n"
        )
    if os.name == "nt":
        pytest.skip("the fake SDPA binary is a shell script")
    script.chmod(script.stat().st_mode | stat.S_IXUSR)
    return str(script)


def test_sdpa_nonzero_exit_raises_solver_error(tmp_path) -> None:
    """A crashing SDPA binary raises SolverError, not a bare parse error."""
    settings = SolverSettings(sdpa_executable=_fake_sdpa(tmp_path, None))
    with pytest.raises(SolverError, match="return code"):
        _basic_problem().solve(2, solver=SolverKind.SDPA, settings=settings)


def test_sdpa_mock_binary_roundtrip(tmp_path) -> None:
    """The SDPA path end to end with a canned solver output."""
    # The problem's blocks are 6x6 and 3x3; the canned output matches.
    row6 = "{" + ", ".join(["1.0"] + ["0.0"] * 5) + "}"
    row3 = "{" + ", ".join(["1.0"] + ["0.0"] * 2) + "}"
    zeros6 = "{" + ", ".join(["0.0"] * 6) + "}"
    zeros3 = "{" + ", ".join(["0.0"] * 3) + "}"
    xmat = "{{" + f"{row6[1:]}\n" + f"{zeros6}\n" * 4 + zeros6 + "}\n"
    xmat3 = "{{" + f"{row3[1:]}\n" + f"{zeros3}\n" + zeros3 + "}\n"
    out_text = (
        "objValPrimal = 1.5000000000e+00\n"
        "objValDual = 1.4999999999e+00\n"
        "phase.value = pdOPT\n"
        "xMat =\n"
        "{\n"
        + xmat
        + xmat3
        + "}\n"
        "yMat =\n"
        "{\n"
        + xmat
        + xmat3
        + "}\n"
    )
    settings = SolverSettings(sdpa_executable=_fake_sdpa(tmp_path, out_text))
    solution = _basic_problem().solve(2, solver=SolverKind.SDPA, settings=settings)
    assert solution.status == "optimal"
    assert solution.primal == pytest.approx(1.5)
    assert solution.dual == pytest.approx(1.5)
    assert solution.x_mat[0].shape == (6, 6)
    assert solution.y_mat[1].shape == (3, 3)
