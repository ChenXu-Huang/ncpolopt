"""Backend-specific tests: error paths and value conventions.

The numerical milestone suites run against every usable backend through the
``solver_kind`` fixture; this file covers what those do not: the SDPA error
path when no binary is present, the CVXOPT (PICOS) value conventions against
a known optimum, and the MOSEK task conversion (which needs the license the
milestone suites cannot count on).
"""

from __future__ import annotations

import shutil
from typing import Any, cast

import numpy as np
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

    x, y = generate_variables("x", 1)[0], generate_variables("y", 1)[0]
    problem = Problem([x, y], objective=x**2 + y**2, equalities=[x - 0.4])
    reference = problem.solve(2, solver=solver_kind)
    eliminated = problem.solve(2, removeequalities=True, solver=solver_kind)
    # A surviving free variable (y) resolves through the elimination basis.
    assert eliminated.dual_value(y) == pytest.approx(
        reference.dual_value(y), abs=1e-5
    )
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


def test_registry_unavailable_module_paths(monkeypatch) -> None:
    """The registry skips unimportable backends and errors clearly."""
    from ncpolopt.solvers import registry

    # A registered backend whose module is not importable is not usable.
    monkeypatch.setattr(
        "importlib.util.find_spec",
        lambda module: None if module == "mosek" else object(),
    )
    assert SolverKind.MOSEK not in registry.available()
    assert SolverKind.CVXPY in registry.available()

    # An explicit request for it raises with the install hint.
    with pytest.raises(SolverError, match="not available"):
        registry._resolve(SolverKind.MOSEK)
    # Unknown names and unregistered kinds raise clear errors.
    with pytest.raises(SolverError, match="Unknown solver"):
        registry._resolve("no-such-solver")
    monkeypatch.delitem(registry._REGISTRY, SolverKind.CLARABEL)
    with pytest.raises(SolverError, match="not available"):
        registry._resolve(SolverKind.CLARABEL)


def test_registry_reports_when_no_backend_available(monkeypatch) -> None:
    """With nothing usable, solving fails with the install hint."""
    from ncpolopt.solvers import registry

    monkeypatch.setattr(registry, "available", lambda: ())
    with pytest.raises(SolverError, match="No solver backend"):
        registry._resolve("auto")


def test_cvxpy_verbose_and_default_fallback(monkeypatch) -> None:
    """The verbose knob forwards; without known solvers CVXPY picks one."""
    import cvxpy

    monkeypatch.setattr(
        cvxpy, "installed_solvers", lambda: ["NOT_A_REAL_SOLVER"]
    )
    solution = _basic_problem().solve(
        2, solver="cvxpy", settings=SolverSettings(verbose=False)
    )
    assert solution.status == "optimal"
    assert abs(solution.primal + 0.75) < 10e-5


def test_cvxpy_infeasible_reports_nan(solver_kind: SolverKind) -> None:
    """A failed solve reports the status and NaN values, no matrices."""
    X = generate_operators("X", 1, hermitian=True)
    problem = Problem(X, objective=X[0], inequalities=[X[0] - 1, -X[0] - 2])
    solution = problem.solve(1, solver=solver_kind)
    # Solvers disagree on the hopeless-model classification (infeasible vs
    # unbounded vs unknown); the contract is: no exception, no values.
    assert solution.status in {"infeasible", "unbounded", "unknown"}
    if solver_kind is not SolverKind.SDPA:
        assert np.isnan(solution.primal)


def test_dual_value_skips_missing_blocks() -> None:
    """dual_value ignores None and empty dual blocks."""
    from ncpolopt.solvers._common import dual_value

    sdp = _basic_problem().relaxation(1).sdp
    none_rows = cast(Any, tuple([None] * len(sdp.blocks)))
    assert dual_value(sdp, none_rows) == sdp.constant_term
    empty = tuple(np.zeros((0, 0)) for _ in sdp.blocks)
    assert dual_value(sdp, empty) == sdp.constant_term


@pytest.mark.skipif(not requires("mosek"), reason="mosek is not installed")
def test_mosek_parameter_namespaces() -> None:
    """Prefixed and type-inferred MOSEK parameters all land."""
    import mosek
    import numpy as np

    from ncpolopt.solvers.mosek_solver import _set_parameter

    try:
        task = mosek.Env().Task(0, 0)
    except Exception:
        pytest.skip("MOSEK refuses to create a task without a license")
    _set_parameter(task, "mosek.iparam.max_num_warnings", 1)
    _set_parameter(task, "iparam.max_num_warnings", 1)
    _set_parameter(task, "dparam.intpnt_tol_rel_gap", 1e-7)
    _set_parameter(task, "sparam.bas_sol_file_name", "unused.sol")
    _set_parameter(task, "max_num_warnings", 1)  # type inference: int
    _set_parameter(task, "intpnt_tol_rel_gap", 1e-7)  # float
    _set_parameter(task, "intpnt_tol_rel_gap", np.float32(1e-6))  # numpy scalar
    _set_parameter(task, "max_num_warnings", np.int64(2))
    with pytest.raises(SolverError, match="cannot infer"):
        _set_parameter(task, "max_num_warnings", b"bytes")


@pytest.mark.skipif(not requires("mosek"), reason="mosek is not installed")
def test_mosek_verbose_and_params_solve() -> None:
    """A verbose solve with parameters exercises the stream printer."""
    settings = SolverSettings(
        verbose=True, mosek_params={"iparam.max_num_warnings": 1}
    )
    solution = _basic_problem().solve(2, solver="mosek", settings=settings)
    assert solution.status == "optimal"
    assert abs(solution.primal + 0.75) < 10e-5


@pytest.mark.skipif(not requires("mosek"), reason="mosek is not installed")
def test_mosek_rejects_complex() -> None:
    """The MOSEK backend refuses complex-valued SDPs."""
    X = generate_operators("X", 1, hermitian=True)[0]
    problem = Problem([X], objective=X, complex_matrix=True)
    with pytest.raises(UnsupportedSdpError, match="complex"):
        problem.solve(1, solver="mosek")


@pytest.mark.skipif(not requires("picos"), reason="picos is not installed")
def test_picos_unknown_option_is_solver_error() -> None:
    """A misspelled PICOS option raises SolverError, not AttributeError."""
    with pytest.raises(SolverError, match="Unknown PICOS option"):
        _basic_problem().solve(
            2, solver="cvxopt", settings=SolverSettings(solver_options={"no_such_option": 1})
        )


def test_sdpa_unknown_option_raises(tmp_path) -> None:
    """An unknown SDPA solver option is a ValueError."""
    settings = SolverSettings(
        sdpa_executable=_fake_sdpa(tmp_path, None), solver_options={"bogus": 1}
    )
    with pytest.raises(ValueError, match="Unknown parameter"):
        _basic_problem().solve(2, solver=SolverKind.SDPA, settings=settings)


def test_sdpa_paramsfile_option(tmp_path) -> None:
    """The paramsfile option is forwarded as -p (and the run then fails)."""
    params = tmp_path / "params.sdpa"
    params.write_text("maxIteration = 100\n")
    settings = SolverSettings(
        sdpa_executable=_fake_sdpa(tmp_path, None),
        solver_options={"paramsfile": str(params)},
    )
    with pytest.raises(SolverError, match="return code"):
        _basic_problem().solve(2, solver=SolverKind.SDPA, settings=settings)


def test_sdpa_complex_problem_unembeds(tmp_path) -> None:
    """A complex problem's doubled solution blocks fold back to complex."""
    row = "{" + ", ".join(["1.0"] + ["0.0"] * 3) + "}"
    zeros = "{" + ", ".join(["0.0"] * 4) + "}"
    mat = "{{" + f"{row[1:]}\n" + f"{zeros}\n" * 2 + zeros + "}\n"
    out_text = (
        "objValPrimal = 0.0\n"
        "objValDual = 0.0\n"
        "phase.value = pdOPT\n"
        "xMat =\n"
        "{\n" + mat + "}\n"
        "yMat =\n"
        "{\n" + mat + "}\n"
    )
    settings = SolverSettings(sdpa_executable=_fake_sdpa(tmp_path, out_text))
    X = generate_operators("X", 1, hermitian=True)[0]
    problem = Problem([X], objective=X, complex_matrix=True)
    solution = problem.solve(1, solver=SolverKind.SDPA, settings=settings)
    assert solution.status == "optimal"
    assert solution.x_mat[0].shape == (2, 2)
    assert solution.x_mat[0].dtype == np.complex128


def test_cvxpy_rejects_complex() -> None:
    """The CVXPY backend refuses complex-valued SDPs."""
    X = generate_operators("X", 1, hermitian=True)[0]
    problem = Problem([X], objective=X, complex_matrix=True)
    with pytest.raises(UnsupportedSdpError, match="complex"):
        problem.solve(1, solver="cvxpy")


def test_cvxpy_constant_only_psd_block() -> None:
    """A constraint erased by substitutions becomes a constant PSD block."""
    X = generate_operators("X", 1, hermitian=True)
    problem = Problem(
        X,
        objective=X[0] ** 2,
        inequalities=[X[0] ** 2 - 1],
        substitutions={X[0] ** 2: 1},
    )
    solution = problem.solve(2, solver="cvxpy")
    assert solution.status == "optimal"
    assert solution.primal == pytest.approx(1.0, abs=1e-5)


def test_sdpa_verbose_path(tmp_path) -> None:
    """Verbose runs stream the solver output (fake binary)."""
    settings = SolverSettings(
        verbose=True, sdpa_executable=_fake_sdpa(tmp_path, None)
    )
    with pytest.raises(SolverError, match="return code"):
        _basic_problem().solve(2, solver=SolverKind.SDPA, settings=settings)


def test_sdpa_missing_output_file_is_solver_error(tmp_path) -> None:
    """A binary that exits 0 without writing output raises SolverError."""
    import os
    import stat

    script = tmp_path / "sdpa-noop"
    script.write_text("#!/bin/sh\nexit 0\n")
    if os.name == "nt":
        pytest.skip("the fake SDPA binary is a shell script")
    script.chmod(script.stat().st_mode | stat.S_IXUSR)
    settings = SolverSettings(sdpa_executable=str(script))
    with pytest.raises(SolverError, match="output file"):
        _basic_problem().solve(2, solver=SolverKind.SDPA, settings=settings)


def test_registry_skips_unregistered_kinds(monkeypatch) -> None:
    """A kind without a registered backend is not usable."""
    from ncpolopt.solvers import registry

    monkeypatch.delitem(registry._REGISTRY, SolverKind.CVXPY)
    assert SolverKind.CVXPY not in registry.available()


def test_mosek_sparam_by_value_type() -> None:
    """A string value without a prefix selects the sparam namespace."""
    pytest.importorskip("mosek")
    import mosek

    from ncpolopt.solvers.mosek_solver import _set_parameter

    try:
        task = mosek.Env().Task(0, 0)
    except Exception:
        pytest.skip("MOSEK refuses to create a task without a license")
    _set_parameter(task, "bas_sol_file_name", "unused.sol")


def test_picos_rejects_blockless_problem() -> None:
    """An empty block list is a clear error, not a backend crash."""
    pytest.importorskip("picos")
    from ncpolopt.sdp_problem import SdpProblem
    from ncpolopt.solvers.picos_solver import convert_to_picos

    with pytest.raises(UnsupportedSdpError, match="at least one block"):
        convert_to_picos(SdpProblem(n_vars=0, blocks=(), obj=np.zeros(0), constant_term=0.0))


def test_clarabel_infeasible_reports_nan() -> None:
    """The direct CLARABEL backend reports NaN values on infeasibility."""
    pytest.importorskip("clarabel")
    X = generate_operators("X", 1, hermitian=True)
    problem = Problem(X, objective=X[0], inequalities=[X[0] - 1, -X[0] - 2])
    solution = problem.solve(1, solver="clarabel")
    assert solution.status == "infeasible"
    assert np.isnan(solution.primal)
    assert np.isnan(solution.dual)


def test_cvxpy_missing_dual_becomes_zero_block(monkeypatch) -> None:
    """A constraint without a dual value yields a zero dual block."""
    from cvxpy.constraints.constraint import Constraint

    from ncpolopt.solvers.cvxpy_solver import solve_with_cvxpy

    sdp = _basic_problem().relaxation(2).sdp
    monkeypatch.setattr(Constraint, "dual_value", property(lambda self: None))
    result = solve_with_cvxpy(sdp, SolverSettings())
    assert result.status == "optimal"
    assert all(np.allclose(block, 0) for block in result.y_mat)


@pytest.mark.skipif(not requires("picos"), reason="picos is not installed")
def test_picos_unsolved_values_report_unknown(monkeypatch) -> None:
    """A solve that returns a status without values reads as unknown/nan."""
    import picos as pic

    class _FakeSolution:
        status = "optimal"

    monkeypatch.setattr(pic.Problem, "solve", lambda self, **kwargs: _FakeSolution())
    solution = _basic_problem().solve(1, solver=SolverKind.CVXOPT)
    # The moment variable was never valued, so no values are reported.
    assert np.isnan(solution.primal)
    assert np.isnan(solution.dual)
