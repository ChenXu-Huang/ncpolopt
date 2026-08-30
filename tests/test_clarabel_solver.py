"""Tests for the direct sparse CLARABEL backend.

The backend builds CLARABEL's native (A, b, cones) directly from the
frozen blockwise COO, in the exact layout CVXPY's CLARABEL
canonicalization produces; these tests pin that equivalence on small
problems and check the backend contract (values, statuses, autodetection
exclusion, complex rejection).
"""

from __future__ import annotations

import numpy as np
import pytest

import ncpolopt as nc
from ncpolopt.solvers.base import (
    SolverKind,
    SolverSettings,
    UnsupportedSdpError,
)
from ncpolopt.solvers.clarabel_solver import canonical_clarabel_data
from ncpolopt.solvers.cvxpy_solver import convert_to_cvxpy
from ncpolopt.solvers.registry import available

clarabel = pytest.importorskip("clarabel")

_TOL = 1e-7


def _quickstart_problem() -> nc.Problem:
    """The package quickstart problem (small, noncommutative)."""
    X = nc.generate_operators("x", 2, hermitian=True)
    return nc.Problem(
        X,
        objective=X[0] * X[1] + X[1] * X[0],
        inequalities=[-X[1] ** 2 + X[1] + 0.5],
        substitutions={X[0] ** 2: X[0]},
    )


def test_matches_cvxpy_backend() -> None:
    """Primal and dual values agree with the CVXPY/CLARABEL path."""
    problem = _quickstart_problem()
    reference = problem.solve(level=2, solver="cvxpy")
    result = problem.solve(level=2, solver="clarabel")
    assert result.status == "optimal"
    assert result.primal == pytest.approx(reference.primal, abs=_TOL)
    assert result.dual == pytest.approx(reference.dual, abs=1e-5)
    assert result.primal == pytest.approx(-0.75, abs=1e-5)


def test_canonical_data_matches_cvxpy_layout() -> None:
    """The (A, b, cones) data is CVXPY's canonical data, bit-identically."""
    import cvxpy as cp

    sdp = _quickstart_problem().relaxation(level=2).sdp
    A, b, cones = canonical_clarabel_data(sdp)

    result = convert_to_cvxpy(sdp).get_problem_data(solver=cp.CLARABEL)
    # cvxpy >= 1.6 returns (data, chain, inverse_data); older versions
    # return the data dict directly.
    data = result[0] if isinstance(result, tuple) else result
    assert np.array_equal(np.asarray(data["b"], dtype=float), b)
    assert (data["A"] != A).nnz == 0
    dims = data["dims"]
    n_nonneg = int(dims.zero) + int(dims.nonneg)
    n_psd = [int(size) for size in dims.psd]
    expected_cones = []
    if n_nonneg:
        expected_cones.append(clarabel.NonnegativeConeT(n_nonneg))
    expected_cones.extend(clarabel.PSDTriangleConeT(size) for size in n_psd)
    assert [str(cone) for cone in cones] == [str(cone) for cone in expected_cones]


def test_solution_extraction_helpers() -> None:
    """The Solution built on the backend exposes moments and SOS data."""
    problem = _quickstart_problem()
    solution = problem.solve(level=2, solver="clarabel")
    x1 = problem.variables[1]
    value = solution.monomial_value(x1)
    assert value == pytest.approx(-0.25, abs=1e-4)
    assert solution.x_mat is not None


def test_autodetection_unchanged() -> None:
    """The backend never enters autodetection, even when importable."""
    assert SolverKind.CLARABEL not in available()


def test_explicit_unavailable_message() -> None:
    """Requesting the backend without clarabel installed names the extra."""
    from ncpolopt.solvers import registry

    assert registry._SOLVER_MODULES[SolverKind.CLARABEL] == "clarabel"
    assert "clarabel" in registry._INSTALL_HINTS[SolverKind.CLARABEL]


def test_complex_sdp_rejected() -> None:
    """Complex-valued SDPs raise instead of silently truncating."""
    Y = nc.generate_operators("y", 1, hermitian=False)
    problem = nc.Problem(
        Y,
        objective=Y[0].adjoint() * Y[0],
        complex_matrix=True,
    )
    with pytest.raises(UnsupportedSdpError, match="real-valued"):
        problem.solve(level=1, solver="clarabel")


def test_verbose_and_solver_options() -> None:
    """SolverSettings reach clarabel.DefaultSettings without errors."""
    problem = _quickstart_problem()
    settings = SolverSettings(
        verbose=False, solver_options={"tol_gap_abs": 1e-8, "tol_gap_rel": 1e-8}
    )
    solution = problem.relaxation(level=2).solve(solver="clarabel", settings=settings)
    assert solution.status == "optimal"
    assert solution.primal == pytest.approx(-0.75, abs=1e-6)
