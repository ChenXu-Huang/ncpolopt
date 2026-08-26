"""The Problem model: defaults, immutability, and the one-call API."""

from __future__ import annotations

import dataclasses

import pytest

from ncpolopt.problem import Problem
from ncpolopt.relaxation import NpaRelaxation
from ncpolopt.solvers.base import SolverError
from ncpolopt.solvers.registry import registered
from ncpolopt.variables import generate_operators


def test_problem_defaults() -> None:
    """The model defaults to a normalized, feasible-form problem."""
    X = generate_operators("X", 1, hermitian=True)
    problem = Problem(X)
    assert problem.objective is None
    assert problem.inequalities is None
    assert problem.equalities is None
    assert problem.normalized is True
    assert problem.complex_matrix is None
    assert problem.parameters is None


def test_problem_is_frozen() -> None:
    """Problems are immutable value objects."""
    X = generate_operators("X", 1, hermitian=True)
    problem = Problem(X, objective=X[0])
    with pytest.raises(dataclasses.FrozenInstanceError):
        problem.objective = X[0] ** 2  # type: ignore[misc]


def test_relaxation_does_not_mutate_problem() -> None:
    """Building a relaxation leaves the model untouched."""
    X = generate_operators("X", 1, hermitian=True)
    problem = Problem(X, objective=X[0], equalities=[X[0] - 1.0])
    relaxation = problem.relaxation(level=1, removeequalities=True)
    assert isinstance(relaxation, NpaRelaxation)
    assert problem.objective == X[0]
    assert problem.equalities == [X[0] - 1.0]
    assert relaxation.level == 1
    assert relaxation.removeequalities is True


def test_solve_delegates_to_relaxation() -> None:
    """Problem.solve builds the relaxation and solves it."""
    X = generate_operators("X", 1, hermitian=True)
    problem = Problem(X, objective=X[0])
    if registered():
        solution = problem.solve(level=1)
        assert solution.status == "optimal"
    else:
        with pytest.raises(SolverError, match="No solver backend"):
            problem.solve(level=1)


def test_solve_passes_solver_kind_through() -> None:
    """An explicit unknown solver kind surfaces as a SolverError."""
    X = generate_operators("X", 1, hermitian=True)
    problem = Problem(X, objective=X[0])
    with pytest.raises(SolverError, match="Unknown solver"):
        problem.solve(level=1, solver="gurobi")
