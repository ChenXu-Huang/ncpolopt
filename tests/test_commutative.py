"""Lasserre-hierarchy tests over commuting variables.

Ported from the legacy Gloptipoly and MaxCut cases. MaxCut
exercises the equality-elimination path: the ``xi**2 - 1`` constraints are
solved algebraically instead of becoming SDP blocks.
"""

from __future__ import annotations

import numpy as np

from ncpolopt.problem import Problem
from ncpolopt.solvers.base import SolverKind
from ncpolopt.variables import generate_variables


def test_six_hump_camel(solver_kind: SolverKind) -> None:
    x = generate_variables("x", 2, commutative=True)
    g0 = (
        4 * x[0] ** 2
        + x[0] * x[1]
        - 4 * x[1] ** 2
        - 2.1 * x[0] ** 4
        + 4 * x[1] ** 4
        + x[0] ** 6 / 3
    )
    solution = Problem(x, objective=g0).solve(3, solver=solver_kind)
    assert abs(solution.primal + 1.0316282672706911) < 10e-5


def test_max_cut(solver_kind: SolverKind) -> None:
    W = (
        np.diag(np.ones(8), 1)
        + np.diag(np.ones(7), 2)
        + np.diag([1, 1], 7)
        + np.diag([1], 8)
    )
    W = W + W.T
    Q = (np.diag(np.dot(np.ones(len(W)).T, W)) - W) / 4
    x = generate_variables("x", len(W), commutative=True)
    equalities = [xi**2 - 1 for xi in x]
    # NOTE: NumPy turns the SymPy variables into an object array; making that
    # explicit keeps the dot product type-checkable.
    x_vector = np.asarray(x, dtype=object)
    objective = -np.dot(x_vector, np.dot(Q, np.transpose(x_vector)))
    problem = Problem(x, objective=objective, equalities=equalities)
    solution = problem.solve(1, removeequalities=True, solver=solver_kind)
    assert abs(solution.primal + 13.5) < 10e-5
