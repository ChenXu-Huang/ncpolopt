"""End-to-end solving tests against small textbook examples.

Ported from the legacy ExampleNoncommutative and ExampleCommutative
cases. Both relaxations solve to -0.75 and
-(1 - sqrt(3)) respectively; the old suite ran them once per solver
backend, which the ``solver_kind`` fixture does automatically.
"""

from __future__ import annotations

from ncpolopt.problem import Problem
from ncpolopt.variables import generate_operators, generate_variables


def test_example_noncommutative(solver_kind: object) -> None:
    X = generate_operators("x", 2, hermitian=True)
    problem = Problem(
        X,
        objective=X[0] * X[1] + X[1] * X[0],
        inequalities=[-X[1] ** 2 + X[1] + 0.5],
        substitutions={X[0] ** 2: X[0]},
    )
    solution = problem.solve(2, solver=solver_kind)
    assert abs(solution.primal + 0.75) < 10e-5


def test_example_commutative(solver_kind: object) -> None:
    x = generate_variables("x", 2, commutative=True)
    problem = Problem(
        x,
        objective=x[0] * x[1] + x[1] * x[0],
        inequalities=[-x[1] ** 2 + x[1] + 0.5],
        substitutions={x[0] ** 2: x[0]},
    )
    solution = problem.solve(2, solver=solver_kind)
    assert abs(solution.primal - (1 - 3**0.5)) < 10e-5
