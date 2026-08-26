"""NPA-hierarchy tests over Bell scenarios.

Ported from Chsh, ChshMixedLevel, ElegantBell, and NietoSilleras
(src.old/tests/test_ncpol2sdpa.py). The last test exercises polynomial
moment equalities (the behaviour constraints) mixed with a MomentEntry
one -- the string DSL entry ``"+0[0,0]-1.0"`` of the old suite becomes
``MomentEntry(0, 0, 0) - 1.0``.
"""

from __future__ import annotations

import numpy as np

from ncpolopt.moment import MomentEntry
from ncpolopt.physics import (
    Probability,
    define_objective_with_I,
    maximum_violation,
    projective_measurement_constraints,
)
from ncpolopt.problem import Problem
from ncpolopt.variables import generate_operators


def test_chsh(solver_kind: object) -> None:
    def expectation_values(measurement: list[list[object]], outcomes):
        exp_values = []
        for k in range(len(measurement)):
            exp_value = 0
            for j in range(len(measurement[k])):
                exp_value += outcomes[k][j] * measurement[k][j]
            exp_values.append(exp_value)
        return exp_values

    E = generate_operators("E", 8, hermitian=True)
    M, outcomes = [], []
    for i in range(4):
        M.append([E[2 * i], E[2 * i + 1]])
        outcomes.append([1, -1])
    A = [M[0], M[1]]
    B = [M[2], M[3]]
    substitutions = projective_measurement_constraints(A, B)
    C = expectation_values(M, outcomes)
    chsh = -(C[0] * C[2] + C[0] * C[3] + C[1] * C[2] - C[1] * C[3])
    problem = Problem(E, objective=chsh, substitutions=substitutions)
    solution = problem.solve(1, solver=solver_kind)
    assert abs(solution.primal + 2 * np.sqrt(2)) < 10e-5


def test_chsh_mixed_level(solver_kind: object) -> None:
    I_matrix = [[0, -1, 0], [-1, 1, 1], [0, 1, -1]]
    P = Probability([2, 2], [2, 2])
    problem = Problem(
        P.get_all_operators(),
        objective=define_objective_with_I(I_matrix, P),
        substitutions=P.substitutions,
        extramonomials=P.get_extra_monomials("AB"),
    )
    solution = problem.solve(1, solver=solver_kind)
    assert abs(solution.primal + (np.sqrt(2) - 1) / 2) < 10e-5


def test_elegant_bell(solver_kind: object) -> None:
    I_matrix = [
        [0, -1.5, 0.5, 0.5, 0.5],
        [0, 1, 1, -1, -1],
        [0, 1, -1, 1, -1],
        [0, 1, -1, -1, 1],
    ]
    violation = maximum_violation(
        [2, 2, 2], [2, 2, 2, 2], I_matrix, 1, solver=solver_kind
    )[0]
    assert abs(violation + np.sqrt(3)) < 10e-5


def test_nieto_silleras(solver_kind: object) -> None:
    p = [
        0.5,
        0.5,
        0.5,
        0.5,
        0.4267766952966368,
        0.4267766952966368,
        0.4267766952966368,
        0.07322330470336313,
    ]
    P = Probability([2, 2], [2, 2])
    behaviour_constraint = [
        P([0], [0], "A") - p[0],
        P([0], [1], "A") - p[1],
        P([0], [0], "B") - p[2],
        P([0], [1], "B") - p[3],
        P([0, 0], [0, 0]) - p[4],
        P([0, 0], [0, 1]) - p[5],
        P([0, 0], [1, 0]) - p[6],
        P([0, 0], [1, 1]) - p[7],
    ]
    behaviour_constraint.append(MomentEntry(0, 0, 0) - 1.0)
    problem = Problem(
        P.get_all_operators(),
        objective=-P([0], [0], "A"),
        momentequalities=behaviour_constraint,
        substitutions=P.substitutions,
        normalized=False,
    )
    solution = problem.solve(1, solver=solver_kind)
    assert abs(solution.primal + 0.5) < 10e-5
