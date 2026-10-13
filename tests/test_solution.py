"""Extraction tests: monomial and dual values, SOS decomposition, ranks.

The solved ExampleNoncommutative relaxation is the reference problem: its
optimum (-0.75) and moment values are stable and its dual solution is
available. The extraction methods are tested against identities that hold
regardless of solver details.
"""

from __future__ import annotations

import numpy as np
import pytest
from sympy import expand

from ncpolopt.problem import Problem
from ncpolopt.sdp_problem import SdpProblem
from ncpolopt.solution import Solution
from ncpolopt.solvers.base import SolverResult
from ncpolopt.variables import generate_operators, generate_variables


@pytest.fixture(scope="module")
def noncommutative_solution() -> Solution:
    """The solved ExampleNoncommutative relaxation."""
    X = generate_operators("x", 2, hermitian=True)
    problem = Problem(
        X,
        objective=X[0] * X[1] + X[1] * X[0],
        inequalities=[-X[1] ** 2 + X[1] + 0.5],
        substitutions={X[0] ** 2: X[0]},
    )
    return problem.solve(2)


def test_status_and_values(noncommutative_solution: Solution) -> None:
    assert noncommutative_solution.status == "optimal"
    assert abs(noncommutative_solution.primal + 0.75) < 10e-5
    # The dual objective of a zero-constant problem equals the primal.
    assert abs(noncommutative_solution.dual - noncommutative_solution.primal) < 10e-4
    assert noncommutative_solution.solution_time >= 0


def test_monomial_value_constant(noncommutative_solution: Solution) -> None:
    assert noncommutative_solution.monomial_value(1) == 1.0
    assert abs(noncommutative_solution.monomial_value(0.5) - 0.5) < 10e-8


def test_monomial_value_substitutions(noncommutative_solution: Solution) -> None:
    X = generate_operators("x", 2, hermitian=True)
    # The rule X0**2 -> X0 makes the two monomials one value.
    v_sq = noncommutative_solution.monomial_value(X[0] ** 2)
    v = noncommutative_solution.monomial_value(X[0])
    assert abs(v_sq - v) < 10e-8
    # A projector value lies between 0 and 1 (values are complex-typed).
    assert -10e-6 <= v.real <= 1 + 10e-6


def test_monomial_value_objective_value(noncommutative_solution: Solution) -> None:
    X = generate_operators("x", 2, hermitian=True)
    objective = X[0] * X[1] + X[1] * X[0]
    assert abs(noncommutative_solution.monomial_value(objective) + 0.75) < 10e-5


def test_dual_value_constant(noncommutative_solution: Solution) -> None:
    # dual_value of the constant monomial is -sum_b tr(Y_b A0_b), i.e. the
    # dual objective minus the constant term (which is zero here).
    assert abs(noncommutative_solution.dual_value(1) - noncommutative_solution.dual) < 10e-4


def test_dual_block_shape(noncommutative_solution: Solution) -> None:
    for block in range(len(noncommutative_solution.sdp.blocks)):
        y = noncommutative_solution.dual_block(block)
        assert y.shape == (
            noncommutative_solution.sdp.blocks[block].size,
        ) * 2


def test_sos_decomposition(noncommutative_solution: Solution) -> None:
    # Only the moment block carries the moment basis; the localizing block
    # would be mislabeled by it and is rejected.
    with pytest.raises(RuntimeError, match="Cannot match block"):
        noncommutative_solution.sos_decomposition(blocks=(0, 1))
    sos = noncommutative_solution.sos_decomposition()
    assert len(sos) == 1
    # Reconstructing the quadratic form of the moment block from its dual
    # matrix must reproduce the SOS term exactly.
    sdp = noncommutative_solution.sdp
    basis = list(sdp.moment_basis)
    y = noncommutative_solution.dual_block(0)
    form = sum(
        y[i, j] * basis[i] * basis[j]
        for i in range(len(basis))
        for j in range(len(basis))
    )
    # Eigen-decomposition round-trips introduce float noise; every
    # coefficient of the difference must vanish to numerical precision.
    difference = expand(sos[0] - form)
    assert all(
        abs(coeff) < 1e-10 for coeff in difference.as_coefficients_dict().values()
    )


def test_solution_ranks() -> None:
    x = generate_variables("x", 2, commutative=True)
    g0 = 4 * x[0] ** 2 + x[0] * x[1] - 4 * x[1] ** 2 + x[0] ** 6 / 3
    solution = Problem(x, objective=g0).solve(3)
    ranks = solution.solution_ranks()
    # Level-1, level-2, and level-3 truncations of a 10-monomial basis.
    assert len(ranks) == 3
    assert all(ranks[i] <= ranks[i + 1] for i in range(len(ranks) - 1))
    # A single base level reports that truncation, plus the full-matrix
    # rank when the moment matrix is larger (mirroring the old behavior).
    single = solution.solution_ranks(baselevel=1)
    assert len(single) == 2
    assert single[0] == ranks[0]
    assert single[1] == np.linalg.matrix_rank(
        solution.x_mat[solution.moment_block_indices[0]]
    )


def test_solution_without_primal_raises() -> None:
    from ncpolopt.problem import Problem as P

    x = generate_variables("x", 1, commutative=True)
    relaxation = P(x, objective=x[0] ** 2).relaxation(1)
    result = SolverResult(
        status="failed",
        primal=float("nan"),
        dual=float("nan"),
        solution_time=0.0,
    )
    solution = Solution(relaxation.sdp, result)
    with pytest.raises(RuntimeError, match="not solved"):
        solution.monomial_value(1)
    with pytest.raises(RuntimeError, match="not solved"):
        solution.dual_value(1)


def test_monomial_value_moment_substitutions(
    noncommutative_solution: Solution,
) -> None:
    """Moment substitutions resolve through the stored rules."""
    from ncpolopt import generate_operators
    from ncpolopt.problem import Problem

    X = generate_operators("x", 2, hermitian=True)
    problem = Problem(
        X,
        objective=X[0],
        substitutions={X[0] ** 2: X[0]},
        momentsubstitutions={X[1]: X[0]},
    )
    solution = problem.solve(2)
    # <x1> is pinned to <x0>; both readings must agree.
    assert solution.monomial_value(X[1]) == pytest.approx(
        solution.monomial_value(X[0]), abs=1e-6
    )
    # A constant times a moment picks up the constant.
    assert solution.monomial_value(2 * X[0]) == pytest.approx(
        2 * solution.monomial_value(X[0]), abs=1e-6
    )


def test_monomial_value_out_of_basis_raises(
    noncommutative_solution: Solution,
) -> None:
    """A monomial beyond the relaxation level is a clear error."""
    X = generate_operators("x", 2, hermitian=True)
    with pytest.raises(RuntimeError, match="could not be found"):
        noncommutative_solution.monomial_value(X[1] ** 5)


def test_extraction_without_dual_raises() -> None:
    """Value extraction without a solved dual raises RuntimeError."""
    from ncpolopt.solvers.base import SolverResult

    sdp = noncommutative_solution_sdp()
    bare = Solution(
        sdp, SolverResult(status="unknown", primal=float("nan"), dual=float("nan"))
    )
    with pytest.raises(RuntimeError, match="no dual solution"):
        bare.dual_value(1)
    with pytest.raises(RuntimeError, match="no dual solution"):
        bare.sos_decomposition()
    with pytest.raises(RuntimeError, match="no primal solution"):
        bare.monomial_value(1)


def noncommutative_solution_sdp() -> SdpProblem:
    """The solved problem's frozen SDP (no solver data attached)."""
    from ncpolopt import generate_operators
    from ncpolopt.problem import Problem

    X = generate_operators("x", 2, hermitian=True)
    problem = Problem(
        X,
        objective=X[0] * X[1] + X[1] * X[0],
        inequalities=[-X[1] ** 2 + X[1] + 0.5],
        substitutions={X[0] ** 2: X[0]},
    )
    return problem.relaxation(2).sdp


def test_sos_decomposition_rejects_non_psd_dual(
    noncommutative_solution: Solution,
) -> None:
    """A dual block with a large negative eigenvalue raises."""
    import dataclasses

    import numpy as np

    y_mat = list(noncommutative_solution.y_mat)
    bad = np.array(y_mat[0], dtype=float)
    bad[0, 0] = -1.0  # break positive semidefiniteness hard
    y_mat[0] = bad
    tampered = dataclasses.replace(
        noncommutative_solution.result, y_mat=tuple(y_mat)
    )
    solution = dataclasses.replace(noncommutative_solution, result=tampered)
    with pytest.raises(RuntimeError, match="negative eigenvalue"):
        solution.sos_decomposition()


def test_sos_decomposition_threshold(noncommutative_solution: Solution) -> None:
    """Terms below the threshold are dropped from the SOS polynomial."""
    sos = noncommutative_solution.sos_decomposition(threshold=1e6)
    assert all(term == 0 for term in sos)


def test_monomial_value_constant_reduction() -> None:
    """Monomials reduced to constants by rules read off the (0, 0) entry."""
    from ncpolopt import generate_operators
    from ncpolopt.problem import Problem

    X = generate_operators("x", 2, hermitian=True)
    problem = Problem(
        X, objective=X[1], substitutions={X[0] ** 2: 1}
    )
    solution = problem.solve(2)
    assert solution.monomial_value(X[0] ** 2) == pytest.approx(1.0)
    # A sum with a constant term picks the constant up inside the loop.
    value = solution.monomial_value(X[0] ** 2 + X[1])
    assert value == pytest.approx(1.0 + solution.monomial_value(X[1]).real)
    # A negated monomial flips the sign.
    assert solution.monomial_value(-X[1]) == pytest.approx(
        -solution.monomial_value(X[1]).real
    )


def test_sos_decomposition_requires_moment_basis() -> None:
    """Without a moment basis the SOS decomposition refuses to guess."""
    import dataclasses

    sdp = noncommutative_solution_sdp()
    no_basis = dataclasses.replace(sdp, moment_basis=())
    bare = Solution(
        no_basis, SolverResult(status="optimal", primal=0.0, dual=0.0, y_mat=(np.eye(2),))
    )
    with pytest.raises(RuntimeError, match="no moment basis"):
        bare.sos_decomposition()


def test_monomial_value_per_term_branches() -> None:
    """The extraction loop handles constants, negatives, and substitutions."""
    from ncpolopt import generate_operators
    from ncpolopt.problem import Problem

    X = generate_operators("x", 2, hermitian=True)
    problem = Problem(
        X,
        objective=X[0] * X[1] + X[1] * X[0],
        substitutions={X[0] ** 2: X[0]},
        momentsubstitutions={X[1]: X[0]},
    )
    solution = problem.solve(2)
    # A term reduced to a constant inside the loop.
    assert solution.monomial_value(X[0] ** 2 - 1) == pytest.approx(
        solution.monomial_value(X[0]) - 1.0
    )
    # A negated term.
    assert solution.monomial_value(X[0] - X[1]) == pytest.approx(
        solution.monomial_value(X[0]) - solution.monomial_value(X[1])
    )
    # A moment-substituted term inside a sum.
    assert solution.monomial_value(X[0] + X[1]) == pytest.approx(
        2 * solution.monomial_value(X[0])
    )


def test_monomial_value_add_pin_value() -> None:
    """A moment pin whose value is a sum resolves term by term."""
    from ncpolopt import generate_operators
    from ncpolopt.problem import Problem

    X = generate_operators("x", 2, hermitian=True)
    problem = Problem(
        X,
        objective=X[0],
        substitutions={},
        momentsubstitutions={X[1]: X[0] + X[0] ** 2 - 1, X[0] ** 2: X[0]},
    )
    solution = problem.solve(2)
    # <x1> -> <x0> + <x0^2> - 1; with x0^2 -> x0 this is 2<x0> - 1.
    expected = 2 * solution.monomial_value(X[0]) - 1
    assert solution.monomial_value(X[1]) == pytest.approx(expected, abs=1e-6)


def test_monomial_value_pinned_term_inside_sum() -> None:
    """A substitution-produced sum with a pinned term resolves per term."""
    from ncpolopt import generate_operators
    from ncpolopt.problem import Problem

    X = generate_operators("x", 2, hermitian=True)
    problem = Problem(
        X,
        objective=X[0],
        substitutions={X[0] * X[1]: X[0] + X[1]},
        momentsubstitutions={X[1]: X[0]},
    )
    solution = problem.solve(2)
    # <x0 x1> -> <x0> + <x1> -> <x0> + <x0>.
    assert solution.monomial_value(X[0] * X[1]) == pytest.approx(
        2 * solution.monomial_value(X[0]), abs=1e-6
    )
