"""Chordal sparsity extension tests.

The SparsePop numerical test is ported from the old suite
(``SparsePop.test_chordal_extension``, src.old/tests/test_ncpol2sdpa.py).
The old pattern matrix was filled with ``random.random()`` draws; the
completion only depends on the nonzero structure, so the old test was
deterministic in practice, and the fill is now pinned to a constant so
the determinism is explicit. The optional chompack path is asserted to
produce the same clique sets.
"""

from __future__ import annotations

import numpy as np
import pytest

from ncpolopt.chordal import (
    find_clique_index,
    find_variable_cliques,
    sliding_cliques,
)
from ncpolopt.problem import Problem
from ncpolopt.variables import generate_variables


def _indicator_matrix(variables: list[object], cliques: list[list[object]]) -> np.ndarray:
    """The clique indicator rows expected by :func:`find_clique_index`.

    Args:
        variables: The problem variables.
        cliques: The variable cliques.

    Returns:
        One indicator row per clique.
    """
    return np.array(
        [[1 if variable in clique else 0 for variable in variables] for clique in cliques]
    )


def test_sparse_pop_chordal_extension() -> None:
    """The chordal relaxation reaches the old SparsePop bound."""
    X = generate_variables("x", 3, commutative=True)
    inequalities = [1 - X[0] ** 2 - X[1] ** 2, 1 - X[1] ** 2 - X[2] ** 2]
    problem = Problem(
        X,
        objective=X[1] - 2 * X[0] * X[1] + X[1] * X[2],
        inequalities=inequalities,
    )
    relaxation = problem.relaxation(level=2, chordal_extension=True)
    # Two moment blocks over the cliques {x0, x1} and {x1, x2}, plus the
    # two localizing blocks of the constraints.
    assert [block.size for block in relaxation.sdp.blocks] == [6, 6, 3, 3]
    solution = relaxation.solve()
    assert abs(solution.primal - (-2.2443690631722637)) < 1e-5


def test_chordal_extension_deterministic() -> None:
    """The completion is deterministic and matches the chompack path."""
    X = generate_variables("x", 4, commutative=True)
    objective = X[0] * X[1] + X[2] * X[3]
    inequalities = [1 - X[0] ** 2 - X[1] ** 2, 1 - X[2] ** 2 - X[3] ** 2]
    first = find_variable_cliques(X, objective, inequalities)
    assert first == find_variable_cliques(X, objective, inequalities)
    assert [[str(variable) for variable in clique] for clique in first] == [
        ["x0", "x1"],
        ["x2", "x3"],
    ]
    pytest.importorskip("chompack")
    pytest.importorskip("cvxopt")
    chompack = find_variable_cliques(
        X, objective, inequalities, method="chompack"
    )
    assert chompack == first


def test_find_clique_index() -> None:
    """The covering clique of a polynomial support is located."""
    X = generate_variables("x", 3, commutative=True)
    cliques = find_variable_cliques(X, X[0] * X[1] + X[1] * X[2])
    indicators = _indicator_matrix(X, cliques)
    assert find_clique_index(X, X[0] * X[1], indicators) == 0
    # x0 and x2 never co-occur in a clique: no clique covers the support.
    assert find_clique_index(X, X[0] * X[2], indicators) == -1


def test_sliding_cliques() -> None:
    """The sliding-window layout combines two half-windows per clique."""
    cliques = sliding_cliques(2, 3)
    assert len(cliques) == 9  # (n - k/2 + 1)^2 pairs of windows
    assert all(len(clique) == 6 for clique in cliques)
    assert all(sum(clique) == 2 for clique in cliques)


def test_chordal_extension_requires_input() -> None:
    """Nothing to extract the pattern from is rejected."""
    X = generate_variables("x", 3, commutative=True)
    with pytest.raises(ValueError, match="nothing to extract"):
        find_variable_cliques(X)


def test_unknown_method_rejected() -> None:
    """An unknown completion method is rejected."""
    X = generate_variables("x", 3, commutative=True)
    with pytest.raises(ValueError, match="Unknown chordal completion"):
        find_variable_cliques(X, X[0] * X[1], method="bogus")
