"""Chordal sparsity extension for sparse SDP relaxations.

Ported from ``chordal_extension`` (src.old/ncpol2sdpa/chordal_extension.py),
which itself follows the MATLAB SparsePOP implementation. The correlative
sparsity pattern of the objective and constraints is completed to a
chordal graph -- via the Cholesky factorization of the pattern matrix --
and the maximal cliques of the completion become the variable sets of a
multipartite relaxation.

The old implementation filled the pattern matrix with ``random.random()``
draws; the factorization pattern is independent of the fill values, so the
output never depended on them, but the draws are gone now: the entries are
pinned to 1.0 and the result is fully deterministic. The optional chompack
path (AMD ordering) remains available through an explicit ``method``
argument instead of the old import-time detection.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .expressions import flatten
from .variables import get_support


def sliding_cliques(k: int, n: int) -> list[list[int]]:
    """The sliding-window clique layout of the SparsePOP sparsity pattern.

    Each clique combines two length-``n`` sliding windows of width
    ``k/2``: the first half of the variables forms one window side, the
    second half the other. The returned indicators have length ``2n``.

    Args:
        k: The window width; the half-width is ``k // 2``.
        n: The number of variables per window side.

    Returns:
        One indicator list per clique.
    """
    one_sides: list[list[int]] = []
    for i in range(n - int(k / 2) + 1):
        side = [0] * n
        for j in range(int(k / 2)):
            side[i + j] = 1
        one_sides.append(side)
    cliques: list[list[int]] = []
    for side1 in one_sides:
        for side2 in one_sides:
            clique = list(side1)
            clique.extend(side2)
            cliques.append(clique)
    return cliques


def _fill_pattern(
    variables: list[Any],
    obj: Any,
    inequalities: Any,
    equalities: Any,
    momentinequalities: Any,
    momentequalities: Any,
) -> np.ndarray:
    """The pattern matrix of the correlative sparsity.

    Two variables correlate if they share a monomial of the objective or
    a constraint; the diagonal is boosted so that the pattern matrix is
    positive definite for the Cholesky completion.

    Args:
        variables: The problem variables the support is indexed against.
        obj: The objective polynomial, or None.
        inequalities: The inequality constraints.
        equalities: The equality constraints.
        momentinequalities: The moment inequalities.
        momentequalities: The moment equalities.

    Returns:
        The pattern matrix.
    """
    n_dim = len(variables)
    rmat = np.eye(n_dim)
    # The fill value is arbitrary (the factorization pattern depends only
    # on the nonzero structure); the old implementation drew a random
    # number per monomial, which made the module import-time dependent on
    # the RNG state without ever changing the result.
    value = 1.0
    if obj is not None:
        for support in get_support(variables, obj):
            nonzeros = np.nonzero(support)[0]
            for i in nonzeros:
                for j in nonzeros:
                    rmat[i, j] = value
    for polynomial in flatten(
        [inequalities, equalities, momentinequalities, momentequalities]
    ):
        support = np.any(get_support(variables, polynomial), axis=0)
        nonzeros = np.nonzero(support)[0]
        for i in nonzeros:
            for j in nonzeros:
                rmat[i, j] = value
    return rmat + 5 * n_dim * np.eye(n_dim)


def _clique_set_from_cholesky(pattern: np.ndarray) -> np.ndarray:
    """The maximal cliques from the Cholesky completion of the pattern.

    Every row of the upper-triangular factor that is not covered by an
    earlier row -- the new variables its pattern introduces -- is a
    maximal clique of the chordal completion.

    Args:
        pattern: The positive definite pattern matrix.

    Returns:
        One indicator row per clique.
    """
    # TODO(chenxu): an approximate minimum degree ordering should precede
    # the Cholesky decomposition to sparsen the completion.
    factor = np.linalg.cholesky(pattern).T
    factor[np.nonzero(factor)] = 1
    remaining_indices = [0]
    for i in range(1, len(pattern)):
        check_set = factor[i, i :]
        one = np.nonzero(check_set)[0]
        n_ones = len(one)
        clique_result = np.dot(factor[:i, i:], check_set.T)
        covered = any(t == n_ones for t in clique_result)
        if not covered:
            remaining_indices.append(i)
    return factor[remaining_indices, :]


def _clique_set_from_chompack(pattern: np.ndarray) -> np.ndarray:
    """The maximal cliques from the chompack symbolic factorization.

    Args:
        pattern: The positive definite pattern matrix.

    Returns:
        One indicator row per clique.

    Raises:
        ImportError: If the optional chompack/cvxopt pair is missing.
    """
    import chompack as cp
    from cvxopt import amd, spmatrix

    n_dim = len(pattern)
    rows, cols = np.nonzero(pattern)
    rmat = spmatrix(
        1.0,
        [int(r) for r in rows],
        [int(c) for c in cols],
        (n_dim, n_dim),
    )
    symbolic = cp.symbolic(rmat, p=amd.order)
    permutation = symbolic.p
    cliques = symbolic.cliques()
    result = np.zeros((len(cliques), n_dim))
    for i, clique in enumerate(cliques):
        for j in range(len(clique)):
            result[i, permutation[cliques[i][j]]] = 1
    return result


def find_clique_index(
    variables: list[Any], polynomial: Any, clique_set: np.ndarray
) -> int:
    """The first clique whose variables cover the support of a polynomial.

    Args:
        variables: The problem variables the support is indexed against.
        polynomial: A polynomial over the variables.
        clique_set: One indicator row per clique.

    Returns:
        The clique index, or -1 if no clique covers the support.
    """
    support = np.any(get_support(variables, polynomial), axis=0)
    support[np.nonzero(support)[0]] = 1
    for i, clique in enumerate(clique_set):
        if np.dot(support, clique) == len(np.nonzero(support)[0]):
            return i
    return -1


def find_variable_cliques(
    variables: list[Any],
    objective: Any = None,
    inequalities: list[Any] | None = None,
    equalities: list[Any] | None = None,
    momentinequalities: list[Any] | None = None,
    momentequalities: list[Any] | None = None,
    method: str = "fill",
) -> list[list[Any]]:
    """The variable cliques of the chordal completion of the problem.

    Args:
        variables: The problem variables.
        objective: The objective polynomial, or None.
        inequalities: The inequality constraints.
        equalities: The equality constraints.
        momentinequalities: The moment inequalities.
        momentequalities: The moment equalities.
        method: The completion algorithm: ``"fill"`` (deterministic
            Cholesky fill, the default) or ``"chompack"`` (AMD ordering
            through the optional chompack package).

    Returns:
        One list of variables per clique.

    Raises:
        ValueError: If nothing is given to extract the pattern from.
        ImportError: If ``method="chompack"`` and the optional
            chompack/cvxopt pair is missing.
    """
    if (
        objective is None
        and inequalities is None
        and equalities is None
        and momentinequalities is None
        and momentequalities is None
    ):
        raise ValueError(
            "There is nothing to extract the chordal structure from!"
        )
    if method == "fill":
        pattern = _fill_pattern(
            variables,
            objective,
            inequalities,
            equalities,
            momentinequalities,
            momentequalities,
        )
        clique_set = _clique_set_from_cholesky(pattern)
    elif method == "chompack":
        pattern = _fill_pattern(
            variables,
            objective,
            inequalities,
            equalities,
            momentinequalities,
            momentequalities,
        )
        clique_set = _clique_set_from_chompack(pattern)
    else:
        raise ValueError(
            f"Unknown chordal completion method {method!r}; expected "
            f"'fill' or 'chompack'."
        )
    return [
        [variables[i] for i in np.nonzero(clique)[0]] for clique in clique_set
    ]
