"""Monomial basis generation and degree bookkeeping."""

from __future__ import annotations

from sympy import S
from sympy.physics.quantum.dagger import Dagger

from ncpolopt.monomials import (
    get_all_monomials,
    get_monomials,
    ncdegree,
    pick_monomials_of_degree,
    pick_monomials_up_to_degree,
    unique,
)
from ncpolopt.variables import generate_operators


def test_get_monomials_level_zero() -> None:
    """Degree 0 yields just the identity."""
    X = generate_operators("X", 2, hermitian=True)
    assert get_monomials(X, 0) == [S.One]


def test_get_monomials_level_minus_one() -> None:
    """Degree -1 yields the empty basis (level 0.5 convention)."""
    X = generate_operators("X", 2, hermitian=True)
    assert get_monomials(X, -1) == []


def test_get_monomials_hermitian_degree_one() -> None:
    """Hermitian operators at degree 1: identity plus the operators."""
    X = generate_operators("X", 2, hermitian=True)
    assert get_monomials(X, 1) == [S.One, X[0], X[1]]


def test_get_monomials_nonhermitian_includes_adjoints() -> None:
    """Non-Hermitian operators bring their adjoints into the basis."""
    a = generate_operators("a", 1)
    monomials = get_monomials(a, 1)
    assert monomials == [S.One, a[0], Dagger(a[0])]


def test_get_monomials_degree_two() -> None:
    """Degree 2 contains all words of length 2 over two operators."""
    X = generate_operators("X", 2, hermitian=True)
    monomials = get_monomials(X, 2)
    assert len(monomials) == 7  # 1 + 2 + 4 words
    assert X[0] * X[1] in monomials
    assert X[0] * X[0] in monomials


def test_get_all_monomials_with_extras() -> None:
    """Extra monomials are appended to the generated basis."""
    X = generate_operators("X", 2, hermitian=True)
    extra = [X[0] * X[1] * X[0]]
    monomials = get_all_monomials(X, extra, None, 2)
    assert extra[0] in monomials


def test_get_all_monomials_removes_substitutions() -> None:
    """Substitution left-hand sides leave the basis and reduce the rest."""
    X = generate_operators("X", 2, hermitian=True)
    monomials = get_all_monomials(X, None, {X[0] ** 2: X[0]}, 2)
    assert X[0] ** 2 not in monomials


def test_ncdegree() -> None:
    """Degree counts operator factors, not numeric ones."""
    a = generate_operators("a", 1)
    assert ncdegree(1) == 0
    assert ncdegree(a[0]) == 1
    assert ncdegree(Dagger(a[0]) ** 2 * a[0]) == 3
    assert ncdegree(a[0] * a[0] + 1) == 2


def test_pick_monomials_by_degree() -> None:
    """Degree selection preserves basis order."""
    X = generate_operators("X", 2, hermitian=True)
    monomials = get_monomials(X, 2)
    assert pick_monomials_of_degree(monomials, 2) == [
        X[0] * X[0],
        X[0] * X[1],
        X[1] * X[0],
        X[1] * X[1],
    ]
    assert pick_monomials_up_to_degree(monomials, 1) == [S.One, X[0], X[1]]


def test_unique_preserves_order() -> None:
    """Deduplication keeps first-occurrence order."""
    assert unique([1, 2, 1, 3, 2]) == [1, 2, 3]
