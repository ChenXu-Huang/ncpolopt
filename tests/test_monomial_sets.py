"""Monomial basis generation and variable-count estimation."""

from __future__ import annotations

import pytest
from sympy import S
from sympy.physics.quantum.dagger import Dagger

from ncpolopt.monomial_sets import estimate_n_vars, generate_monomial_sets
from ncpolopt.variables import generate_operators, generate_variables


def test_level_minus_one_requires_monomials() -> None:
    """Level -1 without extramonomials is an error."""
    X = generate_operators("X", 2, hermitian=True)
    with pytest.raises(ValueError):
        generate_monomial_sets(X, -1, None, {})


def test_level_minus_one_flat_list() -> None:
    """A flat list at level -1 becomes the single monomial basis."""
    X = generate_operators("X", 2, hermitian=True)
    basis = [S.One, X[0], X[1]]
    assert generate_monomial_sets(X, -1, basis, {}) == [basis]


def test_level_minus_one_list_of_lists() -> None:
    """A list of lists at level -1 is taken verbatim as the full layout."""
    X = generate_operators("X", 2, hermitian=True)
    layout = [[S.One, X[0]], [S.One, X[1]]]
    assert generate_monomial_sets(X, -1, layout, {}) == layout


def test_level_zero_generates_identity_only() -> None:
    """Level 0 with commutative substitutions drops higher-degree words."""
    X = generate_operators("X", 2, hermitian=True)
    assert generate_monomial_sets(X, 0, None, {}) == [[S.One]]


def test_single_set_level_one() -> None:
    """Level 1 over two Hermitian operators: 1 + 2 monomials."""
    X = generate_operators("X", 2, hermitian=True)
    (basis,) = generate_monomial_sets(X, 1, None, {})
    assert basis == [S.One, X[0], X[1]]


def test_multipartite_one_basis_per_variable_set() -> None:
    """Each variable set receives its own monomial basis."""
    a = generate_operators("a", 1)
    b = generate_operators("b", 2, hermitian=True)
    bases = generate_monomial_sets([a, b], 1, None, {})
    assert bases == [[S.One, a[0], Dagger(a[0])], [S.One, b[0], b[1]]]


def test_extramonomials_first_entry_mixed_rest_verbatim() -> None:
    """The first extra list joins the generated basis; the rest are their own
    blocks verbatim (multi-block layouts at a positive level)."""
    X = generate_operators("X", 2, hermitian=True)
    bases = generate_monomial_sets(X, 1, [[X[0] * X[1]], [X[1]]], {})
    assert X[0] * X[1] in bases[0]
    assert bases[1] == [X[1]]


def test_estimate_n_vars_normalized() -> None:
    """Normalized: n(n+1)/2 minus the constant term, plus parameters."""
    X = generate_operators("X", 2, hermitian=True)
    (basis,) = generate_monomial_sets(X, 1, None, {})
    assert estimate_n_vars([basis], None, True) == 5  # 3*4/2 - 1
    parameters = generate_variables("p", 2)
    assert estimate_n_vars([basis], parameters, True) == 7


def test_estimate_n_vars_non_normalized() -> None:
    """Without normalization the constant term becomes a free variable."""
    X = generate_operators("X", 2, hermitian=True)
    (basis,) = generate_monomial_sets(X, 1, None, {})
    assert estimate_n_vars([basis], None, False) == 6


def test_estimate_n_vars_rectangular_block() -> None:
    """Rectangular [A, B] blocks span the tensor product, len(A)*len(B).

    The old code sized these blocks by len(A) alone, disagreeing with the
    block generation; the product size is fixed at the source.
    """
    a = generate_operators("a", 3)
    assert estimate_n_vars([[a[:2], a]], None, False) == 6 * 7 // 2
