"""Variable factories and support computations."""

from __future__ import annotations

from sympy import Symbol
from sympy.physics.quantum import HermitianOperator, Operator
from sympy.physics.quantum.dagger import Dagger

from ncpolopt.variables import (
    find_variable_set,
    generate_operators,
    generate_variables,
    get_support,
    get_support_variables,
)


def test_generate_variables_commutative_real() -> None:
    """Commutative variables default to real symbols."""
    x = generate_variables("x", 2)
    assert all(isinstance(v, Symbol) for v in x)
    assert all(v.is_real for v in x)
    assert [str(v) for v in x] == ["x0", "x1"]


def test_generate_variables_single_name() -> None:
    """A single variable keeps the bare name without a digit suffix."""
    (x,) = generate_variables("x")
    assert str(x) == "x"


def test_generate_variables_complex() -> None:
    """hermitian=False yields complex symbols."""
    x = generate_variables("x", 2, hermitian=False)
    assert all(v.is_complex and not v.is_real for v in x)


def test_generate_operators_hermitian() -> None:
    """hermitian=True yields HermitianOperator instances."""
    X = generate_operators("X", 2, hermitian=True)
    assert all(isinstance(v, HermitianOperator) for v in X)


def test_generate_operators_plain() -> None:
    """Plain operators without hermitian=True are non-Hermitian."""
    X = generate_operators("X", 2)
    assert all(isinstance(v, Operator) for v in X)


def test_get_support_simple() -> None:
    """Support counts variable exponents across monomials."""
    X = generate_operators("X", 2, hermitian=True)
    support = get_support(X, X[0] * X[1] + X[0] ** 2)
    assert sorted(support) == [[1, 1], [2, 0]]


def test_get_support_with_adjoint() -> None:
    """Adjoints are counted on the base variable."""
    a = generate_operators("a", 1)
    support = get_support(a, Dagger(a[0]) * a[0])
    assert support == [[1]]


def test_get_support_variables_unwraps_adjoints() -> None:
    """get_support_variables returns base variables, not daggered ones."""
    a = generate_operators("a", 1)
    support = get_support_variables(Dagger(a[0]) * a[0])
    assert support == [a[0], a[0]]


def test_find_variable_set() -> None:
    """The set fully containing a polynomial's variables is selected."""
    a = generate_operators("a", 2)
    b = generate_operators("b", 2)
    assert find_variable_set([a, b], a[0] * a[1]) == 0
    assert find_variable_set([a, b], b[0] + b[1]) == 1
    assert find_variable_set([a, b], a[0] * b[0]) == -1
    assert find_variable_set(a, a[0]) == 0
