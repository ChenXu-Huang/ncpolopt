"""MomentEntry / MomentExpr arithmetic."""

from __future__ import annotations

from ncpolopt.moment import MomentEntry, MomentExpr


def test_entry_arithmetic() -> None:
    """Entries combine into expressions with summed coefficients."""
    a = MomentEntry(0, 0, 0)
    b = MomentEntry(1, 2, 3, coefficient=2.0)
    expr = a + b
    assert isinstance(expr, MomentExpr)
    assert a + b - b == MomentExpr((a,))


def test_zero_coefficients_cancel() -> None:
    """Adding an entry to its negative cancels to the empty expression."""
    a = MomentEntry(2, 1, 1, coefficient=1.0)
    expr = a + (-a)
    assert expr.terms == ()


def test_scalar_multiplication() -> None:
    """Scalars multiply the coefficient of each entry."""
    a = MomentEntry(0, 1, 2, coefficient=1.0)
    assert 3 * a == a * 3 == MomentExpr((MomentEntry(0, 1, 2, coefficient=3.0),))


def test_constant_handling() -> None:
    """Constants are entries at the (0,0,0) position."""
    expr = MomentEntry(0, 0, 0, coefficient=2.0) + 1
    assert expr.is_constant
    assert expr.constant() == 3.0


def test_subtraction_from_constant() -> None:
    """MomentEntry(0,0,0) - 1.0 reproduces the old '+0[0,0]-1.0' DSL."""
    expr = MomentEntry(0, 0, 0) - 1.0
    assert expr.constant() == 0.0
    assert isinstance(expr, MomentExpr)
