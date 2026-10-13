"""MomentEntry / MomentExpr arithmetic."""

from __future__ import annotations

import pytest

from ncpolopt.moment import MomentEntry, MomentExpr


def test_entry_arithmetic() -> None:
    """Entries add/subtract/scale into flat expressions."""
    e1 = MomentEntry(0, 0, 1)
    e2 = MomentEntry(0, 1, 1)
    assert (e1 + e2) == MomentExpr((e1, e2))
    assert (e1 + 2) == MomentExpr((e1, MomentEntry(coefficient=2)))
    assert (2 + e1) == MomentExpr((e1, MomentEntry(coefficient=2)))
    assert (e1 - e2) == MomentExpr((e1, MomentEntry(0, 1, 1, -1.0)))
    assert (e1 - 1) == MomentExpr((e1, MomentEntry(coefficient=-1)))
    assert (1 - e1) == MomentExpr((MomentEntry(coefficient=1), MomentEntry(0, 0, 1, -1.0)))
    assert (e1 * 2) == MomentExpr((MomentEntry(0, 0, 1, 2.0),))
    assert (2 * e1) == MomentExpr((MomentEntry(0, 0, 1, 2.0),))
    assert (-e1) == MomentExpr((MomentEntry(0, 0, 1, -1.0),))


def test_expr_arithmetic_and_zero_combination() -> None:
    """Expressions combine duplicates, drop zeros, and scale."""
    e1 = MomentEntry(0, 0, 1)
    e2 = MomentEntry(0, 1, 1)
    expr = MomentExpr((e1,))
    assert (expr + e2) == MomentExpr((e1, e2))
    assert (expr + MomentExpr((e2,))) == MomentExpr((e1, e2))
    assert (expr + 3) == MomentExpr((e1, MomentEntry(coefficient=3)))
    assert (3 + expr) == MomentExpr((e1, MomentEntry(coefficient=3)))
    assert (expr - 3) == MomentExpr((e1, MomentEntry(coefficient=-3)))
    assert (3 - expr) == MomentExpr(
        (MomentEntry(coefficient=3), MomentEntry(0, 0, 1, -1.0))
    )
    assert (expr * 4) == MomentExpr((MomentEntry(0, 0, 1, 4.0),))
    assert (4 * expr) == MomentExpr((MomentEntry(0, 0, 1, 4.0),))
    # Duplicate positions combine; zero coefficients drop out.
    assert (e1 + (-e1)) == MomentExpr(())
    assert (e1 + e1) == MomentExpr((MomentEntry(0, 0, 1, 2.0),))


def test_unsupported_operands_return_notimplemented() -> None:
    """Arithmetic with unsupported types raises TypeError."""
    e1 = MomentEntry(0, 0, 1)
    expr = MomentExpr((e1,))
    with pytest.raises(TypeError):
        _ = e1 + "x"
    with pytest.raises(TypeError):
        _ = e1 - "x"
    with pytest.raises(TypeError):
        _ = "x" - e1
    with pytest.raises(TypeError):
        _ = e1 * "x"
    with pytest.raises(TypeError):
        _ = "x" * e1
    with pytest.raises(TypeError):
        _ = expr + "x"
    with pytest.raises(TypeError):
        _ = expr - "x"
    with pytest.raises(TypeError):
        _ = "x" - expr
    with pytest.raises(TypeError):
        _ = expr * "x"
    with pytest.raises(TypeError):
        _ = "x" * expr


def test_expr_constant_predicates() -> None:
    """is_constant and constant() reflect the (0, 0, 0) entries."""
    entry = MomentEntry(0, 0, 1)
    constant_expr = MomentExpr((MomentEntry(coefficient=2.0),))
    mixed = MomentExpr((entry, MomentEntry(coefficient=3.0)))
    assert constant_expr.is_constant
    assert not MomentExpr((entry,)).is_constant
    assert not mixed.is_constant
    assert constant_expr.constant() == 2.0
    assert mixed.constant() == 3.0
    assert MomentExpr(()).constant() == 0
