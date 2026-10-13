"""Substitution machinery: matches SymPy's own .subs() on the covered cases."""

from __future__ import annotations

from typing import Any

import pytest
from sympy import I, S, Symbol, expand
from sympy.physics.quantum.dagger import Dagger

from ncpolopt.substitutions import apply_substitutions, fast_substitute
from ncpolopt.variables import generate_operators


def _sympy_fixpoint(monomial, substitutions):
    """Reference implementation: plain .subs() iterated to a fixpoint."""
    original_monomial = monomial
    changed = True
    while changed:
        for lhs, rhs in substitutions.items():
            monomial = monomial.subs(lhs, rhs)
        if original_monomial == monomial:
            changed = False
        original_monomial = monomial
    return monomial


@pytest.fixture(scope="module")
def f() -> list:
    """Two non-Hermitian operators."""
    return generate_operators("f", 2)


@pytest.fixture(scope="module")
def g() -> list:
    """Two non-Hermitian operators, distinct from ``f``."""
    return generate_operators("g", 2)


def test_fast_substitute_dagger_product(f) -> None:
    """Substituting a daggered word into the identical word."""
    monomial = Dagger(f[0]) * f[0]
    lhs = Dagger(f[0]) * f[0]
    rhs = -f[0] * Dagger(f[0])
    assert fast_substitute(monomial, lhs, rhs) == monomial.subs(lhs, rhs)


def test_fast_substitute_word_with_power(f) -> None:
    """The pattern matches a run with an exponent on the right."""
    monomial = Dagger(f[0]) * f[0] ** 2
    lhs = Dagger(f[0]) * f[0]
    rhs = -f[0] * Dagger(f[0])
    assert fast_substitute(monomial, lhs, rhs) == monomial.subs(lhs, rhs)


def test_fast_substitute_power_on_left(f) -> None:
    """The pattern matches a run with an exponent on the left."""
    monomial = Dagger(f[0]) ** 2 * f[0]
    lhs = Dagger(f[0]) * f[0]
    rhs = -f[0] * Dagger(f[0])
    assert fast_substitute(monomial, lhs, rhs) == monomial.subs(lhs, rhs)


def test_fast_substitute_dagger_power(f) -> None:
    """A daggered power pattern with an unmatched left remainder."""
    monomial = Dagger(f[0]) ** 2 * f[0]
    lhs = Dagger(f[0]) ** 2
    rhs = -f[0] * Dagger(f[0])
    assert fast_substitute(monomial, lhs, rhs) == monomial.subs(lhs, rhs)


def test_fast_substitute_commuting_coefficient(f, g) -> None:
    """A commuting numeric coefficient is preserved around the match."""
    monomial = 2 * g[0] ** 3 * g[1] * Dagger(f[0]) ** 2 * f[0]
    lhs = Dagger(f[0]) ** 2
    rhs = -f[0] * Dagger(f[0])
    assert fast_substitute(monomial, lhs, rhs) == monomial.subs(lhs, rhs)


@pytest.mark.parametrize("monomial", [S.One, 5])
def test_fast_substitute_number_or_identity(f, monomial) -> None:
    """Numbers and the identity pass through untouched."""
    lhs = Dagger(f[0]) ** 2
    rhs = -f[0] * Dagger(f[0])
    assert fast_substitute(monomial, lhs, rhs) == monomial


def test_fast_substitute_in_sum(f, g) -> None:
    """Substitution inside a sum applies to every term."""
    monomial = 2 * g[0] ** 3 * g[1] * Dagger(f[0]) ** 2 * f[0] + f[1]
    lhs = Dagger(f[0]) ** 2
    rhs = -f[0] * Dagger(f[0])
    assert fast_substitute(monomial, lhs, rhs) == monomial.subs(lhs, rhs)


def test_fast_substitute_rhs_is_sum_expands(f) -> None:
    """A sum replacement expands the resulting product."""
    monomial = f[1] * Dagger(f[0]) ** 2 * f[0]
    lhs = f[1]
    rhs = 1.0 + f[0]
    assert fast_substitute(monomial, lhs, rhs) == expand(monomial.subs(lhs, rhs))


def test_fast_substitute_iterated(f) -> None:
    """Repeated substitution of a sum replacement stays consistent."""
    monomial = f[1] ** 2 * Dagger(f[0]) ** 2 * f[0]
    lhs = f[1]
    rhs = 1.0 + f[0]
    result = fast_substitute(fast_substitute(monomial, lhs, rhs), lhs, rhs)
    assert result == expand(monomial.subs(lhs, rhs))


def test_apply_substitutions_reaches_fixpoint(f) -> None:
    """apply_substitutions iterates rules until nothing changes."""
    substitutions = {
        Dagger(f[0]) * f[0]: -f[0] * Dagger(f[0]),
        Dagger(f[1]) * f[1]: -f[1] * Dagger(f[1]),
    }
    monomial = Dagger(f[0]) * f[0] * Dagger(f[1]) * f[1]
    assert apply_substitutions(monomial, substitutions) == _sympy_fixpoint(
        monomial, substitutions
    )


def test_apply_substitutions_fermionic_chain() -> None:
    """Fermionic constraints on a two-site Hubbard chain.

    Mirrors the old ``ApplySubstitutions`` test, rule set included: number
    constraints ``a**2 = 0``, the density rule ``a*Dagger(a) -> 1 -
    Dagger(a)*a`` and the cross-site anti-commutation rules. The rule set
    matters -- a naive density rule on its own cycles on the interaction
    term and falls outside the documented working domain of
    ``fast_substitute``.
    """
    from ncpolopt.expressions import flatten

    fu = generate_operators("fu", 2)
    fd = generate_operators("fd", 2)
    _b = flatten([fu, fd])
    substitutions: dict = {}
    for i, op in enumerate(_b):
        substitutions[op**2] = 0
        substitutions[Dagger(op) ** 2] = 0
        substitutions[op * Dagger(op)] = 1.0 - Dagger(op) * op
        for other in _b[i + 1 :]:
            substitutions[op * Dagger(other)] = -Dagger(other) * op
            substitutions[Dagger(op) * other] = -other * Dagger(op)
            substitutions[op * other] = -other * op
            substitutions[Dagger(op) * Dagger(other)] = -Dagger(other) * Dagger(op)
    length, h, U, t = 2, 3.8, -6, 1
    hamiltonian = 0
    for j in range(length):
        hamiltonian += U * (Dagger(fu[j]) * Dagger(fd[j]) * fd[j] * fu[j])
        hamiltonian += -h / 2 * (Dagger(fu[j]) * fu[j] - Dagger(fd[j]) * fd[j])
        for k in ((j + 1) % length,):  # one-dimensional nearest neighbor
            hamiltonian += -t * Dagger(fu[j]) * fu[k] - t * Dagger(fu[k]) * fu[j]
            hamiltonian += -t * Dagger(fd[j]) * fd[k] - t * Dagger(fd[k]) * fd[j]

    # NOTE: SymPy does not expose the factor tuple of ``as_coeff_mul`` in its
    # annotations, so the extraction stays untyped here.
    expanded: Any = expand(hamiltonian)
    monomials = expanded.as_coeff_mul()[1][0].as_coeff_add()[1]
    substituted = sum(apply_substitutions(m, substitutions) for m in monomials)
    reference = sum(_sympy_fixpoint(m, substitutions) for m in monomials)
    assert substituted == expand(reference)


def test_pure_commutative_rule_keeps_noncommutative_tail() -> None:
    """A rule with no noncommutative factors must not swallow them.

    Regression: the early return for a commutative-only rule dropped the
    noncommutative tail of the monomial (``fast_substitute(A*B, x, 2)``
    returned ``1``), silently collapsing mixed classical/operator problems.
    """
    A = generate_operators("A", 1)[0]
    B = generate_operators("B", 1)[0]
    x = Symbol("x")
    assert fast_substitute(A * B, x, 2) == A * B
    assert fast_substitute(x * A, x, 2) == 2 * A
    assert fast_substitute(x**2 * A * B, x**2, 1) == A * B
    y = Symbol("y")
    assert expand(fast_substitute(x * A, x, y + 1) - (y * A + A)) == 0
    assert apply_substitutions(x * A, {x: 2}) == 2 * A


def test_pure_commutative_rule_end_to_end() -> None:
    """A commutative substitution no longer collapses the moment basis."""
    from ncpolopt.problem import Problem
    from ncpolopt.variables import generate_variables

    x = generate_variables("x", 1)[0]
    A = generate_operators("A", 1)[0]
    problem = Problem([x, A], objective=x * A, substitutions={x: 1})
    relaxation = problem.relaxation(level=1)
    assert relaxation.monomial_sets == [[S.One, A, Dagger(A)]]
    # The objective x*A reduces to A: coefficient 1.0 on its moment.
    assert list(relaxation.sdp.obj) == [1.0, 0.0, 0.0, 0.0]


def test_mixed_rule_replaces_whole_pattern() -> None:
    """A rule with commutative and noncommutative parts fires as a unit.

    Regression: the commutative side was substituted globally AND the
    replacement multiplied in full (``fast_substitute(x*B, x*B, 2)`` gave
    ``4``), and the commutative side fired even when the noncommutative
    side never matched.
    """
    A = generate_operators("A", 1)[0]
    B = generate_operators("B", 1)[0]
    x = Symbol("x")
    y = Symbol("y")
    assert fast_substitute(x * B, x * B, 2) == 2
    assert fast_substitute(x * A * B, x * A, 2) == 2 * B
    assert fast_substitute(x**2 * A, x * A, 2) == 2 * x
    assert fast_substitute(x * A, x * A, y * B) == y * B
    # No fire: noncommutative side does not match, or the commutative side
    # is absent.
    assert fast_substitute(x * A, x * B, 2) == x * A
    assert fast_substitute(A, x * A, 2) == A


def test_apply_substitutions_raises_on_nonconverging_rules() -> None:
    """A word-growing rule set raises instead of hanging."""
    import ncpolopt.substitutions as substitutions_module

    A = generate_operators("A", 1)[0]
    B = generate_operators("B", 1)[0]
    monkeypatch = substitutions_module  # cap via module constant
    original = monkeypatch._MAX_SUBSTITUTION_PASSES
    monkeypatch._MAX_SUBSTITUTION_PASSES = 10
    try:
        with pytest.raises(RuntimeError, match="did not stabilize"):
            apply_substitutions(A, {A: A * B})
    finally:
        monkeypatch._MAX_SUBSTITUTION_PASSES = original


def test_separate_scalar_factor_imaginary_and_plain() -> None:
    """Imaginary units fold into the coefficient; plain numbers split off."""
    from ncpolopt.substitutions import separate_scalar_factor

    A = generate_operators("A", 1)[0]
    monomial, coeff = separate_scalar_factor(2 * I * A)
    assert monomial == A and coeff == 2j
    monomial, coeff = separate_scalar_factor(3.5)
    assert monomial == 1 and coeff == 3.5
    monomial, coeff = separate_scalar_factor(-2 * A)
    assert monomial == A and coeff == -2.0


def test_separate_scalar_factor_exact_variant() -> None:
    """_separate_scalar_factor splits a leading numeric factor exactly."""
    from ncpolopt.substitutions import _separate_scalar_factor

    A = generate_operators("A", 1)[0]
    assert _separate_scalar_factor(2 * A) == (A, 2)
    assert _separate_scalar_factor(A) == (A, 1)
    assert _separate_scalar_factor(7) == (1, 7)


def test_simplify_polynomial_number_passthrough() -> None:
    """A numeric polynomial is returned as-is."""
    from ncpolopt.substitutions import simplify_polynomial

    assert simplify_polynomial(5, {}) == 5


def test_apply_substitutions_pure_filters_irrelevant_rules() -> None:
    """With pure=True, rules over foreign variables are dropped."""
    A = generate_operators("A", 1)[0]
    B = generate_operators("B", 1)[0]
    assert apply_substitutions(A, {B: 1}, pure=True) == A
    assert apply_substitutions(A, {A: B}, pure=True) == B


def test_moment_of_entry_string_passthrough() -> None:
    """A string constraint passes through unchanged (legacy marker path)."""
    from ncpolopt.substitutions import moment_of_entry

    assert moment_of_entry((1, 2), [S.One], "raw", {}) == (1, 2, "raw")


def test_fast_substitute_mismatch_break_paths() -> None:
    """Operator/Symbol/exponent mismatches skip to the next position."""
    A = generate_operators("A", 1)[0]
    B = generate_operators("B", 1)[0]
    x = Symbol("x")
    # Operator vs different operator: no fire.
    assert fast_substitute(A, B, 1) == A
    # Exponent too small for the pattern: no fire.
    assert fast_substitute(A**2, A**3, 1) == A**2
    # Symbol in the monomial never matches an operator pattern.
    assert fast_substitute(x * A, x * B, 1) == x * A


def test_simplify_polynomial_cancelling_expression() -> None:
    """An expression cancelling to a number returns the number."""
    from ncpolopt.substitutions import simplify_polynomial

    A = generate_operators("A", 1)[0]
    assert simplify_polynomial(A - A, {}) == 0


def test_fast_substitute_symbol_mismatch_breaks() -> None:
    """A noncommutative Symbol never matches a different pattern factor."""
    A = generate_operators("A", 1)[0]
    B = generate_operators("B", 1)[0]
    s = Symbol("s", commutative=False)
    assert fast_substitute(s * A, B, 1) == s * A
