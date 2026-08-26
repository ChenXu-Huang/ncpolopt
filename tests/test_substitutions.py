"""Substitution machinery: matches SymPy's own .subs() on the covered cases."""

from __future__ import annotations

import pytest
from sympy import S, expand
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

    monomials = expand(hamiltonian).as_coeff_mul()[1][0].as_coeff_add()[1]
    substituted = sum(apply_substitutions(m, substitutions) for m in monomials)
    reference = sum(_sympy_fixpoint(m, substitutions) for m in monomials)
    assert substituted == expand(reference)
