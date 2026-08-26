"""Physics-driven tests: ladder-operator constraint sets and a Hubbard chain.

Ported from HarmonicOscillator and Magnetization
(src.old/tests/test_ncpol2sdpa.py). Magnetization uses a polynomial moment
equality (the particle-number constraint) and extracts a value from the
solution through ``monomial_value``.
"""

from __future__ import annotations

import pytest
from sympy.physics.quantum.dagger import Dagger

from ncpolopt.expressions import flatten
from ncpolopt.physics import (
    bosonic_constraints,
    fermionic_constraints,
    get_neighbors,
)
from ncpolopt.problem import Problem
from ncpolopt.solvers.base import UnsupportedSdpError
from ncpolopt.variables import generate_operators


def test_harmonic_oscillator(solver_kind: object) -> None:
    N = 3
    a = generate_operators("a", N)
    substitutions = bosonic_constraints(a)
    hamiltonian = sum(Dagger(a[i]) * a[i] for i in range(N))
    problem = Problem(a, objective=hamiltonian, substitutions=substitutions)
    solution = problem.solve(1, solver=solver_kind)
    assert abs(solution.primal) < 10e-5


def test_magnetization(solver_kind: object) -> None:
    length, n, h, U, t = 2, 0.8, 3.8, -6, 1
    fu = generate_operators("fu", length)
    fd = generate_operators("fd", length)
    _b = flatten([fu, fd])
    monomials = [list(_b)]
    monomials[-1].extend([Dagger(ci) for ci in _b])
    monomials.append([cj * ci for ci in _b for cj in _b])
    monomials.append([Dagger(cj) * ci for ci in _b for cj in _b])
    monomials[-1].extend([cj * Dagger(ci) for ci in _b for cj in _b])
    monomials.append([Dagger(cj) * Dagger(ci) for ci in _b for cj in _b])
    hamiltonian = 0
    for j in range(length):
        hamiltonian += U * (Dagger(fu[j]) * Dagger(fd[j]) * fd[j] * fu[j])
        hamiltonian += -h / 2 * (Dagger(fu[j]) * fu[j] - Dagger(fd[j]) * fd[j])
        for k in get_neighbors(j, len(fu), width=1):
            hamiltonian += -t * Dagger(fu[j]) * fu[k] - t * Dagger(fu[k]) * fu[j]
            hamiltonian += -t * Dagger(fd[j]) * fd[k] - t * Dagger(fd[k]) * fd[j]
    momentequalities = [n - sum(Dagger(br) * br for br in _b)]
    problem = Problem(
        _b,
        objective=hamiltonian,
        momentequalities=momentequalities,
        substitutions=fermionic_constraints(_b),
        extramonomials=monomials,
    )
    try:
        solution = problem.solve(-1, solver=solver_kind)
    except UnsupportedSdpError:
        # The extra moment matrices have different sizes than the moment
        # matrix; backends that map every variable onto one matrix variable
        # (PICOS) cannot represent the extra blocks.
        pytest.skip(
            f"The {solver_kind.value} backend cannot represent extra moment "
            f"matrices of different sizes"
        )
    s = 0.5 * (
        sum(Dagger(u) * u for u in fu) - sum(Dagger(d) * d for d in fd)
    )
    magnetization = solution.monomial_value(s)
    assert abs(magnetization - 0.021325317328560453) < 10e-5
