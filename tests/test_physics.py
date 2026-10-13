"""Physics-driven tests: ladder-operator constraint sets and a Hubbard chain.

Ported from the legacy HarmonicOscillator and Magnetization cases.
Magnetization uses a polynomial moment
equality (the particle-number constraint) and extracts a value from the
solution through ``monomial_value``.
"""

from __future__ import annotations

import pytest
from sympy.physics.quantum.dagger import Dagger

from ncpolopt.expressions import flatten
from ncpolopt.physics import (
    Probability,
    bosonic_constraints,
    fermionic_constraints,
    get_neighbors,
)
from ncpolopt.problem import Problem
from ncpolopt.solvers.base import SolverKind, UnsupportedSdpError
from ncpolopt.variables import generate_operators


def test_harmonic_oscillator(solver_kind: SolverKind) -> None:
    N = 3
    a = generate_operators("a", N)
    substitutions = bosonic_constraints(a)
    hamiltonian = sum(Dagger(a[i]) * a[i] for i in range(N))
    problem = Problem(a, objective=hamiltonian, substitutions=substitutions)
    solution = problem.solve(1, solver=solver_kind)
    assert abs(solution.primal) < 10e-5


def test_magnetization(solver_kind: SolverKind) -> None:
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


def test_marginal_follows_caller_party_order() -> None:
    """An out-of-order marginal is not silently re-sorted.

    Regression: ``['B', 'A']`` was sorted to ``[0, 1]``, pairing the
    outputs with the wrong parties without any error.
    """
    P = Probability([2, 2], [2, 2])
    # Output 0 on Bob's input 0, output 1 on Alice's input 1.
    result = P([0, 1], [0, 1], ["B", "A"])
    A0 = P.parties[0][1][0]
    B0 = P.parties[1][0][0]
    assert result == B0 * (1 - A0) or result == (1 - A0) * B0


def test_marginal_rejects_bad_indices() -> None:
    """Negative and out-of-range indices raise instead of misbehaving."""
    P = Probability([2, 2], [2, 2])
    with pytest.raises(ValueError, match="projection"):
        P([-1], [0], "A")
    with pytest.raises(ValueError, match="measurement"):
        P([0], [-1], "A")
    with pytest.raises(ValueError, match="party index"):
        P([0], [0], [5])


def test_pauli_constraints() -> None:
    """The Pauli rule set: squares, on-site anticommutation, cross-site commutation."""
    from ncpolopt.physics import pauli_constraints
    from ncpolopt.substitutions import apply_substitutions

    X = generate_operators("X", 2, hermitian=True)
    Y = generate_operators("Y", 2, hermitian=True)
    Z = generate_operators("Z", 2, hermitian=True)
    substitutions = pauli_constraints(X, Y, Z)
    # 6 rules per site (squares + anticommutation) plus 9 per site pair.
    assert len(substitutions) == 2 * 6 + 9
    assert apply_substitutions(X[0] * X[0], substitutions) == 1
    assert apply_substitutions(Y[0] * X[0], substitutions) == -X[0] * Y[0]
    assert apply_substitutions(Z[0] * Y[0], substitutions) == -Y[0] * Z[0]
    assert apply_substitutions(X[1] * X[0], substitutions) == X[0] * X[1]
    assert apply_substitutions(Z[0] * X[1], substitutions) == X[1] * Z[0]


def test_correlator_binary() -> None:
    """The binary correlator is (A0 - A1)(B0 - B1)."""
    from ncpolopt.physics import correlator

    A = [generate_operators("A", 2, hermitian=True)]
    B = [generate_operators("B", 2, hermitian=True)]
    [[corr]] = correlator(A, B)
    expected = (A[0][0] - A[0][1]) * (B[0][0] - B[0][1])
    from sympy import expand

    assert expand(corr - expected) == 0


def test_get_neighbors_2d_and_periodic() -> None:
    """Corner/edge/periodic branches of the lattice neighbor walk."""
    # 3x3 lattice, top-left corner: right and down.
    assert get_neighbors(0, 3) == [1, 3]
    # Right edge: only down.
    assert get_neighbors(2, 3) == [5]
    # Bottom-right corner: nothing.
    assert get_neighbors(8, 3) == []
    # Periodic right edge wraps to the row start; bottom row wraps to top.
    assert get_neighbors(2, 3, periodic=True) == [0, 5]
    assert get_neighbors(7, 3, periodic=True) == [8, 1]
    # Non-square 2x4 lattice visited row-wise.
    assert get_neighbors(5, 2, width=4) == [6]


def test_get_next_neighbors() -> None:
    """Distance-2 neighbors exclude the nearer sites."""
    from ncpolopt.physics import get_next_neighbors

    # distance 1 matches get_neighbors exactly.
    assert get_next_neighbors(0, 3) == get_neighbors(0, 3)
    # 3x3 open lattice from the corner: distance-2 sites are 2, 4, 6.
    assert set(get_next_neighbors(0, 3, distance=2)) == {2, 4, 6}
    # Multiple seeds (forward neighbors only, so corner 8 has none).
    assert set(get_next_neighbors([0, 2], 3, distance=1)) == {1, 3, 5}


def test_define_objective_with_I_argument_forms() -> None:
    """The (A, B) and Probability forms agree; bad arity raises."""
    from ncpolopt.physics import define_objective_with_I

    # CHSH in Collins-Gisin form.
    I_matrix = [[0, 0, 0], [0, 1, 1], [0, 1, -1]]
    P = Probability([2, 2], [2, 2])
    via_probability = define_objective_with_I(I_matrix, P)
    via_lists = define_objective_with_I(I_matrix, P.parties[0], P.parties[1])
    from sympy import expand

    assert expand(via_probability - via_lists) == 0
    with pytest.raises(ValueError, match="two parties"):
        define_objective_with_I(I_matrix, P.parties[0], P.parties[1], P.parties[0])
    with pytest.raises(ValueError, match="two parties"):
        define_objective_with_I(I_matrix)


def test_probability_labels_and_extra_monomials() -> None:
    """Custom labels name the operators; a wrong label count raises."""
    with pytest.raises(ValueError, match="Incorrect number of labels"):
        Probability([2, 2], [2, 2], labels=["A"])
    P = Probability([2], [2], labels=["X", "Y"])
    X0 = P.parties[0][0][0]
    assert str(X0) == "X0"
    # No party named: no extra monomials.
    assert P.get_extra_monomials() == []
    assert P.get_extra_monomials("X")  # one party's worth


def test_probability_call_validation() -> None:
    """Inconsistent calls raise before indexing anything."""
    P = Probability([2, 2], [2, 2])
    with pytest.raises(ValueError, match="does not match the number of outputs"):
        P([0, 1], [0])
    with pytest.raises(ValueError, match="exceeds the number of parties"):
        P([0, 0, 0], [0, 0, 0])
    with pytest.raises(ValueError, match="without defining which"):
        P([0], [0])
    with pytest.raises(ValueError, match="does not match"):
        P([0], [0], ["A", "B"])
