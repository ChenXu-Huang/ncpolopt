"""Relaxation construction: block layout, entries, and elimination.

The assertions pin the exact SDP data produced for hand-computable small
problems: monomial indexing order, entry positions, the two-half equality
blocks, the constant-handling fix, and the basis transform of equality
elimination.
"""

from __future__ import annotations

from typing import Any

import pytest
from sympy import S
from sympy.physics.quantum.dagger import Dagger

from ncpolopt.moment import MomentEntry
from ncpolopt.problem import Problem
from ncpolopt.solvers.base import SolverError
from ncpolopt.solvers.registry import registered
from ncpolopt.variables import generate_operators, generate_variables


def _entries(relaxation, block: int) -> dict[int, dict[int, Any]]:
    """Read a block's COO entries as {position: {variable: coefficient}}."""
    coo = relaxation.sdp.blocks[block].coo
    result: dict[int, dict[int, Any]] = {}
    for k, position, value in zip(coo.row, coo.col, coo.data, strict=True):
        result.setdefault(int(position), {})[int(k)] = value
    return result


def test_level_one_moment_block() -> None:
    """Level-1 moment matrix of two hermitian operators, normalized."""
    X = generate_operators("X", 2, hermitian=True)
    relaxation = Problem(X, objective=X[0]).relaxation(level=1)
    assert relaxation.monomial_sets == [[S.One, X[0], X[1]]]
    assert relaxation.var_offsets == [5]
    sdp = relaxation.sdp
    assert sdp.n_vars == 5
    assert sdp.constant_term == 0.0
    assert list(sdp.obj) == [1.0, 0.0, 0.0, 0.0, 0.0]
    assert sdp.monomial_index[X[0]] == 1
    assert sdp.monomial_index[X[1]] == 2
    assert sdp.monomial_index[X[0] ** 2] == 3
    assert sdp.monomial_index[X[0] * X[1]] == 4
    assert sdp.monomial_index[X[1] ** 2] == 5
    assert S.One not in sdp.monomial_index
    assert sdp.column_locations[1] == (0, 0, 1)
    # Normalization pins M[0,0] to the constant 1.0 instead of a variable.
    assert _entries(relaxation, 0) == {
        0: {0: 1.0},
        1: {1: 1.0},
        2: {2: 1.0},
        4: {3: 1.0},
        5: {4: 1.0},
        8: {5: 1.0},
    }


def test_nonnormalized_top_left_is_variable() -> None:
    """Without normalization M[0,0] becomes a free variable."""
    X = generate_operators("X", 1, hermitian=True)
    relaxation = Problem(X, normalized=False).relaxation(level=1)
    sdp = relaxation.sdp
    assert sdp.n_vars == 3
    assert sdp.column_locations[1] == (0, 0, 0)
    assert _entries(relaxation, 0)[0] == {1: 1.0}


def test_localizing_matrix_entries() -> None:
    """The localizing matrix of a degree-2 inequality is a 1x1 block."""
    X = generate_operators("X", 2, hermitian=True)
    ineq = -X[1] ** 2 + X[1] + 0.5
    relaxation = Problem(X, inequalities=[ineq]).relaxation(level=1)
    sdp = relaxation.sdp
    assert [b.size for b in sdp.blocks] == [3, 1]
    assert sdp.constraint_to_blocks[ineq] == (1,)
    assert _entries(relaxation, 1) == {0: {0: 0.5, sdp.monomial_index[X[1]]: 1.0, sdp.monomial_index[X[1] ** 2]: -1.0}}


def test_rational_constant_in_constraint_not_dropped() -> None:
    """Exact (Rational) constants survive the polynomial split.

    The old ``as_coeff_add()[1]`` idiom moved a Rational constant into the
    coefficient slot and silently dropped it; float constants happened to
    stay among the terms. The new split reads the slot explicitly.
    """
    X = generate_operators("X", 1, hermitian=True)
    relaxation = Problem(X, inequalities=[X[0] - S(1) / 2]).relaxation(level=1)
    entries = _entries(relaxation, 1)
    assert entries[0][0] == pytest.approx(-0.5)
    assert entries[0][1] == pytest.approx(1.0)


def test_equality_scalar_block_pair() -> None:
    """An equality gets two 1x1 blocks with opposite signs."""
    X = generate_operators("X", 2, hermitian=True)
    equality = X[0] ** 2 - X[0]
    relaxation = Problem(X, equalities=[equality]).relaxation(level=1)
    sdp = relaxation.sdp
    assert [b.size for b in sdp.blocks] == [3, 1, 1]
    assert sdp.constraint_to_blocks[equality] == (1, 2)
    assert _entries(relaxation, 1) == {
        0: {sdp.monomial_index[X[0]]: -1.0, sdp.monomial_index[X[0] ** 2]: 1.0}
    }
    assert _entries(relaxation, 2) == {
        0: {sdp.monomial_index[X[0]]: 1.0, sdp.monomial_index[X[0] ** 2]: -1.0}
    }


def test_equality_larger_basis_entry_blocks() -> None:
    """Each localizing entry of an equality lives in its own scalar block."""
    X = generate_operators("X", 2, hermitian=True)
    equality = X[0] - 0.5
    relaxation = Problem(X, equalities=[equality]).relaxation(level=2)
    sdp = relaxation.sdp
    assert [b.size for b in sdp.blocks] == [7] + [1] * 12
    assert sdp.constraint_to_blocks[equality] == (1, 7)
    # Block 1 holds entry (0, 0): X0 - 0.5.
    assert _entries(relaxation, 1)[0] == {0: -0.5, sdp.monomial_index[X[0]]: 1.0}
    # Block 3 holds entry (0, 2): 1*(X0 - 0.5)*X1 = X0*X1 - 0.5*X1.
    assert _entries(relaxation, 3)[0] == {
        sdp.monomial_index[X[0] * X[1]]: 1.0,
        sdp.monomial_index[X[1]]: -0.5,
    }
    # The second half negates the first.
    assert _entries(relaxation, 7)[0] == {0: 0.5, sdp.monomial_index[X[0]]: -1.0}


def test_remove_equalities_eliminates_variable() -> None:
    """X0 = 1 is solved away; the objective value moves to the constant."""
    X = generate_operators("X", 1, hermitian=True)
    relaxation = Problem(
        X, objective=X[0], equalities=[X[0] - 1.0]
    ).relaxation(level=1, removeequalities=True)
    sdp = relaxation.sdp
    assert sdp.n_vars == 1
    assert list(sdp.obj) == [0.0]
    assert sdp.constant_term == 1.0
    entries = _entries(relaxation, 0)
    assert entries[0] == {0: 1.0}  # normalization
    assert entries[1] == {0: 1.0}  # X0 pinned to 1.0
    assert entries[3] == {1: 1.0}  # X0^2 still free


def test_remove_equalities_handles_moment_equalities() -> None:
    """Moment equalities enter the elimination system like polynomials."""
    X = generate_operators("X", 1, hermitian=True)
    meq = MomentEntry(0, 1, 1) - 1.0  # X0^2 = 1
    relaxation = Problem(
        X, objective=MomentEntry(0, 0, 1), momentequalities=[meq]
    ).relaxation(level=1, removeequalities=True)
    sdp = relaxation.sdp
    assert sdp.n_vars == 1
    assert list(sdp.obj) == [1.0]
    assert sdp.constant_term == 0.0
    entries = _entries(relaxation, 0)
    assert entries[1] == {1: 1.0}  # X0 remains free
    assert entries[3] == {0: 1.0}  # X0^2 pinned to 1.0


def test_moment_inequality_scalar_block() -> None:
    """A moment inequality acts on a 1x1 block; M[0,0] = 1 by normalization."""
    X = generate_operators("X", 2, hermitian=True)
    mineq = MomentEntry(0, 0, 0) - 0.5
    relaxation = Problem(X, momentinequalities=[mineq]).relaxation(level=1)
    sdp = relaxation.sdp
    assert [b.size for b in sdp.blocks] == [3, 1]
    assert sdp.constraint_to_blocks[mineq] == (1,)
    assert _entries(relaxation, 1) == {0: {0: 0.5}}


def test_moment_equality_pair() -> None:
    """A moment equality becomes two 1x1 blocks with opposite signs."""
    X = generate_operators("X", 2, hermitian=True)
    meq = MomentEntry(0, 1, 1) - 1.0  # X0^2 = 1
    relaxation = Problem(X, momentequalities=[meq]).relaxation(level=1)
    sdp = relaxation.sdp
    assert [b.size for b in sdp.blocks] == [3, 1, 1]
    assert sdp.constraint_to_blocks[meq] == (1, 2)
    assert _entries(relaxation, 1) == {0: {0: -1.0, sdp.monomial_index[X[0] ** 2]: 1.0}}
    assert _entries(relaxation, 2) == {0: {0: 1.0, sdp.monomial_index[X[0] ** 2]: -1.0}}


def test_parameters_precede_moment_matrix() -> None:
    """Parameters become the first blocks; the first moment variable follows."""
    X = generate_operators("X", 1, hermitian=True)
    p = generate_variables("p", 1)[0]
    relaxation = Problem(X, parameters=[p]).relaxation(level=1)
    sdp = relaxation.sdp
    assert [b.size for b in sdp.blocks] == [1, 2]
    assert sdp.monomial_index[p] == 1
    assert sdp.monomial_index[X[0]] == 2
    assert _entries(relaxation, 0) == {0: {1: 1.0}}


def test_extra_moment_matrix_copy() -> None:
    """A copied extra moment matrix reuses the base variables."""
    X = generate_operators("X", 2, hermitian=True)
    relaxation = Problem(X, extramomentmatrices=[["copy"]]).relaxation(level=1)
    sdp = relaxation.sdp
    assert [b.size for b in sdp.blocks] == [3, 3]
    assert sdp.n_vars == 5
    assert _entries(relaxation, 0) == _entries(relaxation, 1)


def test_extra_moment_matrix_new_variables() -> None:
    """A fresh extra moment matrix gets one variable per upper entry."""
    X = generate_operators("X", 2, hermitian=True)
    relaxation = Problem(X, extramomentmatrices=[[""]]).relaxation(level=1)
    sdp = relaxation.sdp
    assert [b.size for b in sdp.blocks] == [3, 3]
    assert sdp.n_vars == 5 + 6
    assert _entries(relaxation, 1) == {0: {6: 1.0}, 1: {7: 1.0}, 2: {8: 1.0},
                                       4: {9: 1.0}, 5: {10: 1.0}, 8: {11: 1.0}}


def test_extra_moment_matrix_ppt_swaps_entries() -> None:
    """'ppt' transposes the B subsystem of the copied block."""
    a = generate_operators("a", 1)
    b = generate_operators("b", 1)
    relaxation = Problem(
        [a, b],
        extramonomials=[[[S.One, a[0]], [S.One, b[0]]]],
        extramomentmatrices=[["copy", "ppt"]],
        complex_matrix=True,
    ).relaxation(level=-1)
    sdp = relaxation.sdp
    assert [blk.size for blk in sdp.blocks] == [4, 4]
    # The base block keeps a*b at position (0,3) and a*b^dag at (1,2).
    assert _entries(relaxation, 0)[3] == {sdp.monomial_index[a[0] * b[0]]: 1.0}
    assert _entries(relaxation, 0)[6] == {sdp.monomial_index[a[0] * Dagger(b[0])]: 1.0}
    # The copied block is the partial transpose: the two are exchanged.
    assert _entries(relaxation, 1)[3] == {sdp.monomial_index[a[0] * Dagger(b[0])]: 1.0}
    assert _entries(relaxation, 1)[6] == {sdp.monomial_index[a[0] * b[0]]: 1.0}


def test_rectangular_basis_tensor_product() -> None:
    """A rectangular [A, B] basis spans the tensor product, block size lenA*lenB."""
    a = generate_operators("a", 1)
    b = generate_operators("b", 1)
    relaxation = Problem(
        [a, b],
        extramonomials=[[[S.One, a[0]], [S.One, b[0]]]],
        complex_matrix=True,
    ).relaxation(level=-1)
    sdp = relaxation.sdp
    assert [blk.size for blk in sdp.blocks] == [4]
    assert sdp.n_vars == 9  # 10 upper entries, minus the normalized corner
    assert sdp.monomial_index[a[0] * b[0]] == 4


def test_moment_substitution_reuses_variables() -> None:
    """A moment substitution pins an entry to an existing variable."""
    X = generate_operators("X", 2, hermitian=True)
    relaxation = Problem(
        X, momentsubstitutions={X[0] ** 2: X[1]}
    ).relaxation(level=1)
    sdp = relaxation.sdp
    assert sdp.n_vars == 4  # no variable for X0^2: it maps to the X1 variable
    entries = _entries(relaxation, 0)
    assert entries[4] == {2: 1.0}  # M[X0^2] = x_{X1}


def test_substitutions_shrink_the_basis() -> None:
    """Substitution rules reduce the monomial basis before construction."""
    X = generate_operators("X", 1, hermitian=True)
    relaxation = Problem(X, substitutions={X[0] ** 2: 1.0}).relaxation(level=1)
    assert relaxation.monomial_sets == [[S.One, X[0]]]
    sdp = relaxation.sdp
    assert sdp.n_vars == 1
    assert _entries(relaxation, 0)[3] == {0: 1.0}  # M[X0^2] = 1.0


def test_multipartite_two_moment_blocks() -> None:
    """Each variable set of a multipartite problem gets its own block."""
    a = generate_operators("a", 2, hermitian=True)
    b = generate_operators("b", 1, hermitian=True)
    relaxation = Problem([a, b]).relaxation(level=1)
    sdp = relaxation.sdp
    assert [blk.size for blk in sdp.blocks] == [3, 2]
    assert relaxation.var_offsets == [5, 7]
    assert sdp.n_vars == 7


def test_relational_inequality_converted() -> None:
    """Relational inequalities register under their converted (>= 0) form."""
    x = generate_variables("x", 2)
    ineq = x[0] - 0.5 >= 0
    relaxation = Problem(x, inequalities=[ineq]).relaxation(level=1)
    sdp = relaxation.sdp
    assert sdp.constraint_to_blocks[x[0] - 0.5] == (1,)
    assert _entries(relaxation, 1)[0] == {0: -0.5, sdp.monomial_index[x[0]]: 1.0}


def test_objective_as_moment_expression() -> None:
    """The objective may reference moment matrix entries directly."""
    X = generate_operators("X", 2, hermitian=True)
    relaxation = Problem(
        X, objective=MomentEntry(0, 1, 1) - 1.0
    ).relaxation(level=1)
    sdp = relaxation.sdp
    assert sdp.constant_term == -1.0
    assert list(sdp.obj) == [0.0, 0.0, 1.0, 0.0, 0.0]


def test_invalid_level_raises() -> None:
    """Levels below -1 are rejected."""
    X = generate_operators("X", 1, hermitian=True)
    with pytest.raises(ValueError):
        Problem(X).relaxation(level=-2)


def test_level_minus_one_requires_monomials() -> None:
    """Level -1 needs the full monomial bases from the caller."""
    X = generate_operators("X", 1, hermitian=True)
    with pytest.raises(ValueError, match="monomials"):
        Problem(X).relaxation(level=-1)


def test_high_degree_constraint_warns() -> None:
    """Constraints above the relaxation degree produce a warning.

    The monomials must still exist in the basis (via extramonomials here),
    otherwise the constraint cannot be evaluated at all.
    """
    X = generate_operators("X", 2, hermitian=True)
    relaxation = Problem(
        X, inequalities=[X[0] ** 4], extramonomials=[X[0] ** 4]
    ).relaxation(level=1)
    assert any("degree 4" in warning for warning in relaxation.warnings)


def test_uncovered_high_degree_constraint_raises() -> None:
    """A constraint whose monomials are not in the basis is an error."""
    X = generate_operators("X", 2, hermitian=True)
    with pytest.raises(RuntimeError, match="relaxation level"):
        Problem(X, inequalities=[X[0] ** 4]).relaxation(level=1)


def test_solve_without_backend_raises() -> None:
    """Without a registered backend, solving fails with a clear error."""
    X = generate_operators("X", 1, hermitian=True)
    relaxation = Problem(X, objective=X[0]).relaxation(level=1)
    if registered():
        relaxation.solve()
    else:
        with pytest.raises(SolverError, match="No solver backend"):
            relaxation.solve()
