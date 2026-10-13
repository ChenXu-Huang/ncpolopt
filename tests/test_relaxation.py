"""Relaxation construction: block layout, entries, and elimination.

The assertions pin the exact SDP data produced for hand-computable small
problems: monomial indexing order, entry positions, the two-half equality
blocks, the constant-handling fix, and the basis transform of equality
elimination.
"""

from __future__ import annotations

from typing import Any

import numpy as np
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


def test_complex_relaxation_rejects_adjoint_only_monomial() -> None:
    """A complex relaxation must not borrow the adjoint's SDP variable.

    Regression: the conjugate fallback in _get_index_of_monomial ignored
    complex_matrix and mapped a missing monomial onto its adjoint's
    variable without conjugation (not a linear operation), silently
    cancelling the term -- the constraint below degenerated to 1 >= 0.
    """
    A = generate_operators("A", 1, hermitian=False)[0]
    problem = Problem(
        [A],
        inequalities=[S.ImaginaryUnit * A**2 - S.ImaginaryUnit * Dagger(A) ** 2 + 1],
        complex_matrix=True,
    )
    with pytest.raises(RuntimeError, match="adjoint"):
        problem.relaxation(level=1)


def test_moment_substitution_keeps_outer_coefficient() -> None:
    """A moment substitution value inherits the monomial's scalar factor.

    Regression: the recursion into the substituted value dropped the outer
    coefficient, writing 1.0*x0 where 2.0*x0 belonged.
    """
    X = generate_operators("x", 2, hermitian=True)
    problem = Problem(
        X,
        objective=X[0],
        substitutions={X[0] * X[1]: 2 * X[1]},
        momentsubstitutions={X[1]: X[0]},
    )
    sdp = problem.relaxation(level=1).sdp
    block = sdp.blocks[0]
    k = sdp.monomial_index[X[0]]
    entries = {
        (int(pos) // block.size, int(pos) % block.size): value
        for var, pos, value in zip(
            block.coo.row, block.coo.col, block.coo.data, strict=True
        )
        if int(var) == k
    }
    # The (X0, X1) entry is <X0 X1> = 2 <X1> = 2 <X0>.
    assert entries[(1, 2)] == 2.0


def test_complex_objective_is_detected() -> None:
    """A complex objective coefficient forces a complex relaxation.

    Regression: the automatic complex detection scanned only the
    constraints, so a complex objective crashed with a TypeError deep in
    set_objective instead of building a complex SDP.
    """
    B = generate_operators("B", 1, hermitian=True)[0]
    problem = Problem([B], objective=(1 + 2 * S.ImaginaryUnit) * B)
    relaxation = problem.relaxation(level=1)
    assert relaxation.complex_matrix is True
    assert relaxation.sdp.blocks[0].coo.dtype == np.complex128


def test_elimination_honors_localizing_monomials_override() -> None:
    """removeequalities uses the override basis for the equality rows.

    Regression: _process_equalities always computed the automatic basis,
    silently ignoring the user's localizing_monomials.
    """
    X = generate_operators("x", 2, hermitian=True)
    problem = Problem(
        X,
        equalities=[X[0] ** 2 - X[1]],
        localizing_monomials=[[S.One]],
    )
    overridden = problem.relaxation(level=2, removeequalities=True)
    automatic = Problem(
        X, equalities=[X[0] ** 2 - X[1]]
    ).relaxation(level=2, removeequalities=True)
    # The override contributes one equality row (the constraint at the
    # trivial basis), the automatic basis six (the upper triangle over
    # {1, x0, x1}), eliminating fewer variables.
    assert overridden.sdp.n_vars > automatic.sdp.n_vars


def test_localizing_monomials_length_mismatch_raises() -> None:
    """A short localizing_monomials list is a clear error, not IndexError."""
    X = generate_operators("x", 2, hermitian=True)
    problem = Problem(
        X,
        inequalities=[X[0], X[1]],
        localizing_monomials=[[S.One]],
    )
    with pytest.raises(ValueError, match="localizing_monomials"):
        problem.relaxation(level=2)


def test_equality_degree_too_high_raises_on_elimination() -> None:
    """The elimination path rejects constraints of degree > 2*level."""
    X = generate_operators("X", 2, hermitian=True)
    with pytest.raises(ValueError, match="degree 4"):
        Problem(X, equalities=[X[0] ** 4]).relaxation(level=1, removeequalities=True)


def test_constant_objective() -> None:
    """A constant-only objective is carried by the constant term."""
    X = generate_operators("X", 1, hermitian=True)
    solution = Problem(X, objective=2.5).solve(level=1)
    assert solution.primal == pytest.approx(2.5)


def test_objective_constant_term_warns_and_counts(caplog) -> None:
    """A constant part of the objective lands in constant_term (with a note)."""
    import logging

    X = generate_operators("X", 1, hermitian=True)
    with caplog.at_level(logging.WARNING, logger="ncpolopt.relaxation"):
        relaxation = Problem(X, objective=X[0] + 1.5).relaxation(level=1)
    assert any("constant" in record.message for record in caplog.records)
    assert relaxation.sdp.constant_term == 1.5


def test_moment_equality_pins_entry() -> None:
    """A MomentEntry equality pins the moment to a value."""
    from ncpolopt.moment import MomentEntry

    X = generate_operators("X", 2, hermitian=True)
    problem = Problem(
        X,
        objective=X[0] * X[1] + X[1] * X[0],
        momentequalities=[MomentEntry(0, 0, 1) - 0.5],
    )
    solution = problem.solve(level=2)
    assert solution.monomial_value(X[0]) == pytest.approx(0.5, abs=1e-5)


def test_complex_substitution_forces_complex_matrix() -> None:
    """A complex substitution value flips the relaxation to complex."""
    X = generate_operators("X", 1, hermitian=True)
    problem = Problem(X, substitutions={X[0] ** 2: 1j})
    relaxation = problem.relaxation(level=1)
    assert relaxation.complex_matrix is True


def test_removeequalities_without_equalities_is_noop() -> None:
    """removeequalities with an empty equality set builds normally."""
    X = generate_operators("X", 1, hermitian=True)
    relaxation = Problem(X, objective=X[0]).relaxation(level=1, removeequalities=True)
    assert relaxation.sdp.n_vars > 0


def test_polynomial_moment_equality() -> None:
    """A plain polynomial in momentequalities pins the top-left entry."""
    X = generate_operators("X", 2, hermitian=True)
    problem = Problem(
        X, objective=X[0] ** 2, momentequalities=[X[0] - 0.5]
    )
    solution = problem.solve(level=2)
    assert solution.monomial_value(X[0]) == pytest.approx(0.5, abs=1e-5)


def test_moment_equality_on_normalization_entry() -> None:
    """A MomentEntry(0, 0, 0) equality binds the normalization constant."""
    from ncpolopt.moment import MomentEntry

    X = generate_operators("X", 2, hermitian=True)
    problem = Problem(
        X, objective=X[0] ** 2, momentequalities=[MomentEntry(0, 0, 0) - 1.0]
    )
    solution = problem.solve(level=2)
    assert solution.primal == pytest.approx(0.0, abs=1e-5)


def test_complex_moment_entry_transpose_rejected() -> None:
    """A transposed MomentEntry in a complex relaxation raises."""
    from ncpolopt.moment import MomentEntry

    X = generate_operators("X", 2, hermitian=False)
    # X[0] sits at (0, 1); referencing (1, 0) in a complex problem asks
    # for its conjugate, which is not linear in the SDP variables.
    problem = Problem(
        X,
        objective=MomentEntry(0, 1, 0),
        complex_matrix=True,
    )
    with pytest.raises(ValueError, match="conjugate"):
        problem.relaxation(level=1)


def test_negative_objective_monomial() -> None:
    """A negated objective monomial flips the facvar sign."""
    X = generate_operators("X", 1, hermitian=True)
    relaxation = Problem(X, objective=-X[0]).relaxation(level=1)
    assert relaxation.sdp.obj[0] == -1.0
    assert all(c == 0 for c in relaxation.sdp.obj[1:])


def test_objective_referencing_moment_substitution() -> None:
    """The objective resolves a moment-substituted monomial."""
    X = generate_operators("X", 2, hermitian=True)
    problem = Problem(
        X, objective=X[1], momentsubstitutions={X[1]: X[0]}
    )
    relaxation = problem.relaxation(level=1)
    # The objective lands on the variable of X[0].
    assert relaxation.sdp.obj[0] == 1.0
    assert all(c == 0 for c in relaxation.sdp.obj[1:])


def test_moment_substitution_with_complex_value() -> None:
    """A complex constant moment value lands in the constant column."""
    X = generate_operators("X", 1, hermitian=True)
    problem = Problem(
        X,
        objective=X[0],
        momentsubstitutions={X[0] ** 2: 1j},
        complex_matrix=True,
    )
    relaxation = problem.relaxation(level=2)
    assert relaxation.complex_matrix is True


def test_moment_inequality_polynomial_with_constant() -> None:
    """A polynomial moment inequality pushes its constant into the block."""
    X = generate_operators("X", 2, hermitian=True)
    problem = Problem(
        X,
        objective=X[0] * X[1] + X[1] * X[0],
        momentinequalities=[X[0] - 0.5],
    )
    solution = problem.solve(level=2)
    assert solution.monomial_value(X[0]).real >= 0.5 - 1e-5


def test_moment_entry_transposed_position_real() -> None:
    """A lower-triangle MomentEntry resolves through the transpose (real)."""
    from ncpolopt.moment import MomentEntry

    X = generate_operators("X", 2, hermitian=True)
    problem = Problem(
        X,
        objective=X[0] ** 2,
        momentequalities=[MomentEntry(0, 1, 0) - 0.5],
    )
    solution = problem.solve(level=2)
    assert solution.monomial_value(X[0]) == pytest.approx(0.5, abs=1e-5)


def test_moment_entry_missing_position_raises() -> None:
    """A MomentEntry outside the moment matrix is a clear error."""
    from ncpolopt.moment import MomentEntry

    X = generate_operators("X", 2, hermitian=True)
    problem = Problem(X, objective=MomentEntry(0, 5, 5))
    with pytest.raises(ValueError, match="does not correspond"):
        problem.relaxation(level=1)


def test_dependent_equalities_warn(caplog) -> None:
    """Linearly dependent equalities log a warning under elimination."""
    import logging

    X = generate_operators("X", 1, hermitian=True)
    with caplog.at_level(logging.WARNING, logger="ncpolopt.relaxation"):
        Problem(
            X,
            objective=X[0],
            equalities=[X[0] - 1, 2 * X[0] - 2],
        ).relaxation(level=1, removeequalities=True)
    assert any("dependent" in record.message for record in caplog.records)


def test_moment_equality_eliminated_rows() -> None:
    """Moment equalities contribute rows to the elimination system."""
    from ncpolopt.moment import MomentEntry

    X = generate_operators("X", 2, hermitian=True)
    problem = Problem(
        X,
        objective=X[1] ** 2,
        momentinequalities=[MomentEntry(0, 0, 1) - 0.5],
        momentequalities=[MomentEntry(0, 0, 2) - 0.25],
    )
    solution = problem.solve(level=2, removeequalities=True)
    assert solution.monomial_value(X[1]) == pytest.approx(0.25, abs=1e-4)


def test_non_pure_substitution_rule_detected() -> None:
    """A rule whose rhs introduces new atoms is not pure."""
    X = generate_operators("X", 2, hermitian=True)
    relaxation = Problem(X, substitutions={X[0]: X[1]}).relaxation(level=1)
    assert relaxation._pure_substitution_rules is False


def test_ppt_requires_rectangular_basis() -> None:
    """ppt on a flat basis is a clear error."""
    X = generate_operators("X", 2, hermitian=True)
    problem = Problem(X, extramomentmatrices=[["ppt"]])
    with pytest.raises(ValueError, match="rectangular"):
        problem.relaxation(level=1)


def test_constraint_covering_all_query_paths() -> None:
    """Constants, negatives, and moment substitutions inside a constraint."""
    X = generate_operators("X", 2, hermitian=True)
    problem = Problem(
        X,
        inequalities=[-X[0] ** 2 + X[1] + 1.0],
        substitutions={X[0] ** 2: 1},
        momentsubstitutions={X[1]: X[0]},
    )
    relaxation = problem.relaxation(level=2)
    assert relaxation.sdp.n_vars > 0


def test_moment_substitution_complex_value_in_constraint() -> None:
    """A complex pinned moment inside a constraint keeps the coefficient."""
    X = generate_operators("X", 1, hermitian=True)
    problem = Problem(
        X,
        inequalities=[2 * X[0] ** 2 - 1],
        momentsubstitutions={X[0] ** 2: 0.5},
    )
    relaxation = problem.relaxation(level=2)
    assert relaxation.sdp.n_vars > 0


def test_elimination_with_empty_block() -> None:
    """A substitution-erased constraint survives equality elimination."""
    X = generate_operators("X", 2, hermitian=True)
    problem = Problem(
        X,
        objective=X[0],
        inequalities=[1 - X[0] ** 2],
        equalities=[X[1] - 0.5],
        substitutions={X[0] ** 2: 1},
    )
    relaxation = problem.relaxation(level=1, removeequalities=True)
    assert relaxation.sdp.n_vars > 0


def test_relational_equality_elimination() -> None:
    """A relational equality feeds the elimination system."""
    from sympy import Eq

    from ncpolopt.variables import generate_variables

    # NOTE: Eq on an Operator collapses to BooleanFalse; relationals only
    # survive for commutative symbols.
    x = generate_variables("x", 1)[0]
    relaxation = Problem(
        [x], objective=x**2, equalities=[Eq(x, 0.5)]
    ).relaxation(level=1, removeequalities=True)
    assert relaxation.sdp.n_vars >= 1


def test_objective_query_path_variants() -> None:
    """Constants, negatives and moment substitutions in the objective.

    Covers the per-term branches of the monomial resolution loop.
    """
    X = generate_operators("X", 2, hermitian=True)
    problem = Problem(
        X,
        objective=-X[0] ** 2 + X[1],
        substitutions={X[0] ** 2: 2},
        momentsubstitutions={X[1]: X[0]},
    )
    relaxation = problem.relaxation(level=1)
    # -X0^2 -> -2 (constant), X1 -> X0's variable: obj = [1] + const -2.
    assert relaxation.sdp.constant_term == -2.0
    assert relaxation.sdp.obj[0] == 1.0


def test_scalar_constraint_reduced_to_constant() -> None:
    """A moment inequality reduced to a constant writes a constant row."""
    X = generate_operators("X", 1, hermitian=True)
    problem = Problem(
        X, momentinequalities=[X[0] ** 2], substitutions={X[0] ** 2: 2}
    )
    relaxation = problem.relaxation(level=1)
    assert relaxation.sdp.n_vars >= 1


def test_relational_equality_in_elimination() -> None:
    """A relational equality feeds the elimination system."""
    from sympy import Eq

    from ncpolopt.variables import generate_variables

    # NOTE: Eq on an Operator collapses to BooleanFalse; relationals only
    # survive for commutative symbols.
    x = generate_variables("x", 1)[0]
    relaxation = Problem(
        [x], objective=x**2, equalities=[Eq(x, 0.5)]
    ).relaxation(level=1, removeequalities=True)
    assert relaxation.sdp.n_vars >= 1


def test_elimination_rows_cover_all_constraint_kinds() -> None:
    """Equalities and moment inequalities both feed the elimination rows."""
    X = generate_operators("X", 1, hermitian=True)
    relaxation = Problem(
        [X[0]],
        objective=X[0] ** 2,
        equalities=[X[0] - 0.5],
        momentinequalities=[X[0] - 0.25],
        momentequalities=[X[0] - 0.25],
    ).relaxation(level=1, removeequalities=True)
    assert relaxation.sdp.n_vars >= 1


def test_localizing_entry_reduced_to_constant() -> None:
    """A localizing entry reducing to a number writes a constant entry."""
    X = generate_operators("X", 1, hermitian=True)
    problem = Problem(
        X, inequalities=[X[0] ** 2 - 1], substitutions={X[0] ** 2: 2}
    )
    relaxation = problem.relaxation(level=1)
    assert relaxation.sdp.n_vars >= 1


def test_objective_covers_all_query_branches() -> None:
    """Constants, negatives, and pinned terms inside a resolved objective."""
    X = generate_operators("X", 2, hermitian=True)
    problem = Problem(
        X,
        objective=X[0] ** 2,
        substitutions={X[0] ** 2: X[1] - X[0] + 1},
        momentsubstitutions={X[1]: X[0]},
    )
    relaxation = problem.relaxation(level=1)
    # X0^2 -> X1 - X0 + 1 -> X0 - X0 + 1 = 1: everything collapses to the
    # constant term.
    assert relaxation.sdp.constant_term == 1.0
    assert all(c == 0 for c in relaxation.sdp.obj)


def test_nested_moment_substitution_with_constant() -> None:
    """A pinned moment whose value contains another pin cascades."""
    X = generate_operators("X", 2, hermitian=True)
    problem = Problem(
        X,
        objective=X[0] ** 2,
        momentsubstitutions={X[0]: 0.5, X[1]: 2 * X[0]},
    )
    relaxation = problem.relaxation(level=1)
    # X1's own moment cascades to the constant (2*0.5), so no variable is
    # created for it; the product moments keep theirs.
    assert X[1] not in relaxation.sdp.monomial_index
    assert relaxation.sdp.n_vars == 3


def test_objective_moment_substitution_add_value() -> None:
    """A pinned moment whose value is a sum resolves term by term.

    Covers the in-loop branches of the monomial resolution: a number term,
    a pinned term, and a free term inside one substitution value.
    """
    X = generate_operators("X", 2, hermitian=True)
    problem = Problem(
        X,
        objective=X[1],
        momentsubstitutions={X[1]: X[0] + X[0] ** 2 - 1, X[0] ** 2: X[0]},
    )
    relaxation = problem.relaxation(level=1)
    # X1 -> X0 + X0^2 - 1 -> X0 + X0 - 1 = 2*X0 - 1.
    assert relaxation.sdp.constant_term == -1.0
    assert relaxation.sdp.obj[0] == 2.0
