"""SdpBuilder / SdpProblem: freeze invariants and equality elimination."""

from __future__ import annotations

import numpy as np
import pytest
from sympy import Symbol

from ncpolopt.equality_elimination import BasisTransform, eliminate_equalities
from ncpolopt.sdp_problem import SdpBuilder, SparseBlock


def test_freeze_preserves_entries_and_locations() -> None:
    """Freezing keeps entries, monomial index and column locations."""
    x0, x1 = Symbol("x0"), Symbol("x1")
    builder = SdpBuilder([3], normalized=True, complex_matrix=False)
    builder.add_entry(0, 0, 0, 0, 1.0)  # normalized constant in the corner
    k0 = builder.new_variable(x0, 0, 0, 1)
    builder.add_entry(0, 0, 1, k0, 1.0)
    k1 = builder.new_variable(x1, 0, 1, 2)
    builder.add_entry(0, 1, 2, k1, 1.0)
    builder.set_objective([0.0, 1.0, 2.0])
    problem = builder.freeze(1, {}, ())
    assert problem.n_vars == 2
    assert problem.constant_term == 0.0
    assert list(problem.obj) == [1.0, 2.0]
    assert problem.monomial_index == {x0: 1, x1: 2}
    assert problem.column_locations == {1: (0, 0, 1), 2: (0, 1, 2)}
    (block,) = problem.blocks
    assert block.size == 3
    assert block.coo.shape == (3, 9)
    assert block.coo.dtype == np.float64


def test_freeze_requires_full_objective() -> None:
    """set_objective rejects facvar vectors of the wrong length."""
    builder = SdpBuilder([1], normalized=True, complex_matrix=False)
    with pytest.raises(ValueError):
        builder.set_objective([0.0, 1.0])


def test_add_entry_skips_zeros() -> None:
    """Zero coefficients do not create sparse entries."""
    builder = SdpBuilder([2], normalized=True, complex_matrix=False)
    builder.add_entry(0, 0, 0, 0, 0.0)
    k = builder.new_variable(Symbol("x"), 0, 0, 1)
    builder.add_entry(0, 0, 1, k, 1.0)
    (block,) = builder.freeze(1, {}, ()).blocks
    assert block.coo.nnz == 1


def test_block_evaluate_matches_manual_sum() -> None:
    """evaluate computes A0 + sum x_i A_i."""
    builder = SdpBuilder([2], normalized=True, complex_matrix=False)
    builder.add_entry(0, 0, 0, 0, 1.0)
    k = builder.new_variable(Symbol("m"), 0, 0, 1)
    builder.add_entry(0, 0, 1, k, 1.0)
    builder.add_entry(0, 1, 1, k, 0.5)
    problem = builder.freeze(1, {}, ())
    mat = problem.blocks[0].evaluate(np.array([3.0]))
    expected = np.array([[1.0, 3.0], [0.0, 1.5]])
    assert np.allclose(mat, expected)


def test_eliminate_fixes_variable_and_shifts_objective() -> None:
    """Eliminating x1 = 0.5 pins its moment and shifts the objective."""
    x0, x1 = Symbol("x0"), Symbol("x1")
    builder = SdpBuilder([3], normalized=True, complex_matrix=False)
    builder.add_entry(0, 0, 0, 0, 1.0)
    k0 = builder.new_variable(x0, 0, 0, 1)
    builder.add_entry(0, 0, 1, k0, 1.0)
    k1 = builder.new_variable(x1, 0, 0, 2)
    builder.add_entry(0, 0, 2, k1, 1.0)
    builder.set_objective([0.0, 1.0, 2.0])
    # Equality x1 - 0.5 = 0, in the A @ [1, x0, x1] = 0 encoding.
    A = np.array([[-0.5, 0.0, 1.0]])
    transform = eliminate_equalities(A)
    assert transform.n_free == 1
    builder.eliminate(transform)
    problem = builder.freeze(1, {}, ())
    # x1 is fixed to 0.5: c.x = 1*x0 + 2*0.5 = x0 + 1.
    assert problem.n_vars == 1
    assert problem.constant_term == pytest.approx(1.0)
    # The monomial positions survive the change of basis.
    assert problem.monomial_index == {x0: 1, x1: 2}
    assert problem.column_locations == {1: (0, 0, 1), 2: (0, 0, 2)}
    mat = problem.blocks[0].evaluate(np.array([1.0]))
    assert mat[0, 2] == pytest.approx(0.5)
    # The reduced objective evaluated at y == 1 (the point x0 = 1, x1 = 0.5
    # of the original formulation, free-basis signs canonicalized)
    # reproduces the original objective value.
    assert np.allclose(problem.obj @ np.array([1.0]) + problem.constant_term, 2.0)


def test_eliminate_handles_dependent_rows() -> None:
    """Dependent equality rows are detected and handled gracefully."""
    A = np.array([[-0.5, 0.0, 1.0], [-0.5, 0.0, 1.0]])
    transform = eliminate_equalities(A)
    assert transform.dependent_rows
    assert transform.n_free == 1
    assert transform.shift[2] == pytest.approx(0.5)


def test_eliminate_pure_constant_equality_is_identity() -> None:
    """An equality without variables eliminates nothing."""
    A = np.array([[1.0, 0.0, 0.0]])
    transform = eliminate_equalities(A)
    assert transform.n_free == 2
    assert np.allclose(transform.basis, np.eye(2))
    assert np.allclose(transform.shift, [1.0, 0.0, 0.0])


def test_basis_transform_is_frozen() -> None:
    """BasisTransform is immutable."""
    transform = BasisTransform(np.eye(2), np.array([1.0, 0.0, 0.0]))
    with pytest.raises(AttributeError):
        transform.basis = np.zeros((2, 2))


def test_sparse_block_is_frozen() -> None:
    """SparseBlock fields are read-only."""
    block = SparseBlock.__new__(SparseBlock)
    with pytest.raises(AttributeError):
        block.size = 3
