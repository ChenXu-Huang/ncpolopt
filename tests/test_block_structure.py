"""Block structure computation: kinds, sizes, and constraint lookup."""

from __future__ import annotations

from sympy import S
from sympy.physics.quantum.dagger import Dagger

from ncpolopt.block_structure import (
    BlockKind,
    BlockStructure,
    compute_block_structure,
)
from ncpolopt.moment import MomentEntry
from ncpolopt.monomials import get_monomials
from ncpolopt.variables import generate_operators, generate_variables


def _structure(
    *,
    variables,
    level: int,
    monomial_sets,
    inequalities=None,
    equalities=None,
    momentinequalities=None,
    momentequalities=None,
    parameters=None,
    extramomentmatrices=None,
    removeequalities: bool = False,
    localizing_monomials=None,
) -> BlockStructure:
    """Call compute_block_structure with the common defaults filled in."""
    return compute_block_structure(
        variables=variables,
        monomial_sets=monomial_sets,
        level=level,
        inequalities=inequalities,
        equalities=equalities,
        momentinequalities=momentinequalities,
        momentequalities=momentequalities,
        parameters=parameters,
        extramomentmatrices=extramomentmatrices,
        removeequalities=removeequalities,
        localizing_monomials=localizing_monomials,
    )


def test_empty_problem_single_moment_block() -> None:
    """No constraints: just the level-1 moment block over three monomials."""
    X = generate_operators("X", 2, hermitian=True)
    structure = _structure(variables=X, level=1, monomial_sets=[[S.One, X[0], X[1]]])
    assert [b.kind for b in structure.blocks] == [BlockKind.MOMENT]
    assert structure.blocks[0].size == 3
    assert structure.constraint_starting_block == 1
    assert structure.warnings == ()


def test_parameters_precede_moment_blocks() -> None:
    """Each parameter becomes its own 1x1 block before the moment matrix."""
    X = generate_operators("X", 2, hermitian=True)
    parameters = generate_variables("p", 2)
    structure = _structure(
        variables=X,
        level=1,
        monomial_sets=[[S.One, X[0], X[1]]],
        parameters=parameters,
    )
    kinds = [b.kind for b in structure.blocks]
    assert kinds == [BlockKind.PARAMETER, BlockKind.PARAMETER, BlockKind.MOMENT]
    assert structure.constraint_starting_block == 3


def test_degree_two_inequality_at_level_one() -> None:
    """A degree-2 inequality at level 1 localizes with the trivial basis."""
    X = generate_operators("X", 2, hermitian=True)
    ineq = -X[1] ** 2 + X[1] + 0.5
    structure = _structure(
        variables=X, level=1, monomial_sets=[[S.One, X[0], X[1]]], inequalities=[ineq]
    )
    kinds = [b.kind for b in structure.blocks]
    assert kinds == [BlockKind.MOMENT, BlockKind.LOCALIZING]
    assert structure.blocks[1].size == 1
    assert structure.blocks[1].constraint is ineq
    assert structure.constraint_to_blocks[ineq] == (1,)


def test_degree_one_inequality_at_level_two() -> None:
    """A degree-1 inequality at level 2 localizes over the level-1 basis."""
    X = generate_operators("X", 2, hermitian=True)
    basis = get_monomials(X, 2)  # 7 monomials
    ineq = X[0] - 0.5
    structure = _structure(variables=X, level=2, monomial_sets=[basis], inequalities=[ineq])
    assert structure.blocks[1].size == 3  # [1, X0, X1]
    assert structure.blocks[1].localizing_set == (S.One, X[0], X[1])


def test_equality_produces_scalar_block_pair() -> None:
    """An equality yields 2*ln*(ln+1)//2 one-by-one blocks in +/- halves."""
    X = generate_operators("X", 2, hermitian=True)
    equality = X[0] * X[1] - X[1] * X[0]
    structure = _structure(
        variables=X, level=1, monomial_sets=[[S.One, X[0], X[1]]], equalities=[equality]
    )
    kinds = [b.kind for b in structure.blocks]
    assert kinds == [BlockKind.MOMENT] + [BlockKind.SCALAR_EQUALITY] * 2
    assert all(b.size == 1 for b in structure.blocks[1:])
    assert structure.constraint_to_blocks[equality] == (1, 2)
    # First half enforces +eq, second half -eq.
    assert structure.blocks[1].constraint == equality
    assert structure.blocks[2].constraint == -equality


def test_equality_with_larger_localizing_basis() -> None:
    """A degree-1 equality at level 2 localizes over three monomials: 12
    scalar blocks in halves of six."""
    X = generate_operators("X", 2, hermitian=True)
    basis = get_monomials(X, 2)
    equality = X[0] - 0.5
    structure = _structure(variables=X, level=2, monomial_sets=[basis], equalities=[equality])
    scalar_blocks = [b for b in structure.blocks if b.kind == BlockKind.SCALAR_EQUALITY]
    assert len(scalar_blocks) == 12
    assert structure.constraint_to_blocks[equality] == (1, 7)


def test_remove_equalities_skips_blocks() -> None:
    """removeequalities=True eliminates the equality blocks entirely."""
    X = generate_operators("X", 2, hermitian=True)
    equality = X[0] * X[1] - X[1] * X[0]
    structure = _structure(
        variables=X,
        level=1,
        monomial_sets=[[S.One, X[0], X[1]]],
        equalities=[equality],
        removeequalities=True,
    )
    assert [b.kind for b in structure.blocks] == [BlockKind.MOMENT]
    assert structure.constraint_to_blocks == {}


def test_moment_inequality_is_scalar_block() -> None:
    """A MomentExpr inequality becomes a 1x1 scalar inequality block."""
    X = generate_operators("X", 2, hermitian=True)
    mineq = MomentEntry(0, 0, 0) - 1.0
    structure = _structure(
        variables=X,
        level=1,
        monomial_sets=[[S.One, X[0], X[1]]],
        momentinequalities=[mineq],
    )
    assert structure.blocks[1].kind == BlockKind.SCALAR_INEQUALITY
    assert structure.blocks[1].size == 1
    assert structure.constraint_to_blocks[mineq] == (1,)


def test_moment_equality_pair() -> None:
    """A MomentExpr equality yields a pair of scalar blocks."""
    X = generate_operators("X", 2, hermitian=True)
    meq = MomentEntry(0, 0, 0) - 0.5
    structure = _structure(
        variables=X,
        level=1,
        monomial_sets=[[S.One, X[0], X[1]]],
        momentequalities=[meq],
    )
    kinds = [b.kind for b in structure.blocks]
    assert kinds == [BlockKind.MOMENT, BlockKind.SCALAR_EQUALITY, BlockKind.SCALAR_EQUALITY]
    assert structure.constraint_to_blocks[meq] == (1, 2)
    assert structure.blocks[2].constraint == -meq


def test_localizing_monomials_override() -> None:
    """The per-constraint override replaces the automatic basis."""
    X = generate_operators("X", 2, hermitian=True)
    ineq = X[0] - 0.5
    structure = _structure(
        variables=X,
        level=2,
        monomial_sets=[get_monomials(X, 2)],
        inequalities=[ineq],
        localizing_monomials=[[X[0], X[1]]],
    )
    assert structure.blocks[1].size == 2
    assert structure.blocks[1].localizing_set == (X[0], X[1])


def test_degree_warning() -> None:
    """Constraints exceeding the relaxation degree produce a warning."""
    X = generate_operators("X", 2, hermitian=True)
    ineq = X[0] ** 4
    structure = _structure(
        variables=X, level=1, monomial_sets=[[S.One, X[0], X[1]]], inequalities=[ineq]
    )
    assert len(structure.warnings) == 1
    assert "degree 4" in structure.warnings[0]


def test_relational_constraint_registered_under_both_forms() -> None:
    """A relational constraint is registered under its converted form too."""
    # SymPy rejects comparisons of non-real expressions (operators), so the
    # relational input must be built from commutative real variables.
    x = generate_variables("x", 2)
    ineq = x[0] - 0.5 >= 0
    structure = _structure(
        variables=x, level=1, monomial_sets=[[S.One, x[0], x[1]]], inequalities=[ineq]
    )
    assert structure.constraint_to_blocks[ineq] == (1,)
    assert structure.constraint_to_blocks[x[0] - 0.5] == (1,)


def test_extra_moment_matrices_repeat_base_sizes() -> None:
    """Each extra moment matrix repeats the base block sizes."""
    X = generate_operators("X", 2, hermitian=True)
    structure = _structure(
        variables=X,
        level=1,
        monomial_sets=[[S.One, X[0], X[1]]],
        extramomentmatrices=[["copy", "ppt"]],
    )
    kinds = [b.kind for b in structure.blocks]
    assert kinds == [BlockKind.MOMENT, BlockKind.COPY]
    assert structure.blocks[1].impose_ppt
    assert structure.blocks[1].size == 3


def test_product_basis_block_size() -> None:
    """A rectangular monomial set [A, B] spans the tensor product basis."""
    a = generate_operators("a", 1)
    b = generate_operators("b", 1)
    structure = _structure(
        variables=[a, b],
        level=-1,
        monomial_sets=[[[S.One, a[0]], [S.One, b[0]]]],
    )
    assert structure.blocks[0].size == 4


def test_multipartite_constraint_uses_matching_basis() -> None:
    """A constraint over one variable set localizes against its own basis."""
    a = generate_operators("a", 2)
    b = generate_operators("b", 2, hermitian=True)
    structure = _structure(
        variables=[a, b],
        level=2,
        monomial_sets=[[S.One, a[0], Dagger(a[0]), a[1], Dagger(a[1])], [S.One, b[0], b[1]]],
        inequalities=[b[0] - 0.5],
    )
    # One moment matrix per variable set: blocks 0 and 1 hold the ``a`` and
    # ``b`` moment matrices, block 2 is the localizing matrix of the
    # constraint.
    assert structure.blocks[2].size == 3
    # The level-2 basis of ``b`` (not ``a``) provides the localizing set.
    assert structure.blocks[2].localizing_set == (S.One, b[0], b[1])
