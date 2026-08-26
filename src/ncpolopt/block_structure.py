"""Typed description of the diagonal block structure of an SDP relaxation.

Ported from ``SdpRelaxation._calculate_block_structure``
(src.old/ncpol2sdpa/sdp_relaxation.py:848). The old implementation tracked
the layout in two parallel arrays -- ``block_struct`` (block sizes) and
``localizing_monomial_sets`` (an indexed list padded with ``None`` entries,
with each equality's localizing basis duplicated) -- and constraint
processing recovered a block's basis by index arithmetic between the two.
This module replaces the pair with a single typed list of :class:`BlockSpec`
objects; each spec carries the metadata the construction layer needs, so no
parallel bookkeeping survives.

The block order is: parameter blocks, moment matrix blocks, extra moment
matrix blocks, then the constraint blocks (inequalities, moment inequalities,
equalities, moment equalities) in the same order the constraints are
processed downstream.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from sympy import S

from .expressions import convert_relational, flatten
from .moment import MomentExpr, as_moment_expr
from .monomials import ncdegree, pick_monomials_up_to_degree, unique
from .variables import find_variable_set


class BlockKind(Enum):
    """The role of a diagonal block in the SDP relaxation.

    Attributes:
        PARAMETER: One free 1x1 variable per problem parameter.
        MOMENT: The moment matrix over a monomial basis.
        LOCALIZING: The localizing matrix of an inequality constraint.
        SCALAR_INEQUALITY: A 1x1 inequality block over moments.
        SCALAR_EQUALITY: A 1x1 equality block; equalities are enforced by
            pairs of such blocks with opposite signs.
        COPY: An extra moment matrix copying the first moment block.
        NEW: An unconstrained extra moment matrix block.
    """

    PARAMETER = "parameter"
    MOMENT = "moment"
    LOCALIZING = "localizing"
    SCALAR_INEQUALITY = "scalar_inequality"
    SCALAR_EQUALITY = "scalar_equality"
    COPY = "copy"
    NEW = "new"


@dataclass(frozen=True, slots=True)
class BlockSpec:
    """Metadata of one diagonal block of the SDP.

    Attributes:
        kind: The role of the block.
        size: Number of rows (and columns) of the block.
        constraint: The constraint expression the block enforces, in the
            normalized (``>= 0``) form and negated for the second half of an
            equality pair. Only set for localizing and scalar blocks.
        localizing_set: The monomials of the localizing basis. Only set for
            localizing and scalar equality blocks.
        monomial_set_index: Index of the monomial set for moment blocks.
        impose_ppt: Apply the partial transpose to this block after
            construction (extra moment matrices).
    """

    kind: BlockKind
    size: int
    constraint: Any = None
    localizing_set: tuple[Any, ...] = ()
    monomial_set_index: int = -1
    impose_ppt: bool = False


@dataclass(frozen=True, slots=True)
class BlockStructure:
    """The ordered block layout of an SDP relaxation.

    Attributes:
        blocks: Block specs in the order they appear on the block diagonal.
        constraint_to_blocks: Maps each constraint expression to the indices
            of the blocks enforcing it. Inequalities map to a single block
            index; equalities to the pair ``(first, first + half)`` covering
            the + and - copies. Relational inputs are additionally registered
            under their converted (``>= 0``) form.
        warnings: Non-fatal build warnings, e.g. constraints whose degree
            exceeds the relaxation level.
    """

    blocks: tuple[BlockSpec, ...] = ()
    constraint_to_blocks: dict[Any, tuple[int, ...]] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()

    @property
    def constraint_starting_block(self) -> int:
        """Index of the first constraint block (after the moment matrices)."""
        for index, block in enumerate(self.blocks):
            if block.kind in (
                BlockKind.LOCALIZING,
                BlockKind.SCALAR_INEQUALITY,
                BlockKind.SCALAR_EQUALITY,
            ):
                return index
        return len(self.blocks)


def _block_size(monomials: list[Any]) -> int:
    """Size of a moment block over a monomial set (possibly rectangular).

    A rectangular set ``[A, B]`` spans the tensor product basis ``A x B``,
    so its block is ``len(A) * len(B)`` rows tall. The old code returned
    ``len(A)`` here and left the generation code to produce the product --
    an inconsistency the Moroder hierarchy patched by overriding the block
    size. The product size is fixed at the source instead.
    """
    if len(monomials) > 0 and isinstance(monomials[0], list):
        return len(monomials[0]) * len(monomials[1])
    return len(monomials)


def compute_block_structure(
    *,
    variables: Any,
    monomial_sets: list[list[Any]],
    level: int,
    inequalities: list[Any] | None,
    equalities: list[Any] | None,
    momentinequalities: list[Any] | None,
    momentequalities: list[Any] | None,
    parameters: list[Any] | None,
    extramomentmatrices: list[list[str]] | None,
    removeequalities: bool,
    localizing_monomials: list[list[Any]] | None,
) -> BlockStructure:
    """Compute the ordered block layout of the SDP relaxation.

    The localizing basis of a polynomial constraint of degree d at level l
    consists of all basis monomials of degree up to (2l - d)/2; constraints
    exceeding degree 2l yield a warning and the trivial basis.

    Args:
        variables: The problem variables (or list of variable lists).
        monomial_sets: One monomial basis per moment matrix block.
        level: The relaxation level.
        inequalities: Polynomial inequality constraints.
        equalities: Polynomial equality constraints.
        momentinequalities: Inequalities over moment expressions.
        momentequalities: Equalities over moment expressions.
        parameters: Parameter variables, each becoming its own 1x1 block.
        extramomentmatrices: Per extra moment matrix, the option strings
            ``"copy"`` and/or ``"ppt"``.
        removeequalities: If True, equalities are eliminated algebraically
            and produce no blocks.
        localizing_monomials: Per-constraint override of the localizing
            basis; None entries request the automatic basis.

    Returns:
        The block structure with constraint-to-block lookup.
    """
    blocks: list[BlockSpec] = []
    constraint_to_blocks: dict[Any, tuple[int, ...]] = {}
    warnings: list[str] = []

    # Parameter blocks precede everything else; each parameter becomes its
    # own SDP variable, so its block is just a free 1x1 matrix.
    if parameters is not None:
        blocks.extend(BlockSpec(BlockKind.PARAMETER, 1) for _ in parameters)

    # One moment matrix block per monomial set.
    for index, monomials in enumerate(monomial_sets):
        blocks.append(
            BlockSpec(BlockKind.MOMENT, _block_size(monomials), monomial_set_index=index)
        )

    # Extra moment matrices (copy of the first block, or unconstrained new
    # blocks) repeat the base block sizes.
    if extramomentmatrices is not None:
        for options in extramomentmatrices:
            kind = BlockKind.COPY if "copy" in options else BlockKind.NEW
            impose_ppt = "ppt" in options
            for index, monomials in enumerate(monomial_sets):
                blocks.append(
                    BlockSpec(
                        kind,
                        _block_size(monomials),
                        monomial_set_index=index,
                        impose_ppt=impose_ppt,
                    )
                )

    # Constraint blocks, in the same order the constraints are processed
    # downstream: inequalities, moment inequalities, equalities, moment
    # equalities.
    constraints = list(flatten([inequalities]))
    n_inequalities = len(inequalities) if inequalities is not None else 0
    n_polynomial_inequalities = n_inequalities
    if momentinequalities is not None:
        constraints += list(momentinequalities)
        n_inequalities += len(momentinequalities)
    if not removeequalities and equalities is not None:
        constraints += list(flatten([equalities]))
    # Moment equalities are NOT part of the generic list: they get their
    # scalar block pair from the dedicated loop below, and adding them here
    # would double their blocks.

    for k, raw_constraint in enumerate(constraints):
        constraint = as_moment_expr(raw_constraint)
        if isinstance(constraint, MomentExpr):
            # Moment constraints act on 1x1 blocks with the trivial basis.
            localizing_set = [S.One]
        elif k < n_polynomial_inequalities or k >= n_inequalities:
            # A genuine polynomial constraint (inequality or equality):
            # compute its localizing basis from the relaxation degree.
            if constraint.is_Relational:
                constraint = convert_relational(constraint)
            order = ncdegree(constraint)
            if order > 2 * level:
                warnings.append(
                    f"A constraint has degree {order}. Either choose a higher "
                    "level relaxation or ensure that a mixed-order relaxation "
                    "has the necessary monomials"
                )
            localization_order = (2 * level - order) // 2
            if level == -1:
                localization_order = 0
            if (
                localizing_monomials is not None
                and localizing_monomials[k] is not None
            ):
                localizing_set = localizing_monomials[k]
            else:
                index = find_variable_set(variables, constraint)
                localizing_set = pick_monomials_up_to_degree(
                    monomial_sets[index], localization_order
                )
            if len(localizing_set) == 0:
                localizing_set = [S.One]
        else:
            localizing_set = [S.One]
        localizing_set = unique(localizing_set)
        ln = len(localizing_set)
        if k < n_inequalities:
            # Inequality block: the localizing matrix over the basis.
            kind = (
                BlockKind.LOCALIZING
                if not isinstance(constraint, MomentExpr)
                else BlockKind.SCALAR_INEQUALITY
            )
            block_index = len(blocks)
            blocks.append(
                BlockSpec(
                    kind,
                    ln,
                    constraint=constraint,
                    localizing_set=tuple(localizing_set),
                )
            )
            constraint_to_blocks[constraint] = (block_index,)
            if constraint is not raw_constraint:
                constraint_to_blocks[raw_constraint] = (block_index,)
        else:
            # Equality: ln*(ln+1) one-by-one blocks in two halves. The first
            # half enforces M_y(+eq), the second M_y(-eq); together they pin
            # every entry of the localizing matrix to zero. The lookup entry
            # (first, first + half) mirrors the old _constraint_to_block_index
            # convention used by get_dual.
            half = ln * (ln + 1) // 2
            first = len(blocks)
            blocks.extend(
                BlockSpec(
                    BlockKind.SCALAR_EQUALITY,
                    1,
                    constraint=constraint * sign,
                    localizing_set=tuple(localizing_set),
                )
                for sign in (1, -1)
                for _ in range(half)
            )
            constraint_to_blocks[constraint] = (first, first + half)
            if constraint is not raw_constraint:
                constraint_to_blocks[raw_constraint] = (first, first + half)

    # Moment equalities: two 1x1 blocks per constraint (+ and -).
    if not removeequalities and momentequalities is not None:
        for meq in momentequalities:
            first = len(blocks)
            blocks.extend(
                BlockSpec(
                    BlockKind.SCALAR_EQUALITY,
                    1,
                    constraint=meq * sign,
                    localizing_set=(S.One,),
                )
                for sign in (1, -1)
            )
            constraint_to_blocks[meq] = (first, first + 1)

    return BlockStructure(tuple(blocks), constraint_to_blocks, tuple(warnings))
