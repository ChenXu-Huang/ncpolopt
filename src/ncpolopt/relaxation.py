"""SDP relaxation construction: from a symbolic Problem to frozen SDP data.

Ported from the legacy ``SdpRelaxation`` class. The old class was a god
object: it held the symbolic model, built the SDP in one
giant ``lil_matrix``, solved it, and extracted values. Here the model lives
in :class:`~ncpolopt.problem.Problem`, construction fills a per-block
:class:`~ncpolopt.sdp_problem.SdpBuilder`, and solving is delegated to the
solver registry.

The block order, constraint processing order, and the canonical sign
conventions are unchanged so numerical results match the old package; what
changed is the representation (per-block COO instead of global row-offset
arithmetic) and the value model (MomentExpr instead of the string DSL).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from sympy import S
from sympy.core.add import Add

from ._logging import module_logger
from .block_structure import BlockKind, BlockStructure, compute_block_structure
from .equality_elimination import eliminate_equalities
from .expressions import convert_relational, flatten, is_number_type, iscomplex
from .moment import MomentEntry, MomentExpr, as_moment_expr
from .monomial_sets import estimate_n_vars, generate_monomial_sets
from .monomials import ncdegree, pick_monomials_up_to_degree, unique
from .sdp_problem import SdpBuilder, SdpProblem
from .solution import Solution
from .solvers.base import SolverKind, SolverSettings
from .solvers.registry import solve_problem
from .substitutions import (
    apply_substitutions,
    assemble_monomial_and_do_substitutions,
    constant_part,
    is_pure_substitution_rule,
    moment_of_entry,
    poly_elements,
    separate_scalar_factor,
    simplify_polynomial,
)
from .variables import find_variable_set

logger = module_logger(__name__)


def _upper_triangle(size: int) -> list[tuple[int, int]]:
    """All upper-triangle positions in column-major order."""
    return [(row, column) for row in range(size) for column in range(row, size)]


def _entry_position(index: int, size: int) -> tuple[int, int]:
    """The index-th upper-triangle position in column-major order.

    Localizing entries are pushed into their own 1x1 blocks in the same
    order they are iterated: (0,0), (0,1), ..., (0,size-1), (1,1), ...
    """
    row = 0
    while index >= size - row:
        index -= size - row
        row += 1
    return row, row + index


def _localizing_basis(
    constraint: Any,
    variables: Any,
    monomial_sets: list[list[Any]],
    level: int,
) -> list[Any]:
    """The automatic localizing basis of a constraint.

    All basis monomials up to degree ``(2*level - order)//2`` of the
    variable set the constraint lives in; the trivial basis when that
    yields nothing.

    Args:
        constraint: The (converted) constraint polynomial.
        variables: The problem variables (or list of variable lists).
        monomial_sets: One monomial basis per moment matrix block.
        level: The relaxation level.

    Returns:
        The localizing monomials.
    """
    order = ncdegree(constraint)
    if order > 2 * level:
        raise ValueError(
            f"An equality constraint has degree {order}. Choose a higher "
            "level of relaxation."
        )
    localization_order = (2 * level - order) // 2
    if level == -1:
        localization_order = 0
    index = find_variable_set(variables, constraint)
    basis = pick_monomials_up_to_degree(monomial_sets[index], localization_order)
    if len(basis) == 0:
        basis = [S.One]
    return unique(basis)


class Relaxation(ABC):
    """A built SDP relaxation of a Problem, ready to solve.

    Subclasses implement the construction (see :class:`NpaRelaxation` and
    the hierarchy variants); the base class holds the shared state and the
    solve entry point.

    Attributes:
        problem: The symbolic problem this relaxation was built from.
        level: The relaxation level.
        sdp: The frozen SDP data, set once construction finished.
        solution: The most recent solver result, set by :meth:`solve`.
        warnings: Non-fatal construction warnings.
    """

    def __init__(self, problem: Any, level: int) -> None:
        self.problem = problem
        self.level = level
        self.sdp: SdpProblem | None = None
        self.solution: Solution | None = None
        self.moment_block_indices: tuple[int, ...] = (0,)
        self.warnings: list[str] = []

    @abstractmethod
    def _build(self, removeequalities: bool) -> None:
        """Run the full construction pipeline of the hierarchy.

        Args:
            removeequalities: Whether equality constraints are eliminated
                algebraically instead of becoming SDP blocks.
        """

    def solve(
        self,
        solver: SolverKind | str | None = "auto",
        settings: SolverSettings | None = None,
    ) -> Solution:
        """Solve the relaxation with the requested (or detected) backend.

        Args:
            solver: The solver kind, its string value, or None/"auto" for
                the first available backend.
            settings: Backend knobs; defaults to empty settings.

        Returns:
            The solution, also cached in :attr:`solution`.

        Raises:
            SolverError: If no solver backend is available.
        """
        if self.sdp is None:
            raise RuntimeError("The relaxation has not been built.")
        result = solve_problem(self.sdp, solver=solver, settings=settings)
        self.solution = Solution(self.sdp, result, self.moment_block_indices)
        return self.solution


class NpaRelaxation(Relaxation):
    """The standard NPA hierarchy relaxation of a Problem.

    Construction follows the old ``SdpRelaxation.get_relaxation`` flow:
    monomial bases, block structure, parameter and moment matrix blocks,
    extra moment matrices, the objective, and finally the constraint
    blocks with optional equality elimination.

    Attributes:
        monomial_sets: One monomial basis per moment matrix block.
        monomial_index: Maps each moment monomial to its SDP variable.
        var_offsets: The number of SDP variables after each moment block
            (and after each extra moment matrix block).
        complex_matrix: Whether the relaxation uses complex coefficients.
    """

    def __init__(
        self,
        problem: Any,
        level: int,
        *,
        removeequalities: bool = False,
    ) -> None:
        """Build the relaxation of ``problem`` at ``level``.

        Args:
            problem: The symbolic problem to relax.
            level: The relaxation level; -1 uses only ``extramonomials``.
            removeequalities: Eliminate the equality constraints
                algebraically by solving the linear equations they impose.

        Raises:
            ValueError: If the level is below -1.
        """
        super().__init__(problem, level)
        if level < -1:
            raise ValueError("Invalid level of relaxation.")
        self.removeequalities = removeequalities
        self.monomial_sets: list[list[Any]] = []
        self.monomial_index: dict[Any, int] = {}
        self.var_offsets: list[int] = []
        self.complex_matrix = (
            problem.complex_matrix if problem.complex_matrix is not None else False
        )
        self._substitutions: dict[Any, Any] = {}
        self._moment_substitutions: dict[Any, Any] = {}
        self._pure_substitution_rules = True
        self._n_parameter_blocks = 0
        self._builder: SdpBuilder | None = None
        self._position_to_var: dict[tuple[int, int, int], int] = {}
        self._build(removeequalities)

    def _build(self, removeequalities: bool) -> None:
        """Run the full construction pipeline."""
        problem = self.problem
        self._prepare_substitutions()
        self.monomial_sets = self._generate_monomial_sets()
        # Constraints with complex coefficients force a complex-valued SDP.
        for constraint in flatten([problem.inequalities, problem.equalities]):
            if isinstance(constraint, (MomentEntry, MomentExpr)):
                continue
            if constraint.is_Relational:
                constraint = convert_relational(constraint)
            if iscomplex(constraint):
                self.complex_matrix = True
        structure = self._compute_block_structure(
            variables=problem.variables,
            monomial_sets=self.monomial_sets,
            level=self.level,
            inequalities=problem.inequalities,
            equalities=problem.equalities,
            momentinequalities=problem.momentinequalities,
            momentequalities=problem.momentequalities,
            parameters=problem.parameters,
            extramomentmatrices=problem.extramomentmatrices,
            removeequalities=removeequalities,
            localizing_monomials=problem.localizing_monomials,
        )
        self.warnings = list(structure.warnings)
        logger.info(
            "Estimated number of SDP variables: %d",
            estimate_n_vars(self.monomial_sets, problem.parameters, problem.normalized),
        )
        self._builder = SdpBuilder(
            [block.size for block in structure.blocks],
            problem.normalized,
            self.complex_matrix,
        )
        self._n_parameter_blocks = self._add_parameters()
        logger.info("Generating moment matrix...")
        self._generate_all_moment_matrix_blocks()
        self._add_extra_momentmatrices(structure)
        logger.info("Reduced number of SDP variables: %d", self._builder.n_vars)
        # Resolve the (block, i, j) -> variable map for MomentExpr entries
        # and objectives over the finished matrix layout.
        self._position_to_var = {
            position: k for k, position in self._builder.column_locations.items()
        }
        self.set_objective(problem.objective)
        self._process_constraints(structure, removeequalities)
        self.monomial_index = dict(self._builder.monomial_index)
        self.moment_block_indices = tuple(
            range(
                self._n_parameter_blocks,
                self._n_parameter_blocks + len(self.monomial_sets),
            )
        )
        self.sdp = self._builder.freeze(
            self.level,
            structure.constraint_to_blocks,
            tuple(self.monomial_sets[0]) if self.monomial_sets else (),
            self._substitutions,
            self._moment_substitutions,
        )
        logger.info("Number of SDP variables: %d", self.sdp.n_vars)

    def _prepare_substitutions(self) -> None:
        """Copy the substitution rules and detect complex-valued problems."""
        problem = self.problem
        self._substitutions = (
            dict(problem.substitutions) if problem.substitutions is not None else {}
        )
        for lhs, rhs in self._substitutions.items():
            if not is_pure_substitution_rule(lhs, rhs):
                self._pure_substitution_rules = False
            if iscomplex(lhs) or iscomplex(rhs):
                self.complex_matrix = True
        if problem.complex_matrix is not None:
            self.complex_matrix = problem.complex_matrix
        if problem.momentsubstitutions is not None:
            self._moment_substitutions = dict(problem.momentsubstitutions)
            if not self.complex_matrix:
                # In a real-valued problem the moment matrix is symmetric,
                # so each moment substitution also applies to the
                # conjugate monomial.
                for key, value in list(self._moment_substitutions.items()):
                    adjoint = apply_substitutions(key.adjoint(), self._substitutions)
                    self._moment_substitutions[adjoint] = value

    def _generate_monomial_sets(self) -> list[list[Any]]:
        """Generate the monomial bases of the moment matrix blocks.

        The standard NPA hierarchy generates one flat set per variable
        group; hierarchy subclasses override this (the Moroder basis wraps
        the two party bases into a rectangular pair).
        """
        return generate_monomial_sets(
            self.problem.variables,
            self.level,
            self.problem.extramonomials,
            self._substitutions,
        )

    def _compute_block_structure(self, **kwargs: Any) -> BlockStructure:
        """Compute the block structure of the relaxation.

        Hierarchy subclasses override this to adjust the structure after
        the base computation (the steering hierarchy scales the moment
        block sizes by its matrix-variable dimension).

        Args:
            kwargs: The keyword arguments of :func:`compute_block_structure`.

        Returns:
            The computed block structure.
        """
        return compute_block_structure(**kwargs)

    def _add_parameters(self) -> int:
        """Create one free 1x1 block per parameter variable.

        Returns:
            The number of parameter blocks (the index of the first moment
            matrix block).
        """
        parameters = self.problem.parameters
        if parameters is None:
            return 0
        for index, var in enumerate(parameters):
            k = self._builder.new_variable(var, index, 0, 0)
            self._builder.add_entry(index, 0, 0, k, 1.0)
        return len(parameters)

    def _generate_all_moment_matrix_blocks(self) -> None:
        """Build the moment matrix blocks over the monomial sets."""
        block_index = self._n_parameter_blocks
        for monomials in self.monomial_sets:
            if len(monomials) > 0 and isinstance(monomials[0], list):
                # Rectangular set: the moment matrix over the tensor product
                # basis A x B (used by the steering and Moroder hierarchies).
                block_index = self._generate_moment_matrix(
                    monomials[0], monomials[1], block_index
                )
            else:
                block_index = self._generate_moment_matrix(
                    monomials, [S.One], block_index
                )
            self.var_offsets.append(self._builder.n_vars)

    def _generate_moment_matrix(
        self,
        monomialsA: list[Any],
        monomialsB: list[Any],
        block_index: int,
        ppt: bool = False,
    ) -> int:
        """Build one moment matrix block over the (product) basis.

        Args:
            monomialsA: The first factor of the basis.
            monomialsB: The second factor (the identity for flat bases).
            block_index: The block's index in the final structure.
            ppt: Transpose the second factor's indices, i.e. build the
                partial transpose of the moment matrix (used by the Moroder
                hierarchy).

        Returns:
            The index of the next block.
        """
        len_b = len(monomialsB)
        for row_a in range(len(monomialsA)):
            for column_a in range(row_a, len(monomialsA)):
                for row_b in range(len_b):
                    start_column_b = row_b if row_a == column_a else 0
                    for column_b in range(start_column_b, len_b):
                        _, _, _, monomial = assemble_monomial_and_do_substitutions(
                            (row_a, column_a, row_b, column_b),
                            monomialsA,
                            monomialsB,
                            ppt,
                            self._substitutions,
                            self._pure_substitution_rules,
                        )
                        i = row_a * len_b + row_b
                        j = column_a * len_b + column_b
                        self._push_monomial(monomial, block_index, i, j)
        return block_index + 1

    def _push_monomial(self, monomial: Any, block: int, i: int, j: int) -> None:
        """Push one moment monomial into entry (i, j) of a block.

        Moment substitutions apply here: the moment matrix is built over the
        substituted basis, so an entry that reduces to a constant or a known
        moment never creates a new variable.
        """
        substitute = self._moment_substitutions.get(monomial)
        if substitute is not None:
            self._push_monomial(substitute, block, i, j)
            return
        if is_number_type(monomial):
            if i == 0 and j == 0 and not self._builder.normalized:
                # Without normalization the top-left entry is a free
                # variable; with normalization it is pinned to 1.0.
                k = self._builder.new_variable(None, block, 0, 0)
                self._builder.add_entry(block, 0, 0, k, 1.0)
            else:
                self._builder.add_entry(block, i, j, 0, monomial)
        elif monomial.is_Add:
            for element in monomial.args:
                self._push_monomial(element, block, i, j)
        elif monomial != 0:
            for k, coeff in self._process_monomial(monomial, block, i, j):
                self._builder.add_entry(block, i, j, k, coeff)

    def _process_monomial(
        self, monomial: Any, block: int, i: int, j: int
    ) -> list[tuple[int, float]]:
        """Resolve one monomial into (SDP variable, coefficient) entries.

        Moment substitutions apply first; a known monomial maps to its
        variable, a real-valued problem reuses the variable of the conjugate
        monomial, and anything else becomes a new variable whose first
        occurrence records the (block, i, j) position.

        Args:
            monomial: The monomial to resolve.
            block: The block being built.
            i: Row of the entry being built.
            j: Column of the entry being built.

        Returns:
            The (variable, coefficient) pairs to push.
        """
        processed_monomial, coeff = separate_scalar_factor(monomial)
        substitute = self._moment_substitutions.get(processed_monomial)
        if substitute is not None:
            result: list[tuple[int, float]] = []
            args = [substitute] if not isinstance(substitute, Add) else substitute.args
            for arg in args:
                if is_number_type(arg):
                    value = complex(arg) if iscomplex(arg) else float(arg)
                    result.append((0, coeff * value))
                else:
                    result += self._process_monomial(arg, block, i, j)
            return result
        index = self._builder.monomial_index.get(processed_monomial)
        if index is None and not self.complex_matrix:
            # Hermiticity turns symmetry into the conjugate's variable.
            adjoint = apply_substitutions(
                processed_monomial.adjoint(), self._substitutions
            )
            index = self._builder.monomial_index.get(adjoint)
        if index is not None:
            return [(index, coeff)]
        k = self._builder.new_variable(processed_monomial, block, i, j)
        return [(k, coeff)]

    def _add_extra_momentmatrices(self, structure: BlockStructure) -> None:
        """Build the extra moment matrix blocks.

        A block requested as a "copy" replicates the entries of its base
        moment matrix under the same SDP variables; otherwise it is
        unconstrained, with a fresh variable per upper-triangle entry.
        "ppt" then rearranges the block into the partial transpose of the
        moment matrix (bipartite layout).
        """
        if self.problem.extramomentmatrices is None:
            return
        builder = self._builder
        for block_index, spec in enumerate(structure.blocks):
            if spec.kind is BlockKind.COPY:
                source = self._n_parameter_blocks + spec.monomial_set_index
                self._copy_block(source, block_index)
            elif spec.kind is BlockKind.NEW:
                for i in range(spec.size):
                    for j in range(i, spec.size):
                        k = builder.new_variable(None, block_index, i, j)
                        builder.add_entry(block_index, i, j, k, 1.0)
            else:
                continue
            self.var_offsets.append(builder.n_vars)
            if spec.impose_ppt:
                self._impose_ppt(block_index)

    def _copy_block(self, source: int, target: int) -> None:
        """Replicate the entries of one block under the same SDP variables."""
        builder = self._builder
        for k, i, j, value in builder.entries_of(source):
            builder.add_entry(target, i, j, k, value)

    def _impose_ppt(self, block_index: int) -> None:
        """Rearrange a block into the partial transpose (bipartite layout).

        The swap exchanges entry (rowA, rowB; colA, colB) with
        (rowA, colB; colA, rowB) for rowA < colA -- the partial transpose
        of the B subsystem. The diagonal A-blocks are symmetric and stay
        put. The position mapping matches the old ``__impose_ppt``.
        """
        basis = self.monomial_sets[0]
        if not (len(basis) > 0 and isinstance(basis[0], list)):
            raise ValueError(
                "'ppt' requires a rectangular monomial basis [A, B] to "
                "transpose the B subsystem."
            )
        len_a = len(basis[0])
        len_b = len(basis[1])
        swaps: list[tuple[int, int, int, int]] = []
        for row_a in range(len_a):
            for column_a in range(row_a + 1, len_a):
                for row_b in range(len_b):
                    for column_b in range(row_b):
                        swaps.append(
                            (
                                row_a * len_b + row_b,
                                column_a * len_b + column_b,
                                row_a * len_b + column_b,
                                column_a * len_b + row_b,
                            )
                        )
        if not swaps:
            return
        entries = self._builder.entries_of(block_index)
        by_position: dict[tuple[int, int], list[tuple[int, Any]]] = {}
        for k, i, j, value in entries:
            by_position.setdefault((i, j), []).append((k, value))
        for i1, j1, i2, j2 in swaps:
            by_position[(i1, j1)], by_position[(i2, j2)] = (
                by_position[(i2, j2)],
                by_position[(i1, j1)],
            )
        new_entries: list[tuple[int, int, int, Any]] = []
        for (i, j), values in by_position.items():
            for k, value in values:
                new_entries.append((k, i, j, value))
        self._builder.set_entries(block_index, new_entries)

    def set_objective(self, objective: Any | None) -> None:
        """Set the objective of the relaxation.

        The objective may be a polynomial over the problem variables or a
        MomentExpr over the moment matrix entries. A non-zero constant term
        of the polynomial objective is reported: it does not enter the SDP
        objective (the solvers add it back to the reported value).

        Args:
            objective: The objective expression, or None for a zero
                objective.
        """
        if isinstance(objective, (MomentEntry, MomentExpr)):
            facvar = self._moment_expr_facvar(as_moment_expr(objective))
        elif objective is not None:
            facvar = self._get_facvar(
                simplify_polynomial(objective, self._substitutions)
            )
        else:
            facvar = [0] * (self._builder.n_vars + 1)
        if facvar[0] != 0:
            logger.warning(
                "The objective function has a non-zero constant term; it is "
                "not included in the SDP objective."
            )
        self._builder.set_objective(facvar)

    def _get_facvar(self, polynomial: Any) -> list[float]:
        """The dense (n_vars + 1) vector of a polynomial over the moments.

        The entries are indexed by SDP variable, position 0 holding the
        constant term.
        """
        facvar = [0] * (self._builder.n_vars + 1)
        if is_number_type(polynomial):
            facvar[0] = polynomial
            return facvar
        polynomial = polynomial.expand()
        facvar[0] = constant_part(polynomial)
        elements = poly_elements(polynomial)
        for element in elements:
            for k, coeff in self._get_index_of_monomial(element):
                facvar[k] += coeff
        return facvar

    def _get_index_of_monomial(
        self,
        element: Any,
        enablesubstitution: bool = True,
        daggered: bool = False,
    ) -> list[tuple[int, float]]:
        """Resolve a monomial of a constraint or objective to (variable, coeff).

        Moment substitutions, the regular substitution rules, and the
        conjugate fallback all apply, mirroring ``_process_monomial`` but
        without creating variables: constraints and objectives may only
        reference moments of the moment matrix basis.

        Args:
            element: The polynomial element to resolve.
            enablesubstitution: Apply the regular substitution rules.
            daggered: Whether the element is already a conjugate fallback;
                a missing monomial then raises instead of recursing.

        Returns:
            The (variable, coefficient) pairs.
        """
        result: list[tuple[int, float]] = []
        processed_element, coeff1 = separate_scalar_factor(element)
        if processed_element in self._moment_substitutions:
            inner = self._get_index_of_monomial(
                self._moment_substitutions[processed_element], enablesubstitution
            )
            return [(k, coeff * coeff1) for k, coeff in inner]
        if enablesubstitution:
            processed_element = apply_substitutions(
                processed_element, self._substitutions, self._pure_substitution_rules
            )
        if is_number_type(processed_element):
            return [(0, coeff1)]
        if processed_element.is_Add:
            monomials = processed_element.args
        else:
            monomials = [processed_element]
        for monomial in monomials:
            monomial, coeff2 = separate_scalar_factor(monomial)
            coeff = coeff1 * coeff2
            if is_number_type(monomial):
                result.append((0, coeff))
                continue
            if monomial.as_coeff_Mul()[0] < 0:
                monomial = -monomial
                coeff = -1.0 * coeff
            substitute = self._moment_substitutions.get(monomial)
            if substitute is not None:
                # NOTE: the old code looked the substitute value up again as
                # a key and multiplied by an uninitialized k = -1, silently
                # dropping the term. Substituting the value directly is the
                # intended behavior.
                inner = self._get_index_of_monomial(substitute, enablesubstitution)
                result += [(k, coeff * coeff3) for k, coeff3 in inner]
                continue
            try:
                k = self._builder.monomial_index[monomial]
            except KeyError:
                if daggered:
                    raise RuntimeError(
                        f"The requested monomial {monomial} could not be found; "
                        "its degree may exceed the relaxation level."
                    ) from None
                dag_result = self._get_index_of_monomial(
                    monomial.adjoint(), daggered=True
                )
                result += [(k, coeff0 * coeff) for k, coeff0 in dag_result]
            else:
                result.append((k, coeff))
        return result

    def _push_facvar_sparse(
        self, polynomial: Any, block: int, i: int, j: int
    ) -> None:
        """Push a polynomial into entry (i, j) of a block.

        NOTE: do not expand the polynomial before decomposing it --
        ``simplify_polynomial`` already reduced every monomial, and an extra
        ``expand`` here triggers a SymPy bug with powers of daggered
        variables (ported from the old ``__push_facvar_sparse``).
        """
        if is_number_type(polynomial):
            self._builder.add_entry(block, i, j, 0, polynomial)
            return
        constant = constant_part(polynomial)
        if constant != 0:
            self._builder.add_entry(block, i, j, 0, constant)
        for element in poly_elements(polynomial):
            for k, coeff in self._get_index_of_monomial(element):
                if k > -1 and coeff != 0:
                    self._builder.add_entry(block, i, j, k, coeff)

    def _push_moment_expr(self, expr: MomentExpr, block: int) -> None:
        """Push a linear combination of moments into entry (0, 0) of a block."""
        for term in expr.terms:
            for k, factor in self._resolve_position(term):
                self._builder.add_entry(block, 0, 0, k, term.coefficient * factor)

    def _moment_expr_facvar(self, expr: MomentExpr) -> list[float]:
        """The dense (n_vars + 1) vector of a linear combination of moments."""
        facvar = [0] * (self._builder.n_vars + 1)
        for term in expr.terms:
            for k, factor in self._resolve_position(term):
                facvar[k] += term.coefficient * factor
        return facvar

    def _resolve_position(self, term: MomentEntry) -> list[tuple[int, float]]:
        """Resolve a MomentExpr term to (variable, factor) entries.

        The term's block counts moment matrix blocks (parameter blocks are
        not addressable). Lower-triangle references resolve through the
        transposed position; in complex-valued relaxations the value of the
        transpose is a conjugate, not a linear expression, so they are
        rejected. The top-left entry of a normalized moment matrix is the
        pinned normalization constant rather than a variable.

        Args:
            term: One moment matrix entry reference.

        Returns:
            The (variable, factor) pairs; factor 1.0 for a variable and
            1.0 at variable 0 for the normalization constant.
        """
        block = term.block + self._n_parameter_blocks
        position = (block, term.row, term.col)
        variable = self._position_to_var.get(position)
        if variable is not None:
            return [(variable, 1.0)]
        transposed = (block, term.col, term.row)
        variable = self._position_to_var.get(transposed)
        if variable is not None:
            if self.complex_matrix:
                raise ValueError(
                    f"Moment entry ({term.block}, {term.row}, {term.col}) refers "
                    "to the transpose of a complex-valued entry; conjugate "
                    "values are not linear in the SDP variables."
                )
            return [(variable, 1.0)]
        if term.row == 0 and term.col == 0 and self._builder.normalized:
            return [(0, 1.0)]
        raise ValueError(
            f"Moment entry ({term.block}, {term.row}, {term.col}) does not "
            "correspond to an SDP variable."
        )

    def _process_constraints(
        self, structure: BlockStructure, removeequalities: bool
    ) -> None:
        """Push the constraint blocks, optionally eliminating equalities.

        Localizing matrices of inequalities, scalar blocks of moment
        inequalities, and the scalar equality blocks (two halves per
        equality, one scalar block per localizing entry) are filled in
        structure order. With ``removeequalities`` the equality blocks do
        not exist in the structure; their equations are solved and the
        whole draft is rewritten in the reduced basis instead.

        Args:
            removeequalities: Whether the equalities were eliminated from
                the block structure.
        """
        last_group: tuple[Any, ...] | None = None
        entry_index = 0
        for block_index, spec in enumerate(structure.blocks):
            if spec.kind is BlockKind.LOCALIZING:
                monomials = list(spec.localizing_set)
                for row, column in _upper_triangle(len(monomials)):
                    _, _, polynomial = moment_of_entry(
                        (row, column), monomials, spec.constraint, self._substitutions
                    )
                    self._push_facvar_sparse(polynomial, block_index, row, column)
            elif spec.kind is BlockKind.SCALAR_INEQUALITY:
                self._push_moment_expr(spec.constraint, block_index)
            elif spec.kind is BlockKind.SCALAR_EQUALITY:
                if isinstance(spec.constraint, MomentExpr):
                    self._push_moment_expr(spec.constraint, block_index)
                else:
                    # A polynomial equality is enforced entry by entry: the
                    # e-th block of each half holds the e-th entry of the
                    # localizing matrix, pinned to zero.
                    group = (spec.constraint, spec.localizing_set)
                    if group != last_group:
                        last_group = group
                        entry_index = 0
                    row, column = _entry_position(entry_index, len(spec.localizing_set))
                    entry_index += 1
                    _, _, polynomial = moment_of_entry(
                        (row, column),
                        list(spec.localizing_set),
                        spec.constraint,
                        self._substitutions,
                    )
                    self._push_facvar_sparse(polynomial, block_index, 0, 0)
        if removeequalities:
            self._remove_equalities()

    def _remove_equalities(self) -> None:
        """Eliminate the equality constraints by solving the linear system.

        The moment equations are collected as rows of the matrix A; the
        QR-based basis transform rewrites every block and the objective in
        the reduced space (see :func:`eliminate_equalities`).
        """
        A = self._process_equalities()
        if A.shape[0] == 0:
            return
        if min(A.shape) != np.linalg.matrix_rank(A):
            logger.warning(
                "Equality constraints are linearly dependent; results might "
                "be incorrect."
            )
        transform = eliminate_equalities(A)
        self._builder.eliminate(transform)
        logger.info(
            "Number of variables after solving the linear equations: %d",
            self._builder.n_vars,
        )

    def _process_equalities(self) -> np.ndarray:
        """Assemble the rows of the equality system A x + A[:, 0] = 0.

        Each entry of the localizing matrix of each polynomial equality
        contributes one row, then each moment equality one row.

        Returns:
            The matrix A of shape (n_rows, n_vars + 1).
        """
        rows: list[list[float]] = []
        equalities = self.problem.equalities
        if equalities is not None:
            for constraint in flatten([equalities]):
                if constraint.is_Relational:
                    constraint = convert_relational(constraint)
                basis = _localizing_basis(
                    constraint, self.problem.variables, self.monomial_sets, self.level
                )
                for row, column in _upper_triangle(len(basis)):
                    _, _, polynomial = moment_of_entry(
                        (row, column), basis, constraint, self._substitutions
                    )
                    rows.append(self._get_facvar(polynomial))
        momentequalities = self.problem.momentequalities
        if momentequalities is not None:
            for meq in momentequalities:
                meq = as_moment_expr(meq)
                if isinstance(meq, MomentExpr):
                    rows.append(self._moment_expr_facvar(meq))
                else:
                    # A polynomial moment equality constrains the top-left
                    # moment entry directly (the old __process_equalities
                    # pushed ``_get_facvar(meq)`` as its row).
                    rows.append(
                        self._get_facvar(
                            simplify_polynomial(meq, self._substitutions)
                        )
                    )
        if not rows:
            return np.zeros((0, 0))
        dtype = np.complex128 if self.complex_matrix else np.float64
        return np.asarray(rows, dtype=dtype)
