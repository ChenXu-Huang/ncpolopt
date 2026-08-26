"""Extraction of values from a solved SDP relaxation.

The old package glued the solution state onto the relaxation object and
extracted monomial values by recursing through the global F matrix
(``get_xmat_value``, a linear solve per variable). Here the solved data is
an immutable :class:`Solution` wrapping the frozen SDP and the raw
:class:`SolverResult`; every moment monomial has an O(1) value lookup
through the ``column_locations`` table recorded at construction time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sympy import expand

from .expressions import is_number_type
from .monomials import pick_monomials_up_to_degree
from .sdp_problem import SdpProblem
from .solvers.base import SolverResult
from .substitutions import (
    apply_substitutions,
    constant_part,
    poly_elements,
    separate_scalar_factor,
    simplify_polynomial,
)


@dataclass(frozen=True, slots=True)
class Solution:
    """The outcome of solving a relaxation, with value extraction.

    The primal and dual objective values, the status, and the per-block
    solution matrices are the solver result; the extraction methods map
    moment monomials to their values in the solved moment matrix.

    Attributes:
        sdp: The frozen SDP problem that was solved.
        result: The raw solver result.
        moment_block_indices: Indices of the moment matrix blocks, in
            construction order; the first one anchors the constant monomial
            and the rank-loop checks.
    """

    sdp: SdpProblem
    result: SolverResult
    moment_block_indices: tuple[int, ...] = (0,)

    @property
    def status(self) -> str:
        """The solver status."""
        return self.result.status

    @property
    def primal(self) -> float:
        """The primal optimal value, ``c.x* + c0``."""
        return self.result.primal

    @property
    def dual(self) -> float:
        """The dual optimal value, ``-sum_b tr(Y_b @ A0_b) + c0``."""
        return self.result.dual

    @property
    def x_mat(self) -> tuple[np.ndarray, ...]:
        """The per-block primal solution matrices."""
        return self.result.x_mat

    @property
    def y_mat(self) -> tuple[np.ndarray, ...]:
        """The per-block dual solution matrices."""
        return self.result.y_mat

    @property
    def solution_time(self) -> float:
        """The wall-clock solve time in seconds."""
        return self.result.solution_time

    @property
    def _moment_block(self) -> int:
        """The block index of the first moment matrix."""
        return self.moment_block_indices[0] if self.moment_block_indices else 0

    @property
    def _is_complex(self) -> bool:
        """Whether the relaxation uses complex coefficients."""
        return any(
            block.coo.dtype == np.complex128 for block in self.sdp.blocks
        )

    def _require_x_mat(self) -> tuple[np.ndarray, ...]:
        """The primal block matrices, or a clear error when absent."""
        x_mat = self.result.x_mat
        if not x_mat:
            raise RuntimeError(
                "The SDP was not solved successfully and no primal solution "
                "is available."
            )
        return x_mat

    def _require_y_mat(self) -> tuple[np.ndarray, ...]:
        """The dual block matrices, or a clear error when absent."""
        y_mat = self.result.y_mat
        if not y_mat:
            raise RuntimeError(
                "The SDP was not solved successfully and no dual solution "
                "is available."
            )
        return y_mat

    def _constant_value(self, x_mat: tuple[np.ndarray, ...]) -> complex:
        """The value of the constant monomial in the solved moment matrix.

        The top-left entry of the first moment matrix block: pinned to 1.0
        by normalization, a free variable otherwise (its value is whatever
        the solver found).
        """
        return x_mat[self._moment_block][0, 0]

    def _index_of_monomial(
        self, element: Any
    ) -> list[tuple[int, float]]:
        """Resolve a polynomial element to (variable, coefficient) pairs.

        Mirrors the construction-time ``_get_index_of_monomial``: moment
        substitutions, the regular substitution rules, and the conjugate
        fallback for real-valued problems, without creating variables.

        Args:
            element: The polynomial element to resolve.

        Returns:
            The (variable, coefficient) pairs; variable 0 is the constant.

        Raises:
            RuntimeError: If the reduced monomial is not in the moment basis
                (its degree exceeds the relaxation level).
        """
        result: list[tuple[int, float]] = []
        processed_element, coeff1 = separate_scalar_factor(element)
        if processed_element in self.sdp.moment_substitutions:
            inner = self._index_of_monomial(
                self.sdp.moment_substitutions[processed_element]
            )
            return [(k, coeff * coeff1) for k, coeff in inner]
        processed_element = apply_substitutions(
            processed_element, self.sdp.substitutions
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
            if monomial.as_coeff_Mul()[0].is_negative:
                monomial = -monomial
                coeff = -coeff
            substitute = self.sdp.moment_substitutions.get(monomial)
            if substitute is not None:
                inner = self._index_of_monomial(substitute)
                result += [(k, coeff * coeff3) for k, coeff3 in inner]
                continue
            k = self.sdp.monomial_index.get(monomial)
            if k is None and not self._is_complex:
                # Hermiticity turns symmetry into the conjugate's variable.
                k = self.sdp.monomial_index.get(monomial.adjoint())
            if k is None:
                raise RuntimeError(
                    f"The requested monomial {monomial} could not be found; "
                    "its degree may exceed the relaxation level."
                )
            result.append((k, coeff))
        return result

    def monomial_value(
        self, monomial: Any, x_mat: tuple[np.ndarray, ...] | None = None
    ) -> complex:
        """The value of a moment monomial (or polynomial of moments).

        The requested expression is reduced under the substitution rules,
        decomposed into the SDP variables, and evaluated at their solved
        values. Every variable's value is the entry of the primal block
        matrix at the position recorded by ``column_locations`` -- no
        recursive back-solving.

        Args:
            monomial: The polynomial over moments to evaluate.
            x_mat: Per-block primal matrices; the solved ones by default.

        Returns:
            The value of the expression in the solved relaxation.
        """
        if x_mat is None:
            x_mat = self._require_x_mat()
        polynomial = simplify_polynomial(monomial, self.sdp.substitutions)
        if is_number_type(polynomial):
            return complex(polynomial) * self._constant_value(x_mat)
        polynomial = polynomial.expand()
        value = constant_part(polynomial) * self._constant_value(x_mat)
        for element in poly_elements(polynomial):
            for k, coeff in self._index_of_monomial(element):
                if k == 0:
                    value += coeff * self._constant_value(x_mat)
                else:
                    block, row, column = self.sdp.column_locations[k]
                    value += coeff * x_mat[block][row, column]
        return complex(value)

    def dual_value(
        self, monomial: Any, blocks: tuple[int, ...] | list[int] | None = None
    ) -> float:
        """The inner product of the monomial's coefficient matrices and the
        dual solution, restricted to the given blocks.

        With ``blocks`` restricted, the result is the dual contribution of
        those blocks to the value of the monomial.

        Args:
            monomial: The monomial to look up.
            blocks: The blocks to include; all blocks by default.

        Returns:
            The value ``-sum_b tr(Y_b @ A_k,b)`` for the blocks in scope,
            where ``A_k,b`` is the coefficient matrix of the monomial.
        """
        y_mat = self._require_y_mat()
        if blocks is None:
            blocks = range(len(self.sdp.blocks))
        index = (
            0
            if is_number_type(monomial)
            else self.sdp.monomial_index[monomial]
        )
        result = 0.0
        for block in blocks:
            coo = self.sdp.blocks[block].coo
            size = self.sdp.blocks[block].size
            for k, position, value in zip(
                coo.row, coo.col, coo.data, strict=True
            ):
                if int(k) != index:
                    continue
                row, column = divmod(int(position), size)
                result += -value * y_mat[block][row, column]
        return result

    def dual_block(self, block: int) -> np.ndarray:
        """The dual solution matrix of one block.

        Args:
            block: The block index.

        Returns:
            The dual matrix of the block.
        """
        return self._require_y_mat()[block]

    def sos_decomposition(
        self,
        blocks: tuple[int, ...] | list[int] | None = None,
        threshold: float = 0.0,
    ) -> list[Any]:
        """The SOS decomposition of the dual solution.

        Each dual block's spectral decomposition yields a sum of squares of
        the moment basis monomials; terms whose coefficients fall below the
        threshold are dropped.

        Args:
            blocks: The blocks to decompose; all blocks by default.
            threshold: Coefficient threshold below which terms are dropped.

        Returns:
            One polynomial per block.

        Raises:
            RuntimeError: If no dual solution exists.
        """
        y_mat = self._require_y_mat()
        basis = list(self.sdp.moment_basis)
        if not basis:
            raise RuntimeError(
                "Cannot automatically match primal and dual variables: the "
                "problem has no moment basis."
            )
        if blocks is None:
            blocks = range(len(y_mat))
        sos = []
        for block in blocks:
            vals, vecs = np.linalg.eigh(y_mat[block])
            term = 0
            for j, val in enumerate(vals):
                if val < -0.001:
                    raise RuntimeError(
                        f"Large negative eigenvalue {val}; the dual matrix "
                        "of block {block} cannot be positive."
                    )
                if val > 0:
                    sub_term = 0
                    for i, entry in enumerate(vecs[:, j]):
                        sub_term += entry * basis[i]
                    term += val * sub_term**2
            term = expand(term)
            new_term = 0
            for element in poly_elements(term):
                _, coeff = separate_scalar_factor(element)
                if abs(coeff) > threshold:
                    new_term += element
            sos.append(new_term)
        return sos

    def solution_ranks(
        self,
        xmat: np.ndarray | None = None,
        baselevel: int = 0,
    ) -> list[int]:
        """The ranks of the moment matrix at successive levels of the
        relaxation, for rank-loop detection.

        Args:
            xmat: The primal solution of the moment matrix; the solved one
                by default.
            baselevel: When non-zero, only the rank at that level is
                reported.

        Returns:
            The ranks ordered by increasing level; the full matrix rank is
            appended when the moment matrix is larger than the level basis.
        """
        if xmat is None:
            xmat = self._require_x_mat()[self._moment_block]
        ranks = []
        levels = range(1, self.sdp.level + 1) if baselevel == 0 else [baselevel]
        for level in levels:
            base_monomials = pick_monomials_up_to_degree(
                list(self.sdp.moment_basis), level
            )
            ranks.append(
                int(
                    np.linalg.matrix_rank(
                        xmat[: len(base_monomials), : len(base_monomials)]
                    )
                )
            )
        if xmat.shape != (len(base_monomials), len(base_monomials)):
            ranks.append(int(np.linalg.matrix_rank(xmat)))
        return ranks
