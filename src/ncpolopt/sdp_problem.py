"""Frozen SDP data model shared by every solver backend.

The old package represented the whole relaxation as one giant
``scipy.sparse.lil_matrix`` whose rows were block entries and whose columns
were SDP variables (``SdpRelaxation.F``), and every backend reimplemented the
row-offset arithmetic to traverse it. Here the data is split per block: each
:class:`SparseBlock` is a COO array indexed by (SDP variable, linearized
block position), so backends iterate per block without any global indexing.

:class:`SdpBuilder` is the mutable draft accumulated during relaxation
construction; :meth:`SdpBuilder.freeze` turns it into an immutable
:class:`SdpProblem` after optional equality elimination.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.sparse import coo_array

from .equality_elimination import BasisTransform


@dataclass(frozen=True, slots=True)
class SparseBlock:
    """Coefficient matrices of one block of the SDP.

    ``coo`` is a sparse array with one row per SDP variable (row 0 holds the
    constant term) and one column per linearized (row, column) position of
    the size-by-size block matrix, so the block matrix reads
    ``A0 + sum_i x_i * A_i``. Only the upper triangle is stored.

    Attributes:
        coo: The sparse coefficient array, shape (n_vars + 1, size**2).
        size: Number of rows (and columns) of the block.
    """

    coo: coo_array
    size: int

    def constant_matrix(self) -> np.ndarray:
        """Return the constant matrix A0 of the block."""
        return self.evaluate(np.zeros(self.coo.shape[0] - 1))

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """Evaluate the block matrix at the primal solution ``x``.

        Args:
            x: Primal solution, length ``n_vars``.

        Returns:
            The size-by-size matrix ``A0 + sum_i x_i * A_i``.
        """
        size = self.size
        mat = np.zeros((size, size), dtype=self.coo.dtype)
        flat = mat.ravel()
        for k, pos, value in zip(self.coo.row, self.coo.col, self.coo.data, strict=True):
            if k == 0:
                flat[pos] += value
            else:
                flat[pos] += x[k - 1] * value
        return mat


@dataclass(frozen=True, slots=True)
class SdpProblem:
    """An immutable SDP relaxation ready for the solvers.

    Follows the canonical convention ``min c.x + c0`` with block constraints
    ``A0_b + sum_k x_k A_{k,b} >= 0`` (equality blocks enforce ``= 0``).

    Attributes:
        n_vars: Number of SDP variables.
        blocks: The diagonal blocks in order.
        obj: The linear objective ``c``, length ``n_vars``.
        constant_term: The constant ``c0`` of the objective.
        monomial_index: Maps each moment monomial to its SDP variable.
        column_locations: For each SDP variable ``k`` (1-based), the
            (block, row, column) position of the moment matrix entry the
            variable represents. Value extraction becomes O(1): the value of
            a monomial is simply the entry of the primal block matrix at its
            location. This replaces the old recursive back-solve of
            ``get_xmat_value``.
        constraint_to_blocks: Maps each constraint to the indices of the
            blocks enforcing it.
        level: The relaxation level (used by rank-loop detection).
        normalized: Whether the normalization condition was imposed.
        moment_basis: The first monomial set, if the problem has one (used
            by SOS decomposition and rank-loop detection).
        substitutions: The monomial replacement rules the moment matrix was
            built under (used by solution extraction).
        moment_substitutions: The moment replacement rules the moment
            matrix was built under (used by solution extraction).
    """

    n_vars: int
    blocks: tuple[SparseBlock, ...]
    obj: np.ndarray
    constant_term: float
    monomial_index: dict[Any, int] = field(default_factory=dict)
    column_locations: dict[int, tuple[int, int, int]] = field(default_factory=dict)
    constraint_to_blocks: dict[Any, tuple[int, ...]] = field(default_factory=dict)
    level: int = 1
    normalized: bool = True
    moment_basis: tuple[Any, ...] = ()
    substitutions: dict[Any, Any] = field(default_factory=dict)
    moment_substitutions: dict[Any, Any] = field(default_factory=dict)


class SdpBuilder:
    """Mutable draft of the SDP accumulated during relaxation construction.

    Entries are pushed per block with :meth:`add_entry` -- no global
    row-offset arithmetic. New SDP variables are created with
    :meth:`new_variable`, which records the (block, i, j) position of the
    moment matrix entry the variable represents; together with
    :attr:`monomial_index` this makes value extraction O(1) after solving.
    """

    def __init__(
        self, block_sizes: list[int], normalized: bool, complex_matrix: bool
    ) -> None:
        self.block_sizes = list(block_sizes)
        self.normalized = normalized
        self._dtype = np.complex128 if complex_matrix else np.float64
        self._rows: list[list[int]] = [[] for _ in block_sizes]
        self._cols: list[list[int]] = [[] for _ in block_sizes]
        self._data: list[list[Any]] = [[] for _ in block_sizes]
        self._n_vars = 0
        self.column_locations: dict[int, tuple[int, int, int]] = {}
        self.monomial_index: dict[Any, int] = {}
        self.obj_facvar = np.zeros(1, dtype=self._dtype)

    @property
    def n_vars(self) -> int:
        """The current number of SDP variables."""
        return self._n_vars

    @property
    def dtype(self) -> np.dtype:
        """The dtype of the coefficient data (float or complex)."""
        return self._dtype

    def new_variable(
        self, monomial: Any | None, block: int, i: int, j: int
    ) -> int:
        """Create a new SDP variable representing a moment at (block, i, j).

        Args:
            monomial: The moment monomial the variable represents, or None
                for variables without a monomial mapping (unconstrained
                blocks, normalization variables).
            block: The block the variable was created in.
            i: Row within the block.
            j: Column within the block.

        Returns:
            The 1-based index of the new variable.
        """
        self._n_vars += 1
        self.column_locations[self._n_vars] = (block, i, j)
        if monomial is not None:
            self.monomial_index[monomial] = self._n_vars
        return self._n_vars

    def add_entry(
        self, block: int, i: int, j: int, variable: int, coefficient: Any
    ) -> None:
        """Add ``coefficient * x_variable`` at position (i, j) of a block.

        ``variable == 0`` denotes the constant term. Zero coefficients are
        skipped to keep the sparse data clean.

        Args:
            block: The block index.
            i: Row within the block.
            j: Column within the block.
            variable: The SDP variable index (0 = constant).
            coefficient: The coefficient to add.
        """
        if coefficient == 0:
            return
        position = i * self.block_sizes[block] + j
        self._rows[block].append(int(variable))
        self._cols[block].append(int(position))
        self._data[block].append(coefficient)

    def set_objective(self, facvar: list[Any]) -> None:
        """Set the objective from a dense facvar vector.

        Args:
            facvar: Length ``n_vars + 1`` vector whose first entry is the
                constant term of the objective.
        """
        facvar = np.asarray(facvar, dtype=self._dtype)
        if len(facvar) != self._n_vars + 1:
            raise ValueError(
                f"Objective facvar has length {len(facvar)}, expected "
                f"{self._n_vars + 1}."
            )
        self.obj_facvar = facvar

    def entries_of(self, block: int) -> list[tuple[int, int, int, Any]]:
        """The (variable, i, j, coefficient) entries of a block.

        Args:
            block: The block index.

        Returns:
            The entries in storage order.
        """
        size = self.block_sizes[block]
        return [
            (int(k), int(position) // size, int(position) % size, value)
            for k, position, value in zip(
                self._rows[block], self._cols[block], self._data[block], strict=True
            )
        ]

    def set_entries(self, block: int, entries: list[tuple[int, int, int, Any]]) -> None:
        """Replace the entries of a block with new (variable, i, j, value) tuples.

        Used by the partial-transpose rearrangement of extra moment matrices,
        which must move entries between positions of an already built block.

        Args:
            block: The block index.
            entries: The new entries, in storage order.
        """
        size = self.block_sizes[block]
        rows: list[int] = []
        cols: list[int] = []
        data: list[Any] = []
        for k, i, j, value in entries:
            rows.append(int(k))
            cols.append(i * size + j)
            data.append(value)
        self._rows[block] = rows
        self._cols[block] = cols
        self._data[block] = data

    def eliminate(self, transform: BasisTransform) -> None:
        """Rewrite the draft in the reduced basis x = shift + basis.y.

        Every block entry (k, pos) with k >= 1 spreads to the new free
        variables; the constant column absorbs the shift. The monomial index
        and column locations survive unchanged: the position of a moment in
        the (reduced) moment matrix is invariant, only its coefficients
        change.

        Args:
            transform: The basis transform from equality elimination.
        """
        shift = transform.shift
        basis = transform.basis
        n_free = basis.shape[1]
        for block in range(len(self.block_sizes)):
            rows = np.asarray(self._rows[block], dtype=np.int64)
            cols = np.asarray(self._cols[block], dtype=np.int64)
            data = np.asarray(self._data[block], dtype=self._dtype)
            if rows.size == 0:
                continue
            const_part = np.zeros(self.block_sizes[block] ** 2, dtype=self._dtype)
            new_rows: list[int] = []
            new_cols: list[int] = []
            new_data: list[Any] = []
            mask = rows >= 1
            if mask.any():
                k_idx = rows[mask] - 1
                d = data[mask]
                pos = cols[mask]
                # Constant part: every variable's value picks up its shift.
                np.add.at(const_part, pos, d * shift[rows[mask]])
                # Basis part: entry (k, pos) contributes d * basis[k-1, m]
                # to the new variable m at the same position. ``spread`` is
                # (n_entries, n_free) in row-major order, so the flattened
                # variable index runs 1..n_free per entry (tile) while the
                # position repeats per free variable (repeat).
                spread = basis[k_idx] * d[:, None]
                nz = (spread != 0).ravel()
                new_rows = list(np.tile(np.arange(1, n_free + 1), len(k_idx))[nz])
                new_cols = list(np.repeat(pos, n_free)[nz])
                new_data = list(spread.ravel()[nz])
            const_mask = rows == 0
            if const_mask.any():
                np.add.at(const_part, cols[const_mask], data[const_mask])
            const_nz = np.nonzero(const_part)[0]
            self._rows[block] = [0] * len(const_nz) + new_rows
            self._cols[block] = list(const_nz) + new_cols
            self._data[block] = list(const_part[const_nz]) + new_data
        c = self.obj_facvar
        new_c = np.zeros(n_free + 1, dtype=self._dtype)
        new_c[0] = c[0] + c[1:].dot(shift[1:])
        new_c[1:] = basis.T.dot(c[1:])
        self.obj_facvar = new_c
        self._n_vars = n_free

    def freeze(
        self,
        level: int,
        constraint_to_blocks: dict[Any, tuple[int, ...]],
        moment_basis: tuple[Any, ...],
        substitutions: dict[Any, Any] | None = None,
        moment_substitutions: dict[Any, Any] | None = None,
    ) -> SdpProblem:
        """Freeze the draft into an immutable SdpProblem.

        Args:
            level: The relaxation level.
            constraint_to_blocks: Constraint-to-block lookup table.
            moment_basis: The first monomial set (for extraction methods).
            substitutions: The substitution rules the matrix was built
                under; the default keeps the draft free of them.
            moment_substitutions: The moment substitution rules, likewise.

        Returns:
            The frozen problem. The builder must not be used afterwards.
        """
        blocks = tuple(
            SparseBlock(
                coo_array(
                    (
                        np.asarray(self._data[b], dtype=self._dtype),
                        (
                            np.asarray(self._rows[b], dtype=np.int64),
                            np.asarray(self._cols[b], dtype=np.int64),
                        ),
                    ),
                    shape=(self._n_vars + 1, self.block_sizes[b] ** 2),
                ),
                self.block_sizes[b],
            )
            for b in range(len(self.block_sizes))
        )
        return SdpProblem(
            n_vars=self._n_vars,
            blocks=blocks,
            obj=self.obj_facvar[1:].copy(),
            # The constant of a complex-valued objective is still real by
            # Hermiticity; drop any imaginary part to avoid ComplexWarning.
            constant_term=float(np.real(self.obj_facvar[0])),
            monomial_index=dict(self.monomial_index),
            column_locations=dict(self.column_locations),
            constraint_to_blocks=dict(constraint_to_blocks),
            level=level,
            normalized=self.normalized,
            moment_basis=moment_basis,
            substitutions=dict(substitutions or {}),
            moment_substitutions=dict(moment_substitutions or {}),
        )
