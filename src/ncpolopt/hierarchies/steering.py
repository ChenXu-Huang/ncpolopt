"""The steering hierarchy.

Ported from the legacy ``SteeringHierarchy`` class. Each moment of the
basis expands into a ``matrix_var_dim x matrix_var_dim``
block of SDP variables: entry (row, col) of the moment matrix becomes the
sub-block at ``(row*d + r, col*d + c)``, so the moment matrix has size
``len(monomials) * matrix_var_dim``. The sub-block groups number their
variables in the full d x d grid order (variables ``k + r*d + c``), which
keeps the objective extraction formula of the old package verbatim.

The steering hierarchy is complex-valued and unnormalized: the problem must
be created with ``Problem(..., normalized=False, complex_matrix=True)``.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np

from ..block_structure import BlockKind, BlockStructure
from ..expressions import is_number_type, iscomplex
from ..relaxation import NpaRelaxation
from ..substitutions import apply_substitutions


class SteeringHierarchy(NpaRelaxation):
    """Level-by-level steering hierarchy with matrix-variable blocks.

    Every moment monomial carries its own ``d x d`` sub-block of SDP
    variables; off-diagonal sub-block positions get the complex
    coefficient ``1 + i`` so that the complex embedding of the backends
    yields a hermitian moment matrix. The top-left normalization moment
    is expanded like any other (the hierarchy is unnormalized).
    """

    def __init__(
        self,
        problem: Any,
        level: int,
        *,
        matrix_var_dim: int,
        removeequalities: bool = False,
    ) -> None:
        """Build the steering relaxation of ``problem`` at ``level``.

        Args:
            problem: The problem to relax; must be created with
                ``normalized=False`` and ``complex_matrix=True``.
            level: The relaxation level.
            matrix_var_dim: The dimension of the matrix-variable blocks.
            removeequalities: Eliminate equality constraints algebraically.

        Raises:
            ValueError: If the problem is not complex and unnormalized, or
                ``matrix_var_dim`` is not a positive integer.
        """
        if problem.normalized:
            raise ValueError(
                "The steering hierarchy requires an unnormalized problem "
                "(Problem(normalized=False))."
            )
        if problem.complex_matrix is not True:
            raise ValueError(
                "The steering hierarchy requires a complex-valued problem "
                "(Problem(complex_matrix=True))."
            )
        if matrix_var_dim < 1:
            raise ValueError(
                f"matrix_var_dim must be a positive integer, got "
                f"{matrix_var_dim}."
            )
        self.matrix_var_dim = matrix_var_dim
        #: The group-start variables whose sub-blocks already exist; a
        #: second occurrence of the same moment reuses the whole group.
        self._steering_groups: set[int] = set()
        super().__init__(problem, level, removeequalities=removeequalities)

    def _compute_block_structure(self, **kwargs: Any) -> BlockStructure:
        """Scale the moment block sizes by the matrix-variable dimension.

        Args:
            kwargs: The keyword arguments of
                :func:`~ncpolopt.block_structure.compute_block_structure`.

        Returns:
            The block structure with ``matrix_var_dim``-scaled moment
            blocks; the constraint blocks keep their sizes.
        """
        structure = super()._compute_block_structure(**kwargs)
        d = self.matrix_var_dim
        return dataclasses.replace(
            structure,
            blocks=tuple(
                dataclasses.replace(spec, size=spec.size * d)
                if spec.kind is BlockKind.MOMENT
                else spec
                for spec in structure.blocks
            ),
        )

    def _push_monomial(self, monomial: Any, block: int, i: int, j: int) -> None:
        """Push one moment monomial expanded into a matrix-variable sub-block.

        ``i`` and ``j`` are the basis coordinates of the entry; the moment
        spreads over the ``d x d`` sub-block at ``(i*d, j*d)`` with one SDP
        variable per sub-block position.
        """
        substitute = self._moment_substitutions.get(monomial)
        if substitute is not None:
            self._push_monomial(substitute, block, i, j)
            return
        d = self.matrix_var_dim
        if is_number_type(monomial):
            if i == 0 and j == 0 and not self._builder.normalized:
                k = self._builder.new_variable(None, block, 0, 0)
                self._expand_group(k, block, 0, 0, 1.0)
            else:
                self._builder.add_entry(
                    block, i * d, j * d, 0, monomial
                )
        elif monomial.is_Add:
            for element in monomial.args:
                self._push_monomial(element, block, i, j)
        elif monomial != 0:
            for k, coeff in self._process_monomial(
                monomial, block, i * d, j * d
            ):
                self._expand_group(k, block, i, j, coeff)

    def _expand_group(
        self,
        k: int,
        block: int,
        row_a: int,
        column_a: int,
        coeff: Any,
    ) -> None:
        """Create (or reuse) the variable group of one moment's sub-block.

        On first sight the group occupies the ``d**2`` consecutive
        variables ``k + r*d + c`` in grid order; the lower-triangle ones
        stay free (phantoms) and the hermitian mirror of the complex
        embedding fills their positions. Later occurrences of the same
        moment only write entries again, sharing the whole group.

        Args:
            k: The group's first variable (the moment's resolved variable).
            block: The block being built.
            row_a: The basis row of the moment.
            column_a: The basis column of the moment.
            coeff: The scalar factor of the moment.
        """
        d = self.matrix_var_dim
        fresh = k not in self._steering_groups
        for r in range(d):
            for c in range(d):
                position = r * d + c
                if position == 0:
                    variable = k
                elif fresh:
                    variable = self._builder.new_variable(
                        None, block, row_a * d + r, column_a * d + c
                    )
                else:
                    variable = k + position
                if row_a * d + r <= column_a * d + c:
                    # Off the sub-block diagonal the coefficient carries a
                    # unit imaginary part, which the complex embedding turns
                    # into the hermitian mirror below the diagonal.
                    imag_zero = 1 if (row_a != column_a or r != c) else 0
                    value = coeff * (1 + 1j * imag_zero)
                    self._builder.add_entry(
                        block, row_a * d + r, column_a * d + c, variable, value
                    )
        if fresh:
            self._steering_groups.add(k)

    def set_objective(self, objective: Any | None) -> None:
        """Set the objective of the relaxation.

        The steering objective is a ``matrix_var_dim x matrix_var_dim``
        matrix of polynomials whose expectation value enters as the trace
        against the moment matrix; a plain polynomial objective is rejected.

        Args:
            objective: The matrix objective, or None for a zero objective.

        Raises:
            ValueError: If the objective is not a matrix of the required
                shape.
        """
        if objective is not None:
            d = self.matrix_var_dim
            if getattr(objective, "shape", None) != (d, d):
                raise ValueError(
                    f"The steering hierarchy needs a {d}x{d} matrix "
                    f"objective, got {type(objective).__name__}."
                )
            self._builder.set_objective(self._trace_facvar(objective))
        else:
            super().set_objective(objective)

    def _trace_facvar(self, objective: Any) -> list[Any]:
        """The dense objective vector of a matrix-valued steering objective.

        The coefficient of the (r, c) sub-position variable of a moment
        reads off the transposed entry of the moment's coefficient matrix
        (the trace against the standard basis matrix). The constant matrix
        of the objective lands on the (0, 0)-sub-block group -- variable 1,
        the first variable of the unnormalized normalization moment --
        which is what pins the otherwise unnormalized steering matrix.

        Args:
            objective: The ``d x d`` matrix of polynomials.

        Returns:
            The objective vector, indexed by SDP variable.
        """
        d = self.matrix_var_dim
        facvar = [0] * (self._builder.n_vars + 1)
        coefficients: dict[Any, np.ndarray] = {}
        for i in range(d):
            for j in range(d):
                for key, value in objective[i, j].as_coefficients_dict().items():
                    substituted = apply_substitutions(
                        key, self._substitutions, self._pure_substitution_rules
                    )
                    matrix = coefficients.setdefault(
                        substituted, np.zeros((d, d))
                    )
                    matrix[i, j] += value
        for key, matrix in coefficients.items():
            # The hierarchy is unnormalized, so the (0, 0) moment is a free
            # variable like any other moment's group: its first variable is
            # 1, the first variable the builder created.
            k = 1 if is_number_type(key) else self._builder.monomial_index[key]
            for i in range(d):
                for j in range(d):
                    value = matrix[j, i]
                    facvar[k + i * d + j] = (
                        complex(value) if iscomplex(value) else float(value)
                    )
        return facvar
