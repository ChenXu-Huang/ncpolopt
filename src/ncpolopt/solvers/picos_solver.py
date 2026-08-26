"""The PICOS backend (CVXOPT under the hood).

Ported from the legacy ``convert_to_picos``/``solve_with_cvxopt``
functions. The old converter rebuilt each
constraint from picos 1.x internal ``factors`` dictionaries and indexed the
moment vector by SDP variable number, which silently misplaces entries
whenever the variable order differs from the moment-matrix position order.
The new converter builds each block expression from the explicit per-variable
(block, i, j) position table recorded at construction time, and needs no
equality branch: the construction layer encodes equalities as two half-blocks
with opposite signs (the old converter's ``== 0`` branch, which double-advanced
its row offset, is gone with it).

The CVXOPT backend reports every solved point as ``"primal feasible"`` rather
than ``"optimal"``; both map to ``"optimal"``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..sdp_problem import SdpProblem
from ._common import block_matrices, dual_value
from .base import (
    SolverError,
    SolverKind,
    SolverResult,
    SolverSettings,
    UnsupportedSdpError,
)
from .registry import register

_COMPLEX_DTYPE = np.dtype(np.complex128)

#: picos statuses mapped onto the package status vocabulary.
_STATUS_MAP: dict[str, str] = {
    "optimal": "optimal",
    "primal feasible": "optimal",
    "dual feasible": "optimal",
    "infeasible": "infeasible",
    "primal infeasible": "infeasible",
    "dual infeasible": "unbounded",
    "unbounded": "unbounded",
    "unknown": "unknown",
}


@dataclass(frozen=True, slots=True)
class PicosModel:
    """A PICOS problem plus its moment-matrix variables.

    Attributes:
        problem: The PICOS problem.
        moment: The symmetric (or hermitian) variable carrying the moment
            matrix.
        duplicate: An unconstrained copy of the moment variable, when
            ``duplicate_moment_matrix`` was requested; extra constraints
            (e.g. the partial transpose) can be imposed on it without
            touching the moment matrix itself.
    """

    problem: Any
    moment: Any
    duplicate: Any | None = None


def _moment_expression(
    variable: Any, constant: np.ndarray, terms: dict[int, tuple[int, int, np.ndarray]]
) -> Any:
    """Build the affine expression ``constant + sum_k X[i0_k, j0_k] A_k``.

    Args:
        variable: The picos matrix variable whose entries carry the SDP
            variables.
        constant: The constant matrix of the block.
        terms: One (row, column, coefficient matrix) triple per SDP
            variable appearing in the block.

    Returns:
        The picos matrix expression.
    """
    import picos as pic

    expression: Any = pic.Constant(constant)
    for row, column, matrix in terms.values():
        expression = expression + variable[row, column] * pic.Constant(matrix)
    return expression


def convert_to_picos(
    problem: SdpProblem, duplicate_moment_matrix: bool = False
) -> PicosModel:
    """Convert a frozen SDP problem to a PICOS problem.

    One semidefinite variable holds the moment matrix; every block becomes
    one constraint whose entries reference that variable. This mirrors the
    old converter but is exact: each SDP variable's moment position is taken
    from the construction-time ``column_locations`` table instead of being
    guessed from the variable number.

    Args:
        problem: The frozen SDP problem.
        duplicate_moment_matrix: Add an unconstrained copy of the moment
            matrix variable for post-hoc constraints (used by the Moroder
            hierarchy).

    Returns:
        The PICOS model.

    Raises:
        UnsupportedSdpError: If a variable was created outside the first
            moment matrix block, or the problem has no blocks.
    """
    import picos as pic

    if not problem.blocks:
        raise UnsupportedSdpError(
            "The PICOS backend needs at least one block."
        )
    for k in range(1, problem.n_vars + 1):
        if problem.column_locations[k][0] != 0:
            raise UnsupportedSdpError(
                "The PICOS backend maps every variable onto the first moment "
                "matrix; variables created in other blocks (parameter blocks, "
                "rectangular hierarchies) are not supported."
            )
    complex_matrix = any(
        block.coo.dtype == _COMPLEX_DTYPE for block in problem.blocks
    )
    size = problem.blocks[0].size
    dtype = np.complex128 if complex_matrix else np.float64
    if complex_matrix:
        moment = pic.HermitianVariable("X", size)
        duplicate = (
            pic.HermitianVariable("Y", size) if duplicate_moment_matrix else None
        )
    else:
        moment = pic.SymmetricVariable("X", size)
        duplicate = (
            pic.SymmetricVariable("Y", size) if duplicate_moment_matrix else None
        )
    P = pic.Problem()
    for block in problem.blocks:
        block_size = block.size
        constant = np.zeros((block_size, block_size), dtype=dtype)
        terms: dict[int, tuple[int, int, np.ndarray]] = {}
        for k, position, value in zip(
            block.coo.row, block.coo.col, block.coo.data, strict=True
        ):
            i, j = divmod(int(position), block_size)
            if k == 0:
                constant[i, j] += value
                if i != j:
                    constant[j, i] += value
                continue
            row, column = problem.column_locations[int(k)][1:]
            _, _, matrix = terms.setdefault(
                int(k), (row, column, np.zeros((block_size, block_size), dtype=dtype))
            )
            matrix[i, j] += value
            if i != j:
                matrix[j, i] += value
        expression = _moment_expression(moment, constant, terms)
        if block_size > 1:
            P.add_constraint(expression >> 0)
        else:
            # A 1x1 block is a scalar constraint.
            P.add_constraint(expression >= 0)
    nonzero = [(k, c) for k, c in enumerate(problem.obj, start=1) if c != 0]
    if nonzero:
        objective = sum(
            float(np.real(c)) * moment[problem.column_locations[k][1], problem.column_locations[k][2]]
            for k, c in nonzero
        )
        P.set_objective("min", objective)
    return PicosModel(problem=P, moment=moment, duplicate=duplicate)


def solve_with_picos(problem: SdpProblem, settings: SolverSettings) -> SolverResult:
    """Convert the problem to PICOS, solve it, and parse the output.

    Args:
        problem: The frozen SDP problem.
        settings: Backend knobs; ``settings.solver_options`` may contain
            ``"solver"`` (defaults to the CVXOPT solver) and any other PICOS
            option, e.g. ``{"timelimit": 60}``.

    Returns:
        The solver result in the canonical conventions.

    Raises:
        UnsupportedSdpError: If the problem cannot be mapped (see
            :func:`convert_to_picos`).
    """
    model = convert_to_picos(problem)
    P = model.problem
    options = dict(settings.solver_options)
    P.options.solver = str(options.pop("solver", "cvxopt"))
    P.options.verbosity = 1 if settings.verbose else 0
    for name, value in options.items():
        try:
            setattr(P.options, name, value)
        except AttributeError as error:
            raise SolverError(
                f"Unknown PICOS option {name!r}."
            ) from error
    tstart = time.monotonic()
    solution = P.solve()
    solution_time = time.monotonic() - tstart
    status = _STATUS_MAP.get(solution.status, solution.status)
    locations = [problem.column_locations[k] for k in range(1, problem.n_vars + 1)]
    x = np.array([model.moment.value[i0, j0] for _, i0, j0 in locations])
    x_mat = block_matrices(problem, x)
    y_mat = tuple(
        np.asarray(P.get_constraint(i).dual)
        for i in range(len(P.constraints))
    )
    return SolverResult(
        status=status,
        primal=float(np.real(np.dot(problem.obj, x))) + problem.constant_term,
        dual=dual_value(problem, y_mat),
        x_mat=x_mat,
        y_mat=y_mat,
        solution_time=solution_time,
        variables=x,
        raw=P,
    )


register(SolverKind.CVXOPT, solve_with_picos)
