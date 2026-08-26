"""The CVXPY backend.

Ported from ``solve_with_cvxpy`` (src.old/ncpol2sdpa/cvxpy_utils.py). The
old converter rebuilt a giant ``lil_matrix`` per block and re-derived the
row offsets; the new one iterates the per-block COO data directly. Two
behavioral fixes: the old code popped ``"solver"`` out of the caller's
parameter dict (mutating it), and it reported the primal value as the dual.
The canonical value formulas of the package are applied here instead.

Complex-valued SDPs raise :class:`UnsupportedSdpError`; the SDPA and MOSEK
backends handle those.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from ..sdp_problem import SdpProblem
from ._common import block_matrices, dual_value
from .base import (
    SolverKind,
    SolverResult,
    SolverSettings,
    UnsupportedSdpError,
)
from .registry import register

#: cvxpy statuses mapped onto the package status vocabulary.
_STATUS_MAP: dict[str, str] = {
    "optimal": "optimal",
    "optimal_inaccurate": "optimal",
    "infeasible": "infeasible",
    "infeasible_inaccurate": "infeasible",
    "unbounded": "unbounded",
    "unbounded_inaccurate": "unbounded",
}

_COMPLEX_DTYPE = np.dtype(np.complex128)

#: Free solvers, in preference order, tried when the caller does not name
#: a solver. CVXPY's own default picks MOSEK whenever it is installed, which
#: fails on machines without a MOSEK license.
_DEFAULT_SOLVERS = ("CLARABEL", "SCS")


def convert_to_cvxpy(problem: SdpProblem) -> Any:
    """Convert a frozen SDP problem to a CVXPY problem.

    Each block becomes one constraint: a PSD constraint ``sum_ >> 0`` for
    blocks of size > 1, a scalar ``sum_ >= 0`` otherwise. The equality
    blocks need no special handling -- the construction layer encodes an
    equality as two halves with opposite signs, so both halves being
    nonnegative forces zero.

    Args:
        problem: The frozen SDP problem.

    Returns:
        The CVXPY problem; the objective excludes the constant term.

    Raises:
        UnsupportedSdpError: If the problem has complex-valued blocks.
    """
    from cvxpy import Minimize, Problem, Variable

    if any(block.coo.dtype == _COMPLEX_DTYPE for block in problem.blocks):
        raise UnsupportedSdpError(
            "The CVXPY backend does not support complex-valued SDPs; use "
            "MOSEK or the external SDPA backend instead."
        )
    x = Variable(problem.n_vars)
    constraints = []
    for block in problem.blocks:
        size = block.size
        mats: list[np.ndarray] | list[float] = (
            [np.zeros((size, size)) for _ in range(problem.n_vars + 1)]
            if size > 1
            else [0.0] * (problem.n_vars + 1)
        )
        for k, position, value in zip(
            block.coo.row, block.coo.col, block.coo.data, strict=True
        ):
            if size > 1:
                row, column = divmod(int(position), size)
                # Construction stores the upper triangle only; mirror the
                # entry so the constraint matrix is symmetric. CVXPY would
                # otherwise symmetrize ``A >> 0`` as (A + A.T) / 2, halving
                # every off-diagonal coefficient.
                mats[int(k)][row, column] += value
                if row != column:
                    mats[int(k)][column, row] += value
            else:
                mats[int(k)] += value
        if size > 1:
            expression = mats[0]
            for k in range(1, problem.n_vars + 1):
                if np.count_nonzero(mats[k]) > 0:
                    expression = expression + mats[k] * x[k - 1]
            constraints.append(expression >> 0)
        else:
            expression = mats[0]
            for k in range(1, problem.n_vars + 1):
                if mats[k] != 0:
                    expression = expression + mats[k] * x[k - 1]
            constraints.append(expression >= 0)
    objective = sum(
        coefficient * x[k] for k, coefficient in enumerate(problem.obj) if coefficient != 0
    )
    return Problem(Minimize(objective), constraints)


def solve_with_cvxpy(
    problem: SdpProblem, settings: SolverSettings
) -> SolverResult:
    """Convert the problem to CVXPY, solve it, and parse the output.

    Args:
        problem: The frozen SDP problem.
        settings: Backend knobs; ``settings.solver_options`` may contain
            any CVXPY solve keyword, e.g. ``{"solver": "SCS"}`` or SCS
            tolerances.

    Returns:
        The solver result in the canonical conventions.
    """
    cvxpy_problem = convert_to_cvxpy(problem)
    # The frozen settings must not be mutated (the old code popped "solver"
    # out of the caller's dict).
    options = dict(settings.solver_options)
    solver = options.pop("solver", None)
    if solver is None:
        from cvxpy import installed_solvers

        solver = next(
            (name for name in _DEFAULT_SOLVERS if name in installed_solvers()),
            None,
        )
    if settings.verbose is not None:
        options["verbose"] = settings.verbose
    tstart = time.monotonic()
    if solver is not None:
        cvxpy_problem.solve(solver=solver, **options)
    else:
        cvxpy_problem.solve(**options)
    solution_time = time.monotonic() - tstart
    status = _STATUS_MAP.get(cvxpy_problem.status, cvxpy_problem.status)
    x = cvxpy_problem.variables()[0].value
    if x is None:
        return SolverResult(
            status=status,
            primal=float("nan"),
            dual=float("nan"),
            solution_time=solution_time,
            raw=cvxpy_problem,
        )
    x = np.asarray(x)
    x_mat = block_matrices(problem, x)
    y_mat = tuple(
        constraint.dual_value for constraint in cvxpy_problem.constraints
    )
    return SolverResult(
        status=status,
        primal=float(np.real(np.dot(problem.obj, x))) + problem.constant_term,
        dual=dual_value(problem, y_mat),
        x_mat=x_mat,
        y_mat=y_mat,
        solution_time=solution_time,
        variables=x,
        raw=cvxpy_problem,
    )


register(SolverKind.CVXPY, solve_with_cvxpy)
register(SolverKind.SCS, solve_with_cvxpy)
