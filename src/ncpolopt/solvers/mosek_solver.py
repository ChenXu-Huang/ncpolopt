"""The MOSEK backend.

Ported from the legacy ``solve_with_mosek`` function. The old backend was
never exercised numerically (MOSEK needs a license) and its
solution extraction used a hand-rolled splitter (``moseksol_to_xmat``) that
walked a single fused vector with index skips; the new one reads each bar
variable separately and rebuilds the dense matrices from MOSEK's packed
column-major lower triangle.

MOSEK's conic dual requires every scalar variable to occur in some
constraint, so the objective ``c.x`` is encoded in the equality bounds
instead: constraint ``k`` reads ``<-A_k, X> = -c_k`` (equivalent to
``<A_k, X> = c_k``). That makes the dual multipliers ``gety()`` equal the
primal solution ``x`` and the dual slack ``getbarsj = A0 + sum x_k A_k``
equal the primal block matrices, so the shared canonical value formulas
apply unchanged.

The old parameter handling ran ``eval("mosek." + name)`` (bug #7); the
getattr-based :func:`_set_parameter` keeps the same names without executing
code. Complex-valued SDPs raise :class:`UnsupportedSdpError` -- the old
converter reached into PICOS's private ``to_real()`` API for those, which
no longer exists.
"""

from __future__ import annotations

import sys
import time
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


def _streamprinter(text: str) -> None:
    """Forward MOSEK's log stream to the console."""
    sys.stdout.write(text)
    sys.stdout.flush()


def _set_parameter(task: Any, name: str, value: Any) -> None:
    """Set one MOSEK parameter on a task.

    Args:
        task: The MOSEK task.
        name: The parameter name, optionally prefixed (``mosek.``,
            ``iparam.``, ``dparam.``, ``sparam.``). Without a prefix the
            parameter type is inferred from the value's type.
        value: The parameter value.

    Raises:
        SolverError: If the name does not exist or the type cannot be
            inferred.
    """
    import mosek

    if name.startswith("mosek."):
        name = name[7:]
    namespace: Any
    putter_name: str
    if name.startswith("iparam."):
        name, namespace, putter_name = name[7:], mosek.iparam, "putintparam"
    elif name.startswith("dparam."):
        name, namespace, putter_name = name[7:], mosek.dparam, "putdouparam"
    elif name.startswith("sparam."):
        name, namespace, putter_name = name[7:], mosek.sparam, "putstrparam"
    elif isinstance(value, int):
        namespace, putter_name = mosek.iparam, "putintparam"
    elif isinstance(value, float):
        namespace, putter_name = mosek.dparam, "putdouparam"
    elif isinstance(value, str):
        namespace, putter_name = mosek.sparam, "putstrparam"
    else:
        raise SolverError(
            f"Parameter {name!r}: cannot infer the MOSEK parameter type "
            f"from a value of type {type(value).__name__}."
        )
    try:
        parameter = getattr(namespace, name)
    except AttributeError as error:
        raise SolverError(
            f"No MOSEK parameter found named {name!r}; see the MOSEK "
            f"Python API manual for valid parameter names."
        ) from error
    getattr(task, putter_name)(parameter, value)


def _lower_triangle_to_dense(values: list[float], size: int) -> np.ndarray:
    """Rebuild a dense symmetric matrix from MOSEK's packed bar form.

    MOSEK returns each bar-variable solution as the lower triangle in
    column-major order: for every column ``j`` the entries ``(i, j)`` with
    ``i >= j``, hence ``size * (size + 1) / 2`` values.

    Args:
        values: The packed entries.
        size: The matrix dimension.

    Returns:
        The dense symmetric matrix.
    """
    matrix = np.zeros((size, size))
    position = 0
    for column in range(size):
        for row in range(column, size):
            value = values[position]
            matrix[row, column] = value
            matrix[column, row] = value
            position += 1
    return matrix


def convert_to_mosek(problem: SdpProblem) -> Any:
    """Convert a frozen SDP problem to a MOSEK task.

    Args:
        problem: The frozen SDP problem.

    Returns:
        The MOSEK task, not yet optimized.

    Raises:
        UnsupportedSdpError: If the problem has complex-valued blocks.
    """
    import mosek

    if any(block.coo.dtype == _COMPLEX_DTYPE for block in problem.blocks):
        raise UnsupportedSdpError(
            "The MOSEK backend does not support complex-valued SDPs; use "
            "the external SDPA backend instead."
        )
    env = mosek.Env()
    task = env.Task(0, 0)
    # No scalar variables: the objective c.x travels through the constraint
    # bounds, see the module docstring.
    task.appendvars(0)
    task.appendcons(problem.n_vars)
    task.appendbarvars([block.size for block in problem.blocks])
    task.putobjsense(mosek.objsense.minimize)
    for k, coefficient in enumerate(problem.obj, start=1):
        task.putconbound(
            k - 1,
            mosek.boundkey.fx,
            -float(coefficient),
            -float(coefficient),
        )
    for block_index, block in enumerate(problem.blocks):
        size = block.size
        # One sparse symmetric matrix per (block, variable); variables can
        # occur in several blocks, each contributes to the same constraint.
        entries: dict[int, list[tuple[int, int, float]]] = {}
        for k, position, value in zip(
            block.coo.row, block.coo.col, block.coo.data, strict=True
        ):
            i, j = divmod(int(position), size)
            # MOSEK stores the lower triangle: the (row, column) pair of the
            # upper-triangle entry (i, j) is its mirror (j, i).
            if k == 0:
                entries.setdefault(0, []).append((j, i, float(value)))
            else:
                entries.setdefault(int(k), []).append((j, i, -float(value)))
        for k, triples in entries.items():
            barai = [triple[0] for triple in triples]
            baraj = [triple[1] for triple in triples]
            coefficients = [triple[2] for triple in triples]
            symmat = task.appendsparsesymmat(
                size, barai, baraj, coefficients
            )
            if k == 0:
                # The constant matrix enters the objective as <A0, X>.
                task.putbarcj(block_index, [symmat], [1.0])
            else:
                task.putbaraij(k - 1, block_index, [symmat], [1.0])
    return task


def solve_with_mosek(problem: SdpProblem, settings: SolverSettings) -> SolverResult:
    """Convert the problem to MOSEK, solve it, and parse the output.

    Args:
        problem: The frozen SDP problem.
        settings: Backend knobs; ``settings.mosek_params`` holds MOSEK
            parameter name/value pairs (names may carry a ``iparam.`` /
            ``dparam.`` / ``sparam.`` prefix, see :func:`_set_parameter`).

    Returns:
        The solver result in the canonical conventions.

    Raises:
        UnsupportedSdpError: If the problem cannot be mapped (see
            :func:`convert_to_mosek`).
    """
    import mosek

    task = convert_to_mosek(problem)
    if settings.verbose:
        task.set_Stream(mosek.streamtype.log, _streamprinter)
    for name, value in settings.mosek_params.items():
        _set_parameter(task, name, value)
    tstart = time.monotonic()
    task.optimize()
    solution_time = time.monotonic() - tstart
    # NOTE: the enum names differ across MOSEK versions (MOSEK 11 dropped
    # the near_* statuses), so map only the names this version defines.
    status = {
        getattr(mosek.solsta, name): mapped
        for name, mapped in (
            ("optimal", "optimal"),
            ("near_optimal", "optimal"),
            ("prim_infeas_cer", "infeasible"),
            ("prim_and_dual_infeas_cer", "infeasible"),
            ("dual_infeas_cer", "unbounded"),
        )
        if hasattr(mosek.solsta, name)
    }.get(task.getsolsta(mosek.soltype.itr), "unknown")
    # MOSEK 10+ requires the solution selector on gety.
    x = np.asarray(task.gety(mosek.soltype.itr))
    x_mat = block_matrices(problem, x)
    # TODO(ChenXu): the y_mat slot holds MOSEK's primal bar matrices
    # (getbarxj), not the canonical dual matrices, so the dual value
    # computed over them is meaningless (observed far from the primal
    # value). Extract the true dual or stop reporting it on this backend.
    y_mat = tuple(
        _lower_triangle_to_dense(
            task.getbarxj(mosek.soltype.itr, block_index), block.size
        )
        for block_index, block in enumerate(problem.blocks)
    )
    return SolverResult(
        status=status,
        primal=float(np.dot(problem.obj, x)) + problem.constant_term,
        dual=dual_value(problem, y_mat),
        x_mat=x_mat,
        y_mat=y_mat,
        solution_time=solution_time,
        variables=x,
        raw=task,
    )


register(SolverKind.MOSEK, solve_with_mosek)
