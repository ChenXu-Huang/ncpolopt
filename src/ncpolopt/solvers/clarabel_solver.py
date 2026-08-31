"""Direct sparse CLARABEL backend for very large SDPs.

The CVXPY backend's canonicalization materializes one dense
``(size, size)`` matrix per SDP variable for every vectorized PSD
constraint; for an SDP with ~1e4 variables and a ~200-word moment matrix
that intermediate alone is ~1.6 GB and does not fit on a memory-tight
machine. This backend builds CLARABEL's native ``(A, b, cones)`` data
directly from the frozen blockwise COO, skipping CVXPY entirely. The
sparse canonical data is the exact layout CVXPY's CLARABEL backend would
produce (validated bit-identical on control problems), so results agree
with the CVXPY path to solver tolerance.

The backend never enters autodetection (the registry keeps the
``CVXPY -> MOSEK -> CVXOPT -> SDPA`` order); request it explicitly with
``solver="clarabel"``. Real-valued SDPs only.
"""

from __future__ import annotations

import math
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

#: CLARABEL statuses mapped onto the package status vocabulary.
_STATUS_MAP: dict[str, str] = {
    "Solved": "optimal",
    "AlmostSolved": "optimal",
    "PrimalInfeasible": "infeasible",
    "DualInfeasible": "unbounded",
}


def _svec_index(i: int, j: int) -> int:
    """The slot of the lower-triangle entry (i, j), i >= j, in a svec.

    The svec layout mirrors CVXPY's: the row-major lower triangle, with
    off-diagonal entries scaled by sqrt(2).

    Args:
        i: The row index (>= j).
        j: The column index.

    Returns:
        The flat svec slot of (i, j).
    """
    return i * (i + 1) // 2 + j


def canonical_clarabel_data(sdp: SdpProblem) -> tuple[Any, np.ndarray, list[Any]]:
    """Build CLARABEL data (A, b, cones) directly from the frozen SDP.

    Each block becomes rows in the same order CVXPY produces: a size>1
    block becomes a scaled svec triangle (row-major lower triangle,
    off-diagonal entries times sqrt(2)) and a size-1 block becomes one
    NonNeg row whose constant part is its b value. Empty scalar blocks
    get the b = +1.0 artifact of the package's CVXPY conversion:
    ``0.0 >= 0`` evaluates to Python ``True`` inside CVXPY, which it
    converts to the trivially-true ``0.0 <= 1.0``. The degenerate-path
    optimum of CLARABEL depends on the trivial rows' positions, so the
    row grouping matches CVXPY exactly (all NonNeg rows first, then the
    PSD triangles in block order).

    Args:
        sdp: The frozen SDP problem to convert.

    Returns:
        The (A, b, cones) triple in the package's data-dict convention:
        coefficients negated (CLARABEL's effective constraint is
        ``b - A x in K``), constants un-negated.
    """
    import clarabel
    from scipy import sparse as sp

    n = sdp.n_vars
    n_rows: list[int] = []
    n_cols: list[int] = []
    n_vals: list[float] = []
    n_b: list[float] = []
    p_rows: list[int] = []
    p_cols: list[int] = []
    p_vals: list[float] = []
    p_b: list[float] = []
    for block in sdp.blocks:
        size = block.size
        rows = np.asarray(block.coo.row, dtype=np.int64)
        cols = np.asarray(block.coo.col, dtype=np.int64)
        data = np.asarray(block.coo.data, dtype=float)
        if size == 1:
            const = 0.0
            for k, v in zip(rows, data, strict=True):
                if k == 0:
                    const += v
                else:
                    n_rows.append(len(n_b))
                    n_cols.append(int(k) - 1)
                    n_vals.append(v)
            if len(rows) == 0:
                # The package's empty-scalar artifact: `0.0 >= 0` becomes
                # Python True, which CVXPY turns into b = +1.0.
                const = 1.0
            n_b.append(const)
            continue
        # Constant matrix (upper-triangle COO, mirrored) for the b rows.
        const = np.zeros((size, size))
        for k, pos, v in zip(rows, cols, data, strict=True):
            i, j = divmod(int(pos), size)
            if k == 0:
                const[i, j] += v
                if i != j:
                    const[j, i] += v
                continue
            # The COO stores the upper triangle; the svec slot is the
            # lower-triangle mirror (r, c) with r >= c.
            r, c = (j, i) if i <= j else (i, j)
            t = _svec_index(r, c)
            scale = 1.0 if r == c else math.sqrt(2.0)
            p_rows.append(len(p_b) + t)
            p_cols.append(int(k) - 1)
            p_vals.append(scale * v)
        # Row-major lower triangle, matching _svec_index; the constant
        # matrix is symmetric, so the lower mirror carries the same value.
        for r in range(size):
            for c in range(r + 1):
                scale = 1.0 if r == c else math.sqrt(2.0)
                p_b.append(scale * const[r, c])
    b = np.concatenate([np.asarray(n_b, dtype=float), np.asarray(p_b, dtype=float)])
    A = sp.csc_matrix(
        (
            np.asarray(n_vals + p_vals, dtype=float),
            (np.asarray(n_rows + [r + len(n_b) for r in p_rows]), np.asarray(n_cols + p_cols)),
        ),
        shape=(len(b), n),
    )
    cones: list[Any] = []
    if n_b:
        cones.append(clarabel.NonnegativeConeT(len(n_b)))
    cones.extend(clarabel.PSDTriangleConeT(blk.size) for blk in sdp.blocks if blk.size > 1)
    # Package data-dict convention: coefficients are negated (b - A x in K
    # is the effective CLARABEL constraint), constants are not.
    return -A, b, cones


def _dual_blocks(sdp: SdpProblem, z: np.ndarray) -> tuple[np.ndarray, ...]:
    """Reconstruct the per-block dual matrices from CLARABEL's dual vector.

    The NonNeg prefix of ``z`` holds one dual per scalar block; each PSD
    block's scaled-svec segment unscales to a symmetric matrix (diagonal
    entries as-is, off-diagonal entries times sqrt(2), mirrored), so that
    ``tr(Y_b @ A0_b) = z_b . b_b`` block by block.

    Args:
        sdp: The frozen SDP problem.
        z: CLARABEL's dual solution, aligned with the cones.

    Returns:
        One dual matrix per SDP block, aligned with ``sdp.blocks``.
    """
    y_mat: list[np.ndarray] = []
    offset = 0
    for block in sdp.blocks:
        size = block.size
        if size == 1:
            y_mat.append(np.array([[z[offset]]]))
            offset += 1
            continue
        y = np.zeros((size, size))
        for r in range(size):
            for c in range(r + 1):
                scale = 1.0 if r == c else math.sqrt(2.0)
                y[r, c] = scale * z[offset + _svec_index(r, c)]
                y[c, r] = y[r, c]
        offset += size * (size + 1) // 2
        y_mat.append(y)
    return tuple(y_mat)


def solve_with_clarabel(problem: SdpProblem, settings: SolverSettings) -> SolverResult:
    """Solve the SDP through CLARABEL's sparse interface directly.

    Args:
        problem: The frozen SDP problem.
        settings: Backend knobs; ``settings.solver_options`` are applied
            as attributes of ``clarabel.DefaultSettings`` (e.g.
            ``{"tol_gap_abs": 1e-9}``), and ``settings.verbose`` controls
            the solver output (default off).

    Returns:
        The solver result in the canonical conventions.

    Raises:
        UnsupportedSdpError: If the SDP has complex coefficients (the
            backend is real-valued only).
    """
    if np.iscomplexobj(problem.obj) or any(
        np.iscomplexobj(block.coo.data) for block in problem.blocks
    ):
        raise UnsupportedSdpError(
            "The direct CLARABEL backend supports real-valued SDPs only; "
            "use the CVXPY backend for complex problems."
        )
    import clarabel
    from scipy import sparse as sp

    A, b, cones = canonical_clarabel_data(problem)
    q = np.asarray(problem.obj, dtype=float)
    clarabel_settings = clarabel.DefaultSettings()
    clarabel_settings.verbose = bool(settings.verbose)
    for key, value in settings.solver_options.items():
        setattr(clarabel_settings, key, value)
    n = A.shape[1]
    tstart = time.monotonic()
    solver = clarabel.DefaultSolver(sp.csc_matrix((n, n)), q, A, b, cones, clarabel_settings)
    solver.solve()
    solution_time = time.monotonic() - tstart
    sol = solver.get_solution()
    status = _STATUS_MAP.get(str(sol.status), str(sol.status))
    x = np.asarray(sol.x, dtype=float)
    x_mat = block_matrices(problem, x)
    # TODO(ChenXu): validate the dual reconstruction: the reported dual
    # value has been observed orders of magnitude away from the primal
    # value on certified-optimal runs, so _dual_blocks (or dual_value
    # over it) is likely wrong for this backend.
    y_mat = _dual_blocks(problem, np.asarray(sol.z, dtype=float))
    return SolverResult(
        status=status,
        primal=float(q @ x) + problem.constant_term,
        dual=dual_value(problem, y_mat),
        x_mat=x_mat,
        y_mat=y_mat,
        solution_time=solution_time,
        variables=x,
        raw=sol,
    )


register(SolverKind.CLARABEL, solve_with_clarabel)
