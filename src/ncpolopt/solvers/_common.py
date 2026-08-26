"""Canonical value extraction shared by every solver backend.

The package conventions ``primal = c.x* + c0`` and
``dual = -sum_b tr(Y_b @ A0_b) + c0`` are fixed here, once, so that no
backend re-derives (or mis-derives) them. The old package let each backend
grow its own value formulas; three of the four disagreed on the sign of the
constant term, and the CVXPY backend reported the primal value twice.
"""

from __future__ import annotations

import numpy as np

from ..sdp_problem import SdpProblem


def block_matrices(problem: SdpProblem, x: np.ndarray) -> tuple[np.ndarray, ...]:
    """The primal block matrices at the solution ``x``, symmetrized.

    Construction stores only the upper triangle; the full symmetric (real)
    matrix is the mirror image.

    Args:
        problem: The frozen SDP problem.
        x: The primal variable values, length ``n_vars``.

    Returns:
        One dense matrix per block, aligned with ``problem.blocks``.
    """
    x_mat = []
    for block in problem.blocks:
        mat = block.evaluate(x)
        mat = mat + mat.T - np.diag(np.diag(mat))
        x_mat.append(mat)
    return tuple(x_mat)


def dual_value(problem: SdpProblem, y_mat: tuple[np.ndarray, ...]) -> float:
    """The dual objective ``-sum_b tr(Y_b @ A0_b) + c0``.

    Args:
        problem: The frozen SDP problem.
        y_mat: The per-block dual matrices from the backend.

    Returns:
        The dual optimal value.
    """
    value = problem.constant_term
    for block, y in zip(problem.blocks, y_mat, strict=True):
        if y is None:
            continue
        # Scalar (1x1) constraints report their dual as a bare float.
        y = np.atleast_2d(np.asarray(y, dtype=float))
        if y.size == 0:
            continue
        value -= float(np.trace(y @ block.constant_matrix()))
    return value
