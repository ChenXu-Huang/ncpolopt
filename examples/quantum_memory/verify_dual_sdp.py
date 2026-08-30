"""Part A: verify the correlation table and the dual SDPs of the document.

Checks, for the depolarizing-channel example of
``verification-of-memory.md``:

1. The Choi construction ``J_M = (I x M)|phi+><phi+|`` (eq. (3)) equals
   the Werner-state closed form, its fidelity to |phi+> is ``p``, and it
   is positive semidefinite exactly for ``p >= 1/4``.
2. The correlations from eq. (1) (channel picture) and eq. (10) (Choi
   picture) agree entrywise with the closed-form table of eq. (16), and
   sum to one over the outputs of a four-output Bell measurement.
3. The dual pair (eq. (6), max over the Choi state ``J`` of the
   unital-CP map) and (eq. (9), min over ``Z``) both attain the fidelity
   ``p``; the redundant ``Z >= 0`` constraint of eq. (9) is confirmed to
   change nothing.
4. The moment-insertion identity ``<U_{x,y} J V_alpha>_tau =
   p(alpha|x,y)/(8 d)`` used by the NPA-tau reconstruction (word order
   from eq. (10)).

Usage: ``uv run python examples/quantum_memory/verify_dual_sdp.py
[--p-grid 0.25,0.5,1.0] [--p 0.75]``
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (
    DIM,
    bell_projectors,
    choi_fidelity,
    choi_state,
    correlations_direct,
    correlations_via_choi,
    expected_correlation,
    input_projectors,
    werner_state,
)

logger = logging.getLogger(__name__)

TOL = 1e-9


def _check(condition: bool, *context: object) -> None:
    """Fail the verification with context when ``condition`` is false.

    A plain raise instead of ``assert``, so the checks are not stripped
    under ``python -O``.

    Args:
        condition: The verification condition.
        *context: Values reported on failure.

    Raises:
        RuntimeError: If ``condition`` is false.
    """
    if not condition:
        raise RuntimeError(f"verification failed: {context}")


def dual_max_problem_value(p: float) -> float:
    """Solve eq. (6): ``max Tr(J_M J)`` s.t. ``Tr_{A0}(J) = I/2, J >= 0``.

    Args:
        p: The channel parameter.

    Returns:
        The optimal value.

    Raises:
        RuntimeError: If the solver status is not "optimal".
    """
    import cvxpy as cp

    j_m = choi_state(p)
    j = cp.Variable((4, 4), hermitian=True)
    partial = j[0:2, 0:2] + j[2:4, 2:4]
    constraints = [partial == np.eye(2) / DIM, j >> 0]
    problem = cp.Problem(cp.Maximize(cp.real(cp.trace(j_m @ j))), constraints)
    problem.solve(solver=cp.CLARABEL)
    if problem.status != "optimal":
        raise RuntimeError(f"eq. (6) failed with status {problem.status!r}.")
    return float(problem.value)


def dual_min_problem_value(p: float, with_z_psd: bool) -> float:
    """Solve eq. (9): ``min Tr(Z)/d`` s.t. ``I^{A0} x Z - J_M >= 0``.

    Args:
        p: The channel parameter.
        with_z_psd: Also impose the (redundant) ``Z >= 0`` constraint.

    Returns:
        The optimal value.

    Raises:
        RuntimeError: If the solver status is not "optimal".
    """
    import cvxpy as cp

    j_m = choi_state(p)
    z = cp.Variable((2, 2), hermitian=True)
    constraints: list[object] = [cp.kron(np.eye(DIM), z) - j_m >> 0]
    if with_z_psd:
        constraints.append(z >> 0)
    problem = cp.Problem(cp.Minimize(cp.real(cp.trace(z)) / DIM), constraints)
    problem.solve(solver=cp.CLARABEL)
    if problem.status != "optimal":
        raise RuntimeError(f"eq. (9) failed with status {problem.status!r}.")
    return float(problem.value)


def check_correlation_tables(p: float) -> None:
    """Compare eq. (1), eq. (10) and eq. (16) for all inputs."""
    worst = 0.0
    for x in range(4):
        for y in range(4):
            expected = expected_correlation(p, x, y)
            for alpha in range(1):
                direct = correlations_direct(p, x, y, alpha)
                via_choi = correlations_via_choi(p, x, y, alpha)
                worst = max(worst, abs(direct - expected), abs(via_choi - expected))
                _check(abs(direct - expected) <= TOL, p, x, y, direct, expected)
                _check(abs(via_choi - expected) <= TOL, p, x, y, via_choi, expected)
    # Four-output measurement: probabilities sum to one over alpha.
    for x in range(4):
        for y in range(4):
            total = sum(correlations_via_choi(p, x, y, alpha) for alpha in range(4))
            _check(abs(total - 1.0) <= TOL, p, x, y, total)
    logger.info("correlation tables: eq. (1) == eq. (10) == eq. (16), worst diff %.2e", worst)


def check_choi_state(p: float) -> None:
    """Verify eq. (3) = Werner closed form, fidelity p, PSD for p >= 1/4."""
    j_m = choi_state(p)
    _check(np.allclose(j_m, werner_state(p), atol=TOL), "Choi state != Werner state")
    fidelity = choi_fidelity(p)
    _check(abs(fidelity - p) <= TOL, f"Choi fidelity {fidelity} != {p}")
    min_eig = np.linalg.eigvalsh(j_m)[0]
    _check((min_eig >= -TOL) if p >= 0.25 else (min_eig < TOL), p, min_eig)
    if p < 0.25:
        logger.info("p=%.2f < 1/4: J_M indefinite (min eig %.2e), as expected", p, min_eig)


def check_moment_identity(p: float, alpha: int) -> None:
    """Verify the NPA-tau pin identity <U J V>_tau = p(alpha|x,y)/(8d)."""
    j_8 = np.kron(choi_state(p), np.eye(DIM))
    for x in range(4):
        for y in range(4):
            u = np.kron(np.kron(input_projectors()[x].T, np.eye(DIM)), input_projectors()[y])
            v = np.kron(np.eye(DIM), bell_projectors()[alpha])
            moment = np.trace(u @ j_8 @ v) / 8.0
            expected = correlations_via_choi(p, x, y, alpha) / (8.0 * DIM)
            _check(abs(moment - expected) <= TOL, x, y, moment, expected)
    logger.info("moment identity <U_{x,y} J V_alpha>_tau = p(alpha|x,y)/(8d) holds")


def run(p: float) -> None:
    """Run all Part A checks for one channel parameter."""
    logger.info("--- p = %.4f ---", p)
    check_choi_state(p)
    check_correlation_tables(p)
    check_moment_identity(p, alpha=0)
    dual_max = dual_max_problem_value(p)
    dual_min = dual_min_problem_value(p, with_z_psd=True)
    dual_min_no_psd = dual_min_problem_value(p, with_z_psd=False)
    logger.info(
        "eq. (6) max Tr(J_M J) = %.8f | eq. (9) min Tr(Z)/d = %.8f (no Z>=0: %.8f) | "
        "fidelity p = %.8f",
        dual_max,
        dual_min,
        dual_min_no_psd,
        p,
    )
    _check(abs(dual_max - p) <= 1e-6, dual_max, p)
    _check(abs(dual_min - p) <= 1e-6, dual_min, p)
    _check(abs(dual_min_no_psd - dual_min) <= 1e-8, "Z >= 0 is not redundant")


def main() -> None:
    """Parse arguments and run the checks."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--p", type=float, default=None, help="single channel parameter")
    parser.add_argument(
        "--p-grid", type=str, default="0.25,0.3,0.5,0.75,1.0", help="comma-separated grid"
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    values = [args.p] if args.p is not None else [float(v) for v in args.p_grid.split(",")]
    for p in values:
        run(p)
    logger.info("All Part A checks passed.")


if __name__ == "__main__":
    main()
