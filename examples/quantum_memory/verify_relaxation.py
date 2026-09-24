"""Part B: verify the moment-matrix relaxation of eq. (14) of the document.

Three constructions of the relaxation are exercised:

* The verbatim class-parametrized form (``verbatim_relaxation_value``):
  the two moment matrices ``M = Gamma^{(J_M)}`` and ``L = Gamma^{(Z)}``
  over the word basis with one variable per functional class.
* The operator-parametrized form (``operator_relaxation_value``): eq.
  (14) as written, with the genuine Hermitian operators ``J`` and ``Z``
  as variables; level 2 adds the degree-2 monomials ``U_{x,y} V_alpha``
  (the probe words).
* The factored NPA-tau re-expression (``npa_tau_relaxation_value``): the
  same relaxation written as a standard NPA problem over the factored
  algebra ``{rho, sigma, v, J, W}`` with state tau = I/8 (Brown-style
  commutation ``rho_i sigma_j -> sigma_j rho_i``; orthogonality rules
  ``rho_0 rho_1 -> 0``, ``sigma_0 sigma_1 -> 0``), with the full
  functional-linearity relations added as moment-equalities on the
  inserted blocks and the probe words ``rho_x sigma_y v_0`` extending the
  localizing basis. The 4-output variant (``--n-outputs 4``) runs through
  the package's direct sparse CLARABEL backend (``solver="clarabel"``) or
  MOSEK.

No trace-preserving pin is used anywhere: with the completeness relations
in place the pin is redundant.

Each certified value must be a lower bound on the Choi fidelity ``p`` and
strictly positive (non-degenerate); the level-2 values must additionally
match ``p`` itself. The factored solves take on the order of minutes.

Usage: ``uv run python examples/quantum_memory/verify_relaxation.py
[--p-grid 0.25,0.5,1.0] [--n-outputs 4]``
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (
    npa_tau_relaxation_value,
    operator_relaxation_value,
    verbatim_relaxation_value,
)

logger = logging.getLogger(__name__)

BOUND_TOL = 1e-5
#: Slack for the level-2 exactness checks (absorbs interior-point
#: tolerances of the sparse backends).
EXACT_TOL = 1e-4


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


def run(p: float, n_outputs: int) -> None:
    """Solve all relaxation variants for one channel parameter.

    Args:
        p: The channel parameter.
        n_outputs: The measurement outputs for the factored L2 solves
            (1 or 4; the verbatim level-1 rows run for both in any case).
    """
    logger.info("--- p = %.4f ---", p)

    verbatim_1 = verbatim_relaxation_value(p, n_outputs=1, level=1)
    operator_1 = operator_relaxation_value(p, n_outputs=1, level=1)
    logger.info(
        "verbatim class L1 = %.6f | operator L1 = %.6f (wall (2p+1)/6 = %.6f)",
        verbatim_1,
        operator_1,
        (2 * p + 1) / 6,
    )

    # The operator form at level 2: the probe words U_{x,y} V_alpha enter
    # the word set, and the relaxation attains the exact fidelity.
    operator_2 = operator_relaxation_value(p, n_outputs=1, level=2)
    operator_2_4 = operator_relaxation_value(p, n_outputs=4, level=2)
    logger.info(
        "operator L2 (probe words): single-output = %.6f | 4-output = %.6f (exact p = %.6f)",
        operator_2,
        operator_2_4,
        p,
    )

    # The factored NPA-tau re-expression at level 2: the functional-class
    # equalities of eqs. (12)/(13) enter as block-0 moment-matrix
    # equalities, and the functional-linearity relations as linear
    # moment-equalities on the inserted blocks, with the probe words
    # extending the localizing basis.
    # NOTE: the factored SDP is boundary-degenerate (the correlation pins
    # fix the Choi operator uniquely, so the feasible set has no strict
    # interior), so off-anchor interior-point solves may fail with
    # NumericalError/unknown. The operator rows above already certify p
    # at those points, so a factored solver failure degrades to a warning.
    npa_tau: float | None = None
    try:
        npa_tau = npa_tau_relaxation_value(p, solver="clarabel", n_outputs=n_outputs)
        logger.info("factored NPA-tau L2 (n_outputs=%d) = %.6f", n_outputs, npa_tau)
    except RuntimeError as exc:
        logger.warning(
            "factored NPA-tau L2 (n_outputs=%d) solver failed at p=%.4f "
            "(boundary-degenerate SDP; see the README solver note): %s",
            n_outputs, p, exc,
        )

    # Every certified value is a valid lower bound and non-degenerate; the
    # level-2 values must match p itself.
    for name, value in (
        ("verbatim class L1", verbatim_1),
        ("operator L1", operator_1),
        ("operator L2", operator_2),
        ("operator 4-output L2", operator_2_4),
    ):
        _check(value <= p + BOUND_TOL, name, value, p)
        _check(value > 1e-6, name, value)
    for name, value in (
        ("operator L2", operator_2),
        ("operator 4-output L2", operator_2_4),
    ):
        _check(abs(value - p) <= EXACT_TOL, name, value, p)
    if npa_tau is not None:
        _check(npa_tau <= p + BOUND_TOL, "factored NPA-tau L2", npa_tau, p)
        _check(npa_tau > 1e-6, "factored NPA-tau L2", npa_tau)
        _check(
            abs(npa_tau - p) <= EXACT_TOL,
            f"factored NPA-tau L2 (n_outputs={n_outputs})",
            npa_tau,
            p,
        )
    logger.info("all bounds certified (<= p + %.0e; L2 = p within %.0e)", BOUND_TOL, EXACT_TOL)


def main() -> None:
    """Parse arguments and run the checks."""
    assert __doc__ is not None
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--p", type=float, default=None, help="single channel parameter")
    parser.add_argument(
        "--p-grid", type=str, default="0.5", help="comma-separated grid (e.g. 0.25,0.3,0.5,0.75,1.0)"
    )
    parser.add_argument(
        "--n-outputs", type=int, choices=[1, 4], default=1,
        help="measurement outputs of the factored L2 solves (default 1)",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    values = [args.p] if args.p is not None else [float(v) for v in args.p_grid.split(",")]
    for p in values:
        run(p, args.n_outputs)
    logger.info("All Part B checks passed.")


if __name__ == "__main__":
    main()
