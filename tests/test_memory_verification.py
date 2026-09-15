"""Tests for the quantum-memory verification document.

Numerical checks of the claims of ``verification-of-memory.md``
(measurement-device-independent verification of a quantum memory; the
depolarizing-channel example, Rosset-Buscemi-Liang): the correlation table,
the two dual SDPs equal to the Choi fidelity ``p``, the moment-matrix
relaxation of eq. (14) at level 1, and the factored NPA-tau re-expression
of eq. (14) at level 2 with certified non-degenerate values, for both the
single-output and the full four-output Bell measurement.

The NPA-tau re-expression (shared with
``examples/quantum_memory/common.py``, imported below) factors
``U_{x,y} = rho_x (x) I (x) sigma_y`` with the one-direction commutation
``rho_i sigma_j -> sigma_j rho_i``; the orthogonal input states |0> and
|1> make ``rho_0 rho_1`` and ``sigma_0 sigma_1`` identically zero (both
orders), so those words drop out of the SDP basis and prune the relation
set. The regression under test is that the functional class relations
encode into moment-matrix equalities that resolve to SDP variables -- a
missing resolution collapses the objective to a degenerate value.

The single-output L2 certification (two multi-minute solves) and the
4-output factored solves (8-25 minutes each) are marked ``slow`` and
excluded from CI (``uv run pytest -m "not slow"``); run them locally with
``uv run pytest -m slow tests/test_memory_verification.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples" / "quantum_memory"))

import common

DIM = 2  #: The document's d: each qubit subsystem is a qubit.

X = np.array([[0, 1], [1, 0]], dtype=complex)
"""The Pauli-X operator."""

Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
"""The Pauli-Y operator."""

Z = np.array([[1, 0], [0, -1]], dtype=complex)
"""The Pauli-Z operator."""

_STATE_VECTORS: tuple[np.ndarray, ...] = (
    np.array([1.0, 0.0]),
    np.array([0.0, 1.0]),
    np.array([1.0, 1.0]) / np.sqrt(2.0),
    np.array([1.0, 1j]) / np.sqrt(2.0),
)
"""The tomographically complete input states {|0>, |1>, |+>, |+i>}."""

_BELL_VECTORS: tuple[np.ndarray, ...] = (
    np.array([1.0, 0.0, 0.0, 1.0]) / np.sqrt(2.0),  # |phi+>
    np.array([1.0, 0.0, 0.0, -1.0]) / np.sqrt(2.0),  # |phi->
    np.array([0.0, 1.0, 1.0, 0.0]) / np.sqrt(2.0),  # |psi+>
    np.array([0.0, 1.0, -1.0, 0.0]) / np.sqrt(2.0),  # |psi->
)
"""The Bell state vectors on H_{A2} x H_{B2}."""


def input_projectors() -> list[np.ndarray]:
    """Return the projectors of the tomographically complete input set.

    Returns:
        The four projectors rho_x = |v_x><v_x|, x = 0..3.
    """
    return [np.outer(v, v.conj()) for v in _STATE_VECTORS]


def bell_projectors() -> list[np.ndarray]:
    """Return the four Bell projectors Phi^alpha on H_{A2} x H_{B2}.

    Returns:
        The projectors |phi+><phi+|, |phi-><phi-|, |psi+><psi+|, |psi-><psi-|.
    """
    return [np.outer(v, v.conj()) for v in _BELL_VECTORS]


def depolarizing_channel(rho: np.ndarray, p: float) -> np.ndarray:
    """Apply the qubit depolarizing channel, eq. (15) of the document.

    Args:
        rho: A single-qubit density matrix.
        p: The channel parameter in [1/4, 1].

    Returns:
        ``M_p(rho) = p rho + (1-p)/3 (X rho X + Y rho Y + Z rho Z)``.
    """
    return p * rho + (1.0 - p) / 3.0 * (X @ rho @ X + Y @ rho @ Y + Z @ rho @ Z)


def choi_state(p: float) -> np.ndarray:
    """Build the Choi state of the depolarizing channel, eq. (3).

    Args:
        p: The channel parameter.

    Returns:
        ``J_M = (I x M_p)|phi+><phi+|`` as a 4x4 matrix on H_{A0} x H_{A2}.
    """
    x = np.outer(_BELL_VECTORS[0], _BELL_VECTORS[0].conj())
    j_m = np.zeros((4, 4), dtype=complex)
    for a in range(DIM):
        for b in range(DIM):
            j_m[2 * a : 2 * a + 2, 2 * b : 2 * b + 2] = depolarizing_channel(
                x[2 * a : 2 * a + 2, 2 * b : 2 * b + 2], p
            )
    return j_m


def correlations_direct(p: float, x: int, y: int, alpha: int) -> float:
    """Correlation via eq. (1): ``Tr[(M(rho_x) x sigma_y) Phi^alpha]``.

    Args:
        p: The channel parameter.
        x: Alice's input index (0..3).
        y: Bob's input index (0..3).
        alpha: The measurement output (0..3).

    Returns:
        The correlation p(alpha | x, y).
    """
    rho = input_projectors()[x]
    sigma = input_projectors()[y]
    phi = bell_projectors()[alpha]
    m_rho = depolarizing_channel(rho, p)
    return float(np.real(np.trace(np.kron(m_rho, sigma) @ phi)))


def correlations_via_choi(p: float, x: int, y: int, alpha: int) -> float:
    """Correlation via eq. (10): ``d Tr[(rho_x^T x I x sigma_y)(J_M x I)(
    I x Phi^alpha)]``.

    Args:
        p: The channel parameter.
        x: Alice's input index (0..3).
        y: Bob's input index (0..3).
        alpha: The measurement output (0..3).

    Returns:
        The correlation p(alpha | x, y).
    """
    j_8 = np.kron(choi_state(p), np.eye(DIM))
    u = np.kron(np.kron(input_projectors()[x].T, np.eye(DIM)), input_projectors()[y])
    v = np.kron(np.eye(DIM), bell_projectors()[alpha])
    return float(np.real(DIM * np.trace(u @ j_8 @ v)))


def expected_correlation(p: float, x: int, y: int) -> float:
    """Return the closed-form single-output correlation, eq. (16).

    Args:
        p: The channel parameter.
        x: Alice's input index (0..3).
        y: Bob's input index (0..3).

    Returns:
        The table entry p(0 | x, y) for the single-output Bell measurement.

    Raises:
        ValueError: If ``x`` or ``y`` is outside 0..3.
    """
    if not (0 <= x < 4 and 0 <= y < 4):
        raise ValueError(f"Inputs must be in 0..3, got x={x}, y={y}.")
    if x == y:
        if x in (0, 1):
            return (1.0 + 2.0 * p) / 6.0
        if x == 2:
            return (1.0 + 2.0 * p) / 6.0
        return (1.0 - p) / 3.0
    if {x, y} == {0, 1}:
        return (1.0 - p) / 3.0
    return 0.25


def test_correlation_table() -> None:
    """The three forms of the correlation agree with eq. (16) and sum to 1."""
    for p in (0.25, 0.5, 1.0):
        for x in range(4):
            for y in range(4):
                direct = correlations_direct(p, x, y, 0)
                via_choi = correlations_via_choi(p, x, y, 0)
                assert direct == pytest.approx(via_choi, abs=1e-12)
                assert direct == pytest.approx(expected_correlation(p, x, y), abs=1e-12)
        for x in range(4):
            for y in range(4):
                total = sum(correlations_direct(p, x, y, alpha) for alpha in range(4))
                assert total == pytest.approx(1.0, abs=1e-12)


# --- The dual SDPs of eq. (6) and eq. (9) -----------------------------------


def _dual_sdp_values(p: float) -> tuple[float, float, float]:
    """Solve eq. (6) and eq. (9) with and without the redundant Z >= 0.

    Args:
        p: The channel parameter.

    Returns:
        The values of eq. (6), eq. (9), and eq. (9) without ``Z >= 0``.
    """
    import cvxpy as cp

    j_m = choi_state(p)
    j = cp.Variable((4, 4), hermitian=True)
    partial_trace_a0 = sum(j[2 * a : 2 * a + 2, 2 * a : 2 * a + 2] for a in range(DIM))
    constraints = [partial_trace_a0 == np.eye(DIM) / DIM, j >> 0]
    primal = cp.Problem(cp.Maximize(cp.real(cp.trace(j_m @ j))), constraints)
    primal.solve(solver="CLARABEL")
    assert primal.status in ("optimal", "optimal_inaccurate")

    z = cp.Variable((DIM, DIM), hermitian=True)
    dual = cp.Problem(
        cp.Minimize(cp.real(cp.trace(z)) / DIM),
        [cp.kron(np.eye(DIM), z) - j_m >> 0],
    )
    dual.solve(solver="CLARABEL")
    assert dual.status in ("optimal", "optimal_inaccurate")

    z2 = cp.Variable((DIM, DIM), hermitian=True)
    dual_plain = cp.Problem(
        cp.Minimize(cp.real(cp.trace(z2)) / DIM),
        [cp.kron(np.eye(DIM), z2) - j_m >> 0, z2 >> 0],
    )
    dual_plain.solve(solver="CLARABEL")
    assert dual_plain.status in ("optimal", "optimal_inaccurate")
    return float(primal.value), float(dual.value), float(dual_plain.value)


def test_dual_sdps_equal_fidelity() -> None:
    """Eq. (6) and eq. (9) both equal the Choi fidelity p; Z >= 0 is redundant."""
    pytest.importorskip("cvxpy")
    for p in (0.3, 0.5, 0.75):
        primal, dual, dual_plain = _dual_sdp_values(p)
        assert primal == pytest.approx(p, abs=1e-5)
        assert dual == pytest.approx(p, abs=1e-5)
        assert dual_plain == pytest.approx(dual, abs=1e-6)


# --- The eq. (14) relaxation, verbatim at level 1 ---------------------------


def _verbatim_l1_value(p: float, n_outputs: int = 1) -> float:
    """Solve the eq. (14) relaxation verbatim at level 1.

    Delegates to the shared construction in ``common.py`` (the functional
    class parametrization, the correlation pins and the completeness
    relations); the regression under test is that the certified value
    stays a non-degenerate lower bound.

    Args:
        p: The channel parameter.
        n_outputs: Number of measurement outputs (1 or 4).

    Returns:
        The optimal value (the certified fidelity lower bound).
    """
    return common.verbatim_relaxation_value(p, n_outputs=n_outputs, level=1)


def test_relaxation_lower_bound() -> None:
    """The level-1 relaxation is a valid non-degenerate lower bound on p."""
    pytest.importorskip("cvxpy")
    # Measured values (CLARABEL, p = 0.5); update with care if the solvers
    # change. With the completeness relations both output counts sit on
    # the analytic line (2p+1)/6 = 1/3 at p = 0.5.
    expected = {1: 0.33333333, 4: 0.33333333}
    for n_outputs, pinned in expected.items():
        value = _verbatim_l1_value(0.5, n_outputs=n_outputs)
        assert value == pytest.approx(pinned, abs=1e-3)
        assert value <= 0.5 + 1e-5
        assert value > 1e-6


# --- The NPA-tau re-expression: factored, certified values ------------------


@pytest.mark.slow
def test_npa_tau_factored_certified() -> None:
    """The factored NPA-tau L2 relaxation certifies a non-degenerate bound.

    One solve of the shared factored construction at the README table's
    anchor point p = 0.5 (direct sparse CLARABEL backend -- the dense
    cvxpy canonicalization of the 176-word moment matrix does not fit in
    memory). With the completeness moment-equalities (``rho_0 + rho_1 =
    I`` etc. lifted into the inserted blocks) no trace-preserving pin is
    needed: the certified value sits on the analytic line ``(2p+1)/6``
    (= 1/3 at p = 0.5). The value is the measured one; update with care
    if the construction changes. It must improve on the pre-completeness
    verbatim level-1 value (0.0894) and lie in the certified window
    (0, p]. The solve takes minutes, so the test is marked ``slow`` and
    excluded from CI.
    """
    pytest.importorskip("clarabel")
    value = common.npa_tau_relaxation_value(0.5, solver="clarabel")
    assert value == pytest.approx(0.333333, abs=1e-3)
    assert value <= 0.5 + 1e-5
    assert value > 1e-6


# --- The 4-output factored variant: construction sizes and solves -----------


def test_factored_npa_tau_construction() -> None:
    """The factored construction sizes match the measured values.

    The orthogonal-input rules (|0> and |1> are orthogonal, so
    ``rho_0 rho_1 -> 0`` and ``sigma_0 sigma_1 -> 0`` in both orders)
    shrink the SDP: the single-output construction keeps 62 S-basis words
    and 6339 relations (without the rules it was 66 and 3047). The
    relation count includes the functional-class equalities (with the
    constant-equalities tying free class members to pinned classes), the
    completeness moment-equalities lifting ``rho_0 + rho_1 = I`` (likewise
    sigma; ``sum_alpha v_alpha = I`` for four outputs) into the inserted
    blocks, and the Pauli anticommutator moment-equalities
    (``{rho_x, rho_z} = sum_t a_t rho_t``; 72 extramonomials, 176-word
    moment matrix). Pin the measured sizes so a change in the reduction
    rules trips the test; the 4-output variant carries the full Bell
    measurement (14 operators, 113 S-basis words).
    """
    expected = {
        1: (11, 62, 26, 72, 176, 6339, 10727),
        4: (14, 113, 29, 72, 239, 13585, 18425),
    }
    for n_outputs, sizes in expected.items():
        data = common._factored_data(n_outputs)
        relations, extras, localizing = common._factored_classes(n_outputs)
        sdp = common._npa_tau_problem(0.5, n_outputs=n_outputs).relaxation(level=2).sdp
        operators, s_basis, loc, ex, extended, rel, n_vars = sizes
        assert len(data["operators"]) == operators
        assert len(data["s_basis"]) == s_basis
        assert len(localizing) == loc
        assert len(extras) == ex
        assert len(relations) == rel
        assert sdp.blocks[0].size == extended
        assert sdp.n_vars == n_vars


def _npa_tau_4out_value() -> float:
    """Solve the 4-output L2 relaxation in a fresh subprocess.

    The direct sparse CLARABEL backend peaks near 3.5 GB on this problem;
    a long-lived pytest process that already imported cvxpy sits close
    enough to the commit limit that the solve is occasionally killed by
    the OS with no traceback. A fresh process keeps the peak low and the
    test repeatable (the measured value below was produced exactly this
    way).

    Returns:
        The certified L2 value at p = 0.5.
    """
    import subprocess

    example_dir = str(Path(__file__).resolve().parents[1] / "examples" / "quantum_memory")
    code = (
        "import sys, warnings\n"
        "warnings.filterwarnings('ignore')\n"
        f"sys.path.insert(0, {example_dir!r})\n"
        "from common import npa_tau_relaxation_value\n"
        "print(npa_tau_relaxation_value(0.5, solver='cvxpy', n_outputs=4))"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=3600)
    assert result.returncode == 0, f"4-output solve subprocess failed:\n{result.stderr[-2000:]}"
    return float(result.stdout.strip())


@pytest.mark.slow
def test_npa_tau_factored_certified_4out() -> None:
    """The 4-output factored NPA-tau L2 relaxation certifies the same bound.

    The full four-output Bell measurement adds three measurement outputs
    (14 operators instead of 11): at p = 0.5 the L2 value (direct sparse
    backend, one fresh subprocess) coincides with the single-output one on
    the ``(2p+1)/6`` line (= 1/3) -- with the completeness
    moment-equalities in place, neither the extra outputs nor a
    trace-preserving pin add anything. The solve takes 8-25 minutes
    (CLARABEL) or a few minutes (MOSEK), so the test is marked ``slow``
    and excluded from CI; run locally with
    ``uv run pytest -m slow tests/test_memory_verification.py``.
    """
    pytest.importorskip("cvxpy")
    value = _npa_tau_4out_value()
    assert value == pytest.approx(0.333333, abs=1e-3)
    assert value <= 0.5 + 1e-5
    assert value > 1e-6
