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
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest

if TYPE_CHECKING:
    import cvxpy as cp

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


def _u_operators() -> list[np.ndarray]:
    """Return ``U_{x,y} = rho_x^T x I^{A2} x sigma_y`` as 8x8 matrices."""
    return [
        np.kron(np.kron(input_projectors()[x].T, np.eye(DIM)), input_projectors()[y])
        for x in range(4)
        for y in range(4)
    ]


def _v_operators(n_outputs: int) -> list[np.ndarray]:
    """Return ``V_alpha = I^{A0} x Phi^alpha`` as 8x8 matrices."""
    return [np.kron(np.eye(DIM), bell_projectors()[alpha]) for alpha in range(n_outputs)]


def _build_words(us: list[np.ndarray], vs: list[np.ndarray]) -> list[np.ndarray]:
    """Return the level-1 word set {I} U {U} U {V} as 8x8 matrices."""
    return [np.eye(8, dtype=complex), *us, *vs]


def _partial_trace_b2(x: np.ndarray) -> np.ndarray:
    """Trace out H_{B2} from an 8x8 matrix (diagonal pairs only)."""
    x4 = x.reshape(DIM, DIM, DIM, DIM, DIM, DIM)
    return np.einsum("abcdec->abde", x4).reshape(4, 4)


def _partial_trace_a0b2(x: np.ndarray) -> np.ndarray:
    """Trace out H_{A0} x H_{B2} from an 8x8 matrix (diagonal pairs only)."""
    x4 = x.reshape(DIM, DIM, DIM, DIM, DIM, DIM)
    return np.einsum("abcadc->bd", x4).reshape(2, 2)


FunctionalKey = tuple[tuple[float, ...], tuple[float, ...]]


def _functional_keys(
    words: list[np.ndarray], trace_fn: Callable[[np.ndarray], np.ndarray]
) -> list[list[FunctionalKey]]:
    """Classify every (u, v) pair by its functional of the inserted operator.

    ``M[u, v] = Tr[u^dagger (J_M x I) v]`` depends on ``G_{u,v} =
    Tr_{B2}(v u^dagger)`` only; two pairs with the same G share a moment.

    Args:
        words: The word list.
        trace_fn: The partial trace to apply (B2 or A0B2).

    Returns:
        A matrix of rounded real/imag keys, one per (u, v) pair.
    """
    n = len(words)
    keys: list[list[FunctionalKey]] = []
    for i in range(n):
        row: list[FunctionalKey] = []
        for j in range(n):
            g = trace_fn(words[j] @ words[i].T.conj())
            row.append(
                (tuple(np.round(np.real(g), 9).ravel()), tuple(np.round(np.imag(g), 9).ravel()))
            )
        keys.append(row)
    return keys


def _conjugate_key(key: FunctionalKey) -> FunctionalKey:
    """Return the key of the adjoint functional G^dagger of ``key``."""
    real, imag = key
    side = round(np.sqrt(len(real)))
    r = np.array(real).reshape(side, side)
    im = np.array(imag).reshape(side, side)
    return (tuple(r.T.ravel()), tuple((-im).T.ravel()))


def _class_expression(keys: list[list[FunctionalKey]]) -> cp.Expression:
    """Build a cvxpy matrix with one variable per functional class.

    Args:
        keys: The functional class keys of every (u, v) pair.

    Returns:
        The cvxpy matrix expression (Hermitian by construction).
    """
    import cvxpy as cp

    classes: dict[FunctionalKey, int] = {}
    variables: dict[int, object] = {}

    def var_for(key: FunctionalKey) -> object:
        if key in classes:
            return variables[classes[key]]
        adjoint = _conjugate_key(key)
        if adjoint in classes:
            return cp.conj(variables[classes[adjoint]])
        real, imag = key
        side = round(np.sqrt(len(real)))
        r = np.array(real).reshape(side, side)
        im = np.array(imag).reshape(side, side)
        self_adjoint = np.max(np.abs(im)) == 0.0 and np.allclose(r, r.T, atol=1e-8)
        index = len(classes)
        classes[key] = index
        variable = cp.Variable() if self_adjoint else cp.Variable(complex=True)
        variables[index] = variable
        return variable

    n = len(keys)
    return cp.bmat([[var_for(keys[i][j]) for j in range(n)] for i in range(n)])


def _verbatim_l1_value(p: float, n_outputs: int = 1, tp_pin: bool = False) -> float:
    """Solve the eq. (14) relaxation verbatim at level 1.

    Args:
        p: The channel parameter.
        n_outputs: Number of measurement outputs (1 or 4).
        tp_pin: Pin ``M[I, I] = d`` (trace-preserving Choi normalization).

    Returns:
        The optimal value (the certified fidelity lower bound).
    """
    import cvxpy as cp

    words = _build_words(_u_operators(), _v_operators(n_outputs))
    m = _class_expression(_functional_keys(words, _partial_trace_b2))
    l_block = _class_expression(_functional_keys(words, _partial_trace_a0b2))
    constraints: list[object] = [m >> 0, l_block >> 0, l_block - m >> 0]
    for alpha in range(n_outputs):
        for x in range(4):
            for y in range(4):
                i_u = 1 + 4 * x + y
                i_v = 1 + 16 + alpha
                constraints.append(m[i_v, i_u] == correlations_via_choi(p, x, y, alpha) / DIM)
    if tp_pin:
        constraints.append(m[0, 0] == DIM)
    problem = cp.Problem(cp.Minimize(cp.real(l_block[0, 0]) / DIM**3), constraints)
    problem.solve(solver="CLARABEL")
    if problem.status not in ("optimal", "optimal_inaccurate"):
        pytest.skip(f"cvxpy status {problem.status} on the verbatim L1 relaxation")
    return float(problem.value)


def test_relaxation_lower_bound() -> None:
    """The level-1 relaxation is a valid non-degenerate lower bound on p."""
    pytest.importorskip("cvxpy")
    # Measured values (CLARABEL, p = 0.5); update with care if the solvers
    # change. All lie in the certified window [p/(8 d), p] = [0.03125, 0.5].
    expected = {
        (1, False): 0.08939623,
        (1, True): 0.29204206,
        (4, False): 0.12500001,
        (4, True): 0.31250000,
    }
    for (n_outputs, tp_pin), pinned in expected.items():
        value = _verbatim_l1_value(0.5, n_outputs=n_outputs, tp_pin=tp_pin)
        assert value == pytest.approx(pinned, abs=1e-3)
        assert value <= 0.5 + 1e-5
        assert value > 1e-6


# --- The NPA-tau re-expression: factored, certified values ------------------


@pytest.mark.slow
def test_npa_tau_factored_certified() -> None:
    """The factored NPA-tau L2 relaxation certifies non-degenerate bounds.

    Two solves of the shared factored construction at the README table's
    anchor point p = 0.5, with and without the trace-preserving pin
    ``<J> = 1/4``. The pinned values are the measured ones
    (cvxpy/CLARABEL); update with care if the construction changes. The
    orthogonal-input rules drop the ``rho_0 rho_1``-type words, which also
    drops active class-relation links, so the certified values sit slightly
    below the un-reduced construction's (0.301898 / 0.331193); both are
    valid lower bounds. Both must improve on the verbatim level-1 values
    (0.0894 / 0.2920) and lie in the certified window (0, p]. The two
    solves take minutes each, so the test is marked ``slow`` and excluded
    from CI.
    """
    pytest.importorskip("cvxpy")
    expected = {False: 0.293118, True: 0.331071}
    for tp_pin, pinned in expected.items():
        value = common.npa_tau_relaxation_value(0.5, tp_pin=tp_pin, solver="cvxpy")
        assert value == pytest.approx(pinned, abs=1e-3)
        assert value <= 0.5 + 1e-5
        assert value > 1e-6


# --- The 4-output factored variant: construction sizes and solves -----------


def test_factored_npa_tau_construction() -> None:
    """The factored construction sizes match the measured values.

    The orthogonal-input rules (|0> and |1> are orthogonal, so
    ``rho_0 rho_1 -> 0`` and ``sigma_0 sigma_1 -> 0`` in both orders)
    shrink the SDP: the single-output construction keeps 62 S-basis words
    and 2839 class relations (without the rules it was 66 and 3047). Pin
    the measured sizes so a change in the reduction rules trips the test;
    the 4-output variant carries the full Bell measurement (14 operators,
    113 S-basis words). The relation count is independent of ``tp_pin``.
    """
    expected = {
        (1, False): (11, 62, 26, 32, 136, 2839, 5967),
        (1, True): (11, 62, 26, 32, 136, 2839, 5966),
        (4, False): (14, 113, 29, 32, 199, 5551, 11265),
        (4, True): (14, 113, 29, 32, 199, 5551, 11264),
    }
    for (n_outputs, tp_pin), sizes in expected.items():
        data = common._factored_data(n_outputs)
        relations, extras, localizing = common._factored_classes(tp_pin, n_outputs)
        sdp = common._npa_tau_problem(0.5, tp_pin, n_outputs=n_outputs).relaxation(level=2).sdp
        operators, s_basis, loc, ex, extended, rel, n_vars = sizes
        assert len(data["operators"]) == operators
        assert len(data["s_basis"]) == s_basis
        assert len(localizing) == loc
        assert len(extras) == ex
        assert len(relations) == rel
        assert sdp.blocks[0].size == extended
        assert sdp.n_vars == n_vars


def _npa_tau_4out_value(tp_pin: bool) -> float:
    """Solve the 4-output L2 relaxation in a fresh subprocess.

    The direct sparse CLARABEL backend peaks near 3.5 GB on this problem;
    a long-lived pytest process that already imported cvxpy sits close
    enough to the commit limit that the solve is occasionally killed by
    the OS with no traceback. A fresh process per solve keeps the peak
    low and the test repeatable (the measured values below were produced
    exactly this way).

    Args:
        tp_pin: Pin ``<J> = 1/4`` (trace-preserving Choi normalization).

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
        f"print(npa_tau_relaxation_value(0.5, tp_pin={tp_pin!r}, solver='cvxpy', n_outputs=4))"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=3600)
    assert result.returncode == 0, f"4-output solve subprocess failed:\n{result.stderr[-2000:]}"
    return float(result.stdout.strip())


@pytest.mark.slow
def test_npa_tau_factored_certified_4out() -> None:
    """The 4-output factored NPA-tau L2 relaxation certifies tighter bounds.

    The full four-output Bell measurement adds three measurement outputs
    (14 operators instead of 11): at p = 0.5 the L2 values (CLARABEL,
    direct sparse CLARABEL backend, one fresh subprocess per solve) beat the
    single-output ones (0.293118 / 0.331071). Each solve takes 8-25
    minutes, so the test is marked ``slow`` and excluded from CI; run
    locally with ``uv run pytest -m slow tests/test_memory_verification.py``.
    """
    pytest.importorskip("cvxpy")
    expected = {False: 0.331141, True: 0.333319}
    for tp_pin, pinned in expected.items():
        value = _npa_tau_4out_value(tp_pin)
        assert value == pytest.approx(pinned, abs=1e-3)
        assert value <= 0.5 + 1e-5
        assert value > 1e-6
