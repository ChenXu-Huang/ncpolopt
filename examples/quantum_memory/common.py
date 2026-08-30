"""Shared building blocks for the quantum-memory verification scripts.

Implements the operators, channel and correlations of the document
``verification-of-memory.md`` (measurement-device-independent verification
of a quantum memory; the depolarizing-channel example), together with two
SDP constructions that verify its moment-matrix relaxation:

* :func:`verbatim_relaxation_value` -- the relaxation of eq. (14) of the
  document, assembled directly with cvxpy over the word set generated
  from the operator set :math:`S`, for a single-output or the full
  four-output Bell measurement.
* :func:`npa_tau_relaxation_value` -- the same relaxation re-expressed as
  a standard NPA problem on the extended algebra
  :math:`S \\cup \\{J, W\\}` with the maximally mixed state, solved
  through the public ncpolopt API for one output, or through the
  package's direct sparse CLARABEL backend (``solver="clarabel"``) for
  the four-output variant (the dense cvxpy conversion cannot hold its
  199-word moment matrix). The trace pins and the functional-class
  moment-equalities come from the package's operator-insertion helpers
  (:func:`ncpolopt.hierarchies.trace_moment_pins` and
  :func:`ncpolopt.hierarchies.class_moment_equalities`).

Conventions follow the document: :math:`d = 2`, the auxiliary system
:math:`A_0` and the memory output :math:`A_2` are qubits, the Choi state
is :math:`J_M = (I \\otimes M)|\\phi^+\\rangle\\langle\\phi^+|` with
:math:`\\mathrm{Tr}(J_M) = 1`, and 8x8 matrices are laid out as
:math:`A_0 \\otimes A_2 \\otimes B_2`.

The heavy solvers (cvxpy, ncpolopt) are imported lazily inside the
functions that need them, mirroring the lazy-solver invariant of the
package.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
from sympy import S

if TYPE_CHECKING:
    import cvxpy as cp

    from ncpolopt import Problem

FunctionalKey = tuple[tuple[float, ...], tuple[float, ...]]

DIM: int = 2  #: Dimension of each qubit subsystem (the document's d).

# --- Single-qubit operators -------------------------------------------------

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
"""The Bell state vectors on H_{A2} x H_{B2}, in basis {00, 01, 10, 11}."""


def input_projectors() -> list[np.ndarray]:
    """Return the projectors of the tomographically complete input set.

    Returns:
        The four projectors rho_x = |v_x><v_x| (and sigma_y), x = 0..3.
    """
    return [np.outer(v, v.conj()) for v in _STATE_VECTORS]


def bell_projectors() -> list[np.ndarray]:
    """Return the four Bell projectors Phi^alpha on H_{A2} x H_{B2}.

    Returns:
        The projectors |phi+><phi+|, |phi-><phi-|, |psi+><psi+|,
        |psi-><psi-|.
    """
    return [np.outer(v, v.conj()) for v in _BELL_VECTORS]


def depolarizing_channel(rho: np.ndarray, p: float) -> np.ndarray:
    """Apply the qubit depolarizing channel, eq. (15) of the document.

    Args:
        rho: A single-qubit density matrix.
        p: The channel parameter, ``p`` in [1/4, 1].

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
    phi = _BELL_VECTORS[0]
    x = np.outer(phi, phi.conj())
    j_m = np.zeros((4, 4), dtype=complex)
    for a in range(DIM):
        for b in range(DIM):
            j_m[2 * a : 2 * a + 2, 2 * b : 2 * b + 2] = depolarizing_channel(
                x[2 * a : 2 * a + 2, 2 * b : 2 * b + 2], p
            )
    return j_m


def werner_state(p: float) -> np.ndarray:
    """Return the Werner state with fidelity ``p`` to |phi+>.

    Args:
        p: The fidelity parameter in [1/4, 1].

    Returns:
        ``p |phi+><phi+| + (1-p)/3 (I4 - |phi+><phi+|)``.
    """
    phi = _BELL_VECTORS[0]
    phi_plus = np.outer(phi, phi.conj())
    return p * phi_plus + (1.0 - p) / 3.0 * (np.eye(4) - phi_plus)


def choi_fidelity(p: float) -> float:
    """Return the Choi fidelity ``<phi+|J_M|phi+>`` (the document's eq. (4)
    value for the identity extraction map).

    Args:
        p: The channel parameter.

    Returns:
        The fidelity of the Choi state to the maximally entangled state.
    """
    phi_plus = np.outer(_BELL_VECTORS[0], _BELL_VECTORS[0].conj())
    return float(np.real(np.trace(choi_state(p) @ phi_plus)))


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
        if x in (0, 1, 2):
            return (1.0 + 2.0 * p) / 6.0
        return (1.0 - p) / 3.0
    if {x, y} == {0, 1}:
        return (1.0 - p) / 3.0
    return 0.25


# --- Word machinery for the moment-matrix relaxation -----------------------


def u_operators() -> list[np.ndarray]:
    """Return the operators ``U_{x,y} = rho_x^T x I^{A2} x sigma_y``.

    Returns:
        The 16 operators as 8x8 matrices, in index order x*4 + y.
    """
    return [
        np.kron(np.kron(input_projectors()[x].T, np.eye(DIM)), input_projectors()[y])
        for x in range(4)
        for y in range(4)
    ]


def v_operators(n_outputs: int) -> list[np.ndarray]:
    """Return the operators ``V_alpha = I^{A0} x Phi^alpha``.

    Args:
        n_outputs: The number of measurement outputs (1 or 4).

    Returns:
        The projectors as 8x8 matrices.
    """
    return [np.kron(np.eye(DIM), bell_projectors()[alpha]) for alpha in range(n_outputs)]


def build_words(us: Sequence[np.ndarray], vs: Sequence[np.ndarray], level: int) -> list[np.ndarray]:
    """Build the word set over S = {I} U {U} U {V} up to ``level``.

    Words are products of at most ``level`` elements of S, stored as
    explicit 8x8 matrices. Duplicate and zero words are dropped.

    Args:
        us: The U operators (16).
        vs: The V operators (1 or 4).
        level: The word length limit (1 or 2).

    Returns:
        The deduplicated word list; the identity word is the first entry.
    """
    words = [np.eye(8, dtype=complex)]
    words.extend(us)
    words.extend(vs)
    if level >= 2:
        # Products of two level-1 words only; iterating over the live
        # ``words`` list (as originally written) re-appends products of
        # products, compounding to unbounded word length.
        base = words[1:]
        for a in base:
            for b in base:
                w = a @ b
                if np.linalg.norm(w) > 1e-12 and not any(np.allclose(w, x, atol=1e-12) for x in words):
                    words.append(w)
    return words


def partial_trace_b2(x: np.ndarray) -> np.ndarray:
    """Trace out H_{B2} from an 8x8 matrix.

    Args:
        x: An 8x8 matrix on H_{A0} x H_{A2} x H_{B2}.

    Returns:
        The 4x4 partial trace over B2.
    """
    x4 = x.reshape(DIM, DIM, DIM, DIM, DIM, DIM)
    # The partial trace keeps only the diagonal pairs (b2, b2') = b2 == b2';
    # a plain .sum(axis=(2, 5)) would trace over B2 x B2' together.
    return np.einsum("abcdec->abde", x4).reshape(4, 4)


def partial_trace_a0b2(x: np.ndarray) -> np.ndarray:
    """Trace out H_{A0} x H_{B2} from an 8x8 matrix.

    Args:
        x: An 8x8 matrix on H_{A0} x H_{A2} x H_{B2}.

    Returns:
        The 2x2 partial trace over A0 and B2.
    """
    x4 = x.reshape(DIM, DIM, DIM, DIM, DIM, DIM)
    return np.einsum("abcadc->bd", x4).reshape(2, 2)


def _functional_keys(
    words: Sequence[np.ndarray], trace_fn: Callable[[np.ndarray], np.ndarray]
) -> list[list[FunctionalKey]]:
    """Classify every moment (u, v) by its functional of the inserted
    operator.

    ``M[u, v] = Tr[u^dagger (J_M x I) v]`` depends on
    ``G_{u,v} = Tr_{B2}(v u^dagger)`` only; two pairs with the same G
    share a moment. The key is the rounded real/imag part of G.

    Args:
        words: The word list.
        trace_fn: The partial trace to apply (B2 for the M block, A0B2
            for the L block).

    Returns:
        A matrix of keys, one per (u, v) pair.
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
    """Build a cvxpy matrix expression with one variable per functional class.

    Entry ``(i, j)`` becomes the variable of the functional class of
    ``(u_i, u_j)``: a real variable for a self-adjoint functional, a
    complex variable for a paired class (the conjugate class references
    the same variable under ``cp.conj``). The matrix is Hermitian by
    construction, so the PSD constraints are exact.

    Args:
        keys: The functional class keys of every (u, v) pair.

    Returns:
        The cvxpy matrix expression.
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
        # A self-adjoint functional class (G hermitian: real part
        # symmetric, imaginary part antisymmetric) has a real moment, so
        # it takes a real variable; anything else forms a conjugate pair.
        self_adjoint = np.allclose(r, r.T, atol=1e-8) and np.allclose(im, -im.T, atol=1e-8)
        index = len(classes)
        classes[key] = index
        variable = cp.Variable() if self_adjoint else cp.Variable(complex=True)
        variables[index] = variable
        return variable

    n = len(keys)
    return cp.bmat([[var_for(keys[i][j]) for j in range(n)] for i in range(n)])


def verbatim_relaxation_value(
    p: float,
    n_outputs: int = 1,
    level: int = 1,
    tp_pin: bool = False,
    solver: str = "CLARABEL",
) -> float:
    """Solve the document's eq. (14) relaxation verbatim with cvxpy.

    The two moment matrices ``M = Gamma^{(J_M)}`` and ``L = Gamma^{(Z)}``
    are built over the word basis with one variable per functional class
    (entry ``M[u, v]`` depends only on ``G_{u,v} = Tr_{B2}(v u^dagger)``,
    ``L[u, v]`` only on ``H_{u,v} = Tr_{A0 B2}(v u^dagger)``); the
    correlations pin ``M[V_alpha, U_{x,y}] = p(alpha|x,y)/d``, and the
    constraints are ``M >= 0, L >= 0, L - M >= 0`` with objective
    ``L[I, I]/d^3``.

    Args:
        p: The channel parameter.
        n_outputs: Number of measurement outputs (1 or 4).
        level: Word length limit (1 or 2).
        tp_pin: Pin ``M[I, I] = d`` (trace-preserving Choi normalization).
        solver: A cvxpy solver name (default "CLARABEL").

    Returns:
        The optimal value (the certified fidelity lower bound).

    Raises:
        RuntimeError: If the solver status is neither "optimal" nor
            "optimal_inaccurate".
    """
    import cvxpy as cp

    words = build_words(u_operators(), v_operators(n_outputs), level)
    m = _class_expression(_functional_keys(words, partial_trace_b2))
    l_block = _class_expression(_functional_keys(words, partial_trace_a0b2))
    constraints: list[object] = [m >> 0, l_block >> 0, l_block - m >> 0]

    for alpha in range(n_outputs):
        for x in range(4):
            for y in range(4):
                i_u = 1 + 4 * x + y
                i_v = 1 + 16 + alpha
                constraints.append(m[i_v, i_u] == correlations_via_choi(p, x, y, alpha) / DIM)
    if tp_pin:
        constraints.append(m[0, 0] == DIM)

    objective = cp.Minimize(cp.real(l_block[0, 0]) / DIM**3)
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=solver)
    # CLARABEL labels the complex-realified problems "optimal_inaccurate"
    # even when its own gap is ~1e-10; accept both statuses.
    if problem.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"Verbatim relaxation failed with status {problem.status!r}.")
    return float(problem.value)


# --- The factored NPA-tau re-expression (Brown-style) -----------------------
#
# The U operators factor on H_{A0} x H_{A2} x H_{B2} as
# ``U_{x,y} = rho_x (x) I (x) sigma_y``, so the level-2 basis over the
# algebra {rho, sigma, v, J, W} (13 or 14 generators for 1 or 4 outputs)
# collapses under the one-direction commutation ``rho_i sigma_j ->
# sigma_j rho_i`` and the orthogonal-input rules to 136 extended-basis
# words with one output, 199 with four. The degree-3 words ``g U_{x,y}``
# (``g`` in {J, W}) are added as extramonomials: the U-pair entries
# ``<U^dagger g U'>`` (degree 5) then exist in the moment matrix, the
# localizing matrices can run over the ``{1, rho, sigma, v} U {U_{x,y}}``
# basis (26 or 29 words), and the functional-class equalities of
# eqs. (12)/(13) become writable as block-0 moment-equalities. The
# orthogonal input states |0> and |1> make ``rho_0 rho_1`` and
# ``sigma_0 sigma_1`` identically zero (both orders), so those words drop
# out of the basis entirely and shrink the SDP; the certified values below
# are measured under this construction (dropping the rows also drops
# active class-relation links, so the numbers differ slightly from the
# un-reduced construction -- both are valid lower bounds).
# Everything below is independent of the channel parameter ``p`` except the
# moment VALUES, so the relation set is computed once per
# ``(tp_pin, n_outputs)`` and cached.

_FACTORED_DATA: dict[int, dict[Any, Any]] = {}
_FACTORED_CLASSES: dict[tuple[bool, int], tuple[list[object], list[object], list[object]]] = {}


def _factored_data(n_outputs: int = 1) -> dict[Any, Any]:
    """The factored-algebra operators, rules, basis and explicit matrices.

    The algebra is ``{rho, sigma, v, J, W}`` with ``n_outputs`` Bell
    measurement outputs ``v`` and the input set {|0>, |1>, |+>, |+i>}.
    Besides the idempotences ``x^2 -> x``, the mutual orthogonality of the
    measurement outputs (``v_a v_b -> 0`` for ``a != b``) and the
    one-direction commutation ``rho_i sigma_j -> sigma_j rho_i``, the
    rules include the orthogonality of the input states |0> and |1>:
    ``rho_0 rho_1 -> 0`` and ``sigma_0 sigma_1 -> 0`` in both orders.
    Every moment involving such a zero word is identically 0, so the words
    drop out of the SDP basis and shrink the problem (the relaxation's
    optimum shifts slightly -- dropping the rows also drops active
    class-relation links; both constructions yield valid lower bounds).

    Args:
        n_outputs: The number of measurement outputs (1 or 4).

    Returns:
        A dict with the operator list, the substitution rules, the S-basis
        (62 words for one output, 113 for four), the 8x8 S-generator
        matrices, and a helper mapping S-basis words to explicit matrices.

    Raises:
        ValueError: If ``n_outputs`` is not 1 or 4.
    """
    cached = _FACTORED_DATA.get(n_outputs)
    if cached is not None:
        return cached
    if n_outputs not in (1, 4):
        raise ValueError(f"n_outputs must be 1 or 4, got {n_outputs}.")
    from ncpolopt import generate_operators, get_all_monomials

    rhos = generate_operators("rho", 4, hermitian=True)
    sigmas = generate_operators("sigma", 4, hermitian=True)
    vs = generate_operators("v", n_outputs, hermitian=True)
    j, w = generate_operators("jw", 2, hermitian=True)
    operators = [*rhos, *sigmas, *vs, j, w]

    substitutions: dict[object, object] = {}
    for rho in rhos:
        substitutions[rho**2] = rho
    for sigma in sigmas:
        substitutions[sigma**2] = sigma
    for v in vs:
        substitutions[v**2] = v
    for a in range(n_outputs):
        for b in range(n_outputs):
            if a != b:
                substitutions[vs[a] * vs[b]] = 0
    for rho in rhos:
        for sigma in sigmas:
            substitutions[rho * sigma] = sigma * rho
    # The input states |0> and |1> are orthogonal, so rho_0 rho_1 = 0 and
    # sigma_0 sigma_1 = 0 in both orders; the zero words and every moment
    # involving them drop out of the SDP (their values are identically 0).
    substitutions[rhos[0] * rhos[1]] = 0
    substitutions[rhos[1] * rhos[0]] = 0
    substitutions[sigmas[0] * sigmas[1]] = 0
    substitutions[sigmas[1] * sigmas[0]] = 0

    rho_mats = [np.kron(np.kron(pj, np.eye(DIM)), np.eye(DIM)) for pj in input_projectors()]
    sigma_mats = [np.kron(np.kron(np.eye(DIM), np.eye(DIM)), pj) for pj in input_projectors()]
    v_mats = [np.kron(np.eye(DIM), phi) for phi in bell_projectors()[:n_outputs]]

    s_generators = [*rhos, *sigmas, *vs]
    s_mats: dict[object, np.ndarray] = dict(zip(rhos, rho_mats, strict=True))
    s_mats.update(zip(sigmas, sigma_mats, strict=True))
    s_mats.update(zip(vs, v_mats, strict=True))
    s_basis = get_all_monomials(s_generators, None, substitutions, 2)

    def word_matrix(word: Any) -> np.ndarray:
        """The explicit 8x8 matrix of an S-basis word (word order = product order)."""
        if word == 1:
            return np.eye(8, dtype=complex)
        if word in s_mats:
            return s_mats[word]
        mat = np.eye(8, dtype=complex)
        for factor in word.args:
            mat = mat @ s_mats[factor]
        return mat

    _FACTORED_DATA[n_outputs] = {
        "operators": operators,
        "substitutions": substitutions,
        "s_basis": s_basis,
        "s_mats": s_mats,
        "word_matrix": word_matrix,
        "vs": vs,
        "j": j,
        "w": w,
    }
    return _FACTORED_DATA[n_outputs]


def _factored_moments(p: float, tp_pin: bool, n_outputs: int = 1) -> dict[object, object]:
    """Pin every S-moment and the correlations to their tau values.

    The S-moments are pinned to their trace values under the maximally
    mixed state by :func:`ncpolopt.hierarchies.trace_moment_pins`
    (symmetrized to their real part: with ``complex_matrix=False`` the
    consistent pin of a genuinely complex moment is
    ``(m + m.adjoint())/2 = Re Tr[word]/8``, and the real part of the
    true quantum moment matrix stays feasible, so the relaxation remains
    a valid lower bound).

    Args:
        p: The channel parameter.
        tp_pin: Pin ``<J>`` to the trace-preserving Choi normalization.
        n_outputs: The number of measurement outputs (1 or 4).

    Returns:
        The moment substitution dict.
    """
    from ncpolopt.hierarchies import trace_moment_pins
    from ncpolopt.substitutions import apply_substitutions

    data = _factored_data(n_outputs)
    substitutions = data["substitutions"]
    moments: dict[object, object] = trace_moment_pins(
        data["s_basis"],
        data["word_matrix"],
        substitutions=substitutions,
        force_real=True,
    )
    # Correlations: p(alpha|x,y) = 8 d <rho_x sigma_y J v_alpha>.
    for alpha in range(n_outputs):
        for x in range(4):
            for y in range(4):
                key = apply_substitutions(
                    data["operators"][x] * data["operators"][4 + y] * data["j"] * data["vs"][alpha],
                    substitutions,
                )
                moments[key] = correlations_via_choi(p, x, y, alpha) / (8.0 * DIM)
    if tp_pin:
        moments[data["j"]] = np.trace(np.kron(choi_state(p), np.eye(DIM))) / 8.0
    return moments


def _factored_classes(tp_pin: bool, n_outputs: int = 1) -> tuple[list[object], list[object], list[object]]:
    """The p-independent class equalities, extramonomials and localizing basis.

    The pinned-class checks depend only on the moment KEYS, which do not
    change with ``p``, so the relation set is computed once per
    ``(tp_pin, n_outputs)`` pair and cached. The equalities themselves
    are generated by :func:`ncpolopt.hierarchies.class_moment_equalities`
    from a draft build of the relaxation at ``p = 0.5`` (the draft's
    block-0 layout is final: the equality blocks are appended after the
    moment blocks).

    Args:
        tp_pin: Pin ``<J>`` (changes which classes are constants).
        n_outputs: The number of measurement outputs (1 or 4).

    Returns:
        The triple ``(momentequalities, extramonomials, localizing_basis)``.
    """
    cached = _FACTORED_CLASSES.get((tp_pin, n_outputs))
    if cached is not None:
        return cached
    from ncpolopt.hierarchies import class_moment_equalities
    from ncpolopt.substitutions import apply_substitutions

    data = _factored_data(n_outputs)
    operators = data["operators"]
    substitutions = data["substitutions"]
    vs, j, w = data["vs"], data["j"], data["w"]
    rhos, sigmas = operators[:4], operators[4:8]

    u_words = [apply_substitutions(rhos[x] * sigmas[y], substitutions) for x in range(4) for y in range(4)]
    extras = [
        apply_substitutions(g * u_word, substitutions)
        for g in (j, w)
        for u_word in u_words
    ]
    # The localizing matrices run over the {1, rho, sigma, v} U {U_{x,y}}
    # basis (26 or 29 words): the degree-2 U words are admissible because
    # their ``g U`` products are the extramonomials, so every localizing
    # entry monomial ``u^dagger g w`` (degree <= 5) exists as a
    # moment-matrix entry.
    localizing_basis = [S.One, *rhos, *sigmas, *vs, *u_words]

    draft = _assemble_npa_tau_problem(
        data,
        _factored_moments(0.5, tp_pin, n_outputs),
        momentequalities=[],
        extras=extras,
        localizing_basis=localizing_basis,
    )
    momentequalities = class_moment_equalities(
        draft,
        2,
        inserted=((j, partial_trace_b2), (w, partial_trace_a0b2)),
        row_words=list(data["s_basis"]),
        col_words=localizing_basis,
        word_matrix=data["word_matrix"],
    )

    cached = (momentequalities, extras, localizing_basis)
    _FACTORED_CLASSES[(tp_pin, n_outputs)] = cached
    return cached


def _assemble_npa_tau_problem(
    data: dict[Any, Any],
    moments: dict[object, object],
    momentequalities: list[object],
    extras: list[object] | None,
    localizing_basis: list[object] | None,
) -> Problem:
    """Assemble the factored NPA ``Problem`` from its parts.

    Args:
        data: The factored-algebra data of :func:`_factored_data`.
        moments: The moment substitutions of :func:`_factored_moments`.
        momentequalities: The functional-class moment-equalities.
        extras: The extramonomials (the degree-3 words ``g U_{x,y}``).
        localizing_basis: The localizing basis of all three inequalities.

    Returns:
        The built ``ncpolopt.Problem``.
    """
    from ncpolopt import Problem

    j, w = data["j"], data["w"]
    return Problem(
        data["operators"],
        objective=w,
        inequalities=[j, w, w - j],
        substitutions=data["substitutions"],
        momentsubstitutions=moments,
        momentequalities=momentequalities,
        extramonomials=extras,
        localizing_monomials=[localizing_basis] * 3 if localizing_basis else None,
        normalized=True,
        complex_matrix=False,
    )


def _npa_tau_problem(
    p: float,
    tp_pin: bool,
    class_relations: bool = True,
    n_outputs: int = 1,
) -> Problem:
    """Build the factored NPA re-expression of eq. (14).

    The relaxation is a standard NPA problem over the factored algebra
    ``{rho, sigma, v, J, W}`` with the maximally mixed state: every S-moment
    is pinned to its trace value ``Tr[word]/8``, the correlations pin
    ``<rho_x sigma_y J v_alpha> = p(alpha|x,y)/(8d)`` (eq. (10) with
    ``U_{x,y} = rho_x sigma_y``), and the inequalities ``J >= 0, W >= 0,
    W - J >= 0`` produce localizing matrices over the
    ``{1, rho, sigma, v} U {U_{x,y}}`` basis (26 words for one output,
    29 for four). The objective is ``<W>``, which equals
    ``Gamma^{(Z)}_{I,I}/8`` for ``d = 2``.

    Args:
        p: The channel parameter.
        tp_pin: Pin ``<J> = Tr(J_M x I)/8 = 1/4`` (trace-preserving Choi
            normalization).
        class_relations: Add the moment-equalities that encode the
            functional class structure of eq. (12)/(13) of the J- and
            W-localizing blocks as moment-matrix equalities. Without them
            the objective degenerates.
        n_outputs: The number of measurement outputs (1 or 4).

    Returns:
        The built ``ncpolopt.Problem``.
    """
    data = _factored_data(n_outputs)
    moments = _factored_moments(p, tp_pin, n_outputs)
    momentequalities: list[object] = []
    extras: list[object] | None = None
    localizing_basis: list[object] | None = None
    if class_relations:
        momentequalities, extras, localizing_basis = _factored_classes(tp_pin, n_outputs)
    return _assemble_npa_tau_problem(data, moments, momentequalities, extras, localizing_basis)


def npa_tau_relaxation_value(
    p: float,
    tp_pin: bool = False,
    solver: str = "cvxpy",
    class_relations: bool = True,
    n_outputs: int = 1,
) -> float:
    """Solve the factored NPA re-expression of eq. (14) at level 2.

    Args:
        p: The channel parameter.
        tp_pin: Pin the trace-preserving Choi normalization.
        solver: The solver kind to use (default "cvxpy").
        class_relations: Add the functional class relations (default
            True; disable to expose the degenerate form).
        n_outputs: The number of measurement outputs (1 or 4). For four
            outputs the relaxation is solved through the package's direct
            sparse CLARABEL backend (``solver="clarabel"``; the dense
            cvxpy conversion cannot hold the 199-word moment matrix);
            ``solver`` must then be a CLARABEL-backed kind.

    Returns:
        The optimal value (the certified fidelity lower bound).

    Raises:
        ValueError: If ``n_outputs`` is 4 and ``solver`` is not a
            CLARABEL-backed kind.
        RuntimeError: If the solver status is not "optimal".
    """
    problem = _npa_tau_problem(p, tp_pin, class_relations, n_outputs)
    if n_outputs > 1:
        if solver not in ("cvxpy", "CLARABEL", "clarabel"):
            raise ValueError(f"4-output solves require CLARABEL, got {solver!r}.")
        solver = "clarabel"
    solution = problem.solve(level=2, solver=solver)
    if solution.status != "optimal":
        raise RuntimeError(f"NPA-tau relaxation failed with status {solution.status!r}.")
    return float(solution.primal)
