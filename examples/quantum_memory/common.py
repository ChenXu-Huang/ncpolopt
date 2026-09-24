"""Shared building blocks for the quantum-memory verification scripts.

Implements the operators, channel and correlations of the document
``verification-of-memory.md`` (measurement-device-independent verification
of a quantum memory; the depolarizing-channel example), together with
three SDP constructions of its moment-matrix relaxation (eq. (14)):

* :func:`verbatim_relaxation_value` -- the relaxation assembled directly
  with cvxpy, one variable per functional class of the moment entries.
* :func:`operator_relaxation_value` -- eq. (14) as written: the entries
  are traces against the genuine Hermitian operators ``J`` and ``Z``.
* :func:`npa_tau_relaxation_value` -- the same relaxation re-expressed as
  a standard NPA problem on the extended algebra
  :math:`S \\cup \\{J, W\\}` with the maximally mixed state, solved
  through the public ncpolopt API (one output) or the package's direct
  sparse CLARABEL backend (four outputs).

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
    """Return the Choi fidelity ``<phi+|J_M|phi+>`` (eq. (4) for the
    identity extraction map).

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


def x_operators() -> list[np.ndarray]:
    """Return the A0-side projectors ``X_x = rho_x^T x I x I``.

    Returns:
        The four 8x8 matrices; ``U_{x,y} = X_x W_y`` and
        ``X_0 + X_1 = I`` (the input completeness relation).
    """
    return [
        np.kron(np.kron(input_projectors()[x].T, np.eye(DIM)), np.eye(DIM))
        for x in range(4)
    ]


def w_operators() -> list[np.ndarray]:
    """Return the B2-side projectors ``W_y = I x I x sigma_y``.

    Returns:
        The four 8x8 matrices; ``U_{x,y} = X_x W_y`` and
        ``W_0 + W_1 = I`` (the input completeness relation).
    """
    return [
        np.kron(np.kron(np.eye(DIM), np.eye(DIM)), input_projectors()[y])
        for y in range(4)
    ]


def build_words(
    us: Sequence[np.ndarray],
    vs: Sequence[np.ndarray],
    level: int,
    xs: Sequence[np.ndarray] = (),
    ws: Sequence[np.ndarray] = (),
) -> list[np.ndarray]:
    """Build the word set over S = {I} U {U} U {V} U {X} U {W} up to ``level``.

    Words are products of at most ``level`` elements of S, stored as
    explicit 8x8 matrices. Duplicate and zero words are dropped. The X/W
    words are the A0-/B2-side factors of the U operators
    (``U_{x,y} = X_x W_y``); including them makes the completeness
    relations (``X_0 + X_1 = I`` etc.) writable as entrywise linear
    constraints.

    Args:
        us: The U operators (16).
        vs: The V operators (1 or 4).
        level: The word length limit (1 or 2).
        xs: The X operators (4), appended after the V words.
        ws: The W operators (4), appended after the X words.

    Returns:
        The deduplicated word list; the identity word is the first entry.
    """
    words = [np.eye(8, dtype=complex)]
    words.extend(us)
    words.extend(vs)
    words.extend(xs)
    words.extend(ws)
    if level >= 2:
        # Products of two level-1 words only; iterating the live ``words``
        # list would re-append products of products without bound.
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
    # Keep only the diagonal pairs (b2, b2') of the traced-out subsystem.
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
    solver: str = "CLARABEL",
) -> float:
    """Solve the document's eq. (14) relaxation verbatim with cvxpy.

    The two moment matrices ``M = Gamma^{(J_M)}`` and ``L = Gamma^{(Z)}``
    are built over the word basis with one variable per functional class
    (entry ``M[u, v]`` depends only on ``G_{u,v} = Tr_{B2}(v u^dagger)``,
    ``L[u, v]`` only on ``H_{u,v} = Tr_{A0 B2}(v u^dagger)``); the
    correlations pin ``M[V_alpha, U_{x,y}] = p(alpha|x,y)/d``, and the
    constraints are ``M >= 0, L >= 0, L - M >= 0`` with objective
    ``L[I, I]/d^3``. The word set includes the A0-/B2-side factors
    ``X_x``/``W_y`` of the U operators, so the completeness relations are
    imposed entrywise on both blocks (the moments are linear in the word
    on either side):

    ``M[u, X_0] + M[u, X_1] = M[u, I]``,
    ``M[u, W_0] + M[u, W_1] = M[u, I]``,
    ``M[u, U_{0,y}] + M[u, U_{1,y}] = M[u, W_y]``,
    ``M[u, U_{x,0}] + M[u, U_{x,1}] = M[u, X_x]``,

    and ``sum_alpha M[u, V_alpha] = M[u, I]`` for the four-output
    measurement; likewise for ``L``. No trace-preserving pin is applied.

    Args:
        p: The channel parameter.
        n_outputs: Number of measurement outputs (1 or 4).
        level: 1 for the plain word set; >= 2 adds the Pauli product words
            (``X_x X_z``, ``W_y W_z``) and imposes the full complex
            Pauli-decomposition relations ``X_x X_z = sum_t c_t X_t``
            entrywise on both blocks.
        solver: A cvxpy solver name (default "CLARABEL").

    Returns:
        The optimal value (the certified fidelity lower bound).

    Raises:
        RuntimeError: If the solver status is neither "optimal" nor
            "optimal_inaccurate".
    """
    import cvxpy as cp

    words = build_words(
        u_operators(), v_operators(n_outputs), 1, x_operators(), w_operators()
    )
    if level >= 2:
        # Level 2 adds only the Pauli product words (X_x X_z and W_y W_z,
        # needed as the columns of the Pauli relations); the full level-2
        # product closure would canonicalize to tens of GB in cvxpy.
        for mats in (x_operators(), w_operators()):
            for a in range(4):
                for b in range(4):
                    word = mats[a] @ mats[b]
                    if np.linalg.norm(word) > 1e-12 and not any(
                        np.allclose(word, w, atol=1e-12) for w in words
                    ):
                        words.append(word)
    m = _class_expression(_functional_keys(words, partial_trace_b2))
    l_block = _class_expression(_functional_keys(words, partial_trace_a0b2))
    constraints: list[object] = [m >> 0, l_block >> 0, l_block - m >> 0]

    for alpha in range(n_outputs):
        for x in range(4):
            for y in range(4):
                i_u = 1 + 4 * x + y
                i_v = 1 + 16 + alpha
                constraints.append(m[i_v, i_u] == correlations_via_choi(p, x, y, alpha) / DIM)

    # Completeness relations, column-vectorized on both blocks (scalar
    # equalities would explode cvxpy's canonicalization). Word layout:
    # [I, U (16), V (n_outputs), X (4), W (4)].
    i_x = 1 + 16 + n_outputs
    i_w = i_x + 4
    for block in (m, l_block):
        constraints.append(block[:, i_x] + block[:, i_x + 1] == block[:, 0])
        constraints.append(block[:, i_w] + block[:, i_w + 1] == block[:, 0])
        for x in range(4):
            constraints.append(block[:, 1 + 4 * x] + block[:, 1 + 4 * x + 1] == block[:, i_x + x])
        for y in range(4):
            constraints.append(block[:, 1 + y] + block[:, 1 + 4 + y] == block[:, i_w + y])
        if n_outputs == 4:
            constraints.append(sum(block[:, 1 + 16 + a] for a in range(4)) == block[:, 0])

    # The full complex Pauli-decomposition relations between the degree-2
    # and degree-1 words, ``X_x X_z = sum_t c_t X_t`` (transposed product)
    # and ``W_y W_z = sum_t c_t W_t``, column-vectorized on both blocks.
    # Only available at level >= 2, where the products are words.
    if level >= 2:
        coeffs = _pauli_word_coefficients()

        def word_index(mat: np.ndarray) -> int:
            for i, word in enumerate(words):
                if np.allclose(word, mat, atol=1e-12):
                    return i
            raise RuntimeError("product word missing from the level-2 basis")

        x_mats, w_mats = x_operators(), w_operators()
        product_pairs: list[tuple[int, int, np.ndarray]] = []
        for x in range(4):
            for z in range(4):
                if x == z or (x, z) not in coeffs:
                    continue  # diagonal: idempotence is already in the classes
                # X_x X_z = (rho_z rho_x)^T = sum_t coeffs[(z, x)]_t X_t;
                # W_x W_z = sum_t coeffs[(x, z)]_t W_t (no transpose).
                product_pairs.append((word_index(x_mats[x] @ x_mats[z]), i_x, coeffs[(z, x)]))
                product_pairs.append((word_index(w_mats[x] @ w_mats[z]), i_w, coeffs[(x, z)]))
        for idx, base, c in product_pairs:
            for block in (m, l_block):
                constraints.append(
                    block[:, idx] == sum(complex(c[t]) * block[:, base + t] for t in range(4))
                )

    objective = cp.Minimize(cp.real(l_block[0, 0]) / DIM**3)
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=solver)
    # CLARABEL labels the complex-realified problems "optimal_inaccurate"
    # even when its own gap is ~1e-10; accept both statuses.
    if problem.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"Verbatim relaxation failed with status {problem.status!r}.")
    return float(problem.value)


def _realify(mat: np.ndarray) -> np.ndarray:
    """The doubled real symmetric form ``[[Re, -Im], [Im, Re]]`` of a
    Hermitian matrix (PSD iff the complex matrix is PSD)."""
    return np.block([[mat.real, -mat.imag], [mat.imag, mat.real]])


def operator_relaxation_value(
    p: float,
    n_outputs: int = 1,
    level: int = 2,
    solver: str | None = None,
) -> float:
    """Solve the document's eq. (14) relaxation in operator form with cvxpy.

    Unlike :func:`verbatim_relaxation_value` -- which relaxes the moment
    matrices to one free variable per functional class -- this keeps the
    genuine operator degrees of freedom: the entries are
    ``M[u, v] = Tr[G_{u,v} J]`` and ``L[u, v] = Tr[H_{u,v} Z]`` with the
    unknown Hermitian operators ``J`` (4x4) and ``Z`` (2x2) as the SDP
    variables (20 real degrees of freedom). ``M >= 0`` and ``L >= 0`` are
    implied by ``J >= 0`` and ``Z >= 0`` and are not imposed separately.

    The word set is the monomials of the document's operator set
    ``S = {I, U, V}``: level 1 is ``S`` itself; level 2 adds the degree-2
    monomials ``U_{x,y} V_alpha`` (the probe words).

    All complex Hermitian blocks are realified by hand (``[[Re, -Im], [Im,
    Re]]``), so the SDP is real; the correlation pins are equalities on the
    real parts (the pinned functionals are Hermitian, so the imaginary
    parts vanish automatically).

    Args:
        p: The channel parameter.
        n_outputs: Number of measurement outputs (1 or 4).
        level: 1 for the plain word set; >= 2 adds the probe words
            ``U_{x,y} V_alpha``.
        solver: A cvxpy solver name; defaults to MOSEK when installed
            (the 4-output level-2 realified blocks exceed CLARABEL's
            per-cone memory), else CLARABEL.

    Returns:
        The optimal value (the certified fidelity lower bound).

    Raises:
        RuntimeError: If the solver status is neither "optimal" nor
            "optimal_inaccurate".
    """
    import cvxpy as cp

    if solver is None:
        solver = "MOSEK" if "MOSEK" in cp.installed_solvers() else "CLARABEL"

    us = u_operators()
    vs = v_operators(n_outputs)
    words = [np.eye(8, dtype=complex), *us, *vs]
    if level >= 2:
        words.extend(u @ v for u in us for v in vs)
    n = len(words)
    dag = [w.T.conj() for w in words]
    g_mats = [[partial_trace_b2(words[j] @ dag[i]) for j in range(n)] for i in range(n)]
    h_mats = [[partial_trace_a0b2(words[j] @ dag[i]) for j in range(n)] for i in range(n)]

    j_basis = _hermitian_basis(4)
    z_basis = _hermitian_basis(2)
    t_j = cp.Variable(len(j_basis))
    t_z = cp.Variable(len(z_basis))

    # Coefficient matrices: M = sum_k t_j[k] * GM_k with
    # GM_k[i, j] = Tr[G_{ij} B_k]; likewise for L over the Z basis.
    gm = [np.array([[np.trace(g_mats[i][j] @ b) for j in range(n)] for i in range(n)]) for b in j_basis]
    hm = [np.array([[np.trace(h_mats[i][j] @ b) for j in range(n)] for i in range(n)]) for b in z_basis]
    gm = [(m + m.conj().T) / 2 for m in gm]  # exact Hermiticity
    hm = [(m + m.conj().T) / 2 for m in hm]

    d_real = sum(t_z[k] * _realify(hm[k]) for k in range(len(z_basis))) - sum(
        t_j[k] * _realify(gm[k]) for k in range(len(j_basis))
    )
    j_real = sum(t_j[k] * _realify(b) for k, b in enumerate(j_basis))
    z_real = sum(t_z[k] * _realify(b) for k, b in enumerate(z_basis))
    constraints: list[object] = [j_real >> 0, z_real >> 0, d_real >> 0]
    for alpha in range(n_outputs):
        for x in range(4):
            for y in range(4):
                i_u = 1 + 4 * x + y
                i_v = 1 + 16 + alpha
                pin = sum(t_j[k] * gm[k][i_v, i_u].real for k in range(len(j_basis)))
                constraints.append(pin == correlations_via_choi(p, x, y, alpha) / DIM)

    # H_{0,0} = Tr_{A0,B2}(I) = 4 I_2, so L[I, I] = 4 Tr(Z) and the
    # objective L[I, I]/d^3 = Tr(Z)/2, matching eq. (9).
    obj = sum(t_z[k] * hm[k][0, 0].real for k in range(len(z_basis))) / DIM**3
    problem = cp.Problem(cp.Minimize(obj), constraints)
    problem.solve(solver=solver)
    if problem.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"Operator relaxation failed with status {problem.status!r}.")
    return float(problem.value)


# --- The factored NPA-tau re-expression (Brown-style) -----------------------
#
# The U operators factor on H_{A0} x H_{A2} x H_{B2} as
# ``U_{x,y} = rho_x^T (x) I (x) sigma_y``, so the level-2 basis over the
# algebra {rho, sigma, v, J, W} (11 or 14 generators for 1 or 4 outputs)
# collapses under the one-direction commutation ``rho_i sigma_j ->
# sigma_j rho_i`` and the orthogonal-input rules. The degree-3 words
# ``g U_{x,y}`` (``g`` in {J, W}) are added as extramonomials so the
# U-pair entries ``<U^dagger g U'>`` exist in the moment matrix and the
# functional-class equalities of eqs. (12)/(13) become writable as block-0
# moment-equalities. The probe words ``rho_x sigma_y v_0`` (the degree-2
# monomials ``U_{x,y} V_0`` of the document's operator set) and their
# ``g``-products join them as extramonomials, extending the localizing
# basis to ``{1, rho, sigma, v} U {U_{x,y}} U {probe}``. Everything below
# is independent of the channel parameter ``p`` except the moment VALUES,
# so the relation set is computed once per configuration and cached.

_FACTORED_DATA: dict[int, dict[Any, Any]] = {}
_FACTORED_CLASSES: dict[tuple, tuple[list[object], list[object], list[object]]] = {}


def _factored_data(n_outputs: int = 1) -> dict[Any, Any]:
    """The factored-algebra operators, rules, basis and explicit matrices.

    The algebra is ``{rho, sigma, v, J, W}`` with ``n_outputs`` Bell
    measurement outputs ``v`` and the input set {|0>, |1>, |+>, |+i>}.
    Besides the idempotences ``x^2 -> x``, the mutual orthogonality of the
    measurement outputs (``v_a v_b -> 0`` for ``a != b``) and the
    one-direction commutation ``rho_i sigma_j -> sigma_j rho_i``, the
    rules include the orthogonality of the input states |0> and |1>:
    ``rho_0 rho_1 -> 0`` and ``sigma_0 sigma_1 -> 0`` in both orders.

    Args:
        n_outputs: The number of measurement outputs (1 or 4).

    Returns:
        A dict with the operator list, the substitution rules, the
        S-basis, the 8x8 S-generator matrices, the probe words
        ``rho_x sigma_y v_0`` (the degree-2 monomials ``U_{x,y} V_0`` of
        the document's operator set), and a helper mapping S-basis words
        to explicit matrices.

    Raises:
        ValueError: If ``n_outputs`` is not 1 or 4.
    """
    cached = _FACTORED_DATA.get(n_outputs)
    if cached is not None:
        return cached
    if n_outputs not in (1, 4):
        raise ValueError(f"n_outputs must be 1 or 4, got {n_outputs}.")
    from ncpolopt import generate_operators, get_all_monomials
    from ncpolopt.substitutions import apply_substitutions

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
    # sigma_0 sigma_1 = 0 in both orders; the zero words and their moments
    # drop out of the SDP.
    substitutions[rhos[0] * rhos[1]] = 0
    substitutions[rhos[1] * rhos[0]] = 0
    substitutions[sigmas[0] * sigmas[1]] = 0
    substitutions[sigmas[1] * sigmas[0]] = 0

    # Eq. (10) puts rho_x^T on the A0 side, and the transpose matters:
    # rho_3 = |+i><+i| is not symmetric. Without it the correlation pins
    # would force the J-insert moments to those of the non-PSD partial
    # transpose J_M^{T_A0}.
    rho_mats = [np.kron(np.kron(pj.T, np.eye(DIM)), np.eye(DIM)) for pj in input_projectors()]
    sigma_mats = [np.kron(np.kron(np.eye(DIM), np.eye(DIM)), pj) for pj in input_projectors()]
    v_mats = [np.kron(np.eye(DIM), phi) for phi in bell_projectors()[:n_outputs]]

    s_generators = [*rhos, *sigmas, *vs]
    s_mats: dict[object, np.ndarray] = dict(zip(rhos, rho_mats, strict=True))
    s_mats.update(zip(sigmas, sigma_mats, strict=True))
    s_mats.update(zip(vs, v_mats, strict=True))
    s_basis = get_all_monomials(s_generators, None, substitutions, 2)

    # The probe words rho_x sigma_y v_0 (canonically sigma_y rho_x v_0):
    # the factored image of the degree-2 monomials U_{x,y} V_0 of the
    # document's operator set S (eq. (11)), with V_0 the |phi+><phi+|
    # output. Only the V_0 probes are used: one probe per (x, y) keeps the
    # four-output moment matrix solvable.
    probe_words = [
        apply_substitutions(rhos[x] * sigmas[y] * vs[0], substitutions)
        for x in range(4)
        for y in range(4)
    ]

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
        "probe_words": probe_words,
        "vs": vs,
        "j": j,
        "w": w,
    }
    return _FACTORED_DATA[n_outputs]


def _factored_moments(p: float, n_outputs: int = 1) -> dict[object, object]:
    """Pin every S-moment and the correlations to their tau values.

    The S-moments are pinned to their trace values under the maximally
    mixed state by :func:`ncpolopt.hierarchies.trace_moment_pins`
    (symmetrized to their real part: with ``complex_matrix=False`` the
    consistent pin of a genuinely complex moment is
    ``(m + m.adjoint())/2 = Re Tr[word]/8``). No trace-preserving pin is
    applied.

    Args:
        p: The channel parameter.
        n_outputs: The number of measurement outputs (1 or 4).

    Returns:
        The moment substitution dict.
    """
    from ncpolopt.hierarchies import trace_moment_pins
    from ncpolopt.substitutions import apply_substitutions

    data = _factored_data(n_outputs)
    substitutions = data["substitutions"]
    moments: dict[object, object] = trace_moment_pins(
        [*data["s_basis"], *data["probe_words"]],
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
    return moments


def _factored_classes(
    n_outputs: int = 1,
    *,
    probes: bool = True,
    families: tuple[str, ...] = ("linearity",),
) -> tuple[list[object], list[object], list[object]]:
    """The p-independent class equalities, extramonomials and localizing basis.

    The relation set depends only on the moment KEYS, which do not change
    with ``p``, so it is computed once per configuration and cached. The
    class equalities are generated from a draft build of the relaxation at
    ``p = 0.5`` (the draft's block-0 layout is final: the equality blocks
    are appended after the moment blocks).

    Args:
        n_outputs: The number of measurement outputs (1 or 4).
        probes: Add the probe words ``rho_x sigma_y v_0`` (the
            degree-2 monomials ``U_{x,y} V_0`` of the document's
            operator set) to the localizing basis, with the words and
            their ``g``-products as extramonomials.
        families: The moment-equality families to generate:
            ``"linearity"`` (the functional-linearity relations of
            :func:`_functional_linearity_equalities`), ``"classes"``
            (the functional-class equalities of ``class_moment_equalities``),
            ``"completeness"`` (the ``rho_0 + rho_1 = I``-type word
            relations), and ``"pauli"`` (the Pauli anticommutator
            relations). The default keeps only ``"linearity"`` (the
            smallest, linearly independent set).

    Returns:
        The triple ``(momentequalities, extramonomials, localizing_basis)``.
    """
    cache_key = (n_outputs, probes, families)
    cached = _FACTORED_CLASSES.get(cache_key)
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
    probe_words = list(data["probe_words"]) if probes else []
    extras = [
        apply_substitutions(g * u_word, substitutions)
        for g in (j, w)
        for u_word in u_words
    ]
    # The probe words (degree 3) and their g-products (degree 4) must be
    # extramonomials: the level-2 basis only spans degrees <= 2. As
    # extramonomials the probes exist as block-0 rows/columns (their
    # S-moments are trace-pinned) and the g-products provide the columns
    # for the localizing entries <probe^dagger g probe'> (degree 7).
    extras += probe_words
    extras += [
        apply_substitutions(g * probe, substitutions)
        for g in (j, w)
        for probe in probe_words
    ]
    # NOTE: the completeness relations' term monomials ``u^dagger g rho_x``
    # etc. need no generator-level extras: ``g.rho_x`` (degree 2) is already
    # in the level-2 word set.
    if "pauli" in families:
        # The Pauli anticommutator relations need the degree-3 words
        # ``g.(rho_x rho_z)`` / ``g.(sigma_y sigma_z)`` as extramonomials.
        pauli_specs, pauli_words = _pauli_relation_specs(operators)
        extras += [
            apply_substitutions(g * word, substitutions)
            for g in (j, w)
            for word in pauli_words
        ]
    # The localizing matrices run over the {1, rho, sigma, v} U {U_{x,y}}
    # U {probe} basis: the degree-2 U words and degree-3 probe words are
    # admissible because their ``g`` products are the extramonomials, so
    # every localizing entry monomial ``u^dagger g w`` exists as a
    # moment-matrix entry.
    localizing_basis = [S.One, *rhos, *sigmas, *vs, *u_words, *probe_words]
    row_words = [*data["s_basis"], *probe_words]

    draft_problem = _assemble_npa_tau_problem(
        data,
        _factored_moments(0.5, n_outputs),
        momentequalities=[],
        extras=extras,
        localizing_basis=localizing_basis,
    )
    resolver = _moment_resolver(draft_problem)
    momentequalities: list[object] = []
    if "classes" in families:
        momentequalities += class_moment_equalities(
            draft_problem,
            2,
            inserted=((j, partial_trace_b2), (w, partial_trace_a0b2)),
            row_words=row_words,
            col_words=localizing_basis,
            word_matrix=data["word_matrix"],
        )
    if "completeness" in families:
        momentequalities += _completeness_equalities(
            draft_problem, data, localizing_basis, n_outputs, resolver=resolver
        )
    if "linearity" in families:
        momentequalities += _functional_linearity_equalities(
            draft_problem, data, row_words, localizing_basis, resolver=resolver
        )
    if "pauli" in families:
        momentequalities += _pauli_equalities(
            draft_problem, data, localizing_basis, pauli_specs, resolver=resolver
        )

    cached = (momentequalities, extras, localizing_basis)
    _FACTORED_CLASSES[cache_key] = cached
    return cached


def _moment_resolver(
    problem: Problem,
) -> Callable[[object], tuple[int, int] | float | None]:
    """The block-0 position/constant resolver of a draft build of ``problem``.

    Args:
        problem: The draft problem (final pins, extras and localizing
            monomials, without moment equalities).

    Returns:
        A function mapping a moment monomial to a ``(row, col)`` tuple
        (its block-0 SDP variable position), a float (a pinned or
        vanishing moment's value), or None when the monomial has no
        moment-matrix occurrence.

    Raises:
        RuntimeError: If a moment's variable was created outside block 0.
    """
    from ncpolopt.relaxation import NpaRelaxation
    from ncpolopt.substitutions import apply_substitutions

    substitutions = dict(problem.substitutions) if problem.substitutions is not None else {}
    pins = problem.momentsubstitutions or {}
    draft = NpaRelaxation(problem, 2)
    block0 = draft.moment_block_indices[0]

    def resolve(monomial: object) -> tuple[int, int] | float | None:
        """The block-0 position or pinned value of a moment; None if absent.

        A ``(row, col)`` tuple locates the moment's SDP variable in block
        0; a float is the pinned (or vanishing) value. NOTE: the returned
        position may be the creation position of the monomial's adjoint;
        in a real relaxation both resolve to the same variable.
        """
        monomial = apply_substitutions(monomial, substitutions)
        if monomial == 0:
            return 0.0
        if monomial in pins:
            return float(np.real(pins[monomial]))
        adjoint = apply_substitutions(monomial.adjoint(), substitutions)
        if adjoint in pins:
            return float(np.real(np.conj(pins[adjoint])))
        k = draft.monomial_index.get(monomial)
        if k is None:
            k = draft.monomial_index.get(adjoint)
        if k is None:
            return None
        block, r, c = draft.sdp.column_locations[k]
        if block != block0:
            raise RuntimeError(f"monomial {monomial} created in block {block}, not block {block0}")
        return (r, c)

    return resolve


def _linear_span_equalities(
    problem: Problem,
    data: dict[Any, Any],
    localizing_basis: list[object],
    relations: list[list[tuple[float, object]]],
    label: str,
    resolver: Callable[[object], tuple[int, int] | float | None] | None = None,
) -> list[object]:
    """Linear moment equalities lifting linear word relations into the
    inserted (J/W) blocks.

    Each relation is a list of ``(coefficient, factor)`` pairs -- with
    ``factor`` a word over the S-generators, or ``None`` for no factor --
    encoding the matrix-model identity ``sum(coeff * factor) = 0``. For
    every word pair ``(u, w)`` and inserted operator ``g`` in ``{J, W}``
    the moments then obey ``sum(coeff * <u^dagger g factor w>) = 0``.
    Plain polynomial ``equalities`` cannot express this: they only
    constrain block-0 S-moments, which are already trace-pinned (and
    traces satisfy the linear relations automatically). Only pairs
    ``(u, w)`` whose term monomials all occur in the moment matrix
    (block 0) produce a relation.

    Args:
        problem: The draft problem (final pins, extras and localizing
            monomials, without moment equalities).
        data: The factored-algebra data of :func:`_factored_data`.
        localizing_basis: The column words of the localizing blocks.
        relations: The relation specs as ``(coefficient, factor)`` lists.
        label: A tag identifying the relation family in error messages.
        resolver: The block-0 resolver of :func:`_moment_resolver`; built
            from ``problem`` when not given. Pass a shared resolver when
            generating several relation families for the same problem.

    Returns:
        The moment-equality list.

    Raises:
        RuntimeError: If a fully-constant relation is inconsistent with
            the pins.
    """
    from ncpolopt.moment import MomentEntry, MomentExpr

    j, w_op = data["j"], data["w"]
    resolve = resolver if resolver is not None else _moment_resolver(problem)

    equalities: list[object] = []
    for gi, g in enumerate((j, w_op)):
        for ri, rel in enumerate(relations):
            for u in data["s_basis"]:
                ua = u.adjoint()
                for v in localizing_basis:
                    terms: list[object] = []
                    constant = 0.0
                    missing = False
                    for coeff, factor in rel:
                        word = ua * g * v if factor is None else ua * g * factor * v
                        resolved = resolve(word)
                        if resolved is None:
                            missing = True
                            break
                        if isinstance(resolved, tuple):
                            terms.append(MomentEntry(0, resolved[0], resolved[1], coeff))
                        else:
                            constant += coeff * resolved
                    if missing:
                        continue
                    expr = MomentExpr(tuple(terms))
                    if not expr.terms:
                        # Fully constant relation: a consistency check.
                        if abs(constant) > 1e-9:
                            raise RuntimeError(
                                f"inconsistent constant {label} relation "
                                f"(g index {gi}, relation {ri}, u={u}, v={v}): {constant}"
                            )
                        continue
                    if constant != 0.0:
                        terms.append(MomentEntry(coefficient=constant))
                    equalities.append(MomentExpr(tuple(terms)))
    return equalities


def _hermitian_basis(dim: int) -> list[np.ndarray]:
    """A real basis of the ``dim x dim`` Hermitian matrices (``dim**2``
    elements): diagonal units, symmetric and antisymmetric-imaginary
    off-diagonals.

    Args:
        dim: The matrix dimension.

    Returns:
        The basis matrices.
    """
    basis = []
    for a in range(dim):
        mat = np.zeros((dim, dim), dtype=complex)
        mat[a, a] = 1.0
        basis.append(mat)
    for a in range(dim):
        for b in range(a + 1, dim):
            re = np.zeros((dim, dim), dtype=complex)
            re[a, b] = re[b, a] = 1.0
            basis.append(re)
            im = np.zeros((dim, dim), dtype=complex)
            im[a, b] = 1j
            im[b, a] = -1j
            basis.append(im)
    return basis


def _functional_linearity_equalities(
    problem: Problem,
    data: dict[Any, Any],
    row_words: list[object],
    col_words: list[object],
    resolver: Callable[[object], tuple[int, int] | float | None] | None = None,
) -> list[object]:
    """Moment equalities enforcing the functional linearity of the
    inserted (J/W) blocks.

    In the matrix model the inserted-block moment ``<u^dagger g v>``
    equals ``Tr[G_{u,v} T_g]`` with the functional ``G_{u,v} =
    trace_fn(v u^dagger)`` and ``T_g`` the inserted Hermitian operator,
    so the moments are linear in the functional. The relaxation is real,
    so the relations are taken over the real coordinates of the Hermitian
    parts of the functionals (``Re <u^dagger g v> = Tr[(G + G^dagger)/2
    T]``; the coordinate basis is :func:`_hermitian_basis`).

    For each inserted operator the grid of ``(u, v)`` moments is grouped
    by the SDP variable (or pinned constant) it resolves to, a greedy
    pivot set of independent functional vectors is chosen, and every
    other group's relation ``m = sum_a A_a m_a`` is emitted as one moment
    equality. This subsumes the class equalities on the grid (equal
    functionals get equal expansions) but not the word-relation lifts of
    :func:`_completeness_equalities`/:func:`_pauli_equalities`, which
    constrain moments whose column words are products rather than basis
    words.

    Args:
        problem: The draft problem (final pins, extras and localizing
            monomials, without moment equalities).
        data: The factored-algebra data of :func:`_factored_data`.
        row_words: The row words of the class grid (the S-basis plus the
            probe words).
        col_words: The column words (the localizing basis).
        resolver: The block-0 resolver of :func:`_moment_resolver`; built
            from ``problem`` when not given.

    Returns:
        The moment-equality list.

    Raises:
        RuntimeError: If a group of moments sharing one SDP variable has
            inconsistent functionals, or a fully-constant relation is
            inconsistent with the pins.
    """
    from ncpolopt.moment import MomentEntry, MomentExpr
    from ncpolopt.substitutions import apply_substitutions

    resolve = resolver if resolver is not None else _moment_resolver(problem)
    word_matrix = data["word_matrix"]
    substitutions = data["substitutions"]
    j, w_op = data["j"], data["w"]

    def push(
        resolved: tuple[int, int] | float,
        coeff: float,
        terms: list[object],
        constant: list[float],
    ) -> None:
        """Append ``coeff * moment(resolved)`` to the relation in progress."""
        if abs(coeff) < 1e-12:
            return
        if isinstance(resolved, tuple):
            terms.append(MomentEntry(0, resolved[0], resolved[1], float(coeff)))
        else:
            constant[0] += coeff * resolved

    equalities: list[object] = []
    for gi, (g, trace_fn) in enumerate(((j, partial_trace_b2), (w_op, partial_trace_a0b2))):
        # Group the grid entries by their resolved SDP target: the variable
        # position for free moments, the canonical monomial for pinned or
        # vanishing constants (distinct pinned moments may share a VALUE
        # while carrying different functionals, so they must not be
        # merged).
        groups: dict[object, dict[str, Any]] = {}
        basis: list[np.ndarray] | None = None
        for u in row_words:
            mu_dag = word_matrix(u).T.conj()
            for v in col_words:
                g_mat = trace_fn(word_matrix(v) @ mu_dag)
                if basis is None:
                    basis = _hermitian_basis(g_mat.shape[0])
                fvec = np.array([np.trace(g_mat @ b).real for b in basis])
                monomial = apply_substitutions(u.adjoint() * g * v, substitutions)
                resolved = resolve(monomial)
                if resolved is None:
                    continue
                key = resolved if isinstance(resolved, tuple) else ("const", monomial)
                stored = groups.get(key)
                if stored is None:
                    groups[key] = {"f": fvec, "resolved": resolved}
                elif np.max(np.abs(stored["f"] - fvec)) > 1e-7:
                    raise RuntimeError(
                        f"functionally inconsistent moments tied to one variable "
                        f"(g index {gi}, u={u}, v={v}): {stored['f']} vs {fvec}"
                    )
        # Greedy pivots: every non-pivot group gets the relation
        # m_group = sum_a A_a m_pivot_a, exact in exact arithmetic.
        pivot_keys: list[object] = []
        pivot_fvecs: list[np.ndarray] = []
        relations: list[tuple[object, np.ndarray]] = []
        for key, group in sorted(groups.items(), key=lambda item: str(item[0])):
            fvec = group["f"]
            if pivot_fvecs:
                stacked = np.array(pivot_fvecs).T
                coeffs = np.linalg.lstsq(stacked, fvec, rcond=None)[0]
                if np.max(np.abs(stacked @ coeffs - fvec)) < 1e-8:
                    relations.append((key, coeffs))
                    continue
            pivot_keys.append(key)
            pivot_fvecs.append(fvec)
        for key, coeffs in relations:
            terms: list[object] = []
            constant = [0.0]
            push(groups[key]["resolved"], 1.0, terms, constant)
            # The coefficients align with the pivots chosen so far; pivots
            # added later do not participate in this relation.
            for coeff, pivot_key in zip(coeffs, pivot_keys[: len(coeffs)], strict=True):
                push(groups[pivot_key]["resolved"], -float(coeff), terms, constant)
            if not terms:
                # Fully constant relation: a consistency check.
                if abs(constant[0]) > 1e-8:
                    raise RuntimeError(
                        f"inconsistent constant linearity relation (g index {gi}): "
                        f"{constant[0]}"
                    )
                continue
            if abs(constant[0]) > 0:
                terms.append(MomentEntry(coefficient=float(constant[0])))
            equalities.append(MomentExpr(tuple(terms)))
    return equalities


def _completeness_equalities(
    problem: Problem,
    data: dict[Any, Any],
    localizing_basis: list[object],
    n_outputs: int,
    resolver: Callable[[object], tuple[int, int] | float | None] | None = None,
) -> list[object]:
    """Linear moment equalities lifting the generator completeness
    relations into the inserted (J/W) blocks.

    The input projectors satisfy ``rho_0 + rho_1 = I`` (likewise
    ``sigma_0 + sigma_1 = I``), and the four-output Bell measurement
    satisfies ``sum_alpha v_alpha = I``, so for every word pair ``(u, w)``
    and inserted operator ``g`` in ``{J, W}`` the moments obey

    ``<u^dagger g rho_0 w> + <u^dagger g rho_1 w> = <u^dagger g w>``

    and analogously for sigma and v.

    Args:
        problem: The draft problem (final pins, extras and localizing
            monomials, without moment equalities).
        data: The factored-algebra data of :func:`_factored_data`.
        localizing_basis: The column words of the localizing blocks.
        n_outputs: The number of measurement outputs (1 or 4).
        resolver: The block-0 resolver of :func:`_moment_resolver`; built
            from ``problem`` when not given.

    Returns:
        The moment-equality list.
    """
    operators = data["operators"]
    rhos, sigmas = operators[:4], operators[4:8]
    relations: list[list[tuple[float, object]]] = [
        [(1.0, rhos[0]), (1.0, rhos[1]), (-1.0, None)],
        [(1.0, sigmas[0]), (1.0, sigmas[1]), (-1.0, None)],
    ]
    if n_outputs == 4:
        relations.append([(1.0, v) for v in data["vs"]] + [(-1.0, None)])
    return _linear_span_equalities(
        problem, data, localizing_basis, relations, "completeness", resolver=resolver
    )


def _pauli_word_coefficients() -> dict[tuple[int, int], np.ndarray]:
    """The complex coefficients of ``rho_a rho_b = sum_t c_t rho_t``.

    The four input projectors form a complex basis of the 2x2 matrices
    (the Pauli decomposition ``|0><0| = (I+Z)/2``, ``|+><+| = (I+X)/2``
    etc.), so every degree-2 word is a unique complex combination of the
    degree-1 words.

    Returns:
        The coefficient map ``(a, b) -> c`` with ``c[t]`` the coefficient
        of ``rho_t``, for all ordered pairs with nonzero product.

    Raises:
        RuntimeError: If the expansion residual exceeds 1e-9.
    """
    mats = input_projectors()
    basis = np.stack([m.reshape(-1) for m in mats], axis=1)
    coeffs: dict[tuple[int, int], np.ndarray] = {}
    for a in range(4):
        for b in range(4):
            product = mats[a] @ mats[b]
            if np.linalg.norm(product) < 1e-12:
                continue  # the orthogonal pair (0, 1) is identically zero
            c = np.linalg.lstsq(basis, product.reshape(-1), rcond=None)[0]
            residual = np.max(np.abs(basis @ c - product.reshape(-1)))
            if residual > 1e-9:
                raise RuntimeError(f"no Pauli expansion for ({a}, {b}): residual {residual}")
            coeffs[(a, b)] = np.round(c, 9)
    return coeffs


def _pauli_relation_specs(
    operators: list[object],
) -> tuple[list[list[tuple[float, object]]], list[object]]:
    """The anticommutator relations of the input projectors.

    The four input projectors form a complex basis of the 2x2 matrices, so
    every degree-2 word is a linear combination of degree-1 words:
    ``rho_x rho_z = sum_t c_t rho_t`` (the Pauli decomposition
    ``|0><0| = (I+Z)/2`` etc.). The coefficients are genuinely complex, so
    the full relations cannot live in the real (``complex_matrix=False``)
    relaxation; their symmetric parts -- the anticommutators
    ``{rho_x, rho_z} = sum_t a_t rho_t`` with real ``a_t`` -- can: the
    real moment variables are real parts of the complex moments, and
    ``Re`` of the anticommutator identity yields
    ``m(u g rho_x rho_z w) + m(u g rho_z rho_x w) = sum_t a_t m(u g rho_t w)``.

    Args:
        operators: The operator list (rho_0..3, sigma_0..4, ...).

    Returns:
        The relation specs (``(coefficient, factor)`` lists over the
        rho/sigma generators) and the degree-2 words whose ``g`` products
        must be extramonomials for the terms to exist.
    """
    mats = input_projectors()
    basis = np.stack([m.reshape(-1) for m in mats], axis=1)
    specs: list[list[tuple[float, object]]] = []
    extra_words: list[object] = []
    rhos, sigmas = operators[:4], operators[4:8]
    for x in range(4):
        for z in range(x + 1, 4):
            if {x, z} == {0, 1}:
                continue  # orthogonal pair: the anticommutator vanishes
            anticommutator = mats[x] @ mats[z] + mats[z] @ mats[x]
            a = np.linalg.lstsq(basis, anticommutator.reshape(-1), rcond=None)[0]
            residual = np.max(np.abs(basis @ a - anticommutator.reshape(-1)))
            if residual > 1e-9 or np.max(np.abs(a.imag)) > 1e-9:
                raise RuntimeError(
                    f"no real anticommutator relation for ({x}, {z}): residual {residual}"
                )
            a = np.round(a.real, 9)
            for gens in (rhos, sigmas):
                spec: list[tuple[float, object]] = [
                    (1.0, gens[x] * gens[z]),
                    (1.0, gens[z] * gens[x]),
                ]
                spec += [(-float(a[t]), gens[t]) for t in range(4) if a[t] != 0]
                specs.append(spec)
            extra_words += [rhos[x] * rhos[z], rhos[z] * rhos[x]]
            extra_words += [sigmas[x] * sigmas[z], sigmas[z] * sigmas[x]]
    return specs, extra_words


def _pauli_equalities(
    problem: Problem,
    data: dict[Any, Any],
    localizing_basis: list[object],
    specs: list[list[tuple[float, object]]],
    resolver: Callable[[object], tuple[int, int] | float | None] | None = None,
) -> list[object]:
    """Linear moment equalities lifting the Pauli-decomposition
    anticommutator relations into the inserted (J/W) blocks.

    Args:
        problem: The draft problem (final pins, extras and localizing
            monomials, without moment equalities).
        data: The factored-algebra data of :func:`_factored_data`.
        localizing_basis: The column words of the localizing blocks.
        specs: The relation specs of :func:`_pauli_relation_specs`.
        resolver: The block-0 resolver of :func:`_moment_resolver`; built
            from ``problem`` when not given.

    Returns:
        The moment-equality list.
    """
    return _linear_span_equalities(
        problem, data, localizing_basis, specs, "pauli", resolver=resolver
    )


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
        extras: The extramonomials.
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
    class_relations: bool = True,
    n_outputs: int = 1,
    probes: bool = True,
    families: tuple[str, ...] = ("linearity",),
) -> Problem:
    """Build the factored NPA re-expression of eq. (14).

    The relaxation is a standard NPA problem over the factored algebra
    ``{rho, sigma, v, J, W}`` with the maximally mixed state: every S-moment
    is pinned to its trace value ``Tr[word]/8``, the correlations pin
    ``<rho_x sigma_y J v_alpha> = p(alpha|x,y)/(8d)`` (eq. (10) with
    ``U_{x,y} = rho_x sigma_y``), and the inequalities ``J >= 0, W >= 0,
    W - J >= 0`` produce localizing matrices over the
    ``{1, rho, sigma, v} U {U_{x,y}} U {probe}`` basis. The objective is
    ``<W>``, which equals ``Gamma^{(Z)}_{I,I}/8`` for ``d = 2``.

    Args:
        p: The channel parameter.
        class_relations: Add the moment-equalities that encode the
            functional class structure of eq. (12)/(13) of the J- and
            W-localizing blocks. Without them the objective degenerates.
        n_outputs: The number of measurement outputs (1 or 4).
        probes: Add the probe words ``rho_x sigma_y v_0`` to the
            localizing basis (see :func:`_factored_classes`).
        families: The moment-equality families (see
            :func:`_factored_classes`).

    Returns:
        The built ``ncpolopt.Problem``.
    """
    data = _factored_data(n_outputs)
    moments = _factored_moments(p, n_outputs)
    momentequalities: list[object] = []
    extras: list[object] | None = None
    localizing_basis: list[object] | None = None
    if class_relations:
        momentequalities, extras, localizing_basis = _factored_classes(
            n_outputs, probes=probes, families=families
        )
    return _assemble_npa_tau_problem(data, moments, momentequalities, extras, localizing_basis)


def npa_tau_relaxation_value(
    p: float,
    solver: str = "clarabel",
    class_relations: bool = True,
    n_outputs: int = 1,
    probes: bool = True,
    families: tuple[str, ...] = ("linearity",),
    solver_options: dict[str, Any] | None = None,
) -> float:
    """Solve the factored NPA re-expression of eq. (14) at level 2.

    Args:
        p: The channel parameter.
        solver: The solver kind to use. Default and recommended is the
            package's direct sparse CLARABEL backend: the dense cvxpy
            canonicalization of the moment matrix does not fit in
            memory. The MOSEK backend (``solver="mosek"``) is faster
            when a license is available.
        class_relations: Add the moment-equality families (default True;
            disable to expose the degenerate form).
        n_outputs: The number of measurement outputs (1 or 4).
        probes: Add the probe words ``rho_x sigma_y v_0`` (the
            degree-2 monomials ``U_{x,y} V_0`` of the document's
            operator set) to the localizing basis.
        families: The moment-equality families (see
            :func:`_factored_classes`).
        solver_options: Backend knobs forwarded through
            ``SolverSettings.solver_options`` (e.g. CLARABEL's
            ``presolve_enable``).

    Returns:
        The optimal value (the certified fidelity lower bound).

    Raises:
        ValueError: If ``n_outputs`` is 4 and ``solver`` is not a direct
            sparse backend.
        RuntimeError: If the solver status is not "optimal".
    """
    from ncpolopt.solvers.base import SolverSettings

    problem = _npa_tau_problem(p, class_relations, n_outputs, probes, families)
    if n_outputs > 1:
        # NOTE: "cvxpy"/"CLARABEL" are routed to the direct sparse CLARABEL
        # backend, not the dense cvxpy conversion.
        direct = {"cvxpy": "clarabel", "CLARABEL": "clarabel", "clarabel": "clarabel",
                  "mosek": "mosek", "MOSEK": "mosek"}
        if solver not in direct:
            raise ValueError(
                "4-output solves require a direct sparse backend "
                f"('clarabel' or 'mosek'), got {solver!r}."
            )
        solver = direct[solver]
    settings = SolverSettings(solver_options=dict(solver_options or {}))
    solution = problem.solve(level=2, solver=solver, settings=settings)
    if solution.status != "optimal":
        raise RuntimeError(f"NPA-tau relaxation failed with status {solution.status!r}.")
    return float(solution.primal)
