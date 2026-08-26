"""Hierarchy tests: the Moroder, steering, and RDM relaxations.

The Moroder numerical test rebuilds the witness SDP (moment matrix, its
partial transpose, the X - Y + Z == 0 equality with Z[0, 0] == 1) from the
frozen relaxation and solves it with CVXPY; the old 0.1236 baseline was
dropped because the legacy test imposed a full transpose (a no-op on the
symmetric moment matrix) on an unbounded relaxation. A PICOS smoke test
keeps the duplicate-moment-matrix workflow covered (CVXOPT is unreliable
on this model and skips). The steering and RDM hierarchies have no
numerical baseline in the old suite; their construction invariants
(block sizes, variable counts, complex dtype, band positions) are pinned
instead.
"""

from __future__ import annotations

import numpy as np
import pytest
from sympy import Matrix
from sympy.physics.quantum.dagger import Dagger

from ncpolopt.expressions import flatten
from ncpolopt.hierarchies import MoroderHierarchy, RdmHierarchy, SteeringHierarchy
from ncpolopt.physics import Probability, define_objective_with_I
from ncpolopt.problem import Problem
from ncpolopt.variables import generate_operators


def _moment_matrix_data(sdp: object) -> tuple[np.ndarray, list[np.ndarray]]:
    """Reconstruct the moment matrix coefficients of the first block.

    The frozen SDP stores the block in COO form with the SDP variable
    number as row (0 = constant); the COO holds the upper triangle only,
    so every off-diagonal entry materializes its symmetric partner.

    Args:
        sdp: The frozen SdpProblem.

    Returns:
        The constant matrix and one coefficient matrix per SDP variable.
    """
    size = sdp.blocks[0].size
    n = sdp.n_vars
    coo = sdp.blocks[0].coo
    constant = np.zeros((size, size))
    matrices = [np.zeros((size, size)) for _ in range(n)]
    for k, pos, value in zip(coo.row, coo.col, coo.data, strict=True):
        i, j = divmod(int(pos), size)
        target = constant if k == 0 else matrices[int(k) - 1]
        target[i, j] += float(value)
        if i != j:
            target[j, i] += float(value)
    return constant, matrices


def _partial_transpose_perm(dim: int, subsystem: int) -> np.ndarray:
    """The vec-space permutation of a partial transpose.

    ``dim`` is the per-subsystem dimension of the 2 x dim block layout;
    ``subsystem`` selects the factor whose matrix entries are transposed
    (0 or 1). The permutation maps vec(M) to vec(M^T_subsystem): entry
    (i, j) of the selected factor's index pair swaps with its transposed
    partner.

    Args:
        dim: The per-subsystem dimension.
        subsystem: The factor to transpose, 0 or 1.

    Returns:
        The (dim**4, dim**4) permutation matrix.
    """
    size = dim * dim
    perm = np.zeros((size * size, size * size))
    for i in range(dim):
        for j in range(dim):
            for k in range(dim):
                for m in range(dim):
                    if subsystem == 1:
                        row = (i * dim + j) * size + k * dim + m
                        col = (i * dim + m) * size + k * dim + j
                    else:
                        row = (i * dim + j) * size + k * dim + m
                        col = (k * dim + j) * size + i * dim + m
                    perm[row, col] = 1.0
    return perm


def test_moroder_violation() -> None:
    """The Moroder-1 bound with the PPT witness on an I-inequality scenario.

    The old baseline 0.1236 is dropped: the legacy test imposed a
    no-argument ``partial_transpose()`` (a full transpose, which is a
    no-op on the symmetric moment matrix), solved an unbounded relaxation,
    and asserted whatever the unchecked solver returned. The genuine
    partial transpose (on the B subsystem) is solved here with two
    independent backends: CLARABEL -0.72842018, SCS -0.7247 (first-order
    accuracy). The relaxed value lies below the unconstrained problem,
    which is unbounded without the normalization of the witness.
    """
    pytest.importorskip("cvxpy")
    import cvxpy as cp

    I_matrix = [[0, -1, 0], [-1, 1, 1], [0, 1, -1]]
    P = Probability([2, 2], [2, 2])
    problem = Problem(
        [flatten(P.parties[0]), flatten(P.parties[1])],
        objective=define_objective_with_I(I_matrix, P),
        substitutions=P.substitutions,
        normalized=False,
    )
    relaxation = MoroderHierarchy(problem, 1)
    sdp = relaxation.sdp
    size = sdp.blocks[0].size
    assert size == 9
    constant, matrices = _moment_matrix_data(sdp)
    n = sdp.n_vars
    x = cp.Variable(n)
    y = cp.Variable(n)
    z = cp.Variable((size, size), symmetric=True)
    X = constant + sum(x[k] * matrices[k] for k in range(n))
    Y = constant + sum(y[k] * matrices[k] for k in range(n))
    pt = _partial_transpose_perm(3, 1)
    constraints = [
        X >> 0,
        cp.reshape(pt @ cp.vec(Y, order="F"), (size, size), order="F") >> 0,
        cp.reshape(pt @ cp.vec(z, order="F"), (size, size), order="F") >> 0,
        X - Y + z == 0,
        z[0, 0] == 1,
    ]
    prob = cp.Problem(cp.Minimize(sdp.obj @ x + sdp.constant_term), constraints)
    prob.solve(solver=cp.CLARABEL)
    assert prob.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE)
    assert abs(float(prob.value) - (-0.7284201767)) < 1e-3


def test_moroder_picos_witness() -> None:
    """The PICOS conversion exposes the duplicate matrix for the witness.

    The witness constraints are imposed after the conversion, on the
    duplicate moment matrix. CVXOPT is unreliable on this model (it
    returns ``unknown`` states or spurious optima), so the numerical
    assertion only runs when the solver converges.
    """
    pytest.importorskip("picos")
    import picos as pic

    from ncpolopt.solvers.picos_solver import convert_to_picos

    I_matrix = [[0, -1, 0], [-1, 1, 1], [0, 1, -1]]
    P = Probability([2, 2], [2, 2])
    problem = Problem(
        [flatten(P.parties[0]), flatten(P.parties[1])],
        objective=define_objective_with_I(I_matrix, P),
        substitutions=P.substitutions,
        normalized=False,
    )
    relaxation = MoroderHierarchy(problem, 1)
    size = relaxation.sdp.blocks[0].size
    assert size == 9
    model = convert_to_picos(relaxation.sdp, duplicate_moment_matrix=True)
    assert model.duplicate is not None
    X, Y = model.moment, model.duplicate
    Z = pic.SymmetricVariable("Z", size)
    P_problem = model.problem
    P_problem.add_constraint(Y.partial_transpose(1, [3, 3]) >> 0)
    P_problem.add_constraint(Z.partial_transpose(1, [3, 3]) >> 0)
    P_problem.add_constraint(X - Y + Z == 0)
    P_problem.add_constraint(Z[0, 0] == 1)
    P_problem.options.solver = "cvxopt"
    P_problem.options.verbosity = 0
    try:
        P_problem.solve()
    except pic.SolutionFailure:
        pytest.skip("CVXOPT does not converge on this model (unknown state)")
    assert abs(float(P_problem.value) - (-0.7284201767)) < 1e-2


def test_moroder_ppt_during_construction() -> None:
    """The ``ppt`` flag transposes the B subsystem while building."""
    X = generate_operators("x", 1, hermitian=True)
    Y = generate_operators("y", 1, hermitian=True)
    problem = Problem([X, Y], normalized=False)
    relaxation = MoroderHierarchy(problem, 1, ppt=True)
    assert relaxation.sdp.blocks[0].size == 4  # (1 + 1) x (1 + 1) basis


def test_moroder_needs_two_parties() -> None:
    """A single-party problem is rejected."""
    X = generate_operators("x", 2, hermitian=True)
    with pytest.raises(ValueError, match="two parties"):
        MoroderHierarchy(Problem(X), 1)


def test_steering_construction() -> None:
    """The steering hierarchy expands each moment into a d x d sub-block."""
    X = generate_operators("x", 2, hermitian=True)
    problem = Problem(X, normalized=False, complex_matrix=True)
    relaxation = SteeringHierarchy(problem, 1, matrix_var_dim=2)
    sdp = relaxation.sdp
    # The level-1 basis {1, x0, x1} has 6 upper-triangle moments; each
    # expands into a 2x2 group of 4 variables, and the block holds
    # 3 * 2 = 6 rows.
    assert sdp.blocks[0].size == 6
    assert sdp.n_vars == 24
    assert sdp.blocks[0].coo.dtype == np.complex128
    coo = sdp.blocks[0].coo
    positions = {
        (int(k), int(position)): value
        for k, position, value in zip(coo.row, coo.col, coo.data, strict=True)
    }
    # The x0 moment (group at sub-block (0, 1)): every position carries
    # the complex coefficient, the block is hermitian by construction.
    k = sdp.monomial_index[X[0]]
    assert positions[(k, 0 * 6 + 2)] == 1 + 1j
    assert positions[(k + 1, 0 * 6 + 3)] == 1 + 1j
    assert positions[(k + 2, 1 * 6 + 2)] == 1 + 1j
    assert positions[(k + 3, 1 * 6 + 3)] == 1 + 1j
    # The x0^2 group at the diagonal sub-block (1, 1): real on the
    # diagonal, complex off it, the lower-triangle position free.
    k = sdp.monomial_index[X[0] * X[0]]
    assert positions[(k, 2 * 6 + 2)] == 1
    assert positions[(k + 1, 2 * 6 + 3)] == 1 + 1j
    assert positions[(k + 3, 3 * 6 + 3)] == 1
    assert (k + 2, 3 * 6 + 2) not in positions


def test_steering_matrix_objective() -> None:
    """A matrix objective enters through the trace formula."""
    X = generate_operators("x", 2, hermitian=True)
    objective = Matrix([[X[0], 0], [0, 1]])
    problem = Problem(X, objective=objective, normalized=False, complex_matrix=True)
    relaxation = SteeringHierarchy(problem, 1, matrix_var_dim=2)
    sdp = relaxation.sdp
    # The constant matrix [[0, 0], [0, 1]] carries its 1 at sub-position
    # (1, 1) of the (0, 0) group: variable 1 + 1*2 + 1 = 4, whose objective
    # entry sits at obj[3] (freeze drops the facvar constant slot).
    assert sdp.obj[3] == 1.0
    k = sdp.monomial_index[X[0]]
    # X[0] sits at sub-position (0, 0) of its group, its first variable k;
    # the objective entry of variable k sits at obj[k - 1].
    assert sdp.obj[k - 1] == 1.0
    assert sdp.constant_term == 0.0


def test_steering_requires_complex_unnormalized() -> None:
    """The steering problem contract is enforced."""
    X = generate_operators("x", 2, hermitian=True)
    with pytest.raises(ValueError, match="normalized=False"):
        SteeringHierarchy(Problem(X, complex_matrix=True), 1, matrix_var_dim=2)
    with pytest.raises(ValueError, match="complex_matrix=True"):
        SteeringHierarchy(Problem(X, normalized=False), 1, matrix_var_dim=2)


def test_rdm_circulant_degree_one() -> None:
    """The degree-1 circulant layout is the complete second-moment matrix."""
    X = generate_operators("x", 3, hermitian=True)
    problem = Problem(X, normalized=False)
    relaxation = RdmHierarchy(problem, 1, circulant=True)
    sdp = relaxation.sdp
    # The level-1 basis {1, x0, x1, x2}: the normalization variable plus
    # the 9 upper-triangle moments.
    assert sdp.blocks[0].size == 4
    assert sdp.n_vars == 10
    # The old even-N layout dropped the cross-quadrant band pair (1, 2);
    # the complete layout keeps it.
    assert X[0] * X[1] in sdp.monomial_index


def test_rdm_circulant_degree_two() -> None:
    """The degree-2 circulant layout builds the banded fourth-moment matrix.

    The 3x3 grid of 4th-moment products splits into sub-blocks: each
    diagonal sub-block contributes its 6-position upper triangle, each
    off-diagonal sub-block its complete 9-position region.
    """
    X = generate_operators("x", 3, hermitian=True)
    products = [X[i] * X[j] for i in range(3) for j in range(3)]
    problem = Problem(X, normalized=False, extramonomials=[products])
    relaxation = RdmHierarchy(problem, -1, circulant=True)
    sdp = relaxation.sdp
    assert sdp.blocks[0].size == 9
    assert sdp.n_vars == 45  # 3 * 6 + 3 * 9 positions
    coo = sdp.blocks[0].coo
    positions = {
        (int(k), int(position)): value
        for k, position, value in zip(coo.row, coo.col, coo.data, strict=True)
    }
    # The off-diagonal sub-block (0, 1) covers its complete region: the
    # local-lower position (2, 3) carries the moment of the transposed
    # pair rather than zero. The dagger flips the factor order even for
    # hermitian variables (sympy keeps noncommuting factor order).
    transposed_pair = Dagger(products[2]) * products[3]
    assert positions[(sdp.monomial_index[transposed_pair], 2 * 9 + 3)]
    # The diagonal sub-block (0, 0) keeps its upper triangle only.
    assert all(
        (k, 1 * 9 + 0) not in positions for k in range(1, sdp.n_vars + 1)
    )


def test_rdm_circulant_two_band_layout() -> None:
    """The second block uses the two-band cross layout (old m_block 2)."""
    X = generate_operators("x", 3, hermitian=True)
    Y = generate_operators("y", 3, hermitian=True)
    first = [X[i] * X[j] for i in range(3) for j in range(3)]
    second = [X[i] * Y[j] for i in range(3) for j in range(3)]
    second += [Y[i] * Y[j] for i in range(3) for j in range(3)]
    problem = Problem(
        [X, Y], normalized=False, extramonomials=[first, second]
    )
    relaxation = RdmHierarchy(problem, -1, circulant=True)
    sdp = relaxation.sdp
    assert sdp.blocks[0].size == 9
    assert sdp.blocks[1].size == 18
    # Layout A on the first block (45 moments), the two bands (45 each)
    # plus the 9 x 9 cross rectangle (81) on the second.
    assert sdp.n_vars == 45 + 2 * 45 + 81


def test_rdm_circulant_basis_must_be_square() -> None:
    """A non-square degree-2 basis is rejected."""
    X = generate_operators("x", 2, hermitian=True)
    products = [X[i] * X[j] for i in range(2) for j in range(2)]
    problem = Problem(X, normalized=False, extramonomials=[products[:3]])
    with pytest.raises(ValueError, match="square basis"):
        RdmHierarchy(problem, -1, circulant=True)
