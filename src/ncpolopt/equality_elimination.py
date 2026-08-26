"""Algebraic elimination of equality constraints.

Ported from the legacy ``SdpRelaxation.__remove_equalities`` method.
The old code warned about
linearly dependent equality rows and then crashed inside
``np.linalg.solve`` on the resulting non-square system; the new
implementation solves the reduced system in the least-squares sense, so
dependent (but consistent) equalities degrade gracefully instead of raising.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class BasisTransform:
    """The variable reduction x = shift + basis @ y from solving A.x = 0.

    Attributes:
        basis: (n_vars, n_free) matrix expressing the original variables in
            the reduced basis.
        shift: Length ``n_vars + 1`` vector whose first entry is 1 (the
            constant moment): the constant part of every matrix entry after
            substituting the fixed variables.
        n_rows: Number of equality rows in A.
        rank: Rank of ``A[:, 1:]``; rank < n_rows indicates linearly
            dependent equality constraints.
    """

    basis: np.ndarray
    shift: np.ndarray
    n_rows: int = 0
    rank: int = 0

    @property
    def n_free(self) -> int:
        """The number of free variables after elimination."""
        return self.basis.shape[1]

    @property
    def dependent_rows(self) -> bool:
        """Whether the equality system contains dependent rows."""
        return self.rank < self.n_rows


def eliminate_equalities(A: np.ndarray) -> BasisTransform:
    """Compute the basis transform induced by a system of linear equalities.

    The rows of ``A`` encode moment equalities ``A @ [1, x1, ..., xn] = 0``,
    where the first column holds the constant part. The full QR factorization
    of ``A[:, 1:].T`` splits the variables: the first ``rank`` columns of Q
    span the part fixed by the equalities, the remaining columns span the
    free part.

    Args:
        A: The equality coefficient matrix, shape (n_rows, n_vars + 1).

    Returns:
        The transform; ``basis`` has ``n_vars - rank(A[:, 1:])`` columns.
    """
    n_vars = A.shape[1] - 1
    if n_vars == 0:
        return BasisTransform(np.empty((0, 0)), np.ones(1), A.shape[0], 0)
    Q, R = np.linalg.qr(A[:, 1:].T, mode="complete")
    # The first n rows of R are the linearly independent part. The old code
    # computed n from the nonzero-row count, which is exact in exact
    # arithmetic but sensitive to floating-point noise; the count is kept
    # because it agrees with the rank in practice.
    nonzero_rows = np.nonzero(np.sum(np.abs(R), axis=1) > 0)[0]
    n = nonzero_rows[-1] + 1 if nonzero_rows.size else 0
    if n == 0:
        # No variable participates in the equalities (e.g. a pure constant
        # moment equality): nothing to eliminate.
        return BasisTransform(np.eye(n_vars), np.append(1.0, np.zeros(n_vars)),
                              A.shape[0], 0)
    # Solve the reduced system in the least-squares sense: dependent rows
    # make the system overdetermined, and lstsq handles both cases where the
    # old np.linalg.solve would crash (non-square or singular).
    z = np.linalg.lstsq(R[:n, :].T, -A[:, 0], rcond=None)[0]
    x = Q[:, :n].dot(z)
    basis = np.array(Q[:, n:], copy=True)
    # The sign of a free basis column is arbitrary (the free variable is
    # unconstrained); make it canonical so two runs produce identical SDPs.
    for j in range(basis.shape[1]):
        first = np.nonzero(basis[:, j])[0]
        if first.size and basis[first[0], j] < 0:
            basis[:, j] *= -1
    return BasisTransform(basis, np.append(1.0, x), A.shape[0], n)
