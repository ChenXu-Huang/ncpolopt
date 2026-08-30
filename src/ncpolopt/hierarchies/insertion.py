"""Operator-insertion relaxations from an explicit matrix model.

In a moment-insertion relaxation (the Rosset-Buscemi-Liang MDI family,
PRX 8, 021033 (2018)) the moment matrix of a word set ``S`` carries
inserted positive operators -- the Choi state ``J`` of the channel under
test and a bound ``W >= J`` -- and every entry is a linear functional of
the inserted operator: ``Gamma^{(g)}[u, v] = F_g(G_{u,v})`` with
``G_{u,v} = Tr_env(v u^dagger)``. Two entries sharing the functional
``G`` share a moment, which is what keeps the relaxation non-degenerate.

Given an explicit matrix realization of the word algebra (a "matrix
model"), this module generates the two ingredients of that construction
as ordinary :class:`~ncpolopt.problem.Problem` fields:

- :func:`trace_moment_pins` pins the moments over a basis to their trace
  values ``Tr(u^dagger v rho)`` (``momentsubstitutions``);
- :func:`class_moment_equalities` emits the functional-class equalities
  of the inserted-operator localizing blocks as block-0
  ``momentequalities``. The variable positions come from a draft build of
  the problem itself (the equality blocks are appended after the moment
  blocks, so block-0 positions are final), never from a re-simulation of
  the builder's variable-creation order.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
from sympy import S

from ..moment import MomentEntry, MomentExpr
from ..relaxation import NpaRelaxation
from ..substitutions import apply_substitutions

#: A functional-class key: the rounded real and imaginary parts of the
#: flattened functional matrix ``G_{u,v} = trace_fn(v u^dagger)``.
_FunctionalKey = tuple[tuple[float, ...], tuple[float, ...]]


def trace_moment_pins(
    basis: Sequence[Any],
    word_matrix: Callable[[Any], np.ndarray],
    *,
    substitutions: dict[Any, Any] | None = None,
    state: np.ndarray | None = None,
    force_real: bool = False,
    tol: float = 1e-9,
) -> dict[Any, Any]:
    """Pin the moments over a basis to their trace values in a matrix model.

    For every pair ``(u, v)`` of basis words the moment ``<u^dagger v>``
    is pinned to ``Tr(u^dagger v rho)``, where ``rho`` defaults to the
    maximally mixed state on the model's Hilbert space. Moments that
    reduce to 0 or 1 under the substitution rules are skipped (the
    normalization and the zero words need no pin).

    Args:
        basis: The basis words (typically the level-1/2 monomials of the
            operator set without the inserted operators).
        word_matrix: Maps a word to its explicit matrix; the identity
            word must map to the identity matrix.
        substitutions: The word substitution rules of the problem.
        state: The state ``rho`` of the trace; defaults to ``I / dim``
            with ``dim`` read off the identity word's matrix.
        force_real: Keep only the real part of each pin. With
            ``complex_matrix=False`` the moment matrix is real and the
            consistent pin is the symmetrized moment
            ``(m + m.adjoint()) / 2 = Re Tr(u^dagger v rho)`` -- moments
            of a non-self-adjoint word are genuinely complex, but the
            real part of the true quantum moment matrix stays feasible,
            so the relaxation remains a valid bound.
        tol: Revisit tolerance; two pairs reducing to the same key must
            agree within it.

    Returns:
        The ``momentsubstitutions`` dict for :class:`Problem`.

    Raises:
        ValueError: If two basis pairs reduce to the same moment key with
            values differing by more than ``tol``.
    """
    rules = substitutions if substitutions is not None else {}
    if state is None:
        dim = word_matrix(S.One).shape[0]
        state = np.eye(dim, dtype=complex) / dim
    pins: dict[Any, Any] = {}
    for u in basis:
        mu = word_matrix(u)
        for v in basis:
            key = apply_substitutions(u.adjoint() * v, rules)
            if key == 1 or key == 0:
                continue
            value = np.trace(mu.conj().T @ word_matrix(v) @ state)
            if force_real:
                value = value.real
            if key in pins and abs(pins[key] - value) > tol:
                raise ValueError(
                    f"Inconsistent trace pins for {key}: "
                    f"{pins[key]} vs {value}."
                )
            pins[key] = value
    return pins


def _class_key(matrix: np.ndarray, decimals: int) -> _FunctionalKey:
    """The rounded real/imag key of a functional matrix."""
    return (
        tuple(np.round(np.real(matrix), decimals).ravel()),
        tuple(np.round(np.imag(matrix), decimals).ravel()),
    )


def class_moment_equalities(
    problem: Any,
    level: int,
    *,
    inserted: Sequence[tuple[Any, Callable[[np.ndarray], np.ndarray]]],
    row_words: Sequence[Any],
    col_words: Sequence[Any],
    word_matrix: Callable[[Any], np.ndarray],
    decimals: int = 9,
) -> list[MomentExpr]:
    """The functional-class equalities of the inserted-operator blocks.

    Entry ``(u, v)`` of the localizing block of an inserted operator
    ``g`` is the moment ``<u^dagger g v>``, a linear functional of
    ``G_{u,v} = trace_fn(v u^dagger)`` alone; entries sharing the
    functional must share the moment. For each ``(g, trace_fn)`` pair
    this function groups the ``(row, col)`` entries by their functional
    class and emits one block-0 moment equality per non-representative
    member (``MomentEntry(0, r1, c1) - MomentEntry(0, r2, c2)``).

    The positions come from a draft build of ``problem`` at ``level``:
    the SDP variable of an entry monomial is created at its first
    upper-triangle occurrence (or reused from its canonicalized adjoint),
    and the equality blocks are appended after the moment blocks, so the
    draft's block-0 layout is final. Classes whose entries are pinned
    (constant moments, including identically zero words) carry no
    variable and are skipped; classes with a single variable occurrence
    yield a trivially empty equality, mirroring the block count of a
    hand-written relation list.

    Args:
        problem: The problem the relaxation is built from, WITHOUT these
            moment equalities but with the final substitutions, moment
            substitutions, extramonomials and localizing monomials.
        level: The relaxation level.
        inserted: One ``(g, trace_fn)`` pair per inserted operator;
            ``trace_fn`` maps a full matrix to the functional matrix
            (e.g. a partial trace over the traced-out subsystems).
        row_words: The row words of the class grid (the basis whose
            moments are pinned).
        col_words: The column words (the localizing basis).
        word_matrix: Maps a word to its explicit matrix.
        decimals: Rounding decimals of the functional-class keys.

    Returns:
        The ``momentequalities`` list for :class:`Problem`.

    Raises:
        ValueError: If a non-pinned entry monomial (or its adjoint) has
            no moment-matrix occurrence, or its variable was created
            outside block 0.
    """
    substitutions = (
        dict(problem.substitutions) if problem.substitutions is not None else {}
    )
    pins = problem.momentsubstitutions if problem.momentsubstitutions is not None else {}
    draft = NpaRelaxation(problem, level)
    if draft.sdp is None:  # pragma: no cover - _build always sets it
        raise RuntimeError("The draft relaxation has not been built.")
    block0 = draft.moment_block_indices[0]
    row_mats = [word_matrix(w) for w in row_words]
    col_mats = [word_matrix(w) for w in col_words]

    def position(row: int, col: int, g: Any) -> tuple[int, int] | None:
        """The block-0 creation position of the variable of ``u^dagger g v``.

        Returns None for a pinned (constant) class: no variable exists,
        so the class needs no relation.
        """
        monomial = apply_substitutions(
            row_words[row].adjoint() * g * col_words[col], substitutions
        )
        if monomial == 0 or monomial in pins:
            return None
        adjoint = apply_substitutions(monomial.adjoint(), substitutions)
        if adjoint in pins:
            return None
        k = draft.monomial_index.get(monomial)
        if k is None:
            k = draft.monomial_index.get(adjoint)
        if k is None:
            raise ValueError(
                f"No moment-matrix occurrence for monomial {monomial} "
                f"(or its adjoint) at (row={row}, col={col}, g={g})."
            )
        block, r, c = draft.sdp.column_locations[k]
        if block != block0:
            raise ValueError(
                f"The variable of monomial {monomial} was created in "
                f"block {block}, not in moment block {block0}."
            )
        return r, c

    momentequalities: list[MomentExpr] = []
    for g, trace_fn in inserted:
        keys = [
            [_class_key(trace_fn(mv @ mu.T.conj()), decimals) for mv in col_mats]
            for mu in row_mats
        ]
        representatives: dict[_FunctionalKey, tuple[int, int]] = {}
        for row in range(len(row_words)):
            for col in range(len(col_words)):
                key = keys[row][col]
                if key not in representatives:
                    representatives[key] = (row, col)
                    continue
                i0, j0 = representatives[key]
                first = position(row, col, g)
                second = position(i0, j0, g)
                if first is None or second is None:
                    continue
                r1, c1 = first
                r2, c2 = second
                momentequalities.append(
                    MomentEntry(0, r1, c1) - MomentEntry(0, r2, c2)
                )
    return momentequalities
