"""Tests for the operator-insertion helpers of ``hierarchies/insertion.py``.

The matrix model is a single qubit: the projectors P0 = |0><0| and
P1 = |+><+| with the idempotence rules p**2 -> p, plus one inserted
hermitian operator g. Every expectation below is hand-computed from the
explicit 2x2 matrices.
"""

from __future__ import annotations

import numpy as np
import pytest
from sympy import S

import ncpolopt as nc
from ncpolopt.hierarchies import class_moment_equalities, trace_moment_pins
from ncpolopt.substitutions import apply_substitutions

P0 = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=complex)
P1 = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=complex)

_TOL = 1e-9


def _model() -> dict[object, object]:
    """The toy algebra: operators, rules, basis and word matrices."""
    p0, p1 = nc.generate_operators("p", 2, hermitian=True)
    (g,) = nc.generate_operators("g", 1, hermitian=True)
    substitutions = {p0**2: p0, p1**2: p1}
    basis = [S.One, p0, p1]
    mats = {p0: P0, p1: P1}

    def word_matrix(word: object) -> np.ndarray:
        """The explicit 2x2 matrix of a basis word."""
        if word == 1:
            return np.eye(2, dtype=complex)
        return mats[word]

    return {
        "p0": p0,
        "p1": p1,
        "g": g,
        "substitutions": substitutions,
        "basis": basis,
        "word_matrix": word_matrix,
    }


def _problem(model: dict[object, object], momentequalities: list[object]) -> nc.Problem:
    """The inserted-operator problem over the toy algebra."""
    p0, p1, g = model["p0"], model["p1"], model["g"]
    pins = trace_moment_pins(
        model["basis"],
        model["word_matrix"],
        substitutions=model["substitutions"],
        force_real=True,
    )
    return nc.Problem(
        [p0, p1, g],
        objective=g,
        inequalities=[g],
        substitutions=model["substitutions"],
        momentsubstitutions=pins,
        momentequalities=momentequalities,
        extramonomials=[g * p0, g * p1],
        localizing_monomials=[model["basis"]],
        normalized=True,
        complex_matrix=False,
    )


def test_trace_moment_pins_values() -> None:
    """The pins reproduce the hand-computed traces under I/2."""
    model = _model()
    p0, p1 = model["p0"], model["p1"]
    pins = trace_moment_pins(
        model["basis"], model["word_matrix"], substitutions=model["substitutions"]
    )
    assert pins[p0] == pytest.approx(0.5, abs=_TOL)
    assert pins[p1] == pytest.approx(0.5, abs=_TOL)
    # <p0 p1> = Tr(P0 P1)/2 = |<0|+>|^2/2 = 1/4.
    assert pins[p0 * p1] == pytest.approx(0.25, abs=_TOL)
    assert pins[p1 * p0] == pytest.approx(0.25, abs=_TOL)
    # The normalization and zero words need no pin.
    assert S.One not in pins


def test_trace_moment_pins_explicit_state() -> None:
    """A non-maximally-mixed state changes the pins accordingly."""
    model = _model()
    p0 = model["p0"]
    state = P0.copy()  # the pure state |0><0|
    pins = trace_moment_pins(
        model["basis"],
        model["word_matrix"],
        substitutions=model["substitutions"],
        state=state,
    )
    assert pins[p0] == pytest.approx(1.0, abs=_TOL)


def test_trace_moment_pins_force_real() -> None:
    """force_real keeps the symmetrized (real) part of a complex moment."""
    p0, p1 = nc.generate_operators("p", 2, hermitian=True)
    (q,) = nc.generate_operators("q", 1, hermitian=True)
    # |+i><+i| makes Tr(P0 P1 Pi) genuinely complex.
    p_i = np.array([[0.5, -0.5j], [0.5j, 0.5]], dtype=complex)
    substitutions = {p0**2: p0, p1**2: p1, q**2: q}
    mats = {p0: P0, p1: P1, q: p_i}

    def word_matrix(word: object) -> np.ndarray:
        """The explicit 2x2 matrix of a basis word (products expanded)."""
        if word == 1:
            return np.eye(2, dtype=complex)
        if word in mats:
            return mats[word]
        mat = np.eye(2, dtype=complex)
        for factor in word.args:
            mat = mat @ mats[factor]
        return mat

    basis = [S.One, p0, p1, q, p0 * p1]
    complex_pins = trace_moment_pins(basis, word_matrix, substitutions=substitutions)
    real_pins = trace_moment_pins(
        basis, word_matrix, substitutions=substitutions, force_real=True
    )
    key = apply_substitutions((p0 * p1).adjoint() * q, substitutions)
    assert abs(complex_pins[key].imag) > 1e-3
    assert real_pins[key] == pytest.approx(complex_pins[key].real, abs=_TOL)
    assert np.isrealobj(np.asarray(list(real_pins.values()), dtype=float))


def test_trace_moment_pins_clash_raises() -> None:
    """Two pairs reducing to one key with different values are rejected."""
    model = _model()
    p0, p1 = model["p0"], model["p1"]
    # The rule p0 -> p1 identifies the keys of (1, p0) and (1, p1) while
    # their trace values differ.
    with pytest.raises(ValueError, match="Inconsistent trace pins"):
        trace_moment_pins(
            [S.One, p0, p1],
            model["word_matrix"],
            substitutions={p0: p1},
        )


def _class_of(monomial: object, model: dict[object, object]) -> np.ndarray:
    """The functional G = v u^dagger of a monomial u^dagger g v."""
    g = model["g"]
    rules = model["substitutions"]
    for u in model["basis"]:
        for v in model["basis"]:
            if apply_substitutions(u.adjoint() * g * v, rules) == monomial:
                mu = model["word_matrix"](u)
                mv = model["word_matrix"](v)
                return mv @ mu.T.conj()
    raise AssertionError(f"{monomial} is not an inserted entry monomial")


def test_class_moment_equalities_grouping() -> None:
    """The relations connect exactly the same-class block-0 variables."""
    model = _model()
    g = model["g"]
    problem = _problem(model, momentequalities=[])
    relations = class_moment_equalities(
        problem,
        1,
        inserted=[(g, lambda matrix: matrix)],
        row_words=model["basis"],
        col_words=model["basis"],
        word_matrix=model["word_matrix"],
    )
    # The class of P0 has three members ((1,p0), (p0,1), (p0,p0)) and so
    # has the class of P1; the products P0 P1 and P1 P0 are singletons.
    assert len(relations) == 4

    draft = problem.relaxation(level=1)
    sdp = draft.sdp
    var_at = {position: k for k, position in sdp.column_locations.items()}
    monomial_of = {k: m for m, k in sdp.monomial_index.items()}
    nontrivial = []
    for relation in relations:
        if not relation.terms:
            # An adjoint pair shares one variable: the relation between
            # (1, p) and (p, 1) is trivially empty.
            continue
        nontrivial.append(relation)
        assert len(relation.terms) == 2
        first, second = relation.terms
        k1 = var_at[(first.block, first.row, first.col)]
        k2 = var_at[(second.block, second.row, second.col)]
        assert k1 != k2
        assert first.coefficient == 1.0 and second.coefficient == -1.0
        g1 = _class_of(monomial_of[k1], model)
        g2 = _class_of(monomial_of[k2], model)
        assert np.allclose(g1, g2, atol=_TOL)
    # One non-trivial relation per three-member class (P0 and P1); the
    # adjoint-pair relations are the trivial ones.
    assert len(nontrivial) == 2


def test_class_moment_equalities_deterministic() -> None:
    """Two calls produce the same relation list."""
    model = _model()
    g = model["g"]
    problem = _problem(model, momentequalities=[])
    kwargs = {
        "inserted": [(g, lambda matrix: matrix)],
        "row_words": model["basis"],
        "col_words": model["basis"],
        "word_matrix": model["word_matrix"],
    }
    first = class_moment_equalities(problem, 1, **kwargs)
    second = class_moment_equalities(problem, 1, **kwargs)
    assert [r.terms for r in first] == [r.terms for r in second]


def test_class_moment_equalities_missing_monomial() -> None:
    """An entry monomial with no moment-matrix occurrence is an error."""
    model = _model()
    g = model["g"]
    # Without the extramonomials the words g p0 / g p1 never occur as
    # moment-matrix entries.
    problem = nc.Problem(
        [model["p0"], model["p1"], g],
        objective=g,
        inequalities=[g],
        substitutions=model["substitutions"],
        momentsubstitutions=trace_moment_pins(
            model["basis"],
            model["word_matrix"],
            substitutions=model["substitutions"],
            force_real=True,
        ),
        normalized=True,
        complex_matrix=False,
    )
    with pytest.raises(ValueError, match="No moment-matrix occurrence"):
        class_moment_equalities(
            problem,
            1,
            inserted=[(g, lambda matrix: matrix)],
            row_words=model["basis"],
            col_words=model["basis"],
            word_matrix=model["word_matrix"],
        )


def test_insertion_relaxation_solves(solver_kind: object) -> None:
    """The generated problem builds and solves; class moments agree."""
    model = _model()
    g = model["g"]
    draft = _problem(model, momentequalities=[])
    relations = class_moment_equalities(
        draft,
        1,
        inserted=[(g, lambda matrix: matrix)],
        row_words=model["basis"],
        col_words=model["basis"],
        word_matrix=model["word_matrix"],
    )
    problem = _problem(model, momentequalities=relations)
    solution = problem.solve(level=1, solver=solver_kind)
    assert solution.status == "optimal"
    # Feasible point of the model: g = I gives <g> = 1, so the minimum of
    # <g> over g >= 0 with pinned S-moments only is 0; the class relations
    # keep the relaxation bounded.
    assert solution.primal == pytest.approx(0.0, abs=1e-5)
