"""Shared pytest fixtures for the ncpolopt test suite."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest

from ncpolopt.solvers.base import SolverKind
from ncpolopt.solvers.registry import available, requires


def _licensed_mosek() -> bool:
    """Whether MOSEK can acquire a license.

    MOSEK installs without a license and raises when initialized without
    one; tests parametrized over it would fail with a license error instead
    of a numerical one.
    """
    try:
        import mosek

        # NOTE: the MOSEK binding is a compiled extension whose environment
        # methods cannot be resolved statically.
        env: Any = mosek.Env()
        if hasattr(env, "checkoutlicense"):
            # MOSEK 10+: check out the base feature explicitly; the license
            # is otherwise only checked at optimize() time.
            env.checkoutlicense(mosek.feature.pts)
            env.checkinlicense(mosek.feature.pts)
        else:
            env.init()
        if hasattr(env, "dispose"):
            env.dispose()
        return True
    except Exception:
        return False


_LICENSED_MOSEK: bool = _licensed_mosek()
"""Whether MOSEK can acquire a license here, probed once per session.

The probe builds a ``mosek.Env``; repeating it per test is wasted work,
and the solver-kind fixture below already evaluates it at import time.
"""

_MOSEK_SKIP_REASON = (
    "mosek is not installed" if not requires("mosek") else "MOSEK cannot acquire a license"
)

licensed_mosek = pytest.mark.skipif(not _LICENSED_MOSEK, reason=_MOSEK_SKIP_REASON)
"""Skip mark for tests that need a licensed MOSEK, not just the package."""


def _usable_solver_kinds() -> list[SolverKind]:
    """The solver kinds usable in this environment, in detection order.

    cvxpy is part of the dev dependency group, so the default run covers at
    least one backend.
    """
    kinds = list(available())
    if SolverKind.MOSEK in kinds and not _LICENSED_MOSEK:
        kinds.remove(SolverKind.MOSEK)
    return kinds


@pytest.fixture(params=_usable_solver_kinds())
def solver_kind(request: pytest.FixtureRequest) -> SolverKind:
    """The solver kind to run a numerical test against."""
    return request.param


@pytest.fixture(autouse=True)
def clear_sympy_cache() -> Iterator[None]:
    """Reset SymPy's expression cache after every test.

    The old unittest suite cleared the cache in each ``tearDown`` because
    SymPy's ``@cacheit`` memoization keyed by expression hash can leak
    state across test boundaries (the classical issue with ``Operator``
    instances). Keep the same isolation under pytest.
    """
    yield
    from sympy.core.cache import clear_cache

    clear_cache()
