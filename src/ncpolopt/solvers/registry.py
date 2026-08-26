"""Solver registry: backend discovery and dispatch.

Backends register themselves under a :class:`SolverKind`; the registry
picks a backend by explicit request or by the first one whose module is
importable. The old package guessed backend availability by attempting
imports in a fixed order deep inside the solve path; here the check is
isolated so the error message can tell the user exactly which extra to
install.
"""

from __future__ import annotations

import importlib.util
from typing import Any

from .base import (
    SolverBackend,
    SolverError,
    SolverKind,
    SolverResult,
    SolverSettings,
)

#: The supported solver kinds in autodetection order (first wins).
_DETECTION_ORDER: tuple[SolverKind, ...] = (
    SolverKind.CVXPY,
    SolverKind.MOSEK,
    SolverKind.CVXOPT,
    SolverKind.SDPA,
)

#: The importable module name per solver kind.
_SOLVER_MODULES: dict[SolverKind, str] = {
    SolverKind.CVXPY: "cvxpy",
    SolverKind.SCS: "scs",
    SolverKind.MOSEK: "mosek",
    SolverKind.CVXOPT: "cvxopt",
    SolverKind.SDPA: "ncpolopt.solvers.sdpa_solver",
}

_REGISTRY: dict[SolverKind, SolverBackend] = {}

_INSTALL_HINTS: dict[SolverKind, str] = {
    SolverKind.CVXPY: "uv pip install ncpolopt[cvxpy]",
    SolverKind.MOSEK: "uv pip install ncpolopt[mosek]",
    SolverKind.CVXOPT: "uv pip install ncpolopt[cvxopt]",
    SolverKind.SDPA: "point SolverSettings.sdpa_executable at an SDPA binary",
}


def register(kind: SolverKind, backend: SolverBackend) -> None:
    """Register a backend implementation under a solver kind.

    Args:
        kind: The solver kind the backend implements.
        backend: The callable mapping an SdpProblem to a SolverResult.
    """
    _REGISTRY[kind] = backend


def registered() -> tuple[SolverKind, ...]:
    """Return the kinds with a registered backend, in detection order."""
    return tuple(kind for kind in _DETECTION_ORDER if kind in _REGISTRY)


def available() -> tuple[SolverKind, ...]:
    """Return the solver kinds that are usable right now.

    A backend must be registered and its module importable; the external
    SDPA backend additionally needs its binary on PATH (an explicitly
    configured ``settings.sdpa_executable`` still works at solve time even
    when SDPA is not detected here).

    Returns:
        The usable solver kinds in detection order.
    """
    import shutil

    def _usable(kind: SolverKind) -> bool:
        if kind not in _REGISTRY:
            return False
        module = _SOLVER_MODULES[kind]
        if importlib.util.find_spec(module) is None:
            return False
        return shutil.which("sdpa") is not None if kind is SolverKind.SDPA else True

    return tuple(kind for kind in _DETECTION_ORDER if _usable(kind))


#: Backwards-friendly name used by the CLI.
available_solvers = available


def requires(*modules: str) -> bool:
    """Return whether every module is importable.

    Args:
        modules: Module names, e.g. ``"cvxpy"``.

    Returns:
        True if all modules can be imported.
    """
    return all(importlib.util.find_spec(module) is not None for module in modules)


def _resolve(kind: SolverKind | str | None) -> SolverBackend:
    """Resolve a solver request to a registered backend callable.

    Args:
        kind: The requested solver kind, "auto", or None for autodetection.

    Returns:
        The backend callable.

    Raises:
        SolverError: If the kind is unknown, its backend is not registered,
            or no solver is available at all.
    """
    if kind is None or kind == "auto":
        for candidate in available():
            return _REGISTRY[candidate]
        installed = [k.value for k in _REGISTRY if k in _DETECTION_ORDER]
        raise SolverError(
            "No solver backend is available. "
            "Install one of the solver extras, e.g. 'uv pip install "
            "ncpolopt[cvxpy]' (found backends: "
            f"{installed or 'none'})."
        )
    if isinstance(kind, str):
        try:
            kind = SolverKind(kind)
        except ValueError:
            raise SolverError(
                f"Unknown solver {kind!r}; expected one of "
                f"{[k.value for k in _DETECTION_ORDER]} or 'auto'."
            ) from None
    backend = _REGISTRY.get(kind)
    if backend is None:
        raise SolverError(
            f"The {kind.value} solver is not available. "
            f"{_INSTALL_HINTS.get(kind, '')}"
        )
    module = _SOLVER_MODULES.get(kind)
    if module and importlib.util.find_spec(module) is None:
        raise SolverError(
            f"The {kind.value} solver is not available: the module "
            f"{module!r} is not installed. {_INSTALL_HINTS.get(kind, '')}"
        )
    return backend


def solve_problem(
    problem: Any,
    solver: SolverKind | str | None = "auto",
    settings: SolverSettings | None = None,
) -> SolverResult:
    """Solve an SDP relaxation with the requested (or detected) backend.

    Args:
        problem: The frozen :class:`~ncpolopt.sdp_problem.SdpProblem`.
        solver: The solver kind, its string value, or None/"auto" for the
            first available backend.
        settings: Backend knobs; defaults to empty settings.

    Returns:
        The solver result.

    Raises:
        SolverError: If no suitable backend is available.
    """
    if settings is None:
        settings = SolverSettings()
    backend = _resolve(solver)
    return backend(problem, settings)
