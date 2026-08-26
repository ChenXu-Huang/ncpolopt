"""Common types shared by every solver backend.

The old package let each backend grow its own glue code and its own value
formulas; the four backends disagreed on the sign of the objective constant
and on which matrix was ``x_mat``. Here the conventions are fixed once:
``min c.x + c0`` with ``primal = c.x* + c0`` and
``dual = -sum_b tr(Y_b @ A0_b) + c0``, and every backend returns a
:class:`SolverResult` through the same shape.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from ..sdp_problem import SdpProblem


class SolverKind(Enum):
    """The supported SDP solvers.

    ``SDPA`` refers to an external SDPA binary invoked through a .dat-s
    file; the others are Python packages.
    """

    CVXPY = "cvxpy"
    SCS = "scs"
    MOSEK = "mosek"
    CVXOPT = "cvxopt"
    SDPA = "sdpa"


@dataclass(frozen=True, slots=True)
class SolverSettings:
    """Knobs passed through to the solver backend.

    Attributes:
        solver_options: Keyword arguments for the backend's solve call
            (e.g. ``{"solver": "SCS"}`` for CVXPY).
        mosek_params: MOSEK parameter name/value pairs, applied via
            ``task.putparam``.
        sdpa_executable: Path to the SDPA binary for the external backend.
        verbose: Passed to the backend; None leaves the backend default.
    """

    solver_options: dict[str, Any] = field(default_factory=dict)
    mosek_params: dict[str, Any] = field(default_factory=dict)
    sdpa_executable: str | None = None
    verbose: bool | None = None


class SolverError(Exception):
    """Raised when no solver is available or the solve fails."""


class UnsupportedSdpError(SolverError):
    """Raised when a relaxation cannot be mapped onto a backend.

    The complex-valued SDPs the external SDPA path supports through its
    doubled-block convention are not available in every backend; backends
    raise this instead of silently producing a wrong problem.
    """


@dataclass(frozen=True, slots=True)
class SolverResult:
    """The raw outcome of an SDP solve, before solution extraction.

    ``x_mat`` holds the primal block matrices and ``y_mat`` the dual blocks,
    each tuple aligned with :class:`~ncpolopt.sdp_problem.SdpProblem.blocks`.
    Complex blocks are stored as they came out of the backend: real
    (symmetric) arrays, or complex arrays for the backends that support
    them natively.

    Attributes:
        status: The solver status string (e.g. ``"optimal"``).
        primal: The primal optimal value, ``c.x* + c0``.
        dual: The dual optimal value, ``-sum_b tr(Y_b @ A0_b) + c0``.
        x_mat: Tuple of primal block matrices.
        y_mat: Tuple of dual block matrices.
        solution_time: Wall-clock solve time in seconds.
        variables: The primal variable values (length n_vars), if the
            backend exposes them; empty otherwise.
        raw: The backend's native result object, for diagnostics.
    """

    status: str
    primal: float
    dual: float
    x_mat: tuple[np.ndarray, ...] = ()
    y_mat: tuple[np.ndarray, ...] = ()
    solution_time: float = 0.0
    variables: np.ndarray = field(default_factory=lambda: np.array([]))
    raw: Any = None


#: A backend implementation: map an SdpProblem to a SolverResult.
SolverBackend = Callable[["SdpProblem", SolverSettings], SolverResult]
