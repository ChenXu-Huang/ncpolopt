"""Solver backends: common types, the registry, and one module per backend.

Every backend maps a frozen :class:`~ncpolopt.sdp_problem.SdpProblem` onto
its native SDP format, solves it, and returns a :class:`SolverResult`. The
canonical conventions are ``min c.x + c0`` with block constraints
``A0_b + sum_k x_k A_{k,b} >= 0``; the value formulas
``primal = c.x* + c0`` and ``dual = -sum_b tr(Y_b @ A0_b) + c0`` are fixed
here so that no backend re-derives (or mis-derives) them.

Importing this package registers the backends whose code is bundled (their
native solver libraries are imported lazily, at solve time).
"""

from . import (  # noqa: F401  (importing registers backends)
    cvxpy_solver,
    mosek_solver,
    picos_solver,
    sdpa_solver,
)

