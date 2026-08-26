"""ncpolopt: sparse SDP relaxations of polynomial optimization problems.

Builds semidefinite programming (SDP) relaxations -- the NPA hierarchy
for noncommuting operators, the Lasserre hierarchy for commuting
variables -- and solves them with pluggable backends (cvxpy, MOSEK,
cvxopt, SDPA).

The package models an optimization problem as a :class:`Problem` of
operators or variables, adds an objective and constraints, then builds a
:class:`~ncpolopt.relaxation.NpaRelaxation` at a chosen hierarchy level
and solves it to obtain an immutable :class:`~ncpolopt.solution.Solution`.

Quickstart (a small noncommutative problem, see the README for details)::

    import ncpolopt as nc

    X = nc.generate_operators("x", 2, hermitian=True)
    problem = nc.Problem(
        X,
        objective=X[0] * X[1] + X[1] * X[0],
        inequalities=[-X[1] ** 2 + X[1] + 0.5],
        substitutions={X[0] ** 2: X[0]},
    )
    solution = problem.solve(level=2)
    print(solution.primal)  # -0.75
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from ._logging import setup_logging
from .chordal import find_clique_index, find_variable_cliques, sliding_cliques
from .expressions import flatten
from .hierarchies import MoroderHierarchy, RdmHierarchy, SteeringHierarchy
from .moment import MomentEntry, MomentExpr
from .monomials import get_all_monomials, get_monomials
from .physics import (
    Probability,
    bosonic_constraints,
    correlator,
    define_objective_with_I,
    fermionic_constraints,
    generate_measurements,
    get_neighbors,
    get_next_neighbors,
    maximum_violation,
    pauli_constraints,
    projective_measurement_constraints,
)
from .problem import Problem
from .relaxation import NpaRelaxation
from .sdpa_writer import read_sdpa_out, write_dat_s
from .solution import Solution
from .solvers.base import SolverKind, SolverSettings
from .variables import generate_operators, generate_variables, get_support

try:
    __version__ = version("ncpolopt")
except PackageNotFoundError:  # pragma: no cover - editable installs always provide it
    __version__ = "0.0.0"

__all__ = [
    "MomentEntry",
    "MomentExpr",
    "MoroderHierarchy",
    "NpaRelaxation",
    "Probability",
    "Problem",
    "RdmHierarchy",
    "Solution",
    "SolverKind",
    "SolverSettings",
    "SteeringHierarchy",
    "__version__",
    "bosonic_constraints",
    "correlator",
    "define_objective_with_I",
    "fermionic_constraints",
    "find_clique_index",
    "find_variable_cliques",
    "flatten",
    "generate_measurements",
    "generate_operators",
    "generate_variables",
    "get_all_monomials",
    "get_monomials",
    "get_neighbors",
    "get_next_neighbors",
    "get_support",
    "maximum_violation",
    "pauli_constraints",
    "projective_measurement_constraints",
    "read_sdpa_out",
    "setup_logging",
    "sliding_cliques",
    "write_dat_s",
]
