"""Hierarchy variants of the NPA relaxation.

The standard NPA hierarchy is :class:`~ncpolopt.relaxation.NpaRelaxation`;
the variants in this package specialize the moment matrix layout:

- :class:`MoroderHierarchy`: bipartite Bell scenarios, rectangular basis
  of the two parties' measurement operators.
- :class:`SteeringHierarchy`: every moment expands into a
  ``matrix_var_dim x matrix_var_dim`` block of SDP variables.
- :class:`RdmHierarchy`: reduced density matrix band layout for the
  circulant moment matrix.

The :mod:`~ncpolopt.hierarchies.insertion` helpers support the
operator-insertion (MDI) family: they generate the trace pins and the
functional-class moment equalities of an inserted-operator relaxation
from an explicit matrix realization of the word algebra.
"""

from .insertion import class_moment_equalities, trace_moment_pins
from .moroder import MoroderHierarchy
from .rdm import RdmHierarchy
from .steering import SteeringHierarchy

__all__ = [
    "MoroderHierarchy",
    "RdmHierarchy",
    "SteeringHierarchy",
    "class_moment_equalities",
    "trace_moment_pins",
]
