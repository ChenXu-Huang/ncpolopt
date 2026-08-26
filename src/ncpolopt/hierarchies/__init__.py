"""Hierarchy variants of the NPA relaxation.

The standard NPA hierarchy is :class:`~ncpolopt.relaxation.NpaRelaxation`;
the variants in this package specialize the moment matrix layout:

- :class:`MoroderHierarchy`: bipartite Bell scenarios, rectangular basis
  of the two parties' measurement operators.
- :class:`SteeringHierarchy`: every moment expands into a
  ``matrix_var_dim x matrix_var_dim`` block of SDP variables.
- :class:`RdmHierarchy`: reduced density matrix band layout for the
  circulant moment matrix.
"""

from .moroder import MoroderHierarchy
from .rdm import RdmHierarchy
from .steering import SteeringHierarchy

__all__ = ["MoroderHierarchy", "RdmHierarchy", "SteeringHierarchy"]
