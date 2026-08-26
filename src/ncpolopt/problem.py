"""The user-facing model of a noncommutative polynomial optimization problem.

A :class:`Problem` is the symbolic description the user writes; it is a
frozen value object and can be reused. Building an SDP relaxation from it
never mutates it -- use :meth:`Problem.relaxation` (or the one-call
:meth:`Problem.solve`) to construct and solve.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .solvers.base import SolverKind, SolverSettings

if TYPE_CHECKING:
    from .relaxation import NpaRelaxation
    from .solution import Solution


@dataclass(frozen=True, slots=True)
class Problem:
    """Symbolic description of a polynomial optimization problem.

    The model distinguishes polynomials over the problem variables
    (``inequalities``/``equalities``) from linear conditions on the moment
    matrix itself (``momentinequalities``/``momentequalities``, written with
    :class:`~ncpolopt.moment.MomentEntry`). Substitution rules
    (``substitutions``) reduce the generated monomial bases; moment
    substitutions (``momentsubstitutions``) pin moment matrix entries to
    known values before any variable is created.

    The default, ``normalized=True``, imposes the normalization condition:
    the top-left entry of every moment matrix is pinned to 1.0 instead of
    becoming a free variable. This matches the "normalized" convention of
    the NPA hierarchy; steering problems use ``normalized=False``.

    Args:
        variables: The noncommutative operators (or one commutative symbol
            list) the problem is defined over; a list of lists denotes a
            multipartite problem.
        objective: Polynomial or moment expression to minimize; None for a
            feasibility problem.
        inequalities: Polynomial inequalities, each enforced as
            ``>= 0`` by its localizing matrix.
        equalities: Polynomial equalities, each enforced by pairs of scalar
            blocks pinning every localizing entry to zero.
        momentinequalities: Inequalities over moment expressions, e.g.
            ``MomentEntry(0, 0, 0) - 0.5 >= 0``.
        momentequalities: Equalities over moment expressions.
        substitutions: Monomial replacement rules applied to every generated
            monomial basis.
        momentsubstitutions: Moment substitutions applied when building the
            moment matrix, e.g. ``{X[0] ** 2: 1.0}``.
        parameters: Operator variables whose moments are left symbolic, each
            becoming a free 1x1 block of its own.
        extramonomials: Monomials added to the level basis, or (at level
            -1) the full basis. May be a list of lists for multipartite
            problems.
        extramomentmatrices: Per extra moment matrix, the option strings
            ``"copy"`` and/or ``"ppt"``.
        localizing_monomials: Per-constraint override of the localizing
            basis; None entries request the automatic basis.
        normalized: Pin the top-left moment entry to 1.0.
        complex_matrix: Force a complex-valued relaxation; None auto-detects
            from the substitutions and constraint coefficients.
    """

    variables: Any
    objective: Any = None
    inequalities: list[Any] | None = None
    equalities: list[Any] | None = None
    momentinequalities: list[Any] | None = None
    momentequalities: list[Any] | None = None
    substitutions: dict[Any, Any] | None = None
    momentsubstitutions: dict[Any, Any] | None = None
    parameters: list[Any] | None = None
    extramonomials: list[Any] | list[list[Any]] | None = None
    extramomentmatrices: list[list[str]] | None = None
    localizing_monomials: list[list[Any]] | None = None
    normalized: bool = True
    complex_matrix: bool | None = None

    def relaxation(
        self,
        level: int = 1,
        *,
        removeequalities: bool = False,
        chordal_extension: bool = False,
    ) -> NpaRelaxation:
        """Build the SDP relaxation of this problem at ``level``.

        Args:
            level: The relaxation level; -1 uses only the extra monomials.
            removeequalities: Eliminate the equality constraints
                algebraically instead of representing them as SDP blocks.
            chordal_extension: Replace the variable set by the cliques of
                the chordal completion of the correlative sparsity
                pattern, relaxing each clique as its own moment block.

        Returns:
            The built relaxation.
        """
        from .chordal import find_variable_cliques
        from .relaxation import NpaRelaxation

        if chordal_extension:
            variables = find_variable_cliques(
                self.variables,
                self.objective,
                self.inequalities,
                self.equalities,
                self.momentinequalities,
                self.momentequalities,
            )
            problem = dataclasses.replace(self, variables=variables)
        else:
            problem = self
        return NpaRelaxation(problem, level, removeequalities=removeequalities)

    def solve(
        self,
        level: int = 1,
        *,
        removeequalities: bool = False,
        chordal_extension: bool = False,
        solver: SolverKind | str | None = "auto",
        settings: SolverSettings | None = None,
    ) -> Solution:
        """Build the relaxation and solve it in one step.

        Args:
            level: The relaxation level.
            removeequalities: Eliminate equality constraints algebraically.
            chordal_extension: Replace the variable set by the cliques of
                the chordal completion of the correlative sparsity
                pattern (see :meth:`relaxation`).
            solver: The solver kind, its string value, or None/"auto" for
                the first available backend.
            settings: Backend knobs.

        Returns:
            The solution of the relaxation.

        Raises:
            SolverError: If no solver backend is available.
        """
        relaxation = self.relaxation(
            level,
            removeequalities=removeequalities,
            chordal_extension=chordal_extension,
        )
        return relaxation.solve(solver, settings)
