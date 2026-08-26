"""The Moroder hierarchy for bipartite Bell scenarios.

Ported from the legacy ``MoroderHierarchy`` class. The old hierarchy kept
two flat monomial sets and generated the moment
matrix block over their tensor product; here the two party bases are
wrapped into one rectangular set ``[[A, B]]``, which the base machinery of
:class:`~ncpolopt.relaxation.NpaRelaxation` already understands (block size
``|A| * |B|``, partial transpose, variable estimates).

The typical workflow imposes the partial-transpose constraints on the
duplicate moment matrix after conversion with PICOS; the frozen relaxation
is exposed through the standard ``.sdp`` attribute for that route.
"""

from __future__ import annotations

from typing import Any

from ..monomial_sets import generate_monomial_sets
from ..relaxation import NpaRelaxation


class MoroderHierarchy(NpaRelaxation):
    """Level-by-level Moroder relaxation of a bipartite Bell scenario.

    The moment matrix is built over the rectangular basis ``A x B`` of the
    two parties' measurement bases. The partial transpose of the B side can
    be imposed during construction with ``ppt=True``, or post-hoc after
    conversion with PICOS (the standard route, since the witness
    constraints live on a copy of the moment matrix that does not map onto
    the frozen SDP).
    """

    def __init__(
        self,
        problem: Any,
        level: int,
        *,
        ppt: bool = False,
        removeequalities: bool = False,
    ) -> None:
        """Build the Moroder relaxation of ``problem`` at ``level``.

        Args:
            problem: The bipartite Bell problem; its variables must be a
                list of two lists of operators.
            level: The relaxation level.
            ppt: Transpose the second party's subsystem of the moment
                matrix during construction.
            removeequalities: Eliminate equality constraints algebraically.

        Raises:
            ValueError: If the problem has fewer or more than two parties.
        """
        self.ppt = ppt
        super().__init__(problem, level, removeequalities=removeequalities)

    def _generate_monomial_sets(self) -> list[list[Any]]:
        """Generate the two party bases wrapped into a rectangular pair.

        Returns:
            The nested set ``[[A, B]]`` over the parties' measurement
            bases.

        Raises:
            ValueError: If the problem is not bipartite.
        """
        sets = generate_monomial_sets(
            self.problem.variables,
            self.level,
            self.problem.extramonomials,
            self._substitutions,
        )
        if len(sets) != 2:
            raise ValueError(
                "The Moroder hierarchy needs exactly two parties; the "
                "problem's variables must be a list of two lists of "
                "operators."
            )
        # The base machinery expects a rectangular pair [[A, B]]; the flat
        # per-party sets wrap into one nested set.
        return [sets]

    def _generate_all_moment_matrix_blocks(self) -> None:
        """Build the moment matrix block over the rectangular basis.

        The base dispatch generates the square block over a nested set
        without the partial transpose; the Moroder hierarchy passes its
        own ``ppt`` flag through.
        """
        block_index = self._n_parameter_blocks
        for monomials in self.monomial_sets:
            block_index = self._generate_moment_matrix(
                monomials[0], monomials[1], block_index, ppt=self.ppt
            )
            self.var_offsets.append(self._builder.n_vars)
