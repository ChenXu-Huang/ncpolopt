"""The reduced density matrix (RDM) hierarchy.

Ported from ``RdmHierarchy`` (src.old/ncpol2sdpa/rdm_hierarchy.py). The
circulant variant generates the moment matrix in a band layout instead of
the full square: at degree 1 the complete second-moment matrix, at degree 2
the per-sub-block layout of the old ``__fourth_moments`` (the N x N grid
of 4th-moment monomials with its N^2 basis sub-blocks).

Two fixes are applied over the old implementation:

- The layout selection depended on a per-instance call counter
  (``m_block``) that made the result depend on the construction order of
  the blocks (bug #12); here it derives from the block index.
- The old even-N second-moment layout dropped the cross-quadrant band
  pairs and indexed one of its substitution lookups with a negative
  offset; the degree-1 path generates the complete standard moment
  matrix, which coincides with the old layout for odd N.
"""

from __future__ import annotations

import math
from typing import Any

from sympy import S
from sympy.physics.quantum.dagger import Dagger

from ..monomials import ncdegree
from ..relaxation import NpaRelaxation


class RdmHierarchy(NpaRelaxation):
    """Level-by-level RDM relaxation of a Problem.

    The standard NPA relaxation is recovered with ``circulant=False``.
    With ``circulant=True`` the moment matrix is generated in the RDM
    layout instead of the full square: the degree-1 basis yields the
    complete second-moment matrix, the degree-2 basis the banded
    fourth-moment matrix (which needs a square basis of ``N**2`` -- or
    ``2*N**2`` for the two-half layout of the second block -- typically
    passed through ``extramonomials`` at level -1).
    """

    def __init__(
        self,
        problem: Any,
        level: int,
        *,
        circulant: bool = False,
        removeequalities: bool = False,
    ) -> None:
        """Build the RDM relaxation of ``problem`` at ``level``.

        Args:
            problem: The problem to relax.
            level: The relaxation level.
            circulant: Generate the banded circulant moment matrix
                instead of the full square.
            removeequalities: Eliminate equality constraints algebraically.
        """
        self.circulant = circulant
        super().__init__(problem, level, removeequalities=removeequalities)

    def _generate_moment_matrix(
        self,
        monomialsA: list[Any],
        monomialsB: list[Any],
        block_index: int,
        ppt: bool = False,
    ) -> int:
        """Generate one moment matrix block, possibly in the band layout.

        Args:
            monomialsA: The first factor of the basis.
            monomialsB: The second factor (the identity for flat bases).
            block_index: The block's index in the final structure.
            ppt: Transpose the second factor's indices.

        Returns:
            The index of the next block.

        Raises:
            ValueError: If the basis degree cannot be handled by the
                circulant layout.
        """
        if not self.circulant or monomialsB != [S.One]:
            return super()._generate_moment_matrix(
                monomialsA, monomialsB, block_index, ppt=ppt
            )
        degree = max(ncdegree(monomial) for monomial in monomialsA)
        if degree == 1:
            return super()._generate_moment_matrix(
                monomialsA, monomialsB, block_index, ppt=ppt
            )
        if degree == 2:
            return self._fourth_moments(monomialsA, block_index)
        raise ValueError(
            f"Cannot generate a circulant moment matrix with degree-{degree} "
            f"terms."
        )

    def _fourth_moments(self, monomials: list[Any], block_index: int) -> int:
        """The degree-2 circulant layout, selected by the block index.

        The old implementation counted the generated blocks per instance
        (``m_block``); the sequence 1/2/3 of the old code maps onto the
        block index here, so the result no longer depends on the
        construction order of the blocks (bug #12).

        Args:
            monomials: The square basis of the block.
            block_index: The block's index.

        Returns:
            The index of the next block.
        """
        m_block = block_index - self._n_parameter_blocks + 1
        if m_block == 2:
            return self._fourth_moments_two(monomials, block_index)
        if m_block == 1 or m_block == 3:
            return self._fourth_moments_one(monomials, block_index)
        return super()._generate_moment_matrix(monomials, [S.One], block_index)

    def _fourth_moments_one(self, monomials: list[Any], block_index: int) -> int:
        """Layout A: the sub-block regions of the N x N moment grid.

        The basis of ``N**2`` 4th-moment monomials splits into an N x N
        grid of basis sub-blocks; each diagonal sub-block contributes its
        upper triangle, each off-diagonal sub-block its complete region
        (the old ``generate_block_coords`` coords plus their transposed
        partners, which fill the local-lower positions). The remaining
        entries are deliberately zero -- that is the RDM band structure.

        Args:
            monomials: The square basis of the block.
            block_index: The block's index.

        Returns:
            The index of the next block.

        Raises:
            ValueError: If the basis is not a perfect square.
        """
        n = self._grid_size(monomials)
        for block_row in range(n):
            for block_col in range(block_row, n):
                for row in range(n):
                    start_column = row if block_row == block_col else 0
                    for column in range(start_column, n):
                        i = n * block_row + row
                        j = n * block_col + column
                        self._push_monomial(
                            Dagger(monomials[i]) * monomials[j],
                            block_index,
                            i,
                            j,
                        )
        return block_index + 1

    def _fourth_moments_two(self, monomials: list[Any], block_index: int) -> int:
        """Layout B: two diagonal bands around a full cross rectangle.

        The basis holds ``2*N**2`` monomials: the first ``N**2`` form the
        A-side grid, the second the B-side. The moment matrix consists of
        the two A- and B-side band layouts with the complete N x N
        rectangle of cross sub-blocks in between (the old ``m_block == 2``
        layout).

        Args:
            monomials: The basis of the block, ``2*N**2`` elements.
            block_index: The block's index.

        Returns:
            The index of the next block.

        Raises:
            ValueError: If the basis is not twice a perfect square.
        """
        n = int(math.sqrt(len(monomials) // 2))
        if 2 * n * n != len(monomials):
            raise ValueError(
                f"The two-band circulant layout needs a basis of 2*N**2 "
                f"monomials, got {len(monomials)}."
            )
        for offset in (0, n * n):
            for block_row in range(n):
                for block_col in range(block_row, n):
                    for row in range(n):
                        start_column = row if block_row == block_col else 0
                        for column in range(start_column, n):
                            i = offset + n * block_row + row
                            j = offset + n * block_col + column
                            self._push_monomial(
                                Dagger(monomials[i]) * monomials[j],
                                block_index,
                                i,
                                j,
                            )
        # The cross rectangle: every A-side sub-block paired with every
        # B-side sub-block, each contributing its complete region.
        for block_row in range(n):
            for block_col in range(n):
                for row in range(n):
                    for column in range(n):
                        i = n * block_row + row
                        j = n * n + n * block_col + column
                        self._push_monomial(
                            Dagger(monomials[i]) * monomials[j],
                            block_index,
                            i,
                            j,
                        )
        return block_index + 1

    def _grid_size(self, monomials: list[Any]) -> int:
        """The sub-block grid size of a square 4th-moment basis.

        Args:
            monomials: The basis of the block.

        Returns:
            ``sqrt(len(monomials))``.

        Raises:
            ValueError: If the basis is not a perfect square.
        """
        n = int(math.sqrt(len(monomials)))
        if n * n != len(monomials):
            raise ValueError(
                f"The degree-2 circulant layout needs a square basis of "
                f"N**2 monomials, got {len(monomials)}."
            )
        return n
