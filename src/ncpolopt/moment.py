"""Typed references to entries of the moment matrix.

The old package addressed matrix entries through a fragile string DSL such
as ``"+0[0,0]-1.0"`` parsed by hand. This module replaces it with value
objects: :class:`MomentEntry` names one matrix entry and
:class:`MomentExpr` is a linear combination of entries, both supporting
ordinary arithmetic so constraints read naturally.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class MomentEntry:
    """One entry of the moment matrix (by construction-time block index).

    Attributes:
        block: Index of the moment matrix block the entry belongs to.
        row: Row within the block (0-based).
        col: Column within the block (0-based).
        coefficient: The factor multiplying this entry.
    """

    block: int = 0
    row: int = 0
    col: int = 0
    coefficient: complex = 1.0

    def __neg__(self) -> MomentExpr:
        return MomentExpr((MomentEntry(self.block, self.row, self.col, -self.coefficient),))

    def __add__(self, other: object) -> MomentExpr:
        if isinstance(other, MomentEntry):
            return MomentExpr((self, other))
        if isinstance(other, MomentExpr):
            return MomentExpr((self, *other.terms))
        if isinstance(other, (int, float, complex)):
            return MomentExpr((self, MomentEntry(coefficient=other)))
        return NotImplemented

    def __radd__(self, other: object) -> MomentExpr:
        return self.__add__(other)

    def __sub__(self, other: object) -> MomentExpr:
        return self.__add__(-other) if isinstance(other, (MomentEntry, MomentExpr, int, float, complex)) else NotImplemented

    def __rsub__(self, other: object) -> MomentExpr:
        if isinstance(other, (int, float, complex)):
            return MomentExpr((MomentEntry(coefficient=other), -self))
        return NotImplemented

    def __mul__(self, other: object) -> MomentExpr:
        if isinstance(other, (int, float, complex)):
            return MomentExpr((MomentEntry(self.block, self.row, self.col, self.coefficient * other),))
        return NotImplemented

    def __rmul__(self, other: object) -> MomentExpr:
        return self.__mul__(other)


@dataclass(frozen=True, slots=True)
class MomentExpr:
    """A linear combination of moment matrix entries.

    The constant term is represented by the entry with a zero coefficient
    position ``(0, 0, 0)`` whose coefficient carries the constant value.
    """

    terms: tuple[MomentEntry, ...]

    def __post_init__(self) -> None:
        # Combine duplicate positions and drop zero coefficients so that
        # arithmetic does not accumulate garbage terms.
        combined: dict[tuple[int, int, int], complex] = {}
        for term in self.terms:
            key = (term.block, term.row, term.col)
            combined[key] = combined.get(key, 0.0) + term.coefficient
        object.__setattr__(
            self,
            "terms",
            tuple(
                MomentEntry(block, row, col, coeff)
                for (block, row, col), coeff in combined.items()
                if coeff != 0
            ),
        )

    @property
    def is_constant(self) -> bool:
        """Whether this expression is a pure constant."""
        return all(term.block == 0 and term.row == 0 and term.col == 0 for term in self.terms)

    def constant(self) -> complex:
        """The constant part of this expression."""
        return sum(
            term.coefficient
            for term in self.terms
            if term.block == 0 and term.row == 0 and term.col == 0
        )

    def __neg__(self) -> MomentExpr:
        # NOTE: scale the entries directly -- ``-term`` would call
        # MomentEntry.__neg__ and return a nested MomentExpr.
        return MomentExpr(
            tuple(
                MomentEntry(term.block, term.row, term.col, -term.coefficient)
                for term in self.terms
            )
        )

    def __add__(self, other: object) -> MomentExpr:
        if isinstance(other, (MomentEntry, MomentExpr)):
            return MomentExpr((*self.terms, other) if isinstance(other, MomentEntry) else (*self.terms, *other.terms))
        if isinstance(other, (int, float, complex)):
            return MomentExpr((*self.terms, MomentEntry(coefficient=other)))
        return NotImplemented

    def __radd__(self, other: object) -> MomentExpr:
        return self.__add__(other)

    def __sub__(self, other: object) -> MomentExpr:
        return self.__add__(-other) if isinstance(other, (MomentEntry, MomentExpr, int, float, complex)) else NotImplemented

    def __rsub__(self, other: object) -> MomentExpr:
        if isinstance(other, (int, float, complex)):
            return MomentExpr((MomentEntry(coefficient=other),)) + (-self)
        return NotImplemented

    def __mul__(self, other: object) -> MomentExpr:
        if isinstance(other, (int, float, complex)):
            # NOTE: scale the entries directly -- ``term * other`` would call
            # MomentEntry.__mul__ and return a nested MomentExpr.
            return MomentExpr(
                tuple(
                    MomentEntry(term.block, term.row, term.col, term.coefficient * other)
                    for term in self.terms
                )
            )
        return NotImplemented

    def __rmul__(self, other: object) -> MomentExpr:
        return self.__mul__(other)


def combine_entries(entries: Iterable[MomentEntry | MomentExpr]) -> MomentExpr:
    """Sum an iterable of entries or entry expressions into one expression."""
    result = MomentExpr(())
    for entry in entries:
        result = result + entry
    return result


def as_moment_expr(value: MomentEntry | MomentExpr) -> MomentExpr:
    """Wrap a bare MomentEntry into a MomentExpr (no-op otherwise).

    The construction layer consumes moment expressions; accepting a bare
    entry avoids forcing ``MomentEntry(...) + 0`` on the caller.
    """
    if isinstance(value, MomentEntry):
        return MomentExpr((value,))
    return value
