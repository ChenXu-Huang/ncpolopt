"""Monomial basis generation and degree bookkeeping.

A monomial is a word (product) of operators; the moment-matrix construction
needs the ordered list of all words up to a given degree, plus the degree
and support queries used to place them.
"""

from __future__ import annotations

from typing import Any

from sympy import I, Number, Pow, S

from .expressions import is_hermitian, is_number_type
from .substitutions import apply_substitutions, remove_scalar_factor

__all__ = [
    "get_all_monomials",
    "get_monomials",
    "ncdegree",
    "pick_monomials_of_degree",
    "pick_monomials_up_to_degree",
    "unique",
]


def unique(seq: list[Any]) -> list[Any]:
    """Return the elements of a sequence with duplicates removed.

    Preserves first-occurrence order.

    Args:
        seq: The sequence to deduplicate.

    Returns:
        A new list of unique elements.
    """
    seen: dict[Any, int] = {}
    result: list[Any] = []
    for item in seq:
        if item in seen:
            continue
        seen[item] = 1
        result.append(item)
    return result


def get_monomials(variables: list[Any], degree: int) -> list[Any]:
    """Generate all noncommutative monomials of degree up to ``degree``.

    Includes the adjoints of non-Hermitian variables; ``degree == -1``
    yields the empty basis.

    Args:
        variables: The operators to build words from.
        degree: The maximum degree.

    Returns:
        The monomial list, starting with the identity ``1``.
    """
    if degree == -1:
        return []
    if degree == 0:
        # The zero-degree basis is just the identity; the generic loop
        # below starts at degree one and would otherwise over-produce.
        return [S.One]
    if not variables:
        return [S.One]
    _variables = variables[:]
    _variables.insert(0, 1)
    ncmonomials = [S.One]
    ncmonomials.extend(var for var in variables)
    for var in variables:
        if not is_hermitian(var):
            ncmonomials.append(var.adjoint())
    for _ in range(1, degree):
        temp: list[Any] = []
        for var in _variables:
            for new_var in ncmonomials:
                temp.append(var * new_var)
                if var != 1 and not is_hermitian(var):
                    temp.append(var.adjoint() * new_var)
        ncmonomials = unique(temp[:])
    return ncmonomials


def get_all_monomials(
    variables: list[Any],
    extramonomials: list[Any] | None,
    substitutions: dict | None,
    degree: int,
    removesubstitutions: bool = True,
) -> list[Any]:
    """Return the monomial basis up to a degree, with extras and reductions.

    Args:
        variables: The operators to build words from.
        extramonomials: Additional monomials appended to the basis.
        substitutions: Rules whose left-hand sides are removed from the
            basis; the remaining monomials are reduced by the rules.
        degree: The maximum degree.
        removesubstitutions: Apply the substitution reduction.

    Returns:
        The deduplicated monomial basis.
    """
    monomials = get_monomials(variables, degree)
    if extramonomials is not None:
        monomials.extend(extramonomials)
    if removesubstitutions and substitutions is not None:
        monomials = [monomial for monomial in monomials if monomial not in substitutions]
        monomials = [
            remove_scalar_factor(apply_substitutions(monomial, substitutions))
            for monomial in monomials
        ]
    return unique(monomials)


def ncdegree(polynomial: Any) -> int:
    """Return the degree of a (possibly noncommutative) polynomial.

    Args:
        polynomial: The polynomial to measure.

    Returns:
        The maximum degree of its monomials; 0 for constants.
    """
    degree = 0
    if is_number_type(polynomial):
        return degree
    polynomial = polynomial.expand()
    for monomial in polynomial.as_coefficients_dict():
        subdegree = 0
        for variable in monomial.as_coeff_mul()[1]:
            if isinstance(variable, Pow):
                subdegree += variable.exp
            elif not isinstance(variable, Number) and variable != I:
                subdegree += 1
        if subdegree > degree:
            degree = subdegree
    return degree


def pick_monomials_of_degree(monomials: list[Any], degree: int) -> list[Any]:
    """Collect all monomials of exactly the given degree.

    Args:
        monomials: The monomial basis.
        degree: The degree to select.

    Returns:
        The monomials of that degree, in basis order.
    """
    return [monomial for monomial in monomials if ncdegree(monomial) == degree]


def pick_monomials_up_to_degree(monomials: list[Any], degree: int) -> list[Any]:
    """Collect the identity and all monomials of degree up to ``degree``.

    Args:
        monomials: The monomial basis.
        degree: The maximum degree.

    Returns:
        The identity followed by the selected monomials.
    """
    ordered_monomials: list[Any] = []
    if degree >= 0:
        ordered_monomials.append(S.One)
    for deg in range(1, degree + 1):
        ordered_monomials.extend(pick_monomials_of_degree(monomials, deg))
    return ordered_monomials
