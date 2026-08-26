"""Substitution machinery for noncommutative polynomials.

The old ``ncpol2sdpa`` package shipped two substitution paths: a careful
:func:`fast_substitute` that understands the word structure of products of
noncommuting operators, and a fixpoint wrapper :func:`apply_substitutions`
that iterates a rule set until no rule fires any more. Both are ported here
with their semantics preserved, since the moment-matrix construction relies
on them heavily.
"""

from __future__ import annotations

from typing import Any

from sympy import Pow, S, Symbol, expand
from sympy.physics.quantum import Operator

from .expressions import is_adjoint, is_number_type

__all__ = [
    "apply_substitutions",
    "assemble_monomial_and_do_substitutions",
    "constant_part",
    "fast_substitute",
    "is_pure_substitution_rule",
    "moment_of_entry",
    "poly_elements",
    "remove_scalar_factor",
    "separate_scalar_factor",
    "simplify_polynomial",
    "split_commutative_parts",
]


def split_commutative_parts(e: Any) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
    """Split an expression into its leading commutative and remaining parts.

    For a product ``2 * x * A * B`` with commuting ``x`` and operators
    ``A, B``, returns the commutative factors and the noncommutative tail.

    Args:
        e: Any SymPy expression.

    Returns:
        A pair ``(commutative_factors, noncommutative_factors)``; one of
        the tuples may be empty.
    """
    args = e.args if e.is_Mul else (e,)
    for _index, m in enumerate(args):
        if not m.is_commutative:
            break
    else:
        _index += 1
    return args[:_index], args[_index:]


def separate_scalar_factor(element: Any) -> tuple[Any, float]:
    """Separate the scalar coefficient from a term of a polynomial.

    Args:
        element: A single term (monomial with coefficient) or a number.

    Returns:
        A pair ``(monomial, coefficient)`` with the monomial freed of its
        scalar factor (imaginary units folded into the coefficient).
    """
    coeff = 1.0
    monomial = S.One
    if isinstance(element, (int, float, complex)):
        coeff *= element
        return monomial, coeff
    for var in element.as_coeff_mul()[1]:
        if not (var.is_Number or var.is_imaginary):
            monomial = monomial * var
        elif var.is_Number:
            coeff = float(var)
        else:
            coeff = 1j * coeff
    coeff = float(element.as_coeff_mul()[0]) * coeff
    return monomial, coeff


def remove_scalar_factor(monomial: Any) -> Any:
    """Return the monomial with its constant factor stripped."""
    monomial, _ = separate_scalar_factor(monomial)
    return monomial


def _separate_scalar_factor(monomial: Any) -> tuple[Any, Any]:
    """Return ``(monomial, scalar_factor)`` with SymPy-exact semantics.

    Unlike :func:`separate_scalar_factor`, keeps the factor symbolic
    (``S.One`` when there is none), which the support computations need.

    Args:
        monomial: A SymPy monomial or number.

    Returns:
        The monomial divided by its commutative scalar factor, and the
        scalar factor itself.
    """
    if is_number_type(monomial) or monomial == 0:
        return S.One, monomial
    comm_factors, _ = split_commutative_parts(monomial)
    scalar_factor = S.One
    if comm_factors and comm_factors[0].is_Number:
        scalar_factor = comm_factors[0]
    if scalar_factor != 1:
        return monomial / scalar_factor, scalar_factor
    return monomial, scalar_factor


def constant_part(polynomial: Any) -> float:
    """The constant term of a polynomial.

    ``Add.as_coeff_add`` moves the constant into the first slot for exact
    coefficients but leaves it among the terms for floats; the old code
    indexed only the terms and silently dropped constants depending on the
    coefficient types. Reading the slot explicitly makes the behavior
    type-independent.
    """
    if polynomial.is_Mul or is_number_type(polynomial):
        return 0.0
    return polynomial.as_coeff_add()[0]


def poly_elements(polynomial: Any) -> list[Any]:
    """The constituent monomials of a polynomial, constant excluded.

    ``as_coeff_mul()[1][0]`` unwraps a scalar-prefixed Add so the term list
    is plain monomials; each element still carries its own coefficient
    (``2*x`` stays a single element).
    """
    if polynomial.is_Mul:
        return [polynomial]
    return list(polynomial.as_coeff_mul()[1][0].as_coeff_add()[1])


def simplify_polynomial(polynomial: Any, monomial_substitutions: dict) -> Any:
    """Expand a polynomial and substitute each of its monomials.

    Used before processing constraints so every monomial is reduced to the
    basis.

    Args:
        polynomial: A polynomial in the problem variables.
        monomial_substitutions: Rules applied to each constituent monomial.

    Returns:
        The substituted polynomial.
    """
    if isinstance(polynomial, (int, float, complex)):
        return polynomial
    polynomial = (1.0 * polynomial).expand(mul=True, multinomial=True)
    if is_number_type(polynomial):
        return polynomial
    if polynomial.is_Mul:
        elements = [polynomial]
    else:
        elements = polynomial.as_coeff_mul()[1][0].as_coeff_add()[1]
    new_polynomial = 0
    for element in elements:
        monomial, coeff = separate_scalar_factor(element)
        monomial = apply_substitutions(monomial, monomial_substitutions)
        new_polynomial += coeff * monomial
    return new_polynomial


def is_pure_substitution_rule(lhs: Any, rhs: Any) -> bool:
    """Return whether a substitution rule maps within the lhs variable set.

    A rule is pure when every variable of ``rhs`` also occurs in ``lhs``;
    such rules cannot introduce new variables into the moment basis.

    Args:
        lhs: The monomial to replace.
        rhs: The replacement expression.
    """
    if is_number_type(rhs):
        return True
    elements = [rhs] if rhs.is_Mul else rhs.as_coeff_mul()[1][0].as_coeff_add()[1]
    for element in elements:
        monomial, _ = separate_scalar_factor(element)
        for atom in monomial.atoms():
            if atom.is_Number:
                continue
            if not lhs.has(atom):
                return False
    return True


def apply_substitutions(
    monomial: Any, monomial_substitutions: dict, pure: bool = False
) -> Any:
    """Iterate a rule set over a monomial until it stabilizes.

    Args:
        monomial: The expression to reduce.
        monomial_substitutions: Mapping ``lhs -> rhs`` of rules.
        pure: Only apply rules whose variables all occur in the monomial.

    Returns:
        The reduced expression.
    """
    if is_number_type(monomial):
        return monomial
    original_monomial = monomial
    if not pure:
        substitutions = monomial_substitutions
    else:
        substitutions = {}
        for lhs, rhs in monomial_substitutions.items():
            irrelevant = False
            for atom in lhs.atoms():
                if atom.is_Number:
                    continue
                if not monomial.has(atom):
                    irrelevant = True
                    break
            if not irrelevant:
                substitutions[lhs] = rhs
    changed = True
    while changed:
        for lhs, rhs in substitutions.items():
            monomial = fast_substitute(monomial, lhs, rhs)
        if original_monomial == monomial:
            changed = False
        original_monomial = monomial
    return monomial


def fast_substitute(monomial: Any, old_sub: Any, new_sub: Any) -> Any:
    """Substitute one word into another within a product of operators.

    The routine walks the noncommutative factors of ``monomial`` and
    replaces the first contiguous run matching ``old_sub``, keeping the
    leftover factors on either side. It understands daggered factors and
    powers, but only handles restricted cases of noncommutative algebras
    (e.g. no linear combinations on the left-hand side); on failure it
    returns the original expression.

    Args:
        monomial: The expression to substitute into.
        old_sub: The word to be replaced.
        new_sub: The replacement expression.

    Returns:
        The substituted expression; the original when no match is found.
    """
    # Numbers and sums reduce to their constituent terms.
    if is_number_type(monomial):
        return monomial
    if monomial.is_Add:
        return sum(
            fast_substitute(element, old_sub, new_sub)
            for element in monomial.as_ordered_terms()
        )

    comm_factors, ncomm_factors = split_commutative_parts(monomial)
    old_comm_factors, old_ncomm_factors = split_commutative_parts(old_sub)
    if not isinstance(new_sub, (int, float, complex)):
        new_comm_factors, _ = split_commutative_parts(new_sub)
    else:
        new_comm_factors = [new_sub]

    comm_monomial = 1
    is_constant_term = False
    if comm_factors != ():
        if len(comm_factors) == 1 and is_number_type(comm_factors[0]):
            is_constant_term = True
            comm_monomial = comm_factors[0]
        else:
            for comm_factor in comm_factors:
                comm_monomial *= comm_factor
            if old_comm_factors != ():
                comm_old_sub = 1
                for comm_factor in old_comm_factors:
                    comm_old_sub *= comm_factor
                comm_new_sub = 1
                for comm_factor in new_comm_factors:
                    comm_new_sub *= comm_factor
                # NOTE: Pow-valued commutative patterns cannot use
                # ``.subs`` reliably, so the factors are split by hand
                # (a SymPy quirk the old package worked around).
                if isinstance(comm_old_sub, Pow):
                    old_base = comm_old_sub.base
                    if comm_monomial.has(old_base):
                        old_degree = comm_old_sub.exp
                        new_monomial = 1
                        match = False
                        for factor in comm_monomial.as_ordered_factors():
                            if factor.has(old_base):
                                if isinstance(factor, Pow):
                                    degree = factor.exp
                                    if degree >= old_degree:
                                        match = True
                                        new_monomial *= (
                                            old_base ** (degree - old_degree)
                                            * comm_new_sub
                                        )
                                else:
                                    new_monomial *= factor
                            else:
                                new_monomial *= factor
                        if match:
                            comm_monomial = new_monomial
                else:
                    comm_monomial = comm_monomial.subs(comm_old_sub, comm_new_sub)
    if ncomm_factors == () or old_ncomm_factors == ():
        return comm_monomial

    new_var_list: list[Any] = []
    new_monomial = 1
    left_remainder = 1
    right_remainder = 1
    for i in range(len(ncomm_factors) - len(old_ncomm_factors) + 1):
        for j, old_ncomm_factor in enumerate(old_ncomm_factors):
            ncomm_factor = ncomm_factors[i + j]
            # A Symbol in the monomial only matches the same Symbol in the
            # pattern; anything else skips to the next starting position.
            if isinstance(ncomm_factor, Symbol) and (
                isinstance(old_ncomm_factor, Operator)
                or (
                    isinstance(old_ncomm_factor, Symbol)
                    and ncomm_factor != old_ncomm_factor
                )
            ):
                left_remainder, right_remainder = 1, 1
                break
            # An Operator in the monomial matches only the same operator
            # (never a Symbol or a power) in the pattern.
            if isinstance(ncomm_factor, Operator) and (
                (
                    isinstance(old_ncomm_factor, Operator)
                    and ncomm_factor != old_ncomm_factor
                )
                or isinstance(old_ncomm_factor, Pow)
            ):
                left_remainder, right_remainder = 1, 1
                break
            # A daggered factor matches only the identical daggered factor.
            if is_adjoint(ncomm_factor):
                if not is_adjoint(old_ncomm_factor) or ncomm_factor != old_ncomm_factor:
                    left_remainder, right_remainder = 1, 1
                    break
            else:
                if not isinstance(ncomm_factor, Pow):
                    if is_adjoint(old_ncomm_factor):
                        left_remainder, right_remainder = 1, 1
                        break
                else:
                    # Powers split into base and exponent; a partial match
                    # leaves a remainder on the left or right side.
                    if isinstance(old_ncomm_factor, Pow):
                        old_base = old_ncomm_factor.base
                        old_degree = old_ncomm_factor.exp
                    else:
                        old_base = old_ncomm_factor
                        old_degree = 1
                    if old_base != ncomm_factor.base:
                        left_remainder, right_remainder = 1, 1
                        break
                    if old_degree > ncomm_factor.exp:
                        left_remainder, right_remainder = 1, 1
                        break
                    if old_degree < ncomm_factor.exp:
                        if j != len(old_ncomm_factors) - 1:
                            if j != 0:
                                left_remainder, right_remainder = 1, 1
                                break
                            left_remainder = old_base ** (
                                ncomm_factor.exp - old_degree
                            )
                        else:
                            right_remainder = old_base ** (ncomm_factor.exp - old_degree)
        else:
            # The inner loop exhausted without a break: the pattern fits
            # starting at position i. Assemble the substituted product.
            new_monomial = 1
            for var in new_var_list:
                new_monomial *= var
            new_monomial *= left_remainder * new_sub * right_remainder
            for j in range(i + len(old_ncomm_factors), len(ncomm_factors)):
                new_monomial *= ncomm_factors[j]
            new_monomial *= comm_monomial
            break
        # No match at this position: keep the leading factor and retry
        # from the next position.
        new_var_list.append(ncomm_factors[i])
    else:
        if not is_constant_term and comm_factors != ():
            new_monomial = comm_monomial
            for factor in ncomm_factors:
                new_monomial *= factor
        else:
            return monomial
    if not isinstance(new_sub, (float, int, complex)) and new_sub.is_Add:
        return expand(new_monomial)
    return new_monomial


def assemble_monomial_and_do_substitutions(
    arg: tuple[int, int, int, int],
    monomialsA: list[Any],
    monomialsB: list[Any],
    ppt: bool,
    substitutions: dict,
    pure_substitution_rules: bool,
) -> tuple[int, int, int, Any]:
    """Assemble the moment monomial for one entry of a product moment matrix.

    For the Moroder hierarchy, entry ``(rowA, columnA, rowB, columnB)`` of
    the Alice/Bob tensor product maps to the word
    ``monomialsA[rowA].adjoint() * monomialsA[columnA] *
    monomialsB[rowB].adjoint() * monomialsB[columnB]`` (with the Bob pair
    transposed under the partial transposition ``ppt``), reduced by the
    substitution rules.

    Args:
        arg: The quadruple ``(rowA, columnA, rowB, columnB)``.
        monomialsA: The Alice monomial list.
        monomialsB: The Bob monomial list.
        ppt: Apply partial transposition to the Bob subsystem.
        substitutions: Monomial substitution rules.
        pure_substitution_rules: Restrict rules to those over the word's
            own variables.

    Returns:
        The tuple ``(columnA, rowB, columnB, monomial)`` consumed by the
        moment matrix builder.
    """
    rowA, columnA, rowB, columnB = arg
    if (not ppt) or (columnB >= rowB):
        monomial = (
            monomialsA[rowA].adjoint()
            * monomialsA[columnA]
            * monomialsB[rowB].adjoint()
            * monomialsB[columnB]
        )
    else:
        monomial = (
            monomialsA[rowA].adjoint()
            * monomialsA[columnA]
            * monomialsB[columnB].adjoint()
            * monomialsB[rowB]
        )
    monomial = apply_substitutions(monomial, substitutions, pure_substitution_rules)
    return columnA, rowB, columnB, monomial


def moment_of_entry(
    pos: tuple[int, int],
    monomials: list[Any],
    ineq: Any,
    substitutions: dict,
) -> tuple[int, int, Any]:
    """Compute the moment monomial for one entry of a localizing matrix.

    The localizing matrix of a constraint has entries
    ``monomials[row].adjoint() * ineq * monomials[column]``, reduced by the
    substitution rules.

    Args:
        pos: The ``(row, column)`` matrix position.
        monomials: The monomial list of the localizing block.
        ineq: The constraint polynomial.
        substitutions: Substitution rules applied to the product.

    Returns:
        The triple ``(row, column, polynomial)``.
    """
    row, column = pos
    if isinstance(ineq, str):
        return row, column, ineq
    return (
        row,
        column,
        simplify_polynomial(
            monomials[row].adjoint() * ineq * monomials[column], substitutions
        ),
    )
