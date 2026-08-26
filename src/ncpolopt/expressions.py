"""Expression-level helpers: type predicates, relational conversion, strings.

This is the leaf module of the algebra layer: it depends only on SymPy and
the standard library, so the rest of the package can build on it without
import cycles.
"""

from __future__ import annotations

from typing import Any

from sympy import I, adjoint, conjugate

__all__ = [
    "convert_monomial_to_string",
    "convert_relational",
    "flatten",
    "is_adjoint",
    "is_hermitian",
    "is_number_type",
    "iscomplex",
]


def flatten(lol: Any) -> list[Any]:
    """Flatten an arbitrarily nested list of lists into a single list.

    None entries are dropped; tuples are flattened like lists.

    Args:
        lol: Possibly nested list, or any other value.

    Returns:
        A flat list of the non-list leaf values, in depth-first order.
    """
    new_list: list[Any] = []
    for element in lol:
        if element is None:
            continue
        if not isinstance(element, (list, tuple)):
            new_list.append(element)
        elif len(element) > 0:
            new_list.extend(flatten(element))
    return new_list


def is_number_type(exp: Any) -> bool:
    """Return whether the expression is a plain number (or SymPy number).

    Args:
        exp: Any expression.

    Returns:
        True for int/float/complex literals and SymPy numeric expressions.
    """
    return isinstance(exp, (int, float, complex)) or exp.is_number


def is_adjoint(exp: Any) -> bool:
    """Return whether the expression is a Dagger or conjugate wrapper."""
    return isinstance(exp, (adjoint, conjugate))


def is_hermitian(exp: Any) -> bool:
    """Return whether the expression is known to be Hermitian.

    SymPy marks Hermitian operators with ``is_hermitian = True``; plain
    commutative symbols are Hermitian when they are declared real.
    """
    return exp.is_hermitian or (exp.is_hermitian is None and exp.is_complex)


def iscomplex(polynomial: Any) -> bool:
    """Return whether a polynomial has any complex coefficient.

    Args:
        polynomial: A SymPy expression or numeric literal.

    Returns:
        True if expansion exposes a complex literal or the imaginary unit.
    """
    if isinstance(polynomial, (int, float)):
        return False
    if isinstance(polynomial, complex):
        return True
    polynomial = polynomial.expand()
    for monomial in polynomial.as_coefficients_dict():
        for variable in monomial.as_coeff_mul()[1]:
            if isinstance(variable, complex) or variable == I:
                return True
    return False


def convert_relational(relational: Any) -> Any:
    """Convert a SymPy relational into the ``expression >= 0`` form.

    Args:
        relational: A SymPy ``relational`` such as ``x >= 1`` or ``x < 2``.

    Returns:
        The expression equivalent to the constraint ``expression >= 0``.

    Raises:
        ValueError: If the relational operator is not one of the four
            comparison operators.
    """
    rel = relational.rel_op
    if rel in ("==", ">=", ">"):
        return relational.lhs - relational.rhs
    if rel in ("<=", "<"):
        return relational.rhs - relational.lhs
    raise ValueError(f"The relational operation {rel!r} is not implemented!")


def convert_monomial_to_string(monomial: Any) -> str:
    """Render a monomial as the SDPA-style string ``X1^2`` notation.

    ``Dagger(X0)`` becomes ``X0T``, powers use ``^``.
    """
    monomial_str = f"{monomial}"
    monomial_str = monomial_str.replace("Dagger(", "")
    monomial_str = monomial_str.replace(")", "T")
    monomial_str = monomial_str.replace("**", "^")
    return monomial_str
