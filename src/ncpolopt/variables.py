"""Variable and operator factories, and polynomial support computations.

The symbolic core is SymPy: noncommuting operators are
``sympy.physics.quantum.Operator`` / ``HermitianOperator`` instances,
commuting variables are ``sympy.Symbol`` instances.
"""

from __future__ import annotations

from typing import Any

from sympy import Pow, Symbol
from sympy.physics.quantum import HermitianOperator, Operator

from .expressions import flatten, is_adjoint, is_number_type
from .substitutions import _separate_scalar_factor, split_commutative_parts

__all__ = [
    "find_variable_set",
    "generate_operators",
    "generate_variables",
    "get_support",
    "get_support_variables",
]


def generate_variables(
    name: str,
    n_vars: int = 1,
    hermitian: bool | None = None,
    commutative: bool = True,
) -> list[Symbol | Operator]:
    """Generate a number of commuting variables.

    Args:
        name: Prefix of the symbolic names; suffixed with ``0..n_vars-1``
            when ``n_vars > 1``.
        n_vars: Number of variables to generate.
        hermitian: Request real (True) or complex (False) symbols; None
            defaults to real symbols.
        commutative: If False, generate noncommuting operators instead of
            symbols.

    Returns:
        The generated variables.
    """
    variables: list[Symbol | Operator] = []
    for i in range(n_vars):
        var_name = f"{name}{i}" if n_vars > 1 else name
        if commutative:
            if hermitian is None or hermitian:
                variables.append(Symbol(var_name, real=True))
            else:
                variables.append(Symbol(var_name, complex=True))
        elif hermitian is not None and hermitian:
            variables.append(HermitianOperator(var_name))
        else:
            variables.append(Operator(var_name))
    return variables


def generate_operators(
    name: str,
    n_vars: int = 1,
    hermitian: bool | None = None,
    commutative: bool = False,
) -> list[Operator]:
    """Generate a number of noncommuting operators.

    Args:
        name: Prefix of the symbolic names; suffixed with ``0..n_vars-1``
            when ``n_vars > 1``.
        n_vars: Number of operators to generate.
        hermitian: Request ``HermitianOperator`` (True) or plain
            ``Operator`` (False/None).
        commutative: Mark the operators as commuting, which changes how
            SymPy orders factors inside products.

    Returns:
        The generated operators.
    """
    variables: list[Operator] = []
    for i in range(n_vars):
        var_name = f"{name}{i}" if n_vars > 1 else name
        if hermitian is not None and hermitian:
            variables.append(HermitianOperator(var_name))
        else:
            variables.append(Operator(var_name))
        # NOTE: SymPy decides factor ordering by ``is_commutative``, so the
        # flag must be set on each instance right after construction.
        variables[-1].is_commutative = commutative
    return variables


def _monomial_support(variables: list[Any], monomial: Any) -> list[int]:
    """Return the exponent vector of a monomial in the given variables.

    The support counts occurrences of each variable (adjoints counted on
    the adjoint's base).
    """
    tmp_support = [0] * len(variables)
    mon, _ = _separate_scalar_factor(monomial)
    symbolic_support = flatten(split_commutative_parts(mon))
    for s in symbolic_support:
        if isinstance(s, Pow):
            # Powers of operators (e.g. X0**2) count their base as well;
            # restricting to Symbol would drop them.
            base = s.base
            if is_adjoint(base):
                base = base.adjoint()
            tmp_support[variables.index(base)] = s.exp
        elif is_adjoint(s):
            tmp_support[variables.index(s.adjoint())] = 1
        elif isinstance(s, (Operator, Symbol)):
            tmp_support[variables.index(s)] = 1
    return tmp_support


def get_support(variables: list[Any], polynomial: Any) -> list[list[int]]:
    """Compute the support of a polynomial as exponent vectors.

    Args:
        variables: The problem variables the support is indexed against.
        polynomial: A polynomial in those variables.

    Returns:
        One exponent vector per monomial of the expanded polynomial; a
        constant polynomial yields a single all-zero vector.
    """
    if is_number_type(polynomial):
        return [[0] * len(variables)]
    support: list[list[int]] = []
    for monomial in polynomial.expand().as_coefficients_dict():
        support.append(_monomial_support(variables, monomial))
    return support


def get_support_variables(polynomial: Any) -> list[Any]:
    """Return the distinct variables appearing in a polynomial.

    Args:
        polynomial: A polynomial.

    Returns:
        The base variables (adjoints unwrapped) used by the polynomial;
        empty for a constant.
    """
    support: list[Any] = []
    if is_number_type(polynomial):
        return support
    for monomial in polynomial.expand().as_coefficients_dict():
        mon, _ = _separate_scalar_factor(monomial)
        symbolic_support = flatten(split_commutative_parts(mon))
        for s in symbolic_support:
            if isinstance(s, Pow):
                base = s.base
                if is_adjoint(base):
                    base = base.adjoint()
                support.append(base)
            elif is_adjoint(s):
                support.append(s.adjoint())
            elif isinstance(s, Operator):
                support.append(s)
    return support


def find_variable_set(variable_sets: Any, polynomial: Any) -> int:
    """Find which variable set fully contains the polynomial's variables.

    For problems with several sets of operators (e.g. Alice/Bob), returns
    the index of the first set that contains every variable of the
    polynomial.

    Args:
        variable_sets: A list of variable lists, or a single flat list.
        polynomial: The polynomial to place.

    Returns:
        The matching set index, 0 when ``variable_sets`` is flat, or -1 if
        no set contains the polynomial.
    """
    if not isinstance(variable_sets[0], list):
        return 0
    support = set(get_support_variables(polynomial))
    for i, variable_set in enumerate(variable_sets):
        if len(support - set(variable_set)) == 0:
            return i
    return -1
