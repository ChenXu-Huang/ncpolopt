"""Expression helpers: numeric predicates and relational conversion."""

from __future__ import annotations

import numpy as np
import pytest

from ncpolopt.expressions import is_number_type


def test_is_number_type_accepts_numpy_scalars() -> None:
    """NumPy scalar types count as numbers (no AttributeError).

    Regression: ``np.int64``/``np.float32`` carry no ``is_number``
    attribute, so the predicate crashed on them.
    """
    assert is_number_type(np.int64(3))
    assert is_number_type(np.float32(1.5))
    assert is_number_type(np.float64(2.5))
    assert is_number_type(np.complex128(1.0j))
    assert is_number_type(3)
    assert is_number_type(1.5)
    assert is_number_type(1.0j)


def test_is_number_type_rejects_symbols() -> None:
    """Symbols and operators are not numbers."""
    from ncpolopt.variables import generate_operators, generate_variables

    assert not is_number_type(generate_variables("x", 1)[0])
    assert not is_number_type(generate_operators("A", 1)[0])


def test_iscomplex_detects_complex_coefficients() -> None:
    """iscomplex covers plain complex, I-factored, and real cases."""
    import sympy as S

    from ncpolopt.expressions import iscomplex
    from ncpolopt.variables import generate_variables

    x = generate_variables("x", 1)[0]
    assert iscomplex(1.0j)
    assert iscomplex(S.I * x)
    assert iscomplex(1.0 + S.I)
    assert not iscomplex(1.5)
    assert not iscomplex(x + 1)


def test_convert_relational_variants() -> None:
    """All four comparison operators map to the >= 0 form; Ne raises."""
    import sympy as S

    from ncpolopt.expressions import convert_relational
    from ncpolopt.variables import generate_variables

    x = generate_variables("x", 1)[0]
    assert convert_relational(x >= 1) == x - 1
    assert convert_relational(x > 1) == x - 1
    assert convert_relational(x <= 1) == 1 - x
    assert convert_relational(x < 1) == 1 - x
    assert convert_relational(S.Eq(x, 1)) == x - 1
    with pytest.raises(ValueError, match="not implemented"):
        convert_relational(S.Ne(x, 1))


def test_convert_monomial_to_string() -> None:
    """The SDPA-style rendering: Dagger -> T, ** -> ^."""
    from sympy.physics.quantum import Dagger

    from ncpolopt.expressions import convert_monomial_to_string
    from ncpolopt.variables import generate_operators

    A = generate_operators("A0", 1)[0]
    assert convert_monomial_to_string(Dagger(A)) == "A0T"
    assert convert_monomial_to_string(A**2) == "A0^2"
