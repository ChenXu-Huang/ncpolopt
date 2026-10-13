"""Expression helpers: numeric predicates and relational conversion."""

from __future__ import annotations

import numpy as np

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
