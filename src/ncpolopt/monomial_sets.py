"""Generation of monomial bases and SDP variable-count estimation.

Ported from the legacy ``SdpRelaxation.__generate_monomial_sets`` and
``SdpRelaxation._estimate_n_vars`` methods. The old methods mutated
``self.monomial_sets`` in place; the new versions are pure functions of
the inputs.
"""

from __future__ import annotations

from typing import Any

from .monomials import get_all_monomials


def generate_monomial_sets(
    variables: list[Any] | list[list[Any]],
    level: int,
    extramonomials: list[Any] | list[list[Any]] | None,
    substitutions: dict[Any, Any],
) -> list[list[Any]]:
    """Generate one monomial basis per moment matrix block.

    At level -1 no automatic generation happens: the caller must supply the
    full bases. A list of lists is then taken verbatim as the per-block
    specification, a flat list as the single block basis.

    Multi-partite problems pass ``variables`` as a list of variable lists;
    each variable set receives its own monomial basis, paired with the
    corresponding entry of ``extramonomials``.

    Args:
        variables: A list of variables, or a list of such lists.
        level: The relaxation level; -1 skips automatic generation.
        extramonomials: Monomials to add on top of the level basis (or the
            full basis at level -1). May also be a list of such lists.
        substitutions: Monomial replacements applied to the generated bases.

    Returns:
        One monomial list per moment matrix block.

    Raises:
        ValueError: At level -1 without any extramonomials.
    """
    if level == -1:
        if extramonomials is None or not extramonomials:
            raise ValueError(
                "Cannot build relaxation at level -1 without monomials specified."
            )
        if isinstance(extramonomials[0], list):
            return list(extramonomials)
        return [list(extramonomials)]
    if isinstance(variables[0], list):
        result: list[list[Any]] = []
        for k, variable_set in enumerate(variables):
            extra = None
            if extramonomials is not None:
                extra = extramonomials[k]
            result.append(
                get_all_monomials(variable_set, extra, substitutions, level)
            )
        return result
    if (
        extramonomials is not None
        and len(extramonomials) > 0
        and isinstance(extramonomials[0], list)
    ):
        return [get_all_monomials(variables, extramonomials[0], substitutions, level), *extramonomials[1:]]
    return [get_all_monomials(variables, extramonomials, substitutions, level)]


def estimate_n_vars(
    monomial_sets: list[list[Any]],
    parameters: list[Any] | None,
    normalized: bool,
) -> int:
    """Estimate the number of SDP variables before construction.

    A moment block over n monomials contributes n(n + 1)/2 upper-triangular
    entries, minus one when the relaxation is normalized: the constant term is
    pinned to 1.0 in the top-left corner instead of becoming a free variable.

    Args:
        monomial_sets: One monomial list per moment matrix block.
        parameters: Parameter variables, one SDP variable each.
        normalized: Whether the normalization condition is imposed.

    Returns:
        The estimated number of SDP variables.
    """
    n_vars = len(parameters) if parameters is not None else 0
    for monomials in monomial_sets:
        if len(monomials) > 0 and isinstance(monomials[0], list):
            # A rectangular set [A, B] spans the tensor product basis.
            n_monomials = len(monomials[0]) * len(monomials[1])
        else:
            n_monomials = len(monomials)
        # The minus one compensates for the constant term in the top left
        # corner of the moment matrix.
        n_vars += int(n_monomials * (n_monomials + 1) / 2)
        if normalized:
            n_vars -= 1
    return n_vars
