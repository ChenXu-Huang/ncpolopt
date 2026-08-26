"""Physics helpers: ladder operators, measurement projectors, Bell scenarios.

Ported from physics_utils.py (src.old/ncpol2sdpa/physics_utils.py). The
functions generate the constraint sets of bosonic/fermionic ladder operators
and of projective measurements, build Collins-Gisin Bell expressions, and
drive the full maximum-violation workflow. The old ``maximum_violation``
built and solved its own relaxation; here it delegates to
:class:`~ncpolopt.problem.Problem`, and the ``solver``/``settings`` knobs of
the new solve pipeline are exposed.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from sympy import S
from sympy.physics.quantum.dagger import Dagger

from .expressions import flatten
from .problem import Problem
from .solvers.base import SolverKind, SolverSettings
from .variables import generate_operators


def get_neighbors(
    index: int, lattice_length: int, width: int = 0, periodic: bool = False
) -> list[int]:
    """The forward neighbors of a site in a lattice.

    The lattice is ``lattice_length`` sites wide; ``width`` overrides the
    wrap-around of the row (useful for 2D lattices visited row-wise). Only
    forward neighbors are returned, so the pairing is counted once.

    Args:
        index: Linear index of the site.
        lattice_length: Size of the lattice in either dimension.
        width: Row width; defaults to the lattice length.
        periodic: Whether opposite edges are neighbors.

    Returns:
        The neighbor indices in linear index space.
    """
    if width == 0:
        width = lattice_length
    neighbors = []
    coords = divmod(index, width)
    if coords[1] < width - 1:
        neighbors.append(index + 1)
    elif periodic and width > 1:
        neighbors.append(index - width + 1)
    if coords[0] < lattice_length - 1:
        neighbors.append(index + width)
    elif periodic:
        neighbors.append(index - (lattice_length - 1) * width)
    return neighbors


def get_next_neighbors(
    indices: int | list[int],
    lattice_length: int,
    width: int = 0,
    distance: int = 1,
    periodic: bool = False,
) -> list[int]:
    """The forward neighbors of a site at a given distance.

    Each step moves to the forward neighbors of the current frontier; sites
    reachable in fewer steps are excluded, so a site at distance d is never
    reported as its own distance-d neighbor on a periodic lattice.

    Args:
        indices: Linear index or list of indices of the sites.
        lattice_length: Size of the lattice in either dimension.
        width: Row width; defaults to the lattice length.
        distance: The distance of the neighbors.
        periodic: Whether opposite edges are neighbors.

    Returns:
        The neighbor indices at the given distance.
    """
    if not isinstance(indices, list):
        indices = [indices]
    if distance == 1:
        return flatten(
            get_neighbors(index, lattice_length, width, periodic)
            for index in indices
        )
    s1 = set(
        flatten(
            get_next_neighbors(
                get_neighbors(index, lattice_length, width, periodic),
                lattice_length,
                width,
                distance - 1,
                periodic,
            )
            for index in indices
        )
    )
    s2 = set(
        get_next_neighbors(
            indices, lattice_length, width, distance - 1, periodic
        )
    )
    return list(s1 - s2)


def bosonic_constraints(a: list[Any]) -> dict[Any, Any]:
    """Substitution rules that make a list of operators bosonic.

    The commutation relations are ``[a_i, a_j^dagger] = 1`` on-site and
    commuting otherwise; every product is rewritten into the ordered form.

    Args:
        a: The non-Hermitian ladder operators.

    Returns:
        The substitutions defining the bosonic algebra.
    """
    substitutions = {}
    for i, ai in enumerate(a):
        substitutions[ai * Dagger(ai)] = 1.0 + Dagger(ai) * ai
        for aj in a[i + 1 :]:
            substitutions[ai * Dagger(aj)] = Dagger(aj) * ai
            substitutions[Dagger(ai) * aj] = aj * Dagger(ai)
            substitutions[ai * aj] = aj * ai
            substitutions[Dagger(ai) * Dagger(aj)] = Dagger(aj) * Dagger(ai)
    return substitutions


def fermionic_constraints(a: list[Any]) -> dict[Any, Any]:
    """Substitution rules that make a list of operators fermionic.

    Every product is rewritten into the ordered form with the sign of the
    permutation; the on-site anticommutation ``{a_i, a_i^dagger} = 1`` pins
    the occupation number.

    Args:
        a: The non-Hermitian ladder operators.

    Returns:
        The substitutions defining the fermionic algebra.
    """
    substitutions = {}
    for i, ai in enumerate(a):
        substitutions[ai**2] = 0
        substitutions[Dagger(ai) ** 2] = 0
        substitutions[ai * Dagger(ai)] = 1.0 - Dagger(ai) * ai
        for aj in a[i + 1 :]:
            substitutions[ai * Dagger(aj)] = -Dagger(aj) * ai
            substitutions[Dagger(ai) * aj] = -aj * Dagger(ai)
            substitutions[ai * aj] = -aj * ai
            substitutions[Dagger(ai) * Dagger(aj)] = -Dagger(aj) * Dagger(ai)
    return substitutions


def pauli_constraints(
    X: list[Any], Y: list[Any], Z: list[Any]
) -> dict[Any, Any]:
    """Substitution rules that define Pauli spin operators.

    Each site squares to the identity and anticommutes internally; operators
    on different sites commute.

    Args:
        X: The Pauli-X operators, one per site.
        Y: The Pauli-Y operators, one per site.
        Z: The Pauli-Z operators, one per site.

    Returns:
        The substitutions defining the spin algebra.
    """
    substitutions = {}
    n_vars = len(X)
    for i in range(n_vars):
        # They square to the identity.
        substitutions[X[i] * X[i]] = 1
        substitutions[Y[i] * Y[i]] = 1
        substitutions[Z[i] * Z[i]] = 1
        # Anticommutation relations within a site.
        substitutions[Y[i] * X[i]] = -X[i] * Y[i]
        substitutions[Z[i] * X[i]] = -X[i] * Z[i]
        substitutions[Z[i] * Y[i]] = -Y[i] * Z[i]
        # They commute between the sites.
        for j in range(i + 1, n_vars):
            substitutions[X[j] * X[i]] = X[i] * X[j]
            substitutions[Y[j] * Y[i]] = Y[i] * Y[j]
            substitutions[Y[j] * X[i]] = X[i] * Y[j]
            substitutions[Y[i] * X[j]] = X[j] * Y[i]
            substitutions[Z[j] * Z[i]] = Z[i] * Z[j]
            substitutions[Z[j] * X[i]] = X[i] * Z[j]
            substitutions[Z[i] * X[j]] = X[j] * Z[i]
            substitutions[Z[j] * Y[i]] = Y[i] * Z[j]
            substitutions[Z[i] * Y[j]] = Y[j] * Z[i]
    return substitutions


def generate_measurements(party: list[int], label: str) -> list[list[Any]]:
    """Generate hermitian variables that behave like measurement operators.

    One list per measurement input, with one projector fewer than the number
    of outputs: the Collins-Gisin picture drops the last projector of each
    measurement, which is recovered as the identity minus the sum of the
    others.

    Args:
        party: The number of measurement outputs per input.
        label: The prefix of the generated variable names.

    Returns:
        The measurement operators, indexed by input and by output.
    """
    measurements = []
    for i in range(len(party)):
        measurements.append(
            generate_operators(f"{label}{i}", party[i] - 1, hermitian=True)
        )
    return measurements


def projective_measurement_constraints(
    *parties: Any,
) -> dict[Any, Any]:
    """Substitution rules that define projective measurements.

    Projectors of one measurement are idempotent and pairwise orthogonal;
    projectors of different parties commute.

    Args:
        parties: The measurements of each party (or a single list of them).

    Returns:
        The substitutions defining the measurement algebra.
    """
    substitutions = {}
    if isinstance(parties[0][0][0], list):
        parties = parties[0]
    # Idempotency and orthogonality of projectors.
    for party in parties:
        for measurement in party:
            for projector1 in measurement:
                for projector2 in measurement:
                    if projector1 == projector2:
                        substitutions[projector1**2] = projector1
                    else:
                        substitutions[projector1 * projector2] = 0
                        substitutions[projector2 * projector1] = 0
    # Projectors commute between parties in a partition.
    for n1 in range(len(parties)):
        for n2 in range(n1 + 1, len(parties)):
            for measurement1 in parties[n1]:
                for measurement2 in parties[n2]:
                    for projector1 in measurement1:
                        for projector2 in measurement2:
                            substitutions[projector2 * projector1] = (
                                projector1 * projector2
                            )
    return substitutions


def define_objective_with_I(
    I_matrix: Sequence[Sequence[float]], *args: Any
) -> Any:
    """The minimization objective of a Bell inequality in Collins-Gisin form.

    The I matrix is expanded against the measurement operators (or a
    :class:`Probability`); the sign is flipped so that minimizing finds the
    maximum quantum violation.

    Args:
        I_matrix: The I matrix of the Bell inequality in Collins-Gisin
            notation.
        args: Either the measurements of Alice and Bob, or a single
            :class:`Probability` describing them.

    Returns:
        The objective function to minimize.

    Raises:
        ValueError: If the number of arguments is not one or two.
    """
    objective = I_matrix[0][0]
    if len(args) > 2 or len(args) == 0:
        raise ValueError(
            "define_objective_with_I expects the measurements of two parties "
            "or a single Probability object."
        )
    if len(args) == 1:
        A = args[0].parties[0]
        B = args[0].parties[1]
    else:
        A = args[0]
        B = args[1]
    i, j = 0, 1  # Row and column index in I.
    for m_Bj in B:  # Define the first row.
        for Bj in m_Bj:
            objective += I_matrix[i][j] * Bj
            j += 1
    i += 1
    for m_Ai in A:
        for Ai in m_Ai:
            objective += I_matrix[i][0] * Ai
            j = 1
            for m_Bj in B:
                for Bj in m_Bj:
                    objective += I_matrix[i][j] * Ai * Bj
                    j += 1
            i += 1
    return -objective


def correlator(A: list[list[Any]], B: list[list[Any]]) -> list[list[Any]]:
    """The correlator matrix of two parties' measurement operators.

    The correlator of output lists ``(k, l)`` contributes +1 when the
    outputs coincide and -1 otherwise.

    Args:
        A: The measurements of Alice.
        B: The measurements of Bob.

    Returns:
        One row per measurement of Alice, one column per measurement of Bob.
    """
    correlators = []
    for i in range(len(A)):
        correlator_row = []
        for j in range(len(B)):
            corr = 0
            for output_a in range(len(A[i])):
                for output_b in range(len(B[j])):
                    if output_a == output_b:
                        corr += A[i][output_a] * B[j][output_b]
                    else:
                        corr -= A[i][output_a] * B[j][output_b]
            correlator_row.append(corr)
        correlators.append(correlator_row)
    return correlators


def maximum_violation(
    A_configuration: list[int],
    B_configuration: list[int],
    I_matrix: list[list[float]],
    level: int,
    extra: str | list[str] | None = None,
    solver: SolverKind | str | None = "auto",
    settings: SolverSettings | None = None,
) -> tuple[float, float]:
    """Compute the maximum quantum violation of a two-party Bell inequality.

    Builds the Collins-Gisin probability picture of the scenario, flips the
    Bell expression into a minimization objective, and solves the level-
    ``level`` NPA relaxation.

    Args:
        A_configuration: Number of measurement outputs per input for Alice.
        B_configuration: Number of measurement outputs per input for Bob.
        I_matrix: The I matrix of the Bell inequality in Collins-Gisin
            notation.
        level: The relaxation level.
        extra: Party labels of the extra monomials for the level basis
            (e.g. ``"AB"``); None for none.
        solver: The solver kind, its string value, or "auto" for the first
            available backend.
        settings: Backend knobs.

    Returns:
        The primal and dual optimal values.
    """
    P = Probability(A_configuration, B_configuration)
    objective = define_objective_with_I(I_matrix, P)
    extramonomials = P.get_extra_monomials(extra) if extra is not None else None
    problem = Problem(
        P.get_all_operators(),
        objective=objective,
        substitutions=P.substitutions,
        extramonomials=extramonomials,
    )
    solution = problem.solve(level, solver=solver, settings=settings)
    return solution.primal, solution.dual


class Probability:
    """Quantum probabilities p(output|input) of a Bell scenario.

    For a CHSH scenario, instantiate as ``Probability([2, 2], [2, 2])`` and
    call the instance to obtain probabilities in the p(ab...|xy...) notation:

        >>> P = Probability([2, 2], [2, 2])
        >>> P([1, 0], [0, 1])  # p(10|01)
        E0_0*E1_1

    The operators of the last output of each measurement are not generated:
    the Collins-Gisin picture recovers them on call as the identity minus
    the sum of the others.

    Attributes:
        n_parties: The number of parties.
        parties: One list of measurements per party.
        labels: The label of each party.
        substitutions: The projective measurement constraints.
    """

    def __init__(
        self, *args: list[int], labels: list[str] | None = None
    ) -> None:
        """Create the probability picture of a Bell scenario.

        Args:
            args: The input configurations, one list of outputs per party.
            labels: The label of each party; defaults to "A", "B", ...

        Raises:
            ValueError: If the number of labels does not match the parties.
        """
        self.n_parties = len(args)
        if labels is not None and len(labels) != self.n_parties:
            raise ValueError("Incorrect number of labels!")
        self.labels = [chr(ord("A") + i) for i in range(self.n_parties)]
        if labels is not None:
            self.labels = labels
        self.parties = []
        for i, configuration in enumerate(args):
            self.parties.append(
                generate_measurements(configuration, self.labels[i])
            )
        self.substitutions = projective_measurement_constraints(self.parties)

    def get_all_operators(self) -> list[Any]:
        """All operators across all parties and measurements.

        Returns:
            The flat operator list, suitable as the variables of a
            :class:`~ncpolopt.problem.Problem`.
        """
        return flatten(self.parties)

    def _monomial_generator(
        self, monomials: list[Any], label_indices: list[int]
    ) -> list[Any]:
        """All products of operators of the parties at ``label_indices``."""
        if label_indices == []:
            return monomials
        if monomials == []:
            return self._monomial_generator(
                flatten(self.parties[label_indices[0]]), label_indices[1:]
            )
        result = [
            m1 * m2
            for m1 in monomials
            for m2 in flatten(self.parties[label_indices[0]])
        ]
        return self._monomial_generator(result, label_indices[1:])

    def get_extra_monomials(self, *args: Any) -> list[Any]:
        """The monomials over the parties named in ``args``.

        Args:
            args: Party labels (or a single list of them), e.g. ``"AB"``.

        Returns:
            The product monomials, one per element of ``args``.
        """
        if len(args) == 0:
            return []
        if isinstance(args[0], list):
            args = args[0]
        extra_monomials = []
        for s in args:
            label_indices = [self.labels.index(party) for party in s]
            extra_monomials.extend(self._monomial_generator([], label_indices))
        return extra_monomials

    def _convert_marginal_index(
        self, marginal: str | list[str | int]
    ) -> list[int]:
        """Map a marginal specification onto party indices."""
        if isinstance(marginal, str):
            return [self.labels.index(marginal)]
        return sorted(
            self.labels.index(m) if isinstance(m, str) else m for m in marginal
        )

    def __call__(
        self,
        output_: list[int],
        input_: list[int],
        marginal: str | list[str | int] | None = None,
    ) -> Any:
        """The probability p(output|input) as a polynomial of projectors.

        For the CHSH scenario, ``P([1, 0], [0, 1])`` gives p(10|01) and
        ``P([0], [1], ['A'])`` the marginal p_A(0|1). The Collins-Gisin
        convention applies: a requested projector beyond the generated ones
        is replaced by the identity minus the sum of the others.

        Args:
            output_: The conditional output, one entry per party.
            input_: The input conditioned on, one entry per party.
            marginal: The parties the marginal belongs to; None requests the
                full joint probability.

        Returns:
            The probability polynomial over the measurement operators.

        Raises:
            ValueError: If the inputs, outputs, or marginal are inconsistent
                with the scenario.
        """
        if len(output_) != len(input_):
            raise ValueError(
                "The number of inputs does not match the number of outputs!"
            )
        if len(input_) > self.n_parties:
            raise ValueError("The number of inputs exceeds the number of parties!")
        if marginal is None and len(input_) < self.n_parties:
            raise ValueError("Marginal requested, but without defining which!")
        if marginal is None:
            marginal = self._convert_marginal_index(self.labels)
        else:
            marginal = self._convert_marginal_index(marginal)
            if len(marginal) != len(input_):
                raise ValueError(
                    "The number of parties in the marginal does not match "
                    "the number of inputs!"
                )
        result = S.One
        for party, (proj, meas) in enumerate(zip(output_, input_, strict=True)):
            if len(self.parties[marginal[party]]) < meas + 1:
                raise ValueError(
                    f"Invalid measurement index {meas} for party "
                    f"{self.labels[party]}"
                )
            if len(self.parties[marginal[party]][meas]) < proj:
                raise ValueError(
                    f"Invalid projection operator index {proj} for party "
                    f"{self.labels[party]}"
                )
            if len(self.parties[marginal[party]][meas]) == proj:
                # The last projector of the measurement is not part of the
                # Collins-Gisin picture; it is the complement.
                result *= S.One - sum(
                    op for op in self.parties[marginal[party]][meas]
                )
            else:
                result *= self.parties[marginal[party]][meas][proj]
        return result
