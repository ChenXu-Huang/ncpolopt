"""dat-s writing and SDPA output parsing (pure functions, no binary needed).

The SDPA standard form is ``min c.x`` subject to
``sum_k F_k x_k - F0 >= 0``; the constant matrix F0 is therefore written
negated while the variable matrices keep their coefficients. Complex
coefficients are embedded per block into one doubled real block
``[[X, Y], [-Y, X]]`` with ``Z = X + iY``, doubling the variable count
(the imaginary part of variable k is variable ``k + n_vars``).

The parsing side reconstructs the per-block matrices from the row syntax
``xMat = { {{...}, {...}} {...} }`` of the solver output file.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from .sdp_problem import SdpProblem
from .solvers.base import SolverResult

_COMPLEX_DTYPE = np.dtype(np.complex128)


def parse_solution_matrix(iterator: Iterable[str]) -> list[np.ndarray]:
    """Parse the per-block solution matrices of an SDPA output stream.

    Each block matrix is a list of rows ``{a, b, ...}``; a row that also
    closes its block ends with a second ``}``. A bare ``}`` line closes the
    current block, or ends the whole matrix list when no block is open.

    Args:
        iterator: The lines of the SDPA output file, positioned after the
            ``xMat =`` / ``yMat =`` header line.

    Returns:
        One dense matrix per block, in block order.
    """
    matrices: list[np.ndarray] = []
    rows: list[list[float]] = []
    for line in iterator:
        if "}" not in line:
            # Header and opener lines ("xMat =", "{") carry no data.
            continue
        stripped = line.strip()
        if stripped == "}":
            if rows:
                matrices.append(np.asarray(rows))
                rows = []
            else:
                break  # The closing brace of the whole matrix list.
            continue
        numbers = stripped[stripped.rfind("{") + 1 : stripped.find("}")]
        row = [float(n) for n in numbers.split(",") if n.strip()]
        rows.append(row)
        if stripped.find("}") != stripped.rfind("}"):
            # The row also closes its block, e.g. "{0.5}}".
            matrices.append(np.asarray(rows))
            rows = []
    return matrices


def _parse_phase(line: str) -> str:
    """Map an SDPA ``phase.value`` line onto the status vocabulary.

    Args:
        line: The ``phase.value = ...`` line of the output file.

    Returns:
        The canonical status string.
    """
    if "pdOPT" in line:
        return "optimal"
    if "pFEAS" in line:
        return "primal feasible"
    if "pdFEAS" in line:
        return "primal-dual feasible"
    if "dFEAS" in line:
        return "dual feasible"
    if "INF" in line:
        return "infeasible"
    if "UNBD" in line:
        return "unbounded"
    return "unknown"


def read_sdpa_out(filename: str) -> SolverResult:
    """Parse an SDPA output file into a solver result.

    The objective values are reported as stored; the caller adds the
    constant term. Missing objective values or solution matrices mark the
    status as ``"invalid"``.

    Args:
        filename: The path of the ``.out`` file.

    Returns:
        The parsed result; ``primal``/``dual`` are None when missing.
    """
    primal: float | None = None
    dual: float | None = None
    x_mat: tuple[np.ndarray, ...] | None = None
    y_mat: tuple[np.ndarray, ...] | None = None
    status_string: str | None = None
    with open(filename) as file_:
        for line in file_:
            if "objValPrimal" in line:
                primal = float(line.split()[2])
            elif "objValDual" in line:
                dual = float(line.split()[2])
            elif "phase.value" in line:
                status_string = _parse_phase(line)
            elif "xMat =" in line:
                x_mat = tuple(parse_solution_matrix(file_))
            elif "yMat =" in line:
                y_mat = tuple(parse_solution_matrix(file_))
    if None in (primal, dual, status_string, x_mat, y_mat):
        status_string = "invalid"
    return SolverResult(
        status=status_string or "invalid",
        primal=primal,
        dual=dual,
        x_mat=x_mat or (),
        y_mat=y_mat or (),
    )


def write_dat_s(problem: SdpProblem, filename: str) -> None:
    """Write a frozen SDP problem to a .dat-s file.

    Args:
        problem: The frozen SDP problem.
        filename: The output path (conventionally with a ``.dat-s`` suffix).
    """
    complex_matrix = any(
        block.coo.dtype == _COMPLEX_DTYPE for block in problem.blocks
    )
    multiplier = 2 if complex_matrix else 1
    lines: list[list[str]] = [[] for _ in range(multiplier * problem.n_vars + 1)]
    for block_index, block in enumerate(problem.blocks):
        size = block.size
        for k, pos, value in zip(
            block.coo.row, block.coo.col, block.coo.data, strict=True
        ):
            i, j = divmod(int(pos), size)
            if k == 0:
                value = -value
            if not complex_matrix:
                lines[int(k)].append(
                    f"{block_index + 1}\t{i + 1}\t{j + 1}\t{value}"
                )
            else:
                real, imag = float(value.real), float(value.imag)
                if real != 0:
                    lines[int(k)].append(
                        f"{block_index + 1}\t{i + 1}\t{j + 1}\t{real}"
                    )
                    lines[int(k)].append(
                        f"{block_index + 1}\t{i + size + 1}\t"
                        f"{j + size + 1}\t{real}"
                    )
                if imag != 0:
                    lines[int(k) + problem.n_vars].append(
                        f"{block_index + 1}\t{i + 1}\t{j + size + 1}\t{imag}"
                    )
                    lines[int(k) + problem.n_vars].append(
                        f"{block_index + 1}\t{j + 1}\t{i + size + 1}\t{-imag}"
                    )
    with open(filename, "w") as file_:
        file_.write(f'"file {filename} generated by ncpolopt"\n')
        file_.write(f"{multiplier * problem.n_vars} = number of vars\n")
        file_.write(f"{len(problem.blocks)} = number of blocs\n")
        structure = [multiplier * block.size for block in problem.blocks]
        file_.write(
            str(structure).replace("[", "(").replace("]", ")")
            + " = BlocStructure\n"
        )
        # The objective coefficients of a hermitian problem are real; the
        # imaginary part of the dtype is never stored in the file.
        objective = ", ".join(str(float(np.real(c))) for c in problem.obj)
        if multiplier == 2:
            objective += ", " + objective
        file_.write("{" + objective + "}\n")
        for k, entries in enumerate(lines):
            for entry in entries:
                file_.write(f"{k}\t{entry}\n")
