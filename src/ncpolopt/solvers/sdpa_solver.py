"""The external SDPA binary backend.

Ported from the legacy ``solve_with_sdpa`` function. The .dat-s writing
and .out parsing live in :mod:`ncpolopt.sdpa_writer` as pure
functions, so they are testable without the binary; this module only shells
out to ``sdpa`` and applies the canonical constant-term conventions.

The old code allocated a named temporary file whose name was reused after
closing it -- a classic TOCTOU race (bug #9); a :class:`tempfile.TemporaryDirectory`
removes the race. The old ``which()`` helper hand-rolled PATH lookup;
:func:`shutil.which` is the platform-correct replacement (it also finds
``sdpa.exe`` on Windows).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import time

from ..sdp_problem import SdpProblem
from ..sdpa_writer import read_sdpa_out, write_dat_s
from .base import (
    SolverError,
    SolverKind,
    SolverResult,
    SolverSettings,
)
from .registry import register


def solve_with_sdpa(problem: SdpProblem, settings: SolverSettings) -> SolverResult:
    """Write the problem to a .dat-s file, call SDPA, and parse the output.

    Args:
        problem: The frozen SDP problem.
        settings: Backend knobs; ``settings.sdpa_executable`` overrides the
            ``sdpa`` binary, and ``settings.solver_options`` may contain
            ``"paramsfile"`` (path of an SDPA parameter file, passed as
            ``-p``) or ``"executable"`` (legacy alias for the binary path).

    Returns:
        The solver result in the canonical conventions.

    Raises:
        SolverError: If the SDPA binary cannot be found on PATH.
        ValueError: If an unknown key is given in ``solver_options``.
    """
    options = dict(settings.solver_options)
    executable = options.pop("executable", None) or settings.sdpa_executable or "sdpa"
    binary = shutil.which(executable)
    if binary is None:
        raise SolverError(
            f"The SDPA executable {executable!r} was not found on PATH; "
            f"install SDPA and put it on PATH, or pass its location via "
            f"settings.sdpa_executable."
        )
    with tempfile.TemporaryDirectory() as directory:
        dats_filename = os.path.join(directory, "problem.dat-s")
        out_filename = os.path.join(directory, "problem.out")
        write_dat_s(problem, dats_filename)
        command_line = [binary, "-ds", dats_filename, "-o", out_filename]
        for key, value in options.items():
            if key == "paramsfile":
                command_line.extend(["-p", value])
            else:
                raise ValueError(f"Unknown parameter for SDPA: {key}")
        tstart = time.monotonic()
        if settings.verbose:
            subprocess.run(command_line)
        else:
            with open(os.devnull, "w") as devnull:
                subprocess.run(command_line, stdout=devnull, stderr=devnull)
        solution_time = time.monotonic() - tstart
        parsed = read_sdpa_out(out_filename)
    return SolverResult(
        status=parsed.status,
        primal=(
            parsed.primal + problem.constant_term
            if parsed.primal is not None
            else float("nan")
        ),
        dual=(
            parsed.dual + problem.constant_term
            if parsed.dual is not None
            else float("nan")
        ),
        x_mat=parsed.x_mat,
        y_mat=parsed.y_mat,
        solution_time=solution_time,
        raw=parsed,
    )


register(SolverKind.SDPA, solve_with_sdpa)
