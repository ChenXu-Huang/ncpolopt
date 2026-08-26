# ncpolopt — Project Conventions

This file is the project instruction set for AI agents and human
contributors. The architecture is documented in
[docs/architecture.md](docs/architecture.md); read it before touching
structure or the solver backends.

## What this project is

A modern rewrite of the GPL-3 package ncpol2sdpa: sparse SDP relaxations
of polynomial optimization problems with noncommuting variables (NPA
hierarchy and relatives), with four pluggable solver backends.

All code, docstrings, and docs are in **English**; package metadata and
README reflect the public API, and tests are the contract.

## Workflow

```bash
uv sync --group dev        # install core + dev group (pytest, ruff, solvers)
uv run pytest              # must stay green
uv run ruff check .        # must stay clean
```

Python 3.13+, uv build backend (`uv_build`), package metadata in
`pyproject.toml`. Never add a dependency to the core set (numpy, scipy,
sympy only) — solver bindings are optional extras (`cvxpy`, `mosek`,
`cvxopt`, `chordal`) and must stay lazy.

## Architecture invariants

These are traps with hard-won fixes; do not regress them:

1. **Frozen dataclasses everywhere** (`Problem`, `SdpProblem`,
   `SparseBlock`, `Solution`, `MomentEntry`). Never assign through them —
   use `dataclasses.replace`. `Problem.relaxation()` must never mutate its
   `Problem`.
2. **SDP variable numbering is 1-based**; variable 0 is the constant
   placeholder. After `freeze()`, `obj[i]` corresponds to SDP variable
   `i + 1` (the constant slot is dropped).
3. **COO stores the upper triangle only.** Any materialized matrix must
   be symmetrized (`if i != j: target[j, i] += value`); a missing mirror
   entry makes cvxpy/picos report unbounded problems.
4. **The canonical form is fixed once** in `solvers/_common.py`
   (`primal = c·x* + c0`, `dual = -Σ tr(Y @ A0) + c0`). Backends only map
   COO data to native format — never re-derive signs or constants.
5. **Solvers are lazy.** `import ncpolopt` must not import cvxpy/mosek/
   picos/cvxopt. New backends register in `solvers/registry.py`; detection
   order is `CVXPY → MOSEK → CVXOPT → SDPA`. SDPA counts as available only
   when its binary is on PATH; an explicit `settings.sdpa_executable`
   still works at solve time.
6. **Hierarchies are subclasses** of `NpaRelaxation` (`SteeringHierarchy`,
   `MoroderHierarchy`, `RdmHierarchy`), overriding template-method hooks —
   not flags on one class.
7. **Determinism.** No `random` in production paths. Tests must not rely
   on RNG state.

## Code style

Follow the `python-coding-style` skill (the binding authority): Google
style docstrings with `Args:`/`Returns:`/`Raises:` on every public
function and class, full type annotations, class-bound constants
(`cls._UPPERCASE`) for per-class tuning knobs, `TODO(username):` /
`NOTE:` / `FIXME:` task tags, logging instead of `print`, no commented-out
code. Ruff config lives in `pyproject.toml` (E/F/W/I/UP/B/SIM/C4/RUF).

SymPy-specific: noncommutative factors are never reordered, and
`Dagger(A*B) = Dagger(B)*Dagger(A)` flips order — state products
explicitly in tests and code.

## Testing rules

- pytest under `tests/`; numerical tolerance `1e-5`.
- Numerical tests are parametrized over the `solver_kind` fixture from
  `conftest.py` (usable backends only). The SymPy cache is cleared after
  every test by an autouse fixture — keep it.
- **Never assert numbers on the cvxopt/PICOS path**: on Windows it
  returns `unknown` states or spurious optima for feasible/infeasible
  models alike. Skip via `pytest.skip` on `pic.SolutionFailure`; pin
  canonical values (e.g. the Moroder witness -0.7284…) through
  cvxpy/CLARABEL reconstructions instead.
- Every ported bug fix carries a targeted regression test.
