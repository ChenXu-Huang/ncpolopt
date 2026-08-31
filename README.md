# ncpolopt

<p align="center">
  <img src="https://img.shields.io/badge/python-3.13+-blue?style=flat-square" alt="Python 3.13+">
  <img src="https://img.shields.io/badge/license-GPL--3.0-blue?style=flat-square" alt="License: GPL-3.0">
  <img src="https://github.com/ChenXu-Huang/ncpolopt/actions/workflows/ci.yml/badge.svg?style=flat-square" alt="CI status">
<img src="https://img.shields.io/pypi/v/ncpolopt?style=flat-square" alt="PyPI version">
<img src="https://img.shields.io/pypi/dm/ncpolopt?style=flat-square" alt="PyPI downloads">
</p>

Sparse SDP relaxations of polynomial optimization problems with
noncommuting variables — the [NPA hierarchy](https://arxiv.org/abs/0803.4291)
and its relatives. `ncpolopt` is a modern rewrite of the GPL-3
[ncpol2sdpa](https://github.com/peterwittek/ncpol2sdpa) package: the same
hierarchies and the same four solver backends, behind a fresh, typed API
built on dataclasses and lazy solver discovery.

- **NPA hierarchy** for noncommutative operators, plus the
  [Moroder](https://arxiv.org/abs/1305.2630) (PPT), steering, and RDM
  (reduced density matrix) hierarchies, and operator-insertion (MDI-style)
  helpers that generate the trace pins and functional-class moment
  equalities of a moment-insertion relaxation from an explicit matrix
  realization (`trace_moment_pins`, `class_moment_equalities`).
- **Lasserre-style hierarchies** for commuting variables, with an optional
  chordal sparsity extension (SparsePOP-style) that splits the relaxation
  into independent moment blocks per clique.
- **Four solver backends**: [cvxpy](https://www.cvxpy.org/),
  [MOSEK](https://www.mosek.com/), [PICOS/cvxopt](https://picos-api.gitlab.io/),
  and the external [SDPA](https://sdpa.sourceforge.net/) binary — selected
  by name or auto-detected — plus a direct sparse
  [CLARABEL](https://clarabel.org/) backend (`solver="clarabel"`,
  explicit selection only) for SDPs too large for cvxpy's dense
  canonicalization.
- **Moment expressions** (`MomentEntry`) instead of a string DSL for
  writing conditions on the moment matrix directly.

## Installation

Requires Python 3.13+. The core package needs only numpy, scipy, and sympy;
install a solver extra to actually solve:

```bash
uv pip install ncpolopt[cvxpy]        # or: pip install ncpolopt[cvxpy]
```

Available extras: `cvxpy`, `clarabel` (direct sparse backend), `mosek`,
`cvxopt` (PICOS + cvxopt), `chordal` (chompack, only needed for the
chompack completion method), and `all`.
With no solver installed the package still imports and builds relaxations;
only solving raises a `SolverError` telling you which extra to install.

## Quickstart

A tiny noncommutative problem: two Hermitian operators `x0`, `x1`, minimize
the anticommutator `x0*x1 + x1*x0` under the constraint `x1 - x1² + ½ ≥ 0`
and the projection `x0² = x0`:

```python
import ncpolopt as nc

X = nc.generate_operators("x", 2, hermitian=True)
problem = nc.Problem(
    X,
    objective=X[0] * X[1] + X[1] * X[0],
    inequalities=[-X[1] ** 2 + X[1] + 0.5],
    substitutions={X[0] ** 2: X[0]},
)
solution = problem.solve(level=2)
print(solution.primal)   # -0.75
```

## Object model

Work flows through three layers, each a frozen dataclass:

1. **`Problem`** — the symbolic model: variables (a list of lists for
   multipartite problems), objective, polynomial and moment constraints,
   substitution rules. Reusable; building a relaxation never mutates it.
2. **`NpaRelaxation`** — the SDP at a hierarchy level, built by
   `problem.relaxation(level)` or subclass hierarchies
   `MoroderHierarchy`, `SteeringHierarchy`, `RdmHierarchy`. The assembled
   SDP data (`relaxation.sdp`) is a frozen blockwise COO representation.
3. **`Solution`** — the immutable solve result (`primal`, `dual`,
   `status`, `x_mat`, `y_mat`) with extraction helpers:
   `monomial_value()`, `dual_value()`, `dual_block()`,
   `sos_decomposition()`, `solution_ranks()`.

## Command line

```bash
ncpolopt                # version, platform, detected solvers
ncpolopt --version
```

## Development

```bash
uv sync --group dev
uv run pytest
uv run ruff check .
```

Multi-minute solver certification tests are marked `slow` and skipped in
CI (`uv run pytest -m "not slow"`); run them locally with
`uv run pytest -m slow`.

## Examples

[`examples/quantum_memory/`](examples/quantum_memory/README.md) — numerical
verification of a measurement-device-independent quantum-memory
certification (depolarizing channel): correlation tables, dual SDPs, and
moment-matrix relaxations of the NPA-tau form, including a four-output
Bell-measurement variant solved through the direct sparse backends
(CLARABEL by default, or MOSEK with `tp_pin=True`).

## License

GPL-3.0-only. This package is derived from ncpol2sdpa, which is
Copyright (C) 2012-2016 Peter Wittek and contributors.
