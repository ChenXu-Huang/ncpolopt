# Changelog

All notable changes to this project are documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- `ncpolopt.hierarchies.insertion` — operator-insertion (MDI-style)
  helpers: `trace_moment_pins` pins the moments over a basis to their
  trace values in an explicit matrix realization, and
  `class_moment_equalities` generates the functional-class
  moment-equalities of the inserted-operator localizing blocks, reading
  the variable positions off a draft build of the relaxation.
- Direct sparse CLARABEL backend (`solver="clarabel"`, extra
  `ncpolopt[clarabel]`): builds CLARABEL's canonical `(A, b, cones)`
  directly from the frozen blockwise COO for SDPs too large for cvxpy's
  dense canonicalization. Explicit selection only — autodetection order
  is unchanged.
- `examples/quantum_memory/` — verification scripts for the
  measurement-device-independent quantum-memory example (depolarizing
  channel): `common.py` (operators, channel, correlations),
  `verify_dual_sdp.py` (correlation table and dual SDPs) and
  `verify_relaxation.py` (moment-matrix relaxation).
- `tests/test_memory_verification.py` — regression tests pinning the
  quantum-memory example values.
- 4-output Bell-measurement variant of the quantum-memory NPA-tau
  relaxation (`n_outputs=4` in `examples/quantum_memory/common.py`),
  solved through the direct sparse CLARABEL backend (the dense cvxpy
  conversion cannot hold its 199-word moment matrix).
- Orthogonal-input substitution rules (`rho_0 rho_1 -> 0`,
  `sigma_0 sigma_1 -> 0`, both orders) in the quantum-memory example, which
  drop the identically-zero words and shrink the SDP (62 S-basis words and
  2839 class relations for one output, 113 and 5551 for four). Dropping the
  zero rows also drops active class-relation links, so the certified L2
  values shift slightly (e.g. 0.301898 -> 0.293118 at p = 0.5 without the
  TP pin); the test pins and README tables were re-measured under the
  reduced construction, and both constructions yield valid lower bounds.
- `slow` pytest marker for multi-minute solver runs; CI runs
  `pytest -m "not slow"`.

### Changed

- Renamed `docs/architecture.md` → `docs/ARCHITECTURE.md`; updated the
  reference in `AGENTS.md`.

## [0.1.0] — 2026-08-26

### Added

- Noncommutative polynomial optimization: NPA hierarchy and relatives
  (`MoroderHierarchy`, `SteeringHierarchy`, `RdmHierarchy` as
  `NpaRelaxation` subclasses); Lasserre-style hierarchies for commuting
  variables with an optional chordal sparsity extension.
- Three-layer frozen-dataclass object model (`Problem` → `NpaRelaxation`
  → `Solution`), blockwise COO SDP representation, and
  `MomentEntry`/`MomentExpr` replacing the ncpol2sdpa string DSL.
- Four lazy solver backends (CVXPY, MOSEK, PICOS/cvxopt, SDPA) with
  registry-based detection (`CVXPY → MOSEK → CVXOPT → SDPA`) and the
  canonical sign/constant convention fixed once in
  `solvers/_common.py`.
- CLI entry point (`ncpolopt`) reporting version, platform, and detected
  solvers.
- 17 ncpol2sdpa bug fixes, each carrying a regression test.
- PyPI package metadata (extras `cvxpy`, `mosek`, `cvxopt`, `chordal`,
  `all`), CI and release workflows.
- Documentation: `docs/ARCHITECTURE.md`, `AGENTS.md`/`CLAUDE.md`
  conventions.

[0.1.0]: https://github.com/ChenXu-Huang/ncpolopt/releases/tag/v0.1.0
