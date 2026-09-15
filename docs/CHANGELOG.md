# Changelog

All notable changes to this project are documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- Quantum-memory example: the word set now carries the A0-/B2-side
  factors `X_x`/`W_y` of the U operators (`x_operators()`/`w_operators()`
  in `examples/quantum_memory/common.py`), so the generator completeness
  relations (`X_0 + X_1 = I`, `U_{0,y} + U_{1,y} = W_y`, `sum_alpha
  V_alpha = I` etc.) are imposed entrywise on both moment blocks
  (verbatim construction) or lifted into the inserted J/W blocks as
  linear moment-equalities (factored construction,
  `_completeness_equalities`/`_linear_span_equalities`).
- Quantum-memory example: Pauli-decomposition relations of the input
  projectors — full complex identities `X_x X_z = sum_t c_t X_t`
  entrywise in the verbatim level-2 construction (restricted 46-word
  basis), anticommutators `{rho_x, rho_z} = sum_t a_t rho_t` as
  moment-equalities in the factored construction (72 degree-3
  extramonomials). With these relations the certified bound sits on the
  analytic line `(2p+1)/6` at every measured point, for one and four
  outputs alike.

### Changed

- `class_moment_equalities` no longer drops classes whose representative
  entry is pinned (a constant moment): each free class member is now tied
  to the pinned constant by a constant-equality, so pins (including the
  orthogonal-input zero words) reach into the free classes instead of
  weakening the relaxation.
- `npa_tau_relaxation_value` defaults to `solver="clarabel"` (the dense
  cvxpy canonicalization of the 176/239-word moment matrix no longer
  fits in memory); the MOSEK backend now solves the 4-output variant
  without the trace-preserving pin — the completeness moment-equalities
  cure the ill-posedness that stalled its interior point.
- Quantum-memory example: the certified level-1/level-2 bounds moved from
  the degenerate floor `p/(8d)` (unpinned) / TP-pinned values onto the
  analytic line `(2p+1)/6` (exact at p = 0.25); README tables and test
  pins re-measured accordingly.

### Removed

- The `tp_pin` parameter of `verbatim_relaxation_value`,
  `npa_tau_relaxation_value` and their helpers: with the completeness
  relations in place the trace-preserving pin is redundant (pinned and
  unpinned bounds coincide), and the MOSEK `tp_pin=True` requirement is
  gone with it.

## [0.2.0] — 2026-08-31

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
  relaxation (`n_outputs=4` in `examples/quantum_memory/common.py`):
  the dense cvxpy conversion cannot hold its 199-word moment matrix, so
  it is solved through the direct sparse backends — CLARABEL by default,
  or MOSEK (`solver="mosek"`, roughly 6-17x faster at ~40% less memory)
  when a license is available. The MOSEK interior point stalls on the
  unpinned (ill-posed) variant, so `solver="mosek"` requires
  `tp_pin=True`.
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
- `__version_tuple__` on the public API exposing the version components.
- GitHub release workflow creating a Release whose body is the version's
  changelog section.

### Changed

- Renamed `docs/architecture.md` → `docs/ARCHITECTURE.md`; updated the
  reference in `AGENTS.md`.
- The MOSEK backend resolves `solsta` enum names at runtime and passes
  the interior-point solution selector to `gety`, so MOSEK 10/11 API
  changes no longer break solves; the test license probe checks out the
  `pts` feature explicitly (MOSEK 10+ defers the license check to
  `optimize()`).
- CLI `--help` description now shows only the first line of the module
  docstring instead of the raw multiline text.

### Fixed

- `AttributeError` after a successful MOSEK solve on MOSEK 11 (dropped
  `solsta.near_optimal`) and MOSEK 10 (renamed `*_infeasible_cer`
  statuses) — regression-tested in `test_mosek_solves_basic_problem`.

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
[0.2.0]: https://github.com/ChenXu-Huang/ncpolopt/compare/v0.1.0...v0.2.0
