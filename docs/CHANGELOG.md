# Changelog

All notable changes to this project are documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

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
