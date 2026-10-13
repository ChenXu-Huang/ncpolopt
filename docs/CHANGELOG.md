# Changelog

All notable changes to this project are documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Fixed

- CLARABEL backend: the dual reconstruction walked the NonNeg prefix and
  the PSD svec segments of CLARABEL's `z` with a single offset, so any
  problem with scalar blocks (equalities, moment constraints) following
  the moment block got completely misaligned dual matrices; the segments
  are now addressed independently (regression test in
  `test_clarabel_solver.py`).
- MOSEK backend: parameter names prefixed `mosek.` were mangled by a
  wrong strip length; type inference accepts NumPy scalar values
  (`numbers.Integral`/`Real`); infeasible/unbounded statuses report NaN
  values instead of reading the certificate as if it were a solution.
- CLARABEL backend: non-optimal statuses report NaN values instead of the
  last iterate.
- CVXPY backend: a constant-only PSD block (possible when substitutions
  erase a constraint's entries) is wrapped in a CVXPY constant instead of
  hitting NumPy's `>>` bit-shift.
- `__version_tuple__` parses prerelease suffixes (`0.2.2rc1`) instead of
  raising `ValueError`.

### Changed

- `verbosity_to_level` clamps negative values to WARNING (they previously
  fell through to DEBUG).

### Documentation

- Corrected the `Probability` example output, the `get_next_neighbors`
  distance claim, the `maximum_violation` return-value wording, the
  `correlator` full-output requirement, the RDM band-structure wording,
  and the AGENTS.md extras list.

### Testing

- The lazy-import invariant (`import ncpolopt` must not import solver
  libraries) is now guarded by a subprocess test.
- Coverage of the package (`src/ncpolopt`) is now 100%: the CLI, the
  physics helpers (`pauli_constraints`, `correlator`, `get_next_neighbors`,
  `Probability` marginals, `define_objective_with_I`), the moment-entry
  arithmetic, the SDPA subprocess path (mock binary), and the solver error
  paths all carry tests now.
- Removed dead code: `SdpBuilder.dtype`, `moment.combine_entries`, and two
  unreachable branches in `relaxation.py`.

## [0.2.2] — 2026-10-13

### Fixed

- `fast_substitute` mixed rules: a rule with both commutative and
  noncommutative sides (e.g. `x*A -> 2`) double-counted the replacement's
  commutative part (`x*B` gave `4`, not `2`) and applied its commutative
  side even when the noncommutative side never matched; the commutative
  side now divides out of the monomial exactly once and the rule does not
  fire when either side is absent (regression test in
  `test_substitutions.py`).
- `apply_substitutions` raises a clear error instead of hanging forever on
  word-growing rule sets like `{A: A*B}` (capped fixpoint iteration).
- `is_number_type` accepts NumPy scalar types (`np.int64`, `np.float32`,
  ...) instead of raising `AttributeError`.
- `Probability`: a marginal with out-of-order party labels (`['B', 'A']`)
  was silently sorted, pairing outputs with the wrong parties; the caller's
  order is now honored, negative and out-of-range indices raise
  `ValueError`, and the error messages name the right party (regression
  tests in `test_physics.py`).
- `removeequalities` honors `localizing_monomials` overrides for the
  equality rows (the elimination path always computed the automatic
  basis), and a `localizing_monomials` list of the wrong length raises a
  clear `ValueError` instead of a bare `IndexError`; the dependent-rows
  warning now compares the rank against the row count.
- `sos_decomposition` defaults to the moment blocks (other blocks were
  silently labeled with the moment basis) and raises a clear error when a
  block cannot be matched to the basis (nested or scaled bases).
- SDPA backend: a nonzero exit code of the binary raises `SolverError`
  with the return code, and a missing output file no longer surfaces as a
  bare `OSError`; the `phase.value` mapping uses exact tokens (so
  `noINFO` is no longer misread as `infeasible`, and `dINF` maps to
  `unbounded` like the other backends). New tests drive a mock `sdpa`
  binary, covering the subprocess path without the real solver.
- PICOS backend: solver failures (`SolutionFailure`) and unsolved value
  reads (`NotValued`) map onto `status="unknown"`/`"infeasible"` instead
  of raising; `"dual feasible"` no longer reports `"optimal"`, and
  `"near-optimal"` maps to `"optimal"`.
- `solver="scs"` actually drives SCS through CVXPY (the kind was
  registered to the bare CVXPY backend, which picks CLARABEL first).
- `class_moment_equalities` rejects unnormalized problems (its constant
  equalities silently bound to the free normalization variable).
- `SteeringHierarchy` rejects `extramomentmatrices` (the copy/new blocks
  would not be scaled by `matrix_var_dim`), and its variable-group reuse
  looks up the recorded ids instead of assuming consecutive numbering.
- The RDM circulant layouts apply the substitution rules to their words
  (the standard path does), so reduced words share the variables of the
  reduced moments instead of splitting them.

- SDPA complex path: `write_dat_s` wrote the doubled objective as
  `[Re(c), Re(c)]` (a complex coefficient vanished from the file, and real
  coefficients gave the imaginary variables spurious weight) and wrote the
  constant matrix's imaginary entries into variable `n_vars`'s group; the
  objective is now `[Re(c), -Im(c)]` and F0's imaginary part stays in
  group 0. Complex solutions are folded back from the doubled real blocks
  to the original complex block sizes on return (`unembed_block`), so
  value extraction no longer sees only the real parts (regression tests in
  `test_sdpa_writer.py`).
- PICOS backend: complex-valued SDPs are rejected with
  `UnsupportedSdpError` — the per-variable construction cannot express the
  conjugate-mirrored coefficient matrices a complex LMI needs, and the old
  code crashed or built a non-Hermitian problem (regression test in
  `test_solvers.py`). `block_matrices` and `dual_value` now mirror with
  the conjugate transpose and pair in complex arithmetic.
- `_process_monomial`: a moment substitution value lost the monomial's
  outer scalar factor (`2*m` with `m -> x` wrote `x`, not `2*x`)
  (regression test in `test_relaxation.py`).
- Automatic complex detection now covers the objective (polynomial,
  moment-expression and matrix forms) and moment (in)equality constraints;
  a complex objective previously crashed in `set_objective` with a
  `TypeError` (regression test in `test_relaxation.py`).
- `_impose_ppt` tolerates blocks with substitution holes (moments zeroed
  by rules like `a0*a1 -> 0` leave no entry): Moroder + projective
  substitutions + `extramomentmatrices=[["copy", "ppt"]]` builds instead
  of raising `KeyError` (regression test in `test_hierarchies.py`).
- `Solution.dual_value` resolves monomials through the elimination basis
  after `removeequalities` (the monomial index refers to pre-elimination
  variables; their dual contributions spread over the reduced ones via
  `x = shift + basis @ y`), with the transform stored on the frozen
  `SdpProblem` (regression test in `test_solvers.py`).
- Steering objectives: the `(0, 0)` group's constant objective matrix no
  longer assumes variable 1 (wrong when parameter blocks exist), and
  diagonal sub-block groups fold the lower-triangle objective weights onto
  the constrained mirror variables — the phantom variables carry no
  constraint entries, and weighting them made the SDP unbounded
  (regression tests in `test_hierarchies.py`).
- Dual value convention: `dual_value` paired the dual matrices with the
  unsymmetrized (upper-triangle-only) constant matrix, counting every
  off-diagonal constant entry at half weight — the reported dual of
  problems with off-diagonal block constants (substitution-pinned moments,
  `momentsubstitutions`, equality-elimination shifts) was simply wrong.
  `SparseBlock.constant_matrix()` now returns the full symmetric matrix;
  `Solution.dual_value` mirrors the stored entries the same way, so the
  per-monomial dual pairing `-tr(Y @ A_k)` is exact for off-diagonal
  moments as well (regression tests in `test_sdp_problem.py` and
  `test_solvers.py`).
- CLARABEL backend: the svec unscaling multiplied off-diagonal dual
  entries by `sqrt(2)` instead of dividing, inflating `y_mat`
  off-diagonals by 2 (and corrupting `sos_decomposition`); the reported
  dual value only looked right because it cancelled against the
  half-trace bug above. The MOSEK backend's TODO doubting its `y_mat`
  convention is resolved the same way: `getbarxj` was always the
  canonical dual — the mismatch came from the shared `dual_value` bug
  (regression test in `test_clarabel_solver.py`; both TODOs removed).
- `fast_substitute`: a purely commutative rule (e.g. `x -> 2`) applied to
  a monomial with noncommutative factors dropped the noncommutative tail
  entirely (`fast_substitute(A*B, x, 2)` returned `1`), silently
  collapsing mixed classical/operator problems. The tail is now
  reattached, with expansion when the replacement is a sum (regression
  tests in `test_substitutions.py`).
- Complex relaxations: the adjoint fallback in
  `NpaRelaxation._get_index_of_monomial` reused the adjoint's SDP variable
  without conjugation, which is not a linear operation — complex
  constraints could silently degenerate (e.g. `I*A**2 - I*Dagger(A)**2 + 1
  >= 0` became `1 >= 0`). The fallback now raises a clear `RuntimeError`
  in complex relaxations, matching `_process_monomial` and
  `_resolve_position` (regression test in `test_relaxation.py`).
- Equality elimination: the rank of the equality system was computed as
  `last nonzero R row + 1`, which overshoots the true rank whenever a
  dependent row precedes an independent one (e.g. a duplicated equality)
  and silently pinned genuinely free variables. The nonzero rows of the
  unpivoted trapezoidal R are linearly independent, so their count is the
  rank; the fixed/free Q columns are now selected by the actual row
  indices (regression test in `test_sdp_problem.py`).

## [0.2.1] — 2026-09-29

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
  extramonomials). With these relations alone the certified bound still
  sat on the analytic line `(2p+1)/6`; the probe words below are what
  lift the level-2 forms onto `p` itself.
- `operator_relaxation_value` in the quantum-memory example — eq. (14)
  as written, with the unknown Hermitian operators `J` (4x4) and `Z`
  (2x2) as the SDP variables instead of one variable per functional
  class. The word set is the document's `S = {I, U, V}` at level 1; level
  2 adds the degree-2 monomials `U_{x,y} V_alpha`, and the certified value
  then attains the exact Choi fidelity `p` for one and four outputs alike
  (level 1 stays on the `(2p+1)/6` wall). All Hermitian blocks are
  realified by hand, so the SDP is real.
- Quantum-memory example: probe words `rho_x sigma_y v_0` (the factored
  image of the degree-2 monomials `U_{x,y} V_0`) and their `g`-products
  join the extramonomials, extending the localizing basis to 42/45 words
  and the moment matrix to 184/247 words. With them the factored NPA-tau
  level-2 value reaches `p` (0.499999 at p = 0.5) instead of stalling on
  the wall.
- `probes`, `families` and `solver_options` parameters of
  `npa_tau_relaxation_value`: the relation set is now selectable
  (`"linearity"`, `"classes"`, `"completeness"`, `"pauli"`), the new
  default being the functional-linearity family
  (`_functional_linearity_equalities`), and backend knobs are forwarded
  through `SolverSettings.solver_options`.
- `py.typed` marker and a `[tool.pyright]` configuration
  (`typeCheckingMode = "standard"` over `src/ncpolopt`, `tests` and
  `examples`, with `examples` on the extra path), plus a repo-wide
  `.editorconfig` pinning LF, UTF-8 and 4-space indentation.

### Changed

- `class_moment_equalities` no longer drops classes whose representative
  entry is pinned (a constant moment): each free class member is now tied
  to the pinned constant by a constant-equality, so pins (including the
  orthogonal-input zero words) reach into the free classes instead of
  weakening the relaxation.
- `npa_tau_relaxation_value` defaults to `solver="clarabel"` (the dense
  cvxpy canonicalization of the 184/247-word moment matrix no longer
  fits in memory); the MOSEK backend now solves the 4-output variant
  without the trace-preserving pin — the completeness moment-equalities
  cure the ill-posedness that stalled its interior point.
- Quantum-memory example: the factored `rho_x` matrices now carry the
  A0-side transpose of eq. (10) — `rho_3 = |+i><+i|` is not symmetric,
  and without it the correlation pins target the non-PSD partial
  transpose `J_M^{T_A0}` instead of `J_M`.
- Quantum-memory example: the certified level-1/level-2 bounds moved from
  the degenerate floor `p/(8d)` (unpinned) / TP-pinned values onto the
  analytic line `(2p+1)/6`, and then onto the exact `p` for the operator
  and factored level-2 forms once the probe words and the
  functional-linearity family are in place; the verbatim class form still
  sits on the wall, and the README now explains which of the three
  ingredients (probes, the A0-side transpose, cross-class linearity) does
  the work. README tables and test pins re-measured accordingly.
- Type annotations hardened across the package for Pyright's standard
  mode: `facvar`/coefficient vectors are typed `complex` end to end
  (`SdpBuilder`, `NpaRelaxation`, `Solution`), the hierarchy modules
  import `Problem` only under `TYPE_CHECKING` (avoiding the lazy-import
  cycle), `SparseBlock.constant_matrix` resolves SciPy's optional
  `shape` through a cast, and the example's cached factored data and the
  insertion-test model became `TypedDict`s. Solver-side: the SDPA parser
  returns the new `SdpaOutcome` (optional objective values) instead of
  `SolverResult`, the compiled CLARABEL extension is reached through an
  untyped module alias, and CVXPY is imported under `TYPE_CHECKING` so
  the lazy-import rule is unchanged.
- Tests and examples follow the typed interfaces: the memory-verification
  test imports the shared module as `quantum_memory.common` with
  `examples` on `sys.path`, and `SolverKind`/`SdpProblem` replace
  `object` in the test signatures.

### Removed

- The `tp_pin` parameter of `verbatim_relaxation_value`,
  `npa_tau_relaxation_value` and their helpers: with the completeness
  relations in place the trace-preserving pin is redundant (pinned and
  unpinned bounds coincide), and the MOSEK `tp_pin=True` requirement is
  gone with it.

### Fixed

- `MomentEntry.__rsub__` put the whole `-entry` sub-expression into the
  term tuple, so `constant - entry` produced a nested `MomentExpr`
  instead of flat terms (regression test
  `test_reflected_subtraction_stays_flat`).
- CVXPY backend: a constraint's dual value — a bare float for scalar
  blocks, `None` for constraints the solver did not reach — is normalised
  to a block-shaped array, so the dual matrices stay indexable and paired
  with their blocks.
- PICOS backend: a `None` solver status maps to `"unknown"` instead of
  leaking `None` into the status field.

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
[0.2.2]: https://github.com/ChenXu-Huang/ncpolopt/compare/v0.2.1...v0.2.2
[0.2.1]: https://github.com/ChenXu-Huang/ncpolopt/compare/v0.2.0...v0.2.1
