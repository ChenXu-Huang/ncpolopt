# Quantum-memory verification: numerical checks

This directory verifies the SDP derivation of the staged document
`verification-of-memory.md`: measurement-device-independent (MDI) verification
of a quantum memory, in the Rosset-Buscemi-Liang style (PRX 8, 021033 (2018) /
arXiv:1710.04710). The example under test is the depolarizing channel
`M_p(rho) = p rho + (1-p)/3 (X rho X + Y rho Y + Z rho Z)` with the identity
extraction map, whose Choi fidelity is exactly `p`.

Every claim of the document that can be computed is checked either numerically
(Part A, cvxpy/CLARABEL) or through the public ncpolopt API (Part B). The
scripts are intentionally standalone so each part can be re-run independently:

| File | Purpose |
| --- | --- |
| [common.py](common.py) | States, channel, Choi state, correlations (eq. (1) and eq. (10)), eq. (16) table, functional-class machinery, both relaxation constructions |
| [verify_dual_sdp.py](verify_dual_sdp.py) | Part A: correlation tables and the two dual SDPs (eq. (6), eq. (9)) |
| [verify_relaxation.py](verify_relaxation.py) | Part B: the eq. (14) relaxation verbatim (L1) and its factored NPA-tau re-expression (L2, certified) |
| [test_memory_verification.py](../../tests/test_memory_verification.py) | The same checks as self-contained pytest tests |

Run: `uv run python examples/quantum_memory/verify_dual_sdp.py`, likewise
`verify_relaxation.py` (both accept `--p 0.5` or `--p-grid 0.25,0.5,1.0`;
`verify_relaxation.py` also accepts `--n-outputs {1,4}` to select the
single-output or the full four-output Bell measurement), and
`uv run pytest tests/test_memory_verification.py` (the multi-minute
single-output L2 certification and the four-output solves are marked
`slow` and excluded from CI via `uv run pytest -m "not slow"`).

## Part A: correlations and dual SDPs (all passed)

* The three forms of the correlation agree to machine precision: eq. (1)
  `Tr[(M(rho_x) x sigma_y) Phi^alpha]`, eq. (10)
  `d Tr[(rho_x^T x I x sigma_y)(J_M x I)(I x Phi^alpha)]`, and the closed-form
  table eq. (16) (worst deviation ~4e-16 across the p grid, all 16 entries).
* The four-output probabilities sum to one: `sum_alpha p(alpha|x,y) = 1`.
* The Choi state of the depolarizing channel is the Werner state with fidelity
  `p` (eq. (15)), and `<phi+|J_M|phi+> = p` (eq. (4)).
* Strong duality holds: eq. (6) `max Tr(J_M J)` and eq. (9)
  `min Tr(Z)/d` both equal `p` to 1e-8 (see table).

| p | eq. (6) | eq. (9) | eq. (9) without `Z >= 0` |
| --- | --- | --- | --- |
| 0.30 | 0.30000000 | 0.30000000 | 0.30000000 |
| 0.50 | 0.50000000 | 0.50000000 | 0.50000000 |
| 0.75 | 0.75000000 | 0.75000000 | 0.75000000 |
| 1.00 | 0.99999999 | 1.00000000 | 1.00000000 |

The last column confirms the document's `Z >= 0` constraint is redundant:
`I^{A0} x Z >= J_M >= 0` already forces `Z >= 0`.

## Part B: the eq. (14) relaxation (certified)

The relaxation is built twice: verbatim over the word set `S = {I} U {U} U
{V} U {X} U {W}` (one SDP variable per functional class of `G_{u,v} =
Tr_{B2}(v u^dagger)` for the M block and of `H_{u,v} = Tr_{A0B2}(v
u^dagger)` for the L block, pins `M[V_alpha, U_{x,y}] = p(alpha|x,y)/d`,
constraints `M >= 0, L >= 0, L - M >= 0`, objective `L[I, I]/d^3`), and as
a standard NPA problem over the extended algebra `S union {J, W}` with
state `tau = I_8/8` through the public ncpolopt API (see below). The
`X_x = rho_x^T x I x I` and `W_y = I x I x sigma_y` words are the A0-/B2-
side factors of the U operators; including them makes the **completeness
relations** writable as entrywise linear constraints on both blocks:
`M[u, X_0] + M[u, X_1] = M[u, I]`, `M[u, U_{0,y}] + M[u, U_{1,y}] = M[u,
W_y]` (likewise with X/W swapped), and `sum_alpha M[u, V_alpha] = M[u, I]`
for the four-output measurement. No trace-preserving pin is used: with the
completeness relations in place the pin is redundant.

Level-1 values (verbatim form, CLARABEL; all certified lower bounds,
`value <= p + 1e-5` and strictly positive). With the completeness
relations the bound already sits on the analytic line `(2p+1)/6` at level
1, for one and four outputs alike:

| p | single-output L1 | 4-output L1 | `(2p+1)/6` |
| --- | --- | --- | --- |
| 0.25 | **0.250000** | **0.250000** | 0.250000 |
| 0.30 | 0.266667 | 0.266667 | 0.266667 |
| 0.50 | 0.333333 | 0.333333 | 0.333333 |
| 0.75 | 0.416667 | 0.416667 | 0.416667 |
| 1.00 | 0.500000 | 0.500000 | 0.500000 |

Observations:

* **The completeness relations transformed the level-1 bound.** Without
  them (and without any pin) the verbatim L1 bound collapsed toward the
  degenerate floor `p/(8d)` (0.089396 at p = 0.5); with them it equals
  the factored level-2 bound everywhere measured. The missing
  `X_0 + X_1 = I`-type relations were exactly the dilution direction the
  unaugmented relaxation exploited.
* **The bound is exact at the fully depolarizing point** (0.250000 = p at
  p = 0.25) and equals the diagonal correlation `p(0|x,x)` of eq. (16)
  everywhere else: the strongest statement this data set certifies at
  these levels.
* Without the functional-class structure -- treating every moment as an
  independent variable -- the objective degenerates: the pins admit
  `M[I, I] -> 0` with `M[V, V] -> infinity`, so the projection relations
  (U^2 = U etc.) encoded in `G` and `H` are what keep the bound
  non-degenerate.

### Level 2: the factored NPA-tau re-expression (certified)

The correlations pin the degree-2 words `U J V`, so level 2 is the minimum
NPA level. The naive level-2 algebra `{U, V, J, W}` (364 words, ~6e4 SDP
variables, ~60 TB of dense materialization) is not solvable with the current
backends. Following the factored-algebra technique of Brown's user guide on
device-independent conditional-entropy bounds, the `U` operators are
re-expressed as `U_{x,y} = rho_x (x) I (x) sigma_y` on
`H_{A0} x H_{A2} x H_{B2}` with the one-direction commutation
`rho_i sigma_j -> sigma_j rho_i` and the orthogonal-input rules
(`rho_0 rho_1 -> 0`, `sigma_0 sigma_1 -> 0`, both orders -- the input states
|0> and |1> are orthogonal, so those words are identically zero and drop out
of the basis): the level-2 moment matrix collapses to 136 words (176 with
the Pauli extramonomials below), and the
problem solves in minutes (10727 SDP variables, 6339 equality constraints;
~10-20 s construction -- the relation set is p-independent and cached per
output count -- plus a few minutes per solve on the direct sparse CLARABEL
backend; the dense cvxpy canonicalization no longer fits in memory).
Class relations
that reference pinned or zero classes are not dropped:
`class_moment_equalities` emits them as constant-equalities on the free
class members, so the orthogonal-input reduction no longer loses
class-relation links. Two linear word-relation families are lifted into
the inserted blocks as moment-equalities (`_linear_span_equalities` in
[common.py](common.py); plain polynomial `equalities` cannot do this --
they only constrain block-0 S-moments, which are trace-pinned already):

* The **completeness relations** of the generators:
  `rho_0 + rho_1 = I`, `sigma_0 + sigma_1 = I`, and (for four outputs)
  `sum_alpha v_alpha = I` give `<u g rho_0 w> + <u g rho_1 w> = <u g w>`
  etc. for every word pair `(u, w)` whose terms exist in the moment matrix.
* The **Pauli-decomposition anticommutator relations**: the four input
  projectors form a complex basis of the 2x2 matrices, so every degree-2
  word is a linear combination of degree-1 words. The coefficients are
  genuinely complex, so the real relaxation gets the symmetric parts
  `{rho_x, rho_z} = sum_t a_t rho_t` (real `a_t`): 10 anticommutators per
  side, with the degree-3 words `g.(rho_x rho_z)` / `g.(sigma_y sigma_z)`
  added as extramonomials (72 in total).

Certified values (CLARABEL; all lower bounds, `value <= p + 1e-5`, strictly
positive) -- identical to the verbatim level-1 values above, on the
analytic line `(2p+1)/6`:

| p | factored L2 | `(2p+1)/6` |
| --- | --- | --- |
| 0.25 | **0.250000** | 0.250000 |
| 0.30 | 0.266667 | 0.266667 |
| 0.50 | 0.333333 | 0.333333 |
| 0.75 | 0.416667 | 0.416667 |
| 1.00 | 0.500000 | 0.500000 |

Observations:

* **The L2 factored bound equals the verbatim L1 bound everywhere
  measured** (both on the `(2p+1)/6` line): once the completeness
  relations are imposed, level 2 adds nothing further over level 1. The
  factored construction remains the demonstration of the public ncpolopt
  API (and the only route that scales to the four-output measurement).
* **The `(2p+1)/6` wall is robust**: neither a trace-preserving pin
  (measured before its removal) nor the generator completeness relations
  nor the **Pauli-decomposition word relations** go past it (the wall
  equals the diagonal correlation `p(0|x,x)` of eq. (16)). The Pauli
  relations were tested in their strongest form: the verbatim path at
  level 2 with the full complex identities `X_x X_z = sum_t c_t X_t`
  imposed entrywise (restricted word set of 46 words -- the full level-2
  closure's 270 words do not fit a dense cvxpy canonicalization), and the
  factored path with the anticommutator identities `{rho_x, rho_z} = sum_t
  a_t rho_t` as moment-equalities (72 extramonomials, 176-word moment
  matrix, 6339 relations, 10727 SDP variables). Both give exactly
  `(2p+1)/6` (0.333333 at p = 0.5, 0.500000 at p = 1.0). The remaining
  gap `(4p-1)/6` to the exact value `p` is a genuine finite-level effect
  of this relaxation family.
* `removeequalities=True` cannot be used on the class relations: they are
  linearly dependent (conjugate classes share variables in real problems),
  which CLARABEL rejects. The 6339 equalities are passed through as-is.

### The NPA-tau re-expression

`_npa_tau_problem` in [common.py](common.py) writes eq. (14) as a standard
ncpolopt problem over the factored algebra `{rho, sigma, v, J, W}` with the
maximally mixed state, projections `rho^2 -> rho, sigma^2 -> sigma, v^2 -> v`,
the one-direction commutation `rho_i sigma_j -> sigma_j rho_i`, and the
orthogonal-input rules `rho_0 rho_1 -> 0`, `sigma_0 sigma_1 -> 0` (both
orders, for the orthogonal states |0> and |1>). Every moment over the
62-word S-basis `{1, rho, sigma, v}` is pinned to its trace value
`Tr[word]/8` (by `ncpolopt.hierarchies.trace_moment_pins`), the
correlations pin `<rho_x sigma_y J v> = p(alpha|x,y)/(8d)`, and the
objective is `<W>` (which equals `Gamma^{(Z)}_{I,I}/8`). Four details were
hard-won:

* The localizing matrices run over the **26-word basis**
  `{1, rho, sigma, v} U {U_{x,y}}`. The degree-2 `U` words are admissible
  because their `g U` products (32 extramonomials `{J, W} . U_{x,y}`,
  degree 3, plus 40 more `{J, W} . {rho_x rho_z, sigma_y sigma_z}` for the
  Pauli relations) are added as `extramonomials`, so every localizing
  entry monomial `u^dagger g w` (degree <= 5) exists as a moment-matrix
  entry.
* The `J`- and `W`-localizing blocks are tied to the moment matrix by
  **block-0 moment-equalities** encoding the functional classes of eqs.
  (12)/(13) (`G_{u,v} = Tr_{B2}(v u^dagger)` for `J`, `H_{u,v} = Tr_{A0B2}
  (v u^dagger)` for `W`), generated by
  `ncpolopt.hierarchies.class_moment_equalities`. The variable of an entry
  monomial is created at the first upper-triangle occurrence of the monomial
  or of its canonicalized adjoint (the builder reuses the conjugate's
  variable); the helper reads those positions off a draft build of the
  relaxation rather than re-simulating the builder. Pinned (constant)
  classes are **not** dropped: the helper emits a constant-equality tying
  each free class member to the pinned value (dropping them instead
  weakens the relaxation -- the more moments are pinned, the more links
  would be lost).
* These relations are **necessary**: without them the localizing entries
  become independent variables and the objective degenerates.
* The **completeness relations** of the generators are added as linear
  moment-equalities on the inserted blocks (`_completeness_equalities`):
  `rho_0 + rho_1 = I`, `sigma_0 + sigma_1 = I`, and (four outputs)
  `sum_alpha v_alpha = I` become `<u g rho_0 w> + <u g rho_1 w> =
  <u g w>` etc. for every word pair whose terms exist in block 0
  (positions read off the same draft build; pinned terms fold into the
  constant). This family is what lifts the unaugmented bound onto the
  `(2p+1)/6` line, and it cures the ill-posedness that stalled the MOSEK
  interior point on the 4-output variant (now `optimal`,
  0.333333 at p = 0.5).
* The orthogonal-input rules shrink the SDP (62 instead of 66 S-basis
  words); the class relations into the dropped zero classes survive as
  constant-equalities `var = 0`, so the reduction no longer weakens the
  relaxation.

### The four-output Bell measurement

`n_outputs=4` carries the full Bell measurement `{Phi^0, ..., Phi^3}`: the
algebra gains three measurement operators `v_1, v_2, v_3` (14 generators,
113 S-basis words, 239-word level-2 moment matrix, 13585 class,
completeness and Pauli relations, 18425 SDP variables)
and the correlations pin all 64 entries
`<rho_x sigma_y J v_alpha> = p(alpha|x,y)/(8d)`. The 239-word moment matrix
cannot be materialized through the package's dense cvxpy canonicalization
(~1.6 GB), so `npa_tau_relaxation_value` routes the four-output solves
through the package's direct sparse CLARABEL backend (`solver="clarabel"`,
[clarabel_solver.py](../../src/ncpolopt/solvers/clarabel_solver.py): per
block, the upper-triangle COO mirrored to the lower, off-diagonal svec
entries scaled by sqrt(2), rows grouped NonNeg first, PSD triangle cones
after) -- the same SDP, without the dense intermediate. Each CLARABEL
solve takes 8-25 minutes; MOSEK solves the same problem in 1-4 minutes
per point (before the completeness relations were added, MOSEK stalled
with `unknown` on this variant; it now solves, except at p = 1
where the boundary-degenerate interior point still returns `unknown` and
CLARABEL is required, and at p = 0.25 where it overshoots by ~1e-4).

Certified four-output L2 values (direct sparse construction; MOSEK, except
p = 0.25 and p = 1.0 via CLARABEL). All lower bounds within solver
tolerance, `value <= p + 1e-5` up to the marked MOSEK inaccuracies,
strictly positive; they coincide with the single-output values on the
`(2p+1)/6` line:

| p | L2 (4-output) | single-output L1/L2 (reference) |
| --- | --- | --- |
| 0.25 | **0.250000** | 0.250000 |
| 0.30 | 0.266679 | 0.266667 |
| 0.50 | 0.333333 | 0.333333 |
| 0.75 | 0.416691 | 0.416667 |
| 1.00 | 0.500001 | 0.500000 |

(The MOSEK readout at p = 0.25 overshoots by ~1e-4 at this degenerate
point -- 0.250138 -- so the CLARABEL value is reported; at p = 1 MOSEK
returns `unknown` and the CLARABEL value stands in.)

Without the completeness relations the extra outputs genuinely helped the
bound (0.248300 vs 0.209707 at p = 0.25); with them, one-output and
four-output bounds coincide on the `(2p+1)/6` line at every measured
point -- the operator-algebra completeness, not the measurement-outcome
count, was the binding constraint. The bound stays exact at the fully
depolarizing point (0.250000 = p).

## Tests

`tests/test_memory_verification.py` mirrors the two scripts as pytest tests:
the correlation table, the two dual SDPs pinned to `p`, the L1 relaxation
lower bound (with the measured values pinned to +/-1e-3 as a solver-change
tripwire), the factored NPA-tau L2 certification (two solves at p = 0.5,
values pinned to +/-1e-3 in the certified window `(0, p]`, re-using the
shared construction of [common.py](common.py) rather than duplicating it),
and a construction-sizes test pinning the operator/basis/relation/variable
counts so a change in the reduction rules trips the test. The four-output
solves (8-25 minutes each) are marked `slow` and excluded from CI
(`uv run pytest -m "not slow"`); run them locally with
`uv run pytest -m slow tests/test_memory_verification.py`. All numerical
tests use CLARABEL explicitly and follow the project conventions
(tolerance 1e-5, no RNG dependence).
