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

The relaxation is built twice: verbatim over the word set `S = {I} U {U} U {V}`
(one SDP variable per functional class of `G_{u,v} = Tr_{B2}(v u^dagger)` for
the M block and of `H_{u,v} = Tr_{A0B2}(v u^dagger)` for the L block, pins
`M[V_alpha, U_{x,y}] = p(alpha|x,y)/d`, constraints `M >= 0, L >= 0,
L - M >= 0`, objective `L[I, I]/d^3`), and as a standard NPA problem over the
extended algebra `S union {J, W}` with state `tau = I_8/8` through the public
ncpolopt API (see below).

Level-1 values (verbatim form, CLARABEL; all certified lower bounds,
`value <= p + 1e-5` and strictly positive):

| p | single-output | + TP pin | 4-output | + TP pin | degenerate floor `p/(8d)` |
| --- | --- | --- | --- | --- | --- |
| 0.25 | 0.015625 | **0.250000** | 0.015625 | **0.250000** | 0.015625 |
| 0.30 | 0.029196 | 0.258408 | 0.035000 | 0.262500 | 0.018750 |
| 0.50 | 0.089396 | 0.292042 | 0.125000 | 0.312500 | 0.031250 |
| 0.75 | 0.179185 | 0.334084 | 0.265625 | 0.375000 | 0.046875 |
| 1.00 | 0.286725 | 0.376126 | 0.437500 | 0.437500 | 0.062500 |

Observations:

* **The relaxation is a valid but loose lower bound.** Without the
  trace-preserving pin the bound collapses toward the degenerate floor
  `p/(8d)` (at p = 0.25 both variants hit it exactly); the pin
  `M[I, I] = d` (i.e. `Tr(J_M) = 1`) restores a meaningful certificate, and
  at the fully depolarizing point p = 0.25 the pinned relaxation is **exact**
  (0.25 = p).
* **Four outputs beat one output.** The 4-output Bell measurement gives a
  strictly larger certified bound in every row (e.g. 0.125000 vs 0.089396 at
  p = 0.5). The single-output row p = 1.0 (0.286725) shows how much the
  measurement restriction costs.
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
of the basis): the level-2 moment matrix collapses to 136 words, and the
problem solves in minutes (5967 SDP variables, 2839 equality constraints;
~10 s construction -- the relation set is p-independent and cached per
`tp_pin` -- plus 100-290 s per solve on cvxpy/CLARABEL). Dropping the zero
rows also drops active class-relation links, so the certified values below
sit slightly below the un-reduced construction's (0.301898 / 0.331193 at
p = 0.5); both are valid lower bounds on `p`.

Certified values (CLARABEL; all lower bounds, `value <= p + 1e-5`, strictly
positive):

| p | L2 | L2 + TP pin | L1 single-output (reference) |
| --- | --- | --- | --- |
| 0.25 | 0.210473 | **0.250000** | 0.015625 |
| 0.30 | 0.225669 | 0.265717 | 0.029196 |
| 0.50 | 0.293118 | 0.331071 | 0.089396 |
| 0.75 | 0.384767 | 0.413909 | 0.179185 |
| 1.00 | 0.499687 | 0.499644 | 0.286725 |

Observations:

* **The L2 bound dominates the single-output L1 bound at every p** (e.g.
  0.293118 vs 0.089396 at p = 0.5) -- the localizing structure of eq. (14)
  carries exactly the additional information the document derives, and at
  p = 0.25 with the TP pin the bound is exact (0.250000 = p).
* `removeequalities=True` cannot be used on the class relations: they are
  linearly dependent (conjugate classes share variables in real problems),
  which CLARABEL rejects. The 2839 equalities are passed through as-is.

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
  because their `g U` products (the 32 extramonomials `{J, W} . U_{x,y}`,
  degree 3) are added as `extramonomials`, so every localizing entry monomial
  `u^dagger g w` (degree <= 5) exists as a moment-matrix entry.
* The `J`- and `W`-localizing blocks are tied to the moment matrix by
  **block-0 moment-equalities** encoding the functional classes of eqs.
  (12)/(13) (`G_{u,v} = Tr_{B2}(v u^dagger)` for `J`, `H_{u,v} = Tr_{A0B2}
  (v u^dagger)` for `W`), generated by
  `ncpolopt.hierarchies.class_moment_equalities`. The variable of an entry
  monomial is created at the first upper-triangle occurrence of the monomial
  or of its canonicalized adjoint (the builder reuses the conjugate's
  variable); the helper reads those positions off a draft build of the
  relaxation rather than re-simulating the builder, and pinned (constant)
  classes are dropped entirely.
* These relations are **necessary**: without them the localizing entries
  become independent variables and the objective degenerates.
* The orthogonal-input rules shrink the SDP (62 instead of 66 S-basis words,
  2839 instead of 3047 relations) -- but they are **not free**: the dropped
  words also drop active class-relation links, so the certified values shift
  (e.g. 0.301898 -> 0.293118 at p = 0.5 without the TP pin). Both
  constructions are valid lower bounds; the values in this document are
  measured under the reduced one.

### The four-output Bell measurement

`n_outputs=4` carries the full Bell measurement `{Phi^0, ..., Phi^3}`: the
algebra gains three measurement operators `v_1, v_2, v_3` (14 generators,
113 S-basis words, 199-word level-2 moment matrix, 5551 class relations,
11265 SDP variables) and the correlations pin all 64 entries
`<rho_x sigma_y J v_alpha> = p(alpha|x,y)/(8d)`. The 199-word moment matrix
cannot be materialized through the package's dense cvxpy canonicalization
(~1.6 GB), so `npa_tau_relaxation_value` routes the four-output solves
through the package's direct sparse CLARABEL backend (`solver="clarabel"`,
[clarabel_solver.py](../../src/ncpolopt/solvers/clarabel_solver.py): per
block, the upper-triangle COO mirrored to the lower, off-diagonal svec
entries scaled by sqrt(2), rows grouped NonNeg first, PSD triangle cones
after) -- the same SDP, without the dense intermediate. Each solve takes
8-25 minutes.

Certified four-output L2 values (CLARABEL, direct sparse construction; all
lower bounds, `value <= p + 1e-5`, strictly positive):

| p | L2 (4-output) | L2 + TP pin (4-output) | single-output L2 (reference) |
| --- | --- | --- | --- |
| 0.25 | 0.248300 | **0.250000** | 0.210473 |
| 0.30 | 0.264619 | 0.266663 | 0.225669 |
| 0.50 | 0.331141 | 0.333319 | 0.293118 |
| 0.75 | 0.414331 | 0.416647 | 0.384767 |
| 1.00 | 0.500000 | 0.500000 | 0.499687 |

The full measurement strictly dominates the single-output one at every p
(e.g. 0.331141 vs 0.293118 at p = 0.5), and with the TP pin the bound is
exact at the fully depolarizing point (0.250000 = p).

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
