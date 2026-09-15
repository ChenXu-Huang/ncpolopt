# ncpolopt Architecture

This document describes the structure and design of the ncpolopt package.
It is a ground-up rewrite of the GPL-3 package **ncpol2sdpa**: the same
hierarchies and the same four solver backends, behind a fresh API with a
typed, dataclass-based object model.

## Design goals

1. **Three-layer object model** — replace the ~1300-line `SdpRelaxation`
   god object with three responsibilities: the symbolic model, the SDP
   data, the result.
2. **Blockwise COO representation** — one scipy `coo_array` per SDP block
   instead of a single giant `lil_matrix` with recursive inverse lookups.
3. **One canonical SDP convention** — every backend maps COO data to its
   native format; sign/constant handling lives in exactly one place.
4. **Lazy, registry-based solver discovery** — importing the package never
   imports a solver; an empty solver environment still builds relaxations.
5. **Deterministic behavior** — no randomness in production paths (the old
   chordal fill drew `random.random()` without ever depending on it).
6. **17 known bugs of the old package fixed**, each with a regression test
   where practical (see [Ported bug fixes](#ported-bug-fixes)).

## Object model

```
Problem (symbolic, frozen, reusable)
   │  .relaxation(level, removeequalities, chordal_extension)
   ▼
NpaRelaxation  (built SDP + metadata, frozen internally)
   │  .solve(solver, settings)
   ▼
Solution (immutable result + extraction helpers)
```

| Layer | Module | Responsibility |
|---|---|---|
| `Problem` | [problem.py](../src/ncpolopt/problem.py) | Frozen dataclass holding variables, objective, constraints, substitution rules, normalization flag. Building a relaxation never mutates it. |
| `NpaRelaxation` | [relaxation.py](../src/ncpolopt/relaxation.py) | Monomial basis generation, moment matrix construction, constraint processing, objective handling, parameter and extra-moment blocks. |
| `SdpProblem` / `SparseBlock` | [sdp_problem.py](../src/ncpolopt/sdp_problem.py) | Frozen, blockwise COO SDP data; `freeze()`; `column_locations` lookup table. |
| `Solution` | [solution.py](../src/ncpolopt/solution.py) | Frozen result (`primal`, `dual`, `status`, `x_mat`, `y_mat`) with `monomial_value()`, `dual_value()`, `dual_block()`, `sos_decomposition()`, `solution_ranks()`. |

## SDP representation

`SdpProblem.blocks` is a tuple of `SparseBlock`s. Each block holds a scipy
`coo_array` of shape `(n_vars + 1, size**2)`: row = SDP variable index,
column = linearized position `(i, j)` of the matrix entry.

- **Variable numbering is 1-based.** `new_variable` increments the counter
  before returning, and variable 0 is the constant placeholder. `freeze()`
  drops the constant slot (`obj = obj_facvar[1:]`), so after freezing
  `obj[i]` corresponds to SDP variable `i + 1`.
- **Only the upper triangle is stored.** Any backend or extraction path
  that materializes full matrices must symmetrize:
  `if i != j: target[j, i] += value`.
- **`column_locations[k] = (block, i, j)`** gives the O(1) map from an SDP
  variable to its moment-matrix entry — replacing the old recursive
  inverse lookup (`get_xmat_value`, bug #14).

### Canonical form

The package convention, fixed once in
[solvers/_common.py](../src/ncpolopt/solvers/_common.py):

```
min  c·x + c0
s.t. A0_b + Σ_k x_k · Ak_b ⪰ 0        (positive semidefinite blocks)
     A0_b + Σ_k x_k · Ak_b = 0        (equality blocks)
primal = c·x* + c0
dual   = -Σ_b tr(Y_b @ A0_b) + c0
```

Backends never re-derive signs or constant terms — they only convert COO
data to their native format. (The old package had each backend grow its
own value formulas; three of four disagreed on the constant sign and
CVXPY reported the primal value twice, bug #11.)

## Module map

```
src/ncpolopt/
├── __init__.py            public API re-exports, __version__, package docstring
├── __main__.py            CLI: version, platform, available solvers
├── _logging.py            module_logger factory, verbose → WARNING/INFO/DEBUG
│
├── variables.py           generate_operators, generate_variables, get_support
├── monomials.py           get_monomials, get_all_monomials, pick_monomials_up_to_degree, ncdegree
├── substitutions.py       apply_substitutions, fast_substitute, separate_scalar_factor,
│                          simplify_polynomial, split_commutative_parts
├── expressions.py         flatten, is_number_type, convert_relational, moment_of_entry
├── moment.py              MomentEntry / MomentExpr — frozen dataclasses with arithmetic
│                          (replaces the old string DSL "+0[0,0]-1.0", bug #15)
│
├── monomial_sets.py       generate_monomial_sets, estimate_n_vars
├── block_structure.py     BlockKind enum (MOMENT / LOCALIZING / SCALAR_EQUALITY /
│                          SCALAR_INEQUALITY / PARAMETER / COPY / PPT), compute_block_structure
├── sdp_problem.py         SdpProblem / SparseBlock, freeze(), column_locations
├── equality_elimination.py eliminate_equalities → BasisTransform (rank check fix, bug #2)
├── sdpa_writer.py         write_dat_s, read_sdpa_out — pure functions, testable without binary
│
├── problem.py             Problem (user-facing model) + .relaxation() / .solve()
├── relaxation.py          NpaRelaxation: moment matrix build, constraints, objective
├── solution.py            Solution + extraction helpers
├── chordal.py             find_variable_cliques, sliding_cliques, find_clique_index
│                          (deterministic fill, chompack explicit opt-in; bug #13)
├── physics.py             Probability, correlator, maximum_violation,
│                          define_objective_with_I, generate_measurements,
│                          projective/bosonic/fermionic/pauli_constraints, get_neighbors
│
├── hierarchies/           steering.py, moroder.py, rdm.py — subclasses of NpaRelaxation;
│                          insertion.py — operator-insertion (MDI) helpers
└── solvers/               base.py, registry.py, cvxpy_solver.py, clarabel_solver.py,
                           picos_solver.py, mosek_solver.py, sdpa_solver.py, _common.py
```

## Hierarchies

Hierarchies are **subclasses** of `NpaRelaxation`, not enum-selected
variants of one class:

| Class | Module | Behavior |
|---|---|---|
| `SteeringHierarchy` | [hierarchies/steering.py](../src/ncpolopt/hierarchies/steering.py) | `matrix_var_dim` sub-blocks for the assemblage; uses `normalized=False`; the (0,0) moment is a free variable whose first variable is SDP variable 1. |
| `MoroderHierarchy` | [hierarchies/moroder.py](../src/ncpolopt/hierarchies/moroder.py) | Duplicate moment matrix (`extramomentmatrices=["copy"]`) + PPT constraints via the partial-transpose operator on the vec space. |
| `RdmHierarchy` | [hierarchies/rdm.py](../src/ncpolopt/hierarchies/rdm.py) | Circulant banded moment layout; block order computed explicitly from block indices (old `m_block` order dependence, bug #12). |

The operator-insertion (MDI, Rosset–Buscemi–Liang) family is not a
layout variant but a problem-generation pattern, so it lives in
[hierarchies/insertion.py](../src/ncpolopt/hierarchies/insertion.py) as
two factory helpers: `trace_moment_pins` pins the moments over a basis to
their trace values in an explicit matrix realization
(`momentsubstitutions`), and `class_moment_equalities` emits the
functional-class equalities of the inserted-operator localizing blocks as
block-0 `momentequalities`. The equalities need the SDP variable
positions of entry monomials `u† g v`; rather than re-simulating the
builder's variable-creation order, the helper builds a draft relaxation
of the problem (the equality blocks are appended after the moment
blocks, so the draft's block-0 layout is final) and reads the positions
off `monomial_index` + `column_locations`. A class whose representative
entry is pinned (a constant moment, including an identically zero word)
is **not** dropped: each free class member is tied to the pinned constant
by a constant-equality, so pins reach into the free classes instead of
silently weakening the relaxation.

SymPy never reorders noncommutative factors, and `Dagger(A*B)` flips the
order; tests must state products explicitly rather than assuming
commutativity.

## Chordal sparsity extension

`Problem.relaxation(chordal_extension=True)` (also via `Problem.solve`)
replaces the variable set by the cliques of the chordal completion of the
correlative sparsity pattern — SparsePOP style:

1. `_fill_pattern` builds the pattern matrix from the support of the
   objective and constraints (fill value pinned to 1.0, fully
   deterministic).
2. `_clique_set_from_cholesky` (default) factors the pattern and takes
   the maximal cliques from the rows of the factor; `_clique_set_from_chompack`
   offers the AMD-ordered chompack path via an explicit `method` argument.
3. Each clique becomes its own moment block in a multipartite relaxation.

## Solver backends

[solvers/registry.py](../src/ncpolopt/solvers/registry.py) holds the
registry and detection order `CVXPY → MOSEK → CVXOPT → SDPA`. A backend
must be both registered (`register()`) and its module importable
(`available()`); the external SDPA backend additionally requires its
binary on PATH — an explicitly configured `settings.sdpa_executable`
still works at solve time even when SDPA is not detected. The direct
CLARABEL backend is registered but deliberately absent from the
detection order: it is an explicit-selection escape hatch for very large
SDPs (`solver="clarabel"`) and must not change what `"auto"` picks.

All solver imports are lazy; `import ncpolopt` never imports cvxpy,
mosek, picos, or cvxopt. With no solver installed, `available_solvers()`
returns `[]` and `solve()` raises `SolverError` with an install hint.

| Backend | Module | Notes |
|---|---|---|
| CVXPY | [solvers/cvxpy_solver.py](../src/ncpolopt/solvers/cvxpy_solver.py) | Default; never mutates caller dicts (bug #10). |
| CLARABEL | [solvers/clarabel_solver.py](../src/ncpolopt/solvers/clarabel_solver.py) | Direct sparse construction for very large SDPs; explicit selection only; real-valued SDPs only. |
| PICOS/cvxopt | [solvers/picos_solver.py](../src/ncpolopt/solvers/picos_solver.py) | Returns a `PicosModel` so constraints (e.g. PPT in Moroder tests) can be added after construction; fixed duplicate variable name and double row offset (bugs #5, #6). Unreliable on Windows (see Testing). |
| MOSEK | [solvers/mosek_solver.py](../src/ncpolopt/solvers/mosek_solver.py) | Parameter lookup via `getattr(mosek.iparam, name)` instead of `eval` (bug #7); x_mat/y_mat exchange fixed (bug #8). |
| SDPA | [solvers/sdpa_solver.py](../src/ncpolopt/solvers/sdpa_solver.py) | Shells out to the binary inside a `TemporaryDirectory` (TOCTOU race, bug #9); parsing in `sdpa_writer.py` as pure functions. |

### Direct solver data construction (very large SDPs)

The CVXPY backend's canonicalization materializes a dense problem matrix
(`n_vars` x `n_vars`); for SDPs with ~1e4 variables and a 199-word moment
matrix the intermediate alone is ~1.6 GB and does not fit. The CLARABEL
backend (`solver="clarabel"`,
[solvers/clarabel_solver.py](../src/ncpolopt/solvers/clarabel_solver.py))
builds the native CLARABEL data directly from the frozen COO
`SdpProblem`, skipping CVXPY entirely. The mapping encodes the canonical
form from [solvers/_common.py](../src/ncpolopt/solvers/_common.py) once
more, mirrored for CLARABEL's convention `b - A x in K`:

- Per block: scalar blocks become NonNeg rows (constant term into `b`);
  empty scalar blocks get a dummy `b = +1.0` row (the package's
  empty-scalar artifact: `0.0 >= 0` is Python True, which CVXPY turns
  into `b = +1.0`); PSD blocks become row-major **lower**-triangle svec
  rows, off-diagonal entries scaled by `sqrt(2)` (the convention CVXPY
  uses for its PSD triangle cones).
- Rows are grouped [NonNeg first | PSD triangles in block order], then
  `cones = [NonnegativeConeT, *PSDTriangleConeT]`. CLARABEL minimizes
  `1/2 xᵀPx + qᵀx` subject to `b - A x in K`, so `P` is a zero csc,
  `q = obj` as-is, `A` carries the negated coefficient rows (package
  data-dict convention: coefficients negated, constants not) and `b`
  carries the constant offsets. The dual blocks unscale from the svec
  segments of CLARABEL's `z` (off-diagonal entries times `sqrt(2)`).
- Solver statuses map through a small table (`"Solved"` /
  `"AlmostSolved"` -> `"optimal"`, mirroring the CVXPY backend's
  `optimal_inaccurate` folding).

The resulting data is exactly the canonical data the CVXPY backend would
have produced dense — pinned bit-identical by
`tests/test_clarabel_solver.py` — so results agree with the CVXPY path to
solver tolerance.

## Ported bug fixes

The legacy defects this rewrite fixes:

| # | Bug | Fix |
|---|---|---|
| 1 | `flip_sign` dead `startswith("+")` branch | DSL removed |
| 2 | rank check `min(A.shape != rank)` (tuple vs int) | `min(A.shape) != matrix_rank(A)` |
| 3 | both branches of `find_solution_ranks` used `x_mat[0]` | use the passed matrix |
| 4 | `get_dual` ignored the `ymat` argument | use `ymat` |
| 5 | picos duplicate variable name `'X'` | renamed, documented API |
| 6 | picos equality block double row offset | single increment |
| 7 | MOSEK parameters via `eval()` | `getattr(mosek.iparam, ...)` |
| 8 | MOSEK x_mat/y_mat swapped | fixed and verified numerically |
| 9 | named-temp-file naming race | `TemporaryDirectory()` |
| 10 | cvxpy mutated the caller's dict | copy before use |
| 11 | cvxpy reported primal twice | unified dual formula in `_common.py` |
| 12 | RdmHierarchy `m_block` order dependence | explicit block-index computation |
| 13 | chordal `random.random()` fill | deterministic constant fill |
| 14 | recursive inverse `get_xmat_value` | `column_locations` table |
| 15 | string moment DSL | `MomentEntry` / `MomentExpr` |
| 16 | misleading `convert_relational` message | f-string interpolation |
| 17 | `extract_dual_value` sign convention | documented canonical form |

## Testing strategy

pytest under `tests/`, numerical tolerance `1e-5`.

- `conftest.py` provides the `solver_kind` fixture: numerical tests are
  parametrized over the currently usable backends (MOSEK filtered by
  license, SDPA by binary presence). An autouse fixture clears the SymPy
  expression cache after every test.
- **The cvxopt/PICOS path is unreliable on Windows** (returns `unknown`
  states or spurious optima on mathematically infeasible models). Tests
  never assert numbers on it — they skip with `pytest.skip` on
  `pic.SolutionFailure`, and the Moroder witness value is pinned via a
  cvxpy/CLARABEL reconstruction.
- **Multi-minute solver runs carry the `slow` marker** and are excluded
  from CI (`uv run pytest -m "not slow"`): the quantum-memory L2
  certification tests (single-output and 4-output variants, the latter
  8-25 min per solve). Run them locally with
  `uv run pytest -m slow tests/test_memory_verification.py`.
- Each ported bug fix carries a targeted regression test (e.g. MaxCut
  exercises the equality-elimination rank check; `test_sdpa_writer.py`
  round-trips .dat-s text without a binary).

## Directory layout

```
├── AGENTS.md, CLAUDE.md      agent/project instructions (see AGENTS.md)
├── README.md                 quickstart and install
├── pyproject.toml            uv build; GPL-3.0-only; extras cvxpy/mosek/cvxopt/chordal
├── src/ncpolopt/             the package
├── examples/quantum_memory/  MDI quantum-memory verification: common.py (shared
│                             construction on the insertion helpers), verify_dual_sdp.py /
│                             verify_relaxation.py (README.md inside), incl. the
│                             4-output NPA-tau variant solved via solver="clarabel"
└── tests/                    pytest suite (incl. test_memory_verification.py;
                             slow-marked certification runs excluded from CI)
```
