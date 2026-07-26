# Persistence Implementation — Decision & Change Log

Branch: `feat/persistence` · Spec: [`incremental_persistence_spec.md`](incremental_persistence_spec.md)

This log records every non-trivial decision and change during implementation of
the persistent-store layer, with justification. Entries are append-only and
timestamped by phase/step, newest work at the bottom of each phase.

## Ground rules (from CLAUDE.md / project discipline)

- **Spec is the oracle.** Behaviour is dictated by `incremental_persistence_spec.md`
  (already hardened by 3 audit passes). Any deviation must be logged here with a reason.
- **Failure-mode pass is mandatory** after each phase's happy-path lands: the 7
  categories (Security, Idempotency, Atomicity, State transitions, Deterministic
  ordering, Silent success, Caller contract).
- **Adversarial tests**: tests must be able to catch a mutation (oracle = the spec),
  not just exercise the happy path. Crash-safety gets a `kill -9` harness (spec §6).
- **Gate before commit**: build + full test suite green; no `unwrap()`/`panic!` on a
  path reachable from untrusted input or a caller error.
- **Determinism preserved**: the library's headline guarantee. Delta iteration is
  insertion-ordered; no HashMap iteration on a result-affecting path.

## Phase plan

| Phase | Scope (spec §) | Deliverable |
|---|---|---|
| **1** | §3 — in-memory delta | `GraphDelta`, `DeltaAccessor`, `Op` enum, mutation API + semantics (shadow/tombstone, batch validation, side indices, dead-flag edges), unit + property tests |
| **2a** | §4.2/§4.3 — WAL + recovery | WAL frame format (len‖payload crc), append, writer poisoning, replay, torn-tail truncation, MANIFEST, epoch + clean_shutdown, `open()` |
| **2b** | §4.4/§4.4b/§4.5 — compaction + mmap | CSR snapshot format V1, `to_rkyv_v1` + archived CSR accessor, compaction (empty-delta guard, cid), GC, mmap open, `validate` modes + CSR array validation (§5.1), CAS (`expected_seq`) |
| **X** | §6 — crash harness | `kill -9` loop test, property tests, benches |

Sequencing is a **pipeline with hard dependencies** (2 needs 1's `GraphDelta`,
2b needs 2a's WAL). Phases land sequentially; central build+test between each.

## Orchestration approach

- Per phase: a **design** agent produces the exact module/type/method plan from the
  spec; an **implement** agent writes it; orchestrator builds+tests centrally
  (parallel agents can't share a cargo target lock, and worktree isolation is
  unavailable — the session's primary dir isn't a git repo); an **adversarial
  verify** agent (or workflow) reviews against the spec + 7 failure modes.
- Decisions from each stage are distilled here.

---

## Phase 1 — in-memory delta layer

**Scope:** spec §3 only — `GraphDelta`, `DeltaAccessor`, `Op`, mutation semantics,
tests. Deliberately excludes the `ArcSwap`/`PersistentGraph` wrapper, `seq`/CAS,
WAL, snapshot/mmap, Python API → Phase 2. Reads flow through `DeltaAccessor`
implementing the **existing** `GraphAccessor` trait, so traversal is untouched.

**Files:** created `src/delta.rs`; modified `src/lib.rs` (`pub mod delta;` + re-exports),
`Cargo.toml` (`indexmap = "2"` dep, `proptest = "1"` dev-dep). Nothing else touched.

### Decisions (with justification)

- **`IndexMap` for `added_nodes`** (over hand-rolled `Vec`+`HashMap<name,pos>`). Needs
  three things at once: insertion-order iteration (determinism of materialize/replay),
  O(1) name lookup, and order-preserving shadow removal on `RemoveNode`. `shift_remove`
  (never `swap_remove`) gives all three. A hand-rolled positional map would reintroduce
  exactly the index-invalidation trap that forces the `dead` flag on edges — not worth
  hand-maintaining for nodes.
- **Delta edges: append-only `Vec<DeltaEdge>` with a `dead: bool` flag**, never
  `Vec::remove`'d (spec §3.3 rule 2, audit fix). Removal marks dead in place so the `u32`
  positions cached in `out_index`/`in_index` stay valid; physical reclaim happens only at
  compaction (Phase 2b). Guards against the positional-corruption bug the audit flagged.
- **Two-pass `apply(base, ops)`**: pass 1 validates the whole batch against a
  sequentially-simulated intra-batch state (a lazily-seeded `present` map); an `AddEdge`
  to a missing/tombstoned endpoint returns `DeltaError::MissingEndpoint{op_index,…}` and
  leaves the delta byte-identical (atomicity). Pass 2 is infallible replay via private
  mutators (only reached after validation succeeds). Matches §3.3 rule 4.
- **`apply` takes `base: &dyn GraphAccessor`** so the delta stays base-agnostic in storage
  yet can validate endpoints and maintain counters. `materialize` additionally takes an
  explicit `base_node_names: &[String]` — **deviation from the plan's `materialize(base,delta)`**:
  `GraphAccessor` exposes no node-name/edge iterator and the plan forbade changing
  `accessor.rs`, so live base nodes can't be enumerated from `base` alone; the caller
  supplies the name list. (Test-only helper; not on any hot path.)
- **Four O(1) counters** for `node_count`/`edge_count` (`tombstone_base_hits`,
  `added_non_base`, `masked_base_edge_count`, `live_delta_edge_count`), each mutated only
  on a real live↔masked / present↔absent transition, with a `debug_assert` full recompute
  after every `apply` catching drift. Counts add the positive delta term before subtracting
  the masked/tombstone term to keep the `usize` intermediate non-negative.
- **Mask/unmask incident base edges around the tombstone toggle**: mask BEFORE inserting
  the tombstone (node still "present"), unmask AFTER removing it — plus a self-loop dedup
  in the incident-edge walk so a self-edge isn't double-counted.
- **No normalization/clamp of delta nodes' `base_weight`/`noise_penalty`** (§3.3 rule 5 —
  unlike `build_graph`; caller supplies [0,1]); `debug_assert` range only.
- **`DeltaError` impls `Display`+`Error`** (crate has no `thiserror`) while keeping
  `derive(PartialEq)` for test assertions. `Op` derives serde (free; inert now, used by the
  Phase-2 WAL).

### Tests (33 + fuzz)

Unit tests cover every §3.2 resolution row, shadow-vs-tombstone, transition symmetry,
batch validation (accept node+edge-to-it; reject edge-to-tombstoned / missing endpoint;
all-or-nothing), disjointness, counter correctness. Plus a **proptest** property test
(`traverse(base+delta via DeltaAccessor) == traverse(build_graph(materialized))` — delta
is semantically invisible) and a deterministic **400-op LCG fuzz** exercising the debug
counter cross-check where proptest doesn't run.

### Build/test outcome

Compiles clean (one test-only `E0507` move fixed by orchestrator: clone `NodeView.kind`
in an assert). **80 tests green** (79 pass + 1 ignored PageRank bench); 7 integration green.

### Adversarial verification

3-agent adversarial pass (spec-conformance / 7 failure-modes / test-adequacy):

- **spec-conformance: CLEAN** — faithful to §3.2/§3.3, no correctness defect. Both
  agents cross-checked against `accessor.rs`/`builder.rs` and ran differential probes.
- **failure-modes: CLEAN** — all 7 categories, no defect; suite passes in debug AND
  release; a temporary adversarial probe (reverted) also passed.
- **test-adequacy: 3 P2 test-quality findings (no correctness bugs), all fixed:**
  1. Vacuous `||` assertion in `tombstone_masks_incident_base_edges_both_directions`
     (node `a` had no outgoing base edges → OR always true). Replaced with the real
     rule: the *tombstoned* node resolves to `None` and exposes no edges either way.
  2. No focused unit test for delta-edge positional integrity (only proptest caught
     a hypothetical `Vec::remove` regression). Added
     `remove_middle_parallel_delta_edge_keeps_survivors_positionally_correct` (3
     parallel edges, kill the middle, assert survivors keep target/kind/order).
  3. `randomized_op_mix` compared only counts, and the module doc overstated the
     debug-recompute safety net under `--release` (where `debug_assertions=off`
     compiles it out). Extended the test to diff full per-node topology vs
     `build_graph` (so it bites under release too), and softened the doc comment.

**Outcome: Phase 1 code CLEAN by adversarial review; tests hardened to catch the
named mutations.** 81 lib tests (80 pass + 1 ignored bench), 7 integration; delta
suite green in both debug (recompute-invariant active) and release.

**Committed:** `feat/persistence` — Phase 1 complete.
