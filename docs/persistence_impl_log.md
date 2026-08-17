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

**Committed:** `feat/persistence` — Phase 1 complete (`fd098f3`).

---

## Phase 2a — WAL + recovery + PersistentGraph wrapper + CAS

**Scope:** spec §4.1/§4.2/§4.3, §3.5 (concurrency), §3.3.6 (CAS). Durable single-writer
store. **Deferred to 2b** (explicit): CSR V1 snapshot format, compaction, mmap base,
open-time GC, `validate` modes. In 2a the base is an owned `OrpheusGraphInner` snapshotted
with the **existing** `to_rkyv`/`from_rkyv_rebuild` (flat `format_version 0`); the delta
accumulates unbounded until compaction lands in 2b. Python API also deferred (Rust-native
API + Rust tests only).

**Files:** `src/persist/{mod,wal,manifest,error,lock}.rs`; modified `src/lib.rs`, `Cargo.toml`.
**Deps:** `arc-swap`, `crc32fast`, `postcard` (WAL encode), `fs2` (flock), `getrandom`
(epoch); dev: `tempfile`. (`thiserror` appeared only transitively — `PersistError` hand-rolls
`Display`/`Error` to match `DeltaError` style.)

### Decisions (with justification)

- **`apply()` ordering is strict** (§4.2/§3.3.6): poison-check → CAS (`expected_seq` vs
  durable `seq`, atomic under the writer `Mutex`, WAL untouched on conflict) → validate on a
  *cloned* delta (a `DeltaError` never becomes durable) → encode → **WAL append (the commit
  point)** → fsync-per-policy → advance in-memory `seq`/`delta` → **ArcSwap publish**. A panic
  between append and publish loses nothing: the frame owns the seq durably, next `open()`
  reconstructs it.
- **WAL crc covers `len ‖ payload`**, not payload alone (audit fix), and recovery checks
  `len ≤ remaining − 8` **before** slicing/allocating (no allocate-from-unvalidated-len OOM).
- **`scan_wal` returns typed outcomes**: a torn/short/over-long/crc-bad *tail* → truncate
  (`set_len` + fsync + fsync(dir)) with a warning; a crc-valid frame that fails to decode, or
  a real seq **gap**, → hard `Corrupt` error (never guess). Writer poisoning guarantees a torn
  frame is only ever the last one, so tail-truncation can't drop an acked frame.
- **Writer poisoning** is a `bool` in `WalWriter`, touched only under the writer `Mutex`;
  readers never consult it, keeping the reader path lock-free. Any append/fsync failure sets
  it; subsequent `apply()` returns `Poisoned`.
- **epoch = u128 from 16 `getrandom` bytes**, re-minted iff `tail_truncated || !clean_shutdown`;
  MANIFEST rewritten (tmp+fsync+rename+fsync(dir)) at open to land it. `clean_shutdown` set
  true only by `close()`.
- **Single-writer enforced twice**: `fs2` exclusive flock on `LOCK` (cross-process,
  kernel-released on crash) taken *before* recovery, plus in-process `Mutex<Writer>`.
- **Readers lock-free via ArcSwap**: `snapshot()` = `state.load_full()` → a consistent
  `(base, delta, seq, epoch)`; a traverse during `apply()` sees the immutable pre-publish pair.
- **Deviations from plan** (logged): `Writer.base` is `Arc<BaseGraph>` (not
  `Arc<OrpheusGraphInner>`) so the identical `Arc` is shared into each published `GraphState`;
  `Writer` also caches `snapshot_file`/`snapshot_crc32` so MANIFEST rewrites are byte-faithful
  without re-reading disk. `serialization.rs` untouched.

### Build/test outcome

Compiles clean (one test-only `E0277`: added a manual `Debug` for `PersistentGraph` — its
`ArcSwap`/`Mutex`/`DirLock` fields don't derive it). **104 tests green** (+23 persist:
CAS-conflict-no-phantom-frame, poisoning-blocks-applies-reads-survive, recovery-skips-
≤snapshot_seq, torn-tail-truncates-last-only, epoch-reminted-on-unclean-reopen, every-batch-
commits-and-recovers, concurrent-readers-consistent, …).

### Adversarial verification & crash harness

3-agent adversarial pass (spec-conformance §4 / crash-focused 7 failure-modes / test-adequacy)
found 2 P1 + 3 P2 — **all fixed**:

- **P2 recovery re-validated WAL frames** via `delta.apply()` — violates §3.3 rule 4 ("replay
  does NOT re-run validation"). Benign in pure 2a (snapshot_seq always 0) but would brick after
  2b compaction or a validation-rule tightening. **Fix:** added `GraphDelta::replay(base, ops)`
  (PASS-2 mutators only, no validation); recovery now calls it. **Load-bearing for 2b.**
- **P2 `open()` used `create(true)` on wal.log** — a missing WAL under a present MANIFEST was
  silently treated as empty (discarding every committed post-snapshot batch; §7 violation).
  **Fix:** recovery opens the WAL without `create`; a missing WAL → `Corrupt`. Only `create()`
  mints a fresh WAL.
- **P1 `close()` on a poisoned writer** wrote `clean_shutdown=true` and returned `Ok` —
  suppressing the epoch re-mint exactly when a durability write had failed. **Fix:** on poison,
  `close()` writes `clean_shutdown=false` and returns `Poisoned`; never a clean flag on a
  poisoned store.
- **P2 fsync-failure ambiguity**: a write that succeeds then fails its fsync leaves a crc-valid
  frame the kernel may still write back, so `apply`-returned-`Err` does NOT strictly mean
  "not durable". Inherent to fsync-failure semantics without a per-frame commit marker.
  **Fix (honest):** documented on `WalWriter::append` — caller reconciles via `(epoch, seq)`
  on reopen; a per-frame commit/torn-write marker is the proper long-term fix (deferred, noted).
- **P1 no real `kill -9` crash harness** (spec §6's "load-bearing" test) — every "crash" test
  simulated death with in-process `drop`. **Fix:** added `tests/crash_harness.rs` — a child
  re-exec's this test binary (`OG_CRASH_CHILD`), opens the store, applies monotonic marker
  batches, and appends each acked seq (fixed-width, torn-safe) to a ground-truth `acked.log`;
  the parent SIGKILLs at varied offsets, reopens, and asserts: **open always succeeds** after a
  real kill -9; the recovered marker set is a **clean prefix** (marker `seq` present, `seq+1`
  absent — no hole, no phantom); **`seq >= max_acked`** — process death loses nothing acked
  under EveryBatch *or* OnFlush (page cache survives a crash, §4.2); seq monotonic across
  incarnations; epoch re-mints on each unclean reopen; kills also land during recovery
  (mid-open double-fault window) by re-spawning on the same dir. **Passes ×100/policy (200
  kill -9 cycles) in ~4.8 s.** (A harness self-bug — torn `acked.log` append concatenating two
  seqs into a too-high ground truth — was caught at iteration 42 and fixed with fixed-width
  records; the store itself recovered correctly throughout.)

**Outcome: Phase 2a durability proven by a real kill -9 harness; all verification findings
fixed.** 103 lib + 1 crash-harness + 7 integration tests green (crash harness default
24/policy in `cargo test`, ×100 via `OG_CRASH_ITERS`).

**Committed:** `feat/persistence` — Phase 2a complete.

---

## Phase 2b — CSR V1 snapshot + compaction + mmap + GC + validate

**Scope:** spec §4.4/§4.4b/§4.5/§5.1. The on-disk snapshot format V1 (CSR), compaction,
mmap-backed base, open-time + inline GC, and `validate` modes. This is the phase that makes
the base larger-than-RAM-capable and gives archived neighbor lookup petgraph asymptotics.

**Files:** created `src/persist/snapshot.rs`; modified `persist/{mod,manifest,wal}.rs`,
`lib.rs`, `Cargo.toml`. **Dep:** `memmap2`. Legacy flat `serialization.rs` (V0) untouched.

### Decisions (with justification)

- **CSR V1 in a NEW `snapshot.rs`, not `serialization.rs`** — keeps V0 (the ephemeral/Redis
  flat path, `format_version 0`, never mmap-traversed) cleanly separated from V1 (mmap CSR),
  and colocates the unsafe mmap view + `validate` + §5.1 sweep in one auditable place.
  `SerializableGraphV1{nodes, edges: CsrEdge, node_offsets, in_offsets, in_mirror}`.
- **`in_offsets` added alongside `in_mirror`** (plan named only the mirror): true O(degree)
  *incoming* needs a `to_idx`-keyed offset array — `in_mirror` alone would force a scan.
  Outgoing is O(degree) via `node_offsets[idx..idx+1]`; this fixes the §4.4b motivating defect
  (V0's `ArchivedGraphView` did an O(E) filter scan per expansion).
- **`FORMAT_VERSION` 0→1**; the gate still rejects `> 1`, so `open()` reads BOTH. `create()`
  writes V1. `open()` branches: V0 → `from_rkyv_rebuild` → `BaseGraph::Owned` (never mmap'd —
  it lacks the CSR index); V1 → `open_snapshot(mode, validate, prefault)`. Added
  `Writer.format_version` (correctness fix beyond the plan) so `close()`/`open()` never rewrite
  a V0 store's MANIFEST with `format_version=1`.
- **mmap soundness** (`ArchivedCsrView`): read-only `memmap2::Mmap` owned by the struct + a
  `*const ArchivedSerializableGraphV1` derived once from the validated archive. Sound because
  the mapped region's virtual address is independent of where the small `Mmap` handle sits, so
  the pointer survives moves (a *stronger* guarantee than the existing `Vec<u8>`-backed view);
  no `&mut`/interior mutability → `Send`+`Sync` are data-race-free reads. SIGBUS precondition:
  mmap only on local block storage; network / larger-than-RAM-under-pressure callers pass
  `mode=Owned` (read+validate up front, no lazy faulting).
- **`validate` stratified `Full | Crc | None`** with a strict trust contract. **Full**
  (default, mandatory for untrusted snapshots): crc32 + rkyv structural + the **§5.1 sweep
  extended to the CSR arrays** — edge endpoints in range; `node_offsets`/`in_offsets`
  monotonic, in-range, correct `== edges.len()` sentinel, and group-consistent; `in_mirror` a
  true permutation of `0..E` (seen-bitset, no dup/OOB) — all returning `Err`, never panicking.
  **None** is O(1) `access_unchecked`, permitted only for a file this process just wrote.
- **Compaction folds at the current seq** (no new seq) with an **empty-delta NO-OP guard** and
  a **monotonic `cid`** in the filename (`snapshot-{seq:020}-{cid:010}.og`, `cid` in MANIFEST)
  so an equal-seq compaction can never rename over / delete the live snapshot (the brick the
  audit flagged). Order: `to_rkyv_v1` → tmp+fsync+rename+fsync(dir) → MANIFEST tmp+fsync+rename
  +fsync(dir) → WAL truncate (new `WalWriter::truncate`, fsync'd) → publish → delete-old (only
  if `old_path != new_path`). Post-compaction republish honors the open mode (Owned keeps the
  just-built inner; Mmap re-mmaps the fresh file with `Validate::None` — self-written, sound).
- **Auto-compaction** `delta_ops > max(1000, base_nodes/10)`, checked at the end of `apply()`
  under the held writer lock via `compact_locked` (no re-entrancy); bounds the delta, resets
  the counter, never fires on an empty delta.
- **Two-tier GC**: inline post-compaction delete (primary reclaim for the open-once-run-for-
  months workload) + an open-time `read_dir` sweep of unreferenced `snapshot-*.og` (backstop
  for orphans from a crash between publish and delete). Linux `unlink`-safe under a live mmap;
  Windows caveat documented.
- **`open()` kept `(dir, create)`** delegating to a new `open_with(dir, create, mode, validate,
  prefault)` (defaults `mode=Mmap`, `validate=Full`, `prefault=false`).

### Build/test outcome

Compiles clean first try (memmap2 + unsafe CSR view). **128 lib tests** (+24 Phase 2b) + crash
harness + 7 integration green.

### Adversarial verification

3-agent adversarial pass (unsafe/mmap+hostile / compaction-crash / conformance+tests). First
run's agents all hit a session limit (false "clean"); re-run after reset gave real results:

- **unsafe mmap + `ArchivedCsrView` + `validate_v1`: CLEAN** — soundness confirmed (immutable
  mmap, stable `*const` across moves, no interior mutability); an empirical adversarial probe
  (added, run green, reverted) confirmed `validate="full"` rejects every hostile CSR corruption
  (OOB endpoints, non-monotonic offsets, non-permutation `in_mirror`) with `Err`, never a panic.

Found **2 P1 + 2 P2 — all fixed:**

- **P1 `close()` after a post-commit compaction failure rolled the MANIFEST back** to stale
  in-memory `snapshot_*` fields, losing every WAL batch the compaction had already durably
  folded (§7 violation). **Fix:** `compact_locked` now advances the in-memory snapshot pointer
  (`snapshot_seq/file/crc/format_version/compaction_id`) **immediately after the MANIFEST
  rename** (the durable commit), before the WAL-truncate/re-mmap steps that can still fail — so
  a later poison leaves the `Writer` consistent with the committed MANIFEST and a poisoned
  `close()` rebuilds the correct pointer, never rolling back.
- **P1 the `kill -9` harness never exercised compaction** (threshold 1000 unreachable in a
  1–40 ms child life). **Fix:** added `pub fn set_auto_compact_threshold(Option<usize>)` (a
  runtime knob, mirroring `set_fsync_policy`); the crash child sets it to 8, so compaction fires
  every ~8 committed batches and the parent's SIGKILLs land inside `compact_locked`'s non-atomic
  sequence (snapshot write, MANIFEST rename, WAL truncate, re-mmap). The parent also asserts a
  compacted snapshot (`cid > 0`) exists. **Passes ×100/policy (200 kill -9 cycles) with
  compaction firing throughout** — proving compaction crash-safety, not just the logical windows.
- **P2 leftover `.og.tmp` never GC'd** (open-time sweep only matched `.og`). **Fix:** GC now
  also unlinks stray `snapshot-*.og.tmp` and a stale `MANIFEST.json.tmp` (single-writer under
  the lock, never the live file).
- **P2 no test proved the real `compact_locked` truncated the WAL** (removing `wal.truncate()`
  left every test green — recovery skips folded frames anyway). **Fix:** `compact_folds_delta`
  now captures `wal_len` before/after and asserts it shrinks to 0.

**Outcome: Phase 2b verified — unsafe/mmap sound, hostile snapshots rejected without panic,
compaction crash-safe under a real kill -9 harness that now exercises it.** 127 lib + crash
harness + 7 integration tests green.

**Committed:** `feat/persistence` — Phase 2b complete. **All planned phases (1, 2a, 2b) done.**

---

## Final gate & status

- **Build:** clean (`cargo build --release`, PyO3 forward-compat flag).
- **Tests:** 127 lib + 1 crash harness + 7 integration green; crash harness runs
  ×100/policy (200 kill -9 cycles, both fsync policies, compaction firing every ~8 batches)
  via `OG_CRASH_ITERS=100`, default 24 in `cargo test`.
- **Clippy:** all persistence code (`delta`/`persist`/`snapshot`) clean; one pre-existing
  `pybridge` complex-type warning is out of scope.
- **Determinism preserved** (delta insertion order, CSR built deterministically); **no
  `unwrap`/`panic` on any recovery/archived-query path over untrusted bytes.**

**Delivered (spec §3 + §4):** the full embedded persistent-store engine — mutable delta
overlay, durable WAL + crash recovery, optimistic CAS, epoch/clean_shutdown timeline-fork
detection, CSR V1 mmap snapshot (larger-than-RAM-capable, O(degree) archived lookup),
compaction with auto-threshold, two-tier GC, and `validate` trust modes — each phase
adversarially verified (spec-conformance + 7 failure-modes + crash focus) with findings fixed,
and durability proven by a real `kill -9` harness.

**Commits:** `fd098f3` (Phase 1) · `d76f67f` (Phase 2a) · `5a8b974` (Phase 2b) · `4c09dec`
(clippy). Branch `feat/persistence`, not yet merged/pushed — awaits review.

### Full-engine re-audit (all phases composed)

A fresh clean-slate audit of the whole engine (6 finders across delta / WAL+recovery /
concurrency+CAS / compaction+GC / mmap+CSR+validate / cross-phase-API, adversarially verified):
**0 P0, 0 P1, 3 P2** (4 raw, 1 refuted) — no critical/high defect survived cross-phase review.
All 3 P2s fixed (`61aea32`):

- **Determinism (cross-phase):** compaction re-ran `build_graph`, whose base_weight
  max-normalization isn't idempotent — a delta `RemoveNode` of the max-weight node rescaled
  every survivor's `base_weight` at an **unchanged seq**, changing beam/find_path scores
  (violates "same (seq, ctx) → same output"). Fix: `build_graph_prenormalized` (skips
  base_weight normalization; inputs already in [0,1]) used by compaction. Regression test added.
- **Durability:** a WAL truncated/suffix-lost on a frame boundary (external FS fault) was
  silently accepted as a shorter clean log — seq regressed, no epoch re-mint, acked batches
  gone — while `rm wal.log` was correctly `Corrupt`. Fix: a durable `high_seq` high-water in the
  MANIFEST (create/close/compaction/recovery); recovery hard-errors `Corrupt` when
  `last_applied < high_seq`. `#[serde(default)]` for safe degradation. Regression test added.
- **Efficiency:** auto-compaction left `delta_ops` above threshold on failure → a full rebuild
  on every subsequent apply (compaction storm). Fix: reset `delta_ops` on failure.

kill -9 harness still passes ×100/policy after the high-water change (process death never loses
data below the high-water). **Engine is now P0/P1-clean across three per-phase passes + one
whole-engine pass.**

---

## Phase 3 — Python API (PyO3 bindings)

**Scope:** spec §4.6 only — wire the already-hardened Rust `PersistentGraph` into `pybridge`
so it's reachable from Python. No engine-level change; `delta.rs` gets a one-comment-block
change to remove an invariant now enforced at the FFI boundary instead (below).

**Files:** modified `src/pybridge.rs` (bulk of the change: `PyPersistentGraph`, op parsing,
`open`/`create_persistent`, exceptions), `src/lib.rs` (class/function/exception module
registration), `src/delta.rs` (debug_assert removal, see below), `python/orpheusgraph.pyi`
(type stubs — owned by another agent, not edited here). No changes to `persist/`, `delta.rs`
mutation logic, `snapshot.rs`, or the WAL/recovery/CAS/compaction engine itself.

### Decisions (with justification)

- **Op-dict schema mirrors the spec verbatim.** `apply(ops, expected_seq=None)` takes a list
  of dicts keyed by `"op"` (`upsert_node` / `remove_node` / `add_edge` / `remove_edge`,
  matching §4.6's example exactly), parsed up front into `Vec<Op>` before anything touches the
  writer — a bad op anywhere in the batch raises before a WAL frame is written, preserving the
  all-or-nothing batch contract (§3.3 rule 4) at the Python boundary too. Required fields
  (`name`, `from`/`to`/`kind` on edges) reject empty strings — an empty name would otherwise
  inject a phantom node into traversal (same contract as the existing overlay parser).
- **Boundary weight validation via `require_unit`** — `base_weight`/`noise_penalty` on
  `upsert_node`/`add_edge` are checked against `[0.0, 1.0]` at the FFI boundary and rejected
  with `ValueError` if out of range, *before* the value ever reaches `GraphDelta`. This let the
  **stale `debug_assert` in `GraphDelta::upsert_node` be removed**: that assert pre-dates the
  Python API and would have been the only enforcement point, but it's the wrong layer post-#6 —
  the audit-#6 determinism guarantee requires that whatever value *does* enter the delta (e.g.
  from a Rust caller that bypasses the FFI check) is stored **verbatim**, so the live view and
  a post-compaction rebuild (`build_graph_prenormalized`) return the identical weight at the
  same `seq`. An assert/clamp inside `GraphDelta` would diverge those two paths (or crash a
  query-adjacent path on release-mode-invisible debug builds) instead of just documenting the
  contract. Enforcement moved from "assert deep inside the engine" to "reject loud at the one
  boundary that can't be bypassed by a well-behaved caller" — `delta.rs` now only comments the
  invariant, `pybridge.rs`'s `require_unit` is the actual gate for Python callers.
- **GIL released across `apply`/`flush`/`compact`/`close` and all traversal methods**
  (`py.allow_threads`), matching the existing `OrpheusGraph` binding convention — a Python
  caller with multiple threads (or an async loop offloading to a thread pool) doesn't stall
  other Python work during a WAL fsync or a compaction fold.
- **`close()` consumes the store via `Option<PersistentGraph>`** (`pg: Option<...>` on
  `PyPersistentGraph`) — Rust's `PersistentGraph::close(self)` takes ownership (flush + mark
  clean shutdown), but the Python object must keep living after `.close()`/`__exit__` returns.
  `.take()` moves the store out for the one real close; a second `close()` (or use after
  `__exit__`) finds `None` and is a no-op — idempotent by construction, not by a manual flag.
  A dropped-without-close store still releases its flock + mmap; it's just not marked clean, so
  the next `open()` re-mints the epoch (the same process-death path the crash harness covers).
- **Custom `ConflictError` / `CorruptError`** (`pyo3::create_exception!`, registered on the
  module) so callers can `except` CAS conflicts and corruption precisely instead of parsing
  message strings. `map_persist_err` fans the Rust `PersistError` enum out to the closest
  Python type: `Conflict → ConflictError`, `NotFound → FileNotFoundError`, `Delta →
  ValueError`, `Poisoned`/`LockHeld → RuntimeError`, `Corrupt`/`UnsupportedVersion →
  CorruptError`, `Io → OSError` — one mapping point, not scattered per call site.
- **`create_persistent(dir, nodes, edges)`** runs `build_graph` once and hands the result
  straight to `PersistentGraph::create` as the initial base — a one-pass seed for a
  freshly-built graph, instead of the alternative of `open(create=True)` followed by an
  `apply()` that replays every node/edge through the delta (and would defer PageRank/pagerank
  bypass, §3.4). Refuses to clobber an existing store (delegates to the Rust `create`'s
  existing-store check).

### Build/test outcome

Compiles clean. No dedicated Rust `#[test]` module for the bindings yet (PyO3 boundary code is
conventionally exercised from the Python side); verified via a manual smoke script covering
`open(create=True)`, a multi-op `apply`, read-API parity (`get_node`/`outgoing_edges`/
`beam_traverse`/`find_path`) against the just-applied state, a CAS conflict + successful retry,
an out-of-range-weight rejection, `flush`+`compact`, `close`+reopen durability, the `with`
context-manager form, `create_persistent` seeding, and the `FileNotFoundError` path on a
missing store with `create=False` — all passed. A proper `pytest` suite is not yet part of this
phase's delivered scope.

**Delivered (spec §4.6):** `orpheusgraph.open`, `orpheusgraph.create_persistent`,
`PersistentGraph` (`.seq`, `.epoch`, `.apply`, `.flush`, `.compact`,
`.set_fsync_policy`, `.set_auto_compact_threshold`, `.close`, context-manager protocol, and the
full `OrpheusGraph`-parity read API), `orpheusgraph.ConflictError`, `orpheusgraph.CorruptError`.
The durable engine (Phases 1/2a/2b) is unchanged; this phase only adds a Python-reachable
surface over it.

### Audit #5 (deeper edges + regression check on the #4 fixes)

5 finders (recent-fix-regressions / arithmetic-limits / rare-interleavings / recovery-corruption
-deep / delta-property-deep), adversarially verified: **0 P0, 2 P1, 1 new P2** — and both P1s
were **regressions introduced by the audit-#4 `high_seq` fix** (a good catch on the fix itself).
All fixed (`554792a`):

- **P1 × 2 — `high_seq` over-reported durability under OnFlush → false brick.** high_seq was
  set from the in-memory `seq`, but an OnFlush append is only write-through (not fsync'd), so
  `seq` isn't durable. A crash-recovery reopen (which replays page-cache frames) and a
  poisoned/flush-failed `close()` both durably recorded `high_seq = seq`; a subsequent
  *legitimate* power-loss of the un-fsync'd tail then tripped the `last_applied < high_seq`
  guard and permanently bricked an otherwise-recoverable store. **Fix:** `Writer.durable_seq` —
  the highest seq actually on stable storage, advanced ONLY by a real fsync (`append` now
  reports whether it fsync'd; flush/clean-close/compaction advance it; recovery sets it to the
  on-disk high_seq and no longer promotes to `last_applied`). `high_seq` is written from
  `durable_seq` everywhere, so it never covers un-fsync'd frames. Two regression tests pin both
  directions (un-fsync'd tail loss reopens gracefully; genuinely-durable loss is still Corrupt).
- **P2 — MANIFEST had no integrity checksum.** A bit-rotted `snapshot_seq`/`high_seq` (the
  snapshot bytes have a crc; the MANIFEST didn't) would silently skip WAL frames. Added a crc32
  over the integrity-critical fields, stamped on write, verified on read. Regression test added.

Refuted: 2 (arithmetic-limits and a rare-interleaving claim didn't survive verification). The
durable_seq change passed the kill -9 harness ×100/policy unchanged. **132 lib + crash harness +
7 integration green.**

Lesson logged: the audit-#4 high_seq fix conflated "acked" (in-memory) with "durable"
(fsync'd) — the exact distinction OnFlush is built around. `durable_seq` makes it explicit.

### Audit #6 (maximal — 9 finders, two-lens verification, property/fuzz emphasis)

The biggest pass: 9 finders (durable_seq state machine, MANIFEST checksum, unsafe mmap, delta
fuzz, recovery byte-fuzz, concurrency, arithmetic, compaction-crash-deep, API/panics) with
**perspective-diverse verification** — every finding vetted by TWO lenses (a correctness lens
that live-repros, a skeptic lens that tries to refute), surviving only if BOTH agree.
**0 P0 / 0 P1 survived; 5 P2 + 1 split.** All 5 fixed (`6d39573`):

- **SOUNDNESS (the standout — safe code → UB):** public `open_with(Validate::None)` reached
  `rkyv::access_unchecked` on caller-provided bytes. Now the public entry upgrades `None → Crc`
  (checked access, no UB); `None` stays valid only on the internal self-written re-mmap.
- **MANIFEST checksum 0-sentinel collision:** a legit crc of 0 disabled its own guard. Write
  remaps computed 0 → 1.
- **`validate=Full` accepted `offsets[0] != 0`** → orphaned leading edges. Now requires
  `offsets[0] == 0` (canonical CSR base).
- **Compaction sanitized an out-of-[0,1] delta weight** while the live view returns it raw → a
  same-seq divergence. `build_graph_prenormalized` now stores weights exactly as given.
- **high_seq detector granularity documented** (its floor is the last close/compaction; an
  external truncation of only fsync-since-last-close frames isn't caught — a defense-in-depth
  bound, core durability unaffected; per-batch MANIFEST rewrites rejected as too costly).

The split finding (one lens P1, one refuted) was the same high_seq-lags class → resolved by the
documentation. 3 regression tests added. **135 lib + crash harness ×100 + 7 integration green.**

**Audit trail:** #1–#3 per-phase (clean/minor) · #4 whole-engine (3 P2) · #5 (2 P1 = regressions
in #4's fix + 1 P2) · #6 maximal (0 P0/P1, 5 P2). Severity has strictly decreased and the last
two P0/P1-free passes were the two hardest. The engine is durable (real kill -9 ×100), sound
(unsafe mmap vetted + the one safe→UB hole closed), and deterministic (same seq,ctx → same output,
now incl. out-of-contract weights across compaction).

**Deferred (out of the implemented scope, spec §4.6 / future):**
- ~~**Python API** (`open`/`apply`/`flush`/`compact` PyO3 bindings for `PersistentGraph`) — the
  store is Rust-native today; wiring it into `pybridge` + Orpheus is a distinct Phase 3.~~
  **DELIVERED — see "Phase 3 — Python API (PyO3 bindings)" below.**
- ~~**Temporal validity + edge-level ACL** (format headroom reserved; roadmap).~~
  **DELIVERED — see "Deferred features delivered" below.**
- ~~**fsync-failure per-frame commit marker** (documented limitation: `apply`-Err ≠ strictly
  not-durable under an fsync error; reconcile via `(epoch, seq)` on reopen).~~
  **DELIVERED (COMMIT sidecar) — see "Deferred features delivered" + "Honest apply()
  durability contract" below. The limitation is now precisely bounded, not removed.**
- ~~**Power-loss harness** (`dm-flakey`/VM) — the `kill -9` harness proves process-death
  durability; power-loss (page-cache loss) is the spec's separately-scoped case.~~
  **DELIVERED as a deterministic userspace fault-injection `Vfs` (`FaultVfs`), stronger than
  `dm-flakey` (no root, deterministic, CI-able) — see "Deferred features delivered" below.**

---

## §6 Benchmarks + name-index redesign (Phase 3 follow-up)

**Scope:** spec §6 (the phase-plan's `X` row — crash harness, property tests, benches —
still had no benches after Phase 3). Added `benches/bench_persist.rs` (criterion) to
measure the perf claims the persistent store was designed around: `apply(batch=100)`,
warm `open()` (mmap V1 vs the legacy V0 `from_rkyv` zero-copy view), `DeltaAccessor`
traversal overhead at delta 0/1/10%, and CSR-mmap-beam vs owned petgraph vs the legacy
V0 archived O(E) scan. Writing the benchmark surfaced two real findings, both fixed; the
second led to an on-disk format redesign.

**Files:** `benches/bench_persist.rs` (new); `src/delta.rs` (DeltaAccessor allocation
fix); `src/persist/snapshot.rs` (V1 → V2 format: persisted name index); `src/persist/mod.rs`
(V2 wiring, legacy-format rejection); `src/persist/manifest.rs` (`FORMAT_VERSION` 1 → 2);
`Cargo.toml` (bench target registration).

### Decisions (with justification)

- **`DeltaAccessor` allocation fix.** `outgoing_neighbors`/`incoming_neighbors` built a
  3-`String` tuple key (`(from, to, kind)`) per base edge to probe `removed_edges` —
  unconditionally, even when `removed_edges`/`removed_nodes` were EMPTY, which is the
  common case for an additive (no-removal) delta. Changed both to reuse the base
  `Vec<NeighborView>` in place and `retain()` it, running the tuple-key mask check only
  when the relevant tombstone set is actually non-empty (`get_node`'s shadow/tombstone
  checks got the same empty-set short-circuit). An additive delta now pays no incremental
  allocation on this path beyond the base call itself. **Measured:** beam traversal over a
  10%-delta dropped from **1.44× to 1.33×** vs the plain owned accessor (target ≤1.3×;
  1.33× lands at-target within benchmark noise).
- **Warm-open index redesign.** The bench exposed that warm `open()` was dominated by an
  eager O(N) `name -> idx` `HashMap` build performed on every open — walking every node and
  allocating a `String` per name, which faults every node into RAM. That defeats the entire
  point of the mmap base (larger-than-RAM lazy paging, §4.5): an open that touches all N
  nodes up front is no better than loading the whole file.
  - **First attempt — binary search — tried and discarded.** `nodes` is already
    name-sorted (§4.4b), so binary-searching it directly on lookup removes the O(N)
    open-time build with no format change. This fixed open, but every `idx_of` call became
    `O(log N)` string comparisons against archived bytes, and CSR-beam calls `idx_of` on
    every expansion — traversal regressed from the ≤1.5× target to **1.69× owned
    petgraph**. Trading a one-time open cost for a per-lookup cost paid on every traversal
    is the wrong trade for a hot-path-first library.
  - **Chosen: persist the index in the snapshot itself.** Moved `name -> idx` resolution
    into the on-disk format as `name_buckets`: an open-addressing hash table (FNV-1a,
    linear probing, `M` = next power of two `> N`), built once at write time (deterministic:
    fixed hash + name-sorted insertion order → byte-identical output across two
    serializations of the same graph) and read as a zero-copy, `O(1)`-expected probe over
    the mapped bytes. This is the only option that wins on BOTH axes at once: open no longer
    walks every node (`idx_of` touches only the probed bucket page + the one matching node),
    and lookup during traversal stays O(1) instead of O(log N).
- **Format bump to V2** (`SerializableGraphV2`, `FORMAT_VERSION = 2`). `name_buckets` is a
  new persisted field, so it changes the on-disk layout — a V1 reader can't interpret V2
  bytes. `open()`'s legacy branch (`0 | 1`) now returns `PersistError::Corrupt` instead of
  attempting to read/upgrade: this is pre-release with no shipped V1 stores to preserve, so
  silent migration isn't worth the complexity. The fix for a stale store is "recreate it."
  (V0 was already never-mmap'd/owned-only; V1 is now equally unsupported, not just
  downgraded.)
- **Trust-boundary addition for `name_buckets`** (`validate_csr`, extends the §5.1 sweep).
  A hostile/bit-rotted table can never cause UB — every access is `.get()`-bounded and
  probing is capped at `M` steps — but it could make lookups silently MISS an existing
  node, which `Validate::Full` must catch for untrusted/shared snapshots. Three checks, all
  required together: **(a)** `M` is a power of two and `> N` (mask arithmetic valid, and at
  least one `EMPTY_BUCKET` exists so every probe terminates); **(b)** occupied slots form a
  bijection with `0..N` (no duplicate/out-of-range index, no phantom entries); **(c)**
  reachability — probing from `fnv1a(name[i])` reaches bucket `i` before hitting an
  `EMPTY_BUCKET`, for every node `i` (implemented as an O(1)-per-node EMPTY-prefix-sum
  check of the equivalent linear-probe invariant "no EMPTY slot on the cyclic path
  `[home(i), slot(i))`" — see the verification pass below for why NOT a probe walk with a
  budget). (a)+(b) alone do not imply (c): a table can be a valid
  bijection and still strand a node behind an empty slot reachable only from a different
  probe start. A table built by `build_name_buckets` satisfies all three by construction.
  Noted in-code: FNV-1a is not collision-resistant, so adversarially-collided names are out
  of the integrity threat model — consistent with the rest of §5.1 (structural safety, not
  hash-flooding resistance).

### Measured results (50K nodes, release build)

- `apply(batch=100)`: **~27.7 µs**.
- **Warm open** (mmap V2, `Validate::Crc`): **2.15 ms** — vs the old eager-`HashMap`-build
  path **6.51 ms** (−67%; measured BEFORE that path was deleted — the committed bench can
  no longer reproduce this baseline, only the V2/V0 rows) and vs V0 `from_rkyv` **4.70 ms**
  (V2 is **~2.2×** faster than V0). `Validate::Full` open: **3.57 ms** (adds the O(N+M)
  `name_buckets` sweep on top of crc + the CSR §5.1 sweep — mandatory for untrusted
  snapshots, optional for a self-written store).
- **CSR-beam** (mmap V2): **7.47 µs = 1.24× owned petgraph** (under the ≤1.5× target); the
  legacy V0 archived view's O(E) per-expansion scan is **697 µs (~94× slower)** — confirms
  the CSR layout + persisted index together fix the §4.4b motivating defect. (The discarded
  binary-search variant measured 1.69×, for comparison.)
- Delta overhead: empty delta ≈**1.03×** (~0 overhead, as designed); 10% delta **1.33×**
  (the allocation fix above).

**Outcome:** the persisted `name_buckets` table delivers O(1) open + O(1)-expected lookup +
byte-determinism (fixed FNV-1a + name-sorted order) simultaneously — the combination the
binary-search attempt could not get in one structure. Larger-than-RAM lazy paging is
preserved for `Validate::None` opens and for post-open lookups/traversal; `Crc`/`Full`
opens crc-hash the whole file and therefore fault every page by design (stated accurately
in the in-code `Validate` docs — the win is that no O(N) index build runs on ANY open).
All four §6 targets (apply, warm-open improvement, delta-overhead ≤1.3×, CSR-beam ≤1.5×)
are met or at-target within noise.

### Verification pass (pre-commit, 2 independent adversarial reviewers)

One P1 and four P2s, all fixed before commit:

- **P1 — write/validate asymmetry on `name_buckets` (CONFIRMED with a 17-name repro).**
  `build_name_buckets` bounds nothing at write time (any probe-cluster length succeeds),
  but reachability check (c) enforced a cumulative probe budget of `8N+16` at read time.
  Names clustering into one hash home (17 same-bucket names = 153 cumulative probes > 152
  budget) produced a store that `create()`/`compact()` wrote successfully and the DEFAULT
  `open()` (`Validate::Full`) then rejected as `Corrupt` — durably-written data that
  cannot be read back, and a violation of the `Validate::Crc` soundness comment's "we only
  ever write semantically-valid V2" invariant. **Fix:** replaced the budgeted probe walk
  with an exact check of the linear-probe invariant — node `i` is reachable iff the cyclic
  path `[home(i), slot(i))` contains no `EMPTY_BUCKET` — computed in O(1) per node from an
  EMPTY-slot prefix sum built during the (b) bijection pass. This is equivalence, not
  approximation (names are unique per the strict-sort check, hits are name-confirmed), it
  accepts every table `build_name_buckets` can emit by construction (insertions land on
  the first EMPTY after home and slots are never emptied), and it hard-bounds validation
  at O(N+M) so hostile clustering cannot DoS it — no budget needed on either side.
  Regression test: `full_accepts_pathologically_clustered_names` (mines 17 FNV-colliding
  names, asserts a self-written store passes `Validate::Full`).
- **P2** — check (a) asserted only `M == bucket_capacity(N)`; the power-of-two/`> N` mask
  precondition held transitively through `bucket_capacity`'s internals. Now asserted
  independently so a future load-factor change cannot silently break the `& (M-1)` mask.
- **P2** — `bench_apply_batch` timed the 100-op batch `.clone()` inside `b.iter`; moved to
  `iter_batched` setup so the apply figure measures the durable-write path only
  (re-measured post-fix: **~27.6 µs**, statistically unchanged — the clone was noise).
- **P2** — stale-doc sweep: bench comments still describing the deleted O(N) open-time
  name-index build; `Cargo.toml`'s memmap2 rationale saying "V1"; the spec's §4.4b
  compatibility section + MANIFEST example still normatively describing V1 (amended to V2
  with a pointer here); the lazy-paging outcome overstatement above (reworded).
- Clean bill otherwise: byte-determinism (no HashMap order leaks; metadata key-sorted;
  PageRank summation order canonicalized by the builder name-sort), no wrong-node return
  possible from a hostile table (name-confirmed hits + uniqueness), legacy 0|1 rejection,
  mmap bounds-checking, delta.rs retain-masking equivalence, and zero Cyrillic across the
  diff and all branch commits were verified explicitly.

---

## Deferred features delivered (2026-08-15 → 08-17)

The three items parked in "Deferred (out of the implemented scope)" above were implemented
in one push, orchestrated per feature with an adversarial verification round each. Format is
now `format_version = 3` (the COMMIT sidecar is the version-bumping addition; the CSR byte
layout is unchanged and still named "V2"). Legacy 0/1/2 are rejected at `open()`.

### A. fsync-failure per-frame commit marker — the COMMIT sidecar (`src/persist/commit.rs`)
An authoritative durable-commit high-water mark recovery trusts INSTEAD of "whatever crc-valid
WAL frames survived". Two-slot ping-pong at fixed offsets, each slot
`magic‖version‖gen‖epoch‖committed_seq‖crc32` (44 B); a reader picks the crc-valid slot with the
highest **`gen`** (a monotonic write counter — `epoch` is a random u128 and can't order across a
re-mint; `committed_seq` ties on the open-time re-stamp). Ack protocol: WAL append → WAL fsync
(per policy) → on success advance the marker (write slot + fsync) → only then `apply()` returns
`Ok`. Recovery cap `commit_hw = max(COMMIT.committed_seq, snapshot_seq)`; crc-valid WAL frames
beyond it are the ambiguity artifact → truncated + counted in `RecoveryReport`. `durable_seq` now
moves only through the marker; MANIFEST `high_seq` consistency-checked against it. This CLOSES the
audit-#6 "MANIFEST high_seq granularity" limitation (the marker is the cheap per-fsync high-water
that comment said would be too costly). **Behavior change:** strict semantics — recovered state is
EXACTLY the committed prefix, so an `OnFlush` write-through "lucky tail" that survives a `kill -9`
is now discarded (it was never fsync-acknowledged). Its residual limit — a COMMIT-fsync that fails
but whose slot persists — is bounded, not removed; see "Honest apply() durability contract" below.
Audit: the marker's `durable_seq`/`high_seq`/`epoch`/`clean_shutdown` interactions were traced;
one design deviation (`gen` freshness key over `(epoch, committed_seq)`) with justification.

### B. Power-loss harness — a deterministic userspace fault FS (`Vfs` + `FaultVfs`)
Instead of the spec's `dm-flakey`/VM (root, nondeterministic), a `Vfs` seam over the persist
layer's write-and-durability ops (prod = `RealVfs`, a zero-cost `std::fs` passthrough; the hot read
path never touches it). The test `FaultVfs` (RocksDB `FaultInjectionTestFS` pattern) passes every op
through to the real FS while shadowing last-fsync'd state + an un-synced journal; `power_cut()`
deterministically restores the real directory to the crash image (byte-granular torn writes,
undone-un-fsync'd renames/creates/removes) and `fail_next_fsync(path, persist)` models the
fails-but-persists ambiguity. The harness (`tests/powerloss_harness.rs`) runs seeded multi-generation
cuts (`OG_POWERLOSS_ITERS`-scalable) plus targeted cases: crown-jewel (WAL fsync fails but persists →
Err'd frame invisible + counted), mid-COMMIT-slot torn, snapshot_seq-floor mid-compaction,
rename-before-`fsync_dir` undo. 1600+ randomized cuts found NO bug in the marker/recovery core.

### C. Temporal validity + edge-level ACL (query-time filtering)
Every edge representation (`EdgeInput`/`EdgeData`, delta `Op`, ephemeral `SerializableEdge`, `CsrEdge`,
`NeighborView`) gains `valid_from`/`valid_to: Option<u64>` (half-open `[from, to)`, `None` = unbounded)
and `acl: Vec<String>` (empty = public; sorted+deduped at every ingest funnel for byte-determinism).
`DynamicContext` gains `as_of: Option<u64>` (`None` = temporal filter off) and `principals: Vec<String>`.
One shared oracle `DynamicContext::is_edge_visible(valid_from, valid_to, acl)` — temporal passes iff
`as_of` is `None` or in `[from, to)`; ACL passes iff `acl` is empty or intersects `principals` (empty
principals ⇒ only public edges pass, **fail-closed**) — applied at exactly the two query-layer choke
points (`overlay::neighbors_with_overlay` feeding beam/find_path/contextual_subgraph, and
`multi_beam_intersection`'s own reconstruction). Accessors stay ctx-free; **PageRank and all base
metrics run over ALL edges** (filtering is query-time only). `to_rkyv_v2`'s edge sort key was extended
to `(…, valid_from, valid_to, acl)` so two parallel edges differing only in those fields still
serialize byte-identically. Design notes: builder SKIPS a `valid_from > valid_to` edge (matching its
unknown-endpoint precedent) while delta `apply` HARD-REJECTS the batch; raw `outgoing_edges`/
`incoming_edges` are ctx-free UNFILTERED adjacency by design (documented in `.pyi` — do not surface to
end users under an access-control assumption); `add_edge` is always a parallel edge, so tightening a
public edge needs `remove_edge` then re-add (documented in `delta.rs`). Adversarial review: no
P0/P1, ACL confirmed fail-closed across all four ops both directions; five P2 doc/hardening items.

**Follow-up noted (not yet done):** `committed_seq()`/`recovery_report()` are Rust-only; a Python
caller reconciles an `apply`-`Err` via `seq` after reopen (== `committed_seq`), but cannot read the
dropped-frame count. Exposing both to Python is a clean future addition.

---

## Honest apply() durability contract (2026-08-17, COMMIT-marker verification round)

**Scope:** a targeted verification pass over the COMMIT sidecar (`commit.rs`) + power-loss
harness added to close the fsync-failure ambiguity. It found the sidecar's docstring
OVER-CLAIMING what the marker buys, empirically reproduced the gap, and replaced the
unachievable guarantee with an honest, still-strong contract. No format change; the marker
stays.

### The bug (CONFIRMED, reproduced)

The COMMIT marker was documented as making `apply()`-`Err` mean "not durable / invisible on
reopen" ("A frame is below the marker ONLY after its fsync succeeded … above-marker == not
durable against either"). That is FALSE for one direction:

- Ack protocol (`apply`, mod.rs): WAL append → WAL fsync (per policy) → on success
  `advance_commit` (write COMMIT slot + fsync) → only then return `Ok`.
- Trace: WAL frame `N`'s fsync SUCCEEDS (frame durable), then the COMMIT slot's OWN `fsync`
  returns `Err` **while the fully-written 44-byte slot still persists** (kernel writeback).
  `advance_commit` poisons the writer and propagates `Err`, so `apply()` returns `Err` and the
  in-memory seq stays at `N-1` — BUT the on-disk marker now reads `committed_seq = N`.
- On reopen: the highest-`gen` crc-valid slot wins → `commit_hw = N` → the (durably fsync'd)
  frame `N` is REPLAYED and VISIBLE. So an `apply()`-`Err` batch is visible — violating the old
  docstring and the implied "apply-Err ⇒ invisible / recovered == exactly the committed prefix".

### Theory: the ambiguity is irreducible; the marker RELOCATES it, not removes it

fsync-fails-but-persists cannot be distinguished from fsync-success at recovery time by ANY
on-disk means, for ANY single fsync. The marker genuinely removes the WAL-frame directions
(a WAL fsync that fails, or a never-fsync'd `OnFlush` tail, does not advance the marker → its
frame stays above-marker and is discarded — the `crown_jewel` test), and it strengthens the
`Ok` side. But it simply MOVES the identical property onto the COMMIT slot's own fsync. Adding
slots + crc defends only against a TORN slot, never against a fully-written slot whose fsync
merely erred. Therefore "apply-Err ⇒ invisible" is UNACHIEVABLE and was not pursued (no WAL
reformat, no extra slots).

### Resolution: the honest contract

- **`apply()`-`Ok` ⇒ durable** — the WAL frame is fsync'd AND a marker covering it is fsync'd;
  survives process death and power loss. (Under `OnFlush` the marker advances at the next
  `flush`/`close`/compaction; until then an applied batch is visible but `> committed_seq()`.)
  Kept as a strong assertion.
- **`apply()`-`Err` ⇒ durability INDETERMINATE** — the batch may be visible on reopen (this
  COMMIT-fsync-persist path) or invisible (the WAL-fsync-fails path). What recovery ALWAYS
  guarantees: (1) the store opens to a CONSISTENT committed prefix (batches are all-or-nothing —
  no torn/half-applied batch), and (2) no `Ok`-acked batch is ever lost. The caller reconciles
  an `Err` by reading `committed_seq()` / `recovery_report()` after reopen, NOT by blind retry
  (re-applying a non-idempotent batch on the persist path double-applies).

Docs corrected to state this exactly: the `commit.rs` module docstring (WAL directions still
fixed, COMMIT-slot direction now called out as irreducible; "below-marker" == "acked durable on
the Ok path", not "every fsync literally succeeded"), the Rust `PersistentGraph::apply` doc,
the Python `apply` docstring (`pybridge.rs` + `orpheusgraph.pyi` — Python reconciles by reopening
and reading `seq`, its committed-prefix equivalent).

### New test + oracle tightening (the gap that hid this)

- **New targeted test** `commit_fsync_persist_makes_apply_err_indeterminate_but_prefix_consistent`
  (powerloss harness): arms `fail_next_fsync("COMMIT", persist=true)` on the seq-3 apply, asserts
  `apply` Errs + the writer poisons, then drops-without-close and reopens. It asserts the
  ACHIEVABLE invariant — NOT invisibility: (a) open succeeds (not `Corrupt`); (b) `committed_seq()`
  is EXACTLY the pre-batch seq (2) OR the batch seq (3), nothing else; (c) the recovered content
  equals the materialization of exactly that prefix (root + one node/edge per committed batch, by
  exact `node_count`/`edge_count` + per-batch presence — no torn edge); (d) both acked batches
  survive. Under drop-all it deterministically takes the VISIBLE direction (marker persisted →
  `commit_hw = 3`, r = 3).
- **Oracle tightening.** The multi-generation `verify()` was strengthened: an explicit
  consistent-prefix materialization check (recovered content == prefix-`r` materialization,
  set-for-set via presence-of-all + exact counts) and an explicit, independent NO-ACKED-LOSS
  loop (every `seq <= max_committed` present and fully materialized — `max_committed` = the
  running max of `committed_seq()`, the correct "acked & durable" high-water since an `OnFlush`
  `apply`-Ok is not yet durable). No existing assertion weakened.
- **Fault mix.** The COMMIT-fsync-persist injection (persist=true on the COMMIT file) was ADDED
  to the multi-generation loop — the previously-unexercised visible-after-Err direction (the
  actual gap that hid the bug). Under the strengthened, correct oracle it PASSES (consistent
  prefix). Determinism preserved (fixed seeds, `OG_POWERLOSS_ITERS`-scalable, seed printed on
  failure). Verified: harness green at `OG_POWERLOSS_ITERS=200`.
- **Teeth (mutation-verified, reverted).** (A) Replaying one frame PAST the marker
  (`rec.seq > commit_hw` → `> commit_hw + 1`) — a leaked phantom — the multi-gen oracle catches
  it (`committed_seq != seq`). (B) Dropping the frame AT the marker (`>` → `>= commit_hw`) — a
  hole / acked-loss — both the new targeted test and the harness catch it (`recovered seq < commit
  marker`). Confirms the strengthened oracle fails a real prefix-consistency regression.

### NOTE — deferred simplification (option b, not a correctness necessity)

Fold the commit high-water INTO the WAL frame (a single fsync domain): stamp the committed-seq
high-water into the WAL frame itself instead of a separate fsync'd sidecar. This collapses the
TWO ambiguity windows (WAL-frame fsync + COMMIT-slot fsync) into ONE and halves the `EveryBatch`
fsync count, delivering the SAME achievable contract (Ok ⇒ durable / Err ⇒ indeterminate /
consistent prefix + no acked loss) with less surface area. Deferred as a future simplification —
the current two-fsync marker is already correct under the honest contract, so this is an
optimization, not a fix.
