# Incremental Mutability & Persistence — Design Spec

> Status: **DRAFT v1** (2026-07-23) · Target: orpheusgraph v0.2.x
> Scope: Phase 1 (in-memory delta layer) + Phase 2 (WAL + snapshot persistence, mmap)

---

## 1. Motivation

Today the graph is a **derived cache artifact**: it is materialized from a source
DB, serialized to Redis, and any change — however small — kills the whole graph
and triggers a full rebuild (`build_graph`: ~48 ms for 50K nodes, plus the source
DB extraction which dominates). Its lifecycle is governed by TTL and
invalidation, which forces cache-engineering machinery on every consumer
(stampede locks, generation counters, cold-start penalties).

This spec inverts the lifecycle: the graph becomes a **long-lived embedded
store** that owns its state. A process opens it once, mutations apply in
microseconds through a delta layer, durability comes from a write-ahead log +
snapshots, and requests borrow consistent views. The per-request
`DynamicContext` is unchanged — it remains the ephemeral *view* mechanism
(boosts, overlays, filters), strictly separated from durable *state*.

```
                 durable state                    per-request view
        ┌──────────────────────────────┐   ┌───────────────────────────┐
        │  base (immutable, archived)  │   │  DynamicContext           │
        │  + GraphDelta (mutable)      │ + │  boosts / overlay / noise │
        │  + WAL + snapshots on disk   │   │  ephemeral, never stored  │
        └──────────────────────────────┘   └───────────────────────────┘
                 apply() writes                  ctx per traversal call
```

### Goals

- `apply()` mutations without full rebuild; base graph stays immutable.
- Crash-safe on-disk persistence; warm start via zero-copy snapshot load.
- Single-writer / multi-reader concurrency, lock-free reads.
- **Zero changes to traversal code** — reuse the `GraphAccessor` seam.
- Deterministic results preserved: same (state seq, ctx) → same output.
- Snapshot format versioned from day one so later features (temporal validity,
  edge metadata, ACL tags) do not break compatibility.

### Non-goals (this iteration)

- Multi-writer, cross-process writers, distributed anything.
- Query language, server protocol (a facade can wrap the store later).
- Temporal validity / edge ACL themselves — only format headroom for them.
- Incremental PageRank (recomputed at compaction only; see §4.5).

---

## 2. Architecture Overview

The design reuses the existing seam: `beam_traverse`, `find_path` and
`contextual_subgraph` already access the graph exclusively through the
`GraphAccessor` trait, and `neighbors_with_overlay` already merges a base with
virtual entries. A persistent delta is "an overlay owned by the graph instead
of the request".

```
   ┌────────────────────────────  PersistentGraph  ───────────────────────────┐
   │                                                                          │
   │  ArcSwap<GraphState>          GraphState {                               │
   │        │                          base:  Arc<dyn GraphAccessor>,  ← owned│
   │        │  (atomic swap on             or ArchivedGraph over mmap         │
   │        │   apply/compact)         delta: Arc<GraphDelta>,                │
   │        ▼                          seq:   u64,                            │
   │  readers grab Arc —           }                                          │
   │  consistent for whole call                                               │
   │                                                                          │
   │  Writer (single, &mut self):                                             │
   │    apply(batch) ──→ WAL append ──→ new GraphState published              │
   │    compact()    ──→ materialize base+delta → build_graph → snapshot      │
   └──────────────────────────────────────────────────────────────────────────┘
```

Traversal receives a `DeltaAccessor` implementing `GraphAccessor`; no traversal
file changes.

---

## 3. Phase 1 — In-Memory Delta Layer

### 3.1 Data structures

```rust
/// Mutable overlay owned by the graph. Insertion-ordered for determinism.
pub struct GraphDelta {
    /// Added or upserted nodes. IndexMap: name → NodeData (insertion order).
    added_nodes: IndexMap<String, NodeData>,
    /// Added edges, in insertion order.
    added_edges: Vec<(String, String, EdgeData)>,
    /// Node tombstones. Masks the node AND all incident base edges.
    removed_nodes: HashSet<String>,
    /// Edge tombstones, keyed by (from, to, kind).
    removed_edges: HashSet<(String, String, String)>,
}
```

Rationale for `IndexMap`/`Vec` over `HashMap`: delta iteration order must be
deterministic so that WAL replay and repeated traversals produce identical
results (determinism is a headline library guarantee).

### 3.2 `DeltaAccessor`

```rust
pub struct DeltaAccessor<'a> {
    base: &'a dyn GraphAccessor,
    delta: &'a GraphDelta,
}

impl GraphAccessor for DeltaAccessor<'_> { ... }
```

Resolution rules:

| Query | Rule |
|---|---|
| `get_node(name)` | tombstoned → `None`; else delta node; else base node |
| `outgoing_neighbors(n)` | base edges, minus edges whose target is tombstoned, minus tombstoned edges; **then** delta edges from `n` (skipping tombstoned targets), in insertion order |
| `incoming_neighbors(n)` | symmetric |
| `node_count()` | `base − |removed∩base| + |added∖base|` (maintained counters, O(1)) |
| `edge_count()` | same approach |

Delta edge lookup must not be a linear scan of `added_edges` on the hot path:
maintain two side indices `out_index: HashMap<String, Vec<u32>>` and
`in_index: HashMap<String, Vec<u32>>` (values = positions in `added_edges`,
preserving insertion order). Built incrementally on apply; O(1) amortized.

### 3.3 Mutation API and semantics

Operations (also the WAL op enum, see §4.2):

```rust
enum Op {
    UpsertNode(NodeData),          // add OR replace; last-write-wins
    RemoveNode { name: String },
    AddEdge { from: String, to: String, edge: EdgeData },
    RemoveEdge { from: String, to: String, kind: String },
}
```

Locked-down semantics (decided now, so they don't drift later):

1. **Upsert, last-write-wins.** Adding an existing node replaces its data.
   Makes WAL replay trivially idempotent and the API forgiving. Replacing a
   *base* node = tombstone-in-base + entry in `added_nodes` (one op).
2. **`RemoveNode` masks the node and all incident base edges.** Incident
   *delta* edges are physically dropped from `added_edges` indices (cheap —
   they are in the side indices).
3. **State transitions are symmetric.** `UpsertNode` after `RemoveNode` clears
   the tombstone; the node re-appears with the new data. Same for edges.
4. **`AddEdge` with an unknown endpoint is an error**, not a silent skip.
   Batch semantics: the whole batch is validated first, then applied —
   all-or-nothing (see §4.2 batch framing). No partial application.
5. **Weight semantics unchanged:** delta nodes carry `base_weight` /
   `noise_penalty` exactly like build-time nodes; normalization is NOT applied
   to delta values (documented: caller provides values in [0,1]).

### 3.4 PageRank staleness

`pagerank_weight` for delta nodes defaults to `0.0` and is only recomputed at
compaction (which calls `build_graph`). Documented consequences:

- A fresh delta node never gets the high-pagerank `max_fan_out` bypass.
- God-object detection lags until compaction.

This is acceptable staleness: PageRank is a *modifier*, not a correctness
input. If a caller needs the bypass immediately, `semantic_boosts` already
provides one (existing behavior).

### 3.5 Concurrency

```rust
struct GraphState {
    base: Arc<BaseGraph>,      // enum: Owned(OrpheusGraphInner) | Archived(Mmap)
    delta: Arc<GraphDelta>,
    seq: u64,
}
// publisher
state: ArcSwap<GraphState>
```

- **Readers**: `let s = state.load_full();` — one atomic load per call; the
  whole traversal runs against that immutable pair. Lock-free, no contention.
- **Writer**: single (`&mut self` in Rust; a `Mutex` guard on the Python side
  since PyO3 objects are shared). `apply()` clones the current delta
  (copy-on-write of small structures), applies the batch, publishes a new
  `GraphState` with `seq + 1`.
- Cost: delta clone is O(delta size). Fine because compaction bounds delta
  size (§4.4). If profiling ever shows otherwise, swap `IndexMap` for `im`-rs
  persistent maps — API unchanged.

---

## 4. Phase 2 — Durability (WAL + Snapshot + mmap)

### 4.1 On-disk layout

```
graph_dir/
  LOCK                   # flock'd for the writer's lifetime (see below)
  MANIFEST.json          # current state pointer (atomic rename to update)
  snapshot-{seq:020}.og  # rkyv bytes of SerializableGraph (existing format)
  wal.log                # ops after snapshot seq
```

**Single-writer enforcement.** `open()` takes an exclusive `flock` on `LOCK`
and holds it until `close()`/process death. A second writer opening the same
dir fails fast with a clear error instead of silently interleaving WAL frames
(guaranteed corruption). The lock is advisory-OS-level, so it also survives
crashed processes (kernel releases flock on death — no stale-lockfile
recovery dance).

**Rename durability.** Every tmp+fsync+rename sequence (snapshot, MANIFEST)
is followed by `fsync(graph_dir)` — on ext4 and friends the rename itself
lives in the directory's page cache until the directory is fsynced; skipping
this is the classic "MANIFEST points at a snapshot that vanished after power
loss" bug.

`MANIFEST.json`:

```json
{
  "format_version": 1,
  "snapshot_file": "snapshot-00000000000000000042.og",
  "snapshot_seq": 42,
  "snapshot_crc32": 3735928559,
  "created_by": "orpheusgraph 0.2.0"
}
```

`format_version` is checked on open; unknown major version → hard error with a
clear message. **Headroom rule:** future additions (edge `metadata`,
`valid_from`/`valid_to`, ACL tags) bump the snapshot struct behind a version
gate; WAL op enum is `#[non_exhaustive]`-style — unknown op tag on replay of a
*newer* version → hard error (never silent skip).

### 4.2 WAL format

Binary framing, one frame per `apply()` **batch** (postcard-encoded):

```
frame := [len: u32 LE] [crc32: u32 LE] [payload]
payload := postcard(WalRecord { seq: u64, ops: Vec<Op> })
```

- **Batch = atomic unit.** A schema diff arrives as a set of ops; framing the
  batch as one record makes replay all-or-nothing and matches §3.3 rule 4.
- `crc32` covers `payload`. `seq` strictly increases by 1 per frame.
- **Write vs fsync semantics.** Every `apply()` writes its frame through to the
  OS (`write()`; any userspace buffer is flushed per batch — no frame ever
  lives only in process memory). Consequently a *process* crash (kill -9,
  panic, OOM) loses nothing under any policy — the OS page cache survives and
  is written back by the kernel. Only a *machine* failure (power loss, kernel
  panic) can lose the tail written after the last fsync.
- fsync policy (config): `EveryBatch` | `EveryN(n)` | `OnFlush` (default).
  Under `OnFlush`, fsync happens on explicit `flush()` and always on
  `close()`; compaction durability is independent (tmp+fsync+rename, §4.4).
  Reads never touch the WAL — the policy concerns mutations only.
  Rationale for the default: the primary workload is schema/knowledge sync,
  not payments — losing the post-flush tail to a power cut means re-sync, not
  data loss. Callers with stricter needs opt into `EveryBatch` and pay fsync
  latency per batch. A Redis-AOF-style `Interval(duration)` policy is a
  possible later addition (needs a background thread; out of scope v1 —
  flushing at job boundaries is the idiomatic embedded pattern).

### 4.3 Recovery (open)

```
open(dir):
  1. read MANIFEST.json; verify format_version
  2. load snapshot:
       mmap file → verify crc32 → rkyv::access (zero-copy ArchivedGraph)
       + build name→index side map (existing from_bytes path, ~5.6 ms @50K)
  3. replay wal.log frames where frame.seq > snapshot_seq:
       - crc mismatch or truncated frame → torn tail from a crash:
         TRUNCATE wal at that offset, log a WARNING with dropped-frame count
         (never silent), stop replay
       - seq gap or seq ≤ snapshot_seq → corruption → hard error (do not guess)
  4. publish GraphState { base: Archived(mmap), delta: replayed, seq: last }
  5. GC: delete snapshot-*.og not referenced by MANIFEST (orphans from
     interrupted compactions), log what was removed
```

`open()` on an empty/missing dir creates it and starts with an empty graph at
`seq = 0` (explicit `create: bool` flag to catch typo'd paths — opening a
nonexistent dir without `create=True` is an error, not a silent empty graph).

### 4.4 Compaction

Trigger: manual `compact()` or automatic when
`delta_ops > max(1000, 10% of base node count)` (config).

```
compact():
  1. materialize: iterate base+delta through DeltaAccessor →
     Vec<NodeInput>, Vec<EdgeInput>
  2. build_graph(...)            # recomputes normalization + PageRank (~48 ms)
  3. to_rkyv → write snapshot-{new_seq}.og.tmp → fsync → rename
  4. write MANIFEST.json.tmp (pointing at new snapshot) → fsync → rename
  5. rotate WAL: truncate wal.log (frames ≤ new_seq are now in the snapshot)
  6. publish GraphState { base: new, delta: empty, seq: new_seq }
```

Crash-safety by rename atomicity:

- Crash after 3, before 4 → MANIFEST still points at the old snapshot; the new
  file is an orphan → removed by GC on next open. State intact.
- Crash after 4, before 5 → new snapshot is live; WAL frames with
  `seq ≤ snapshot_seq` are skipped by the replay filter. State intact.

Compaction runs on the writer; readers keep serving the old `GraphState`
until step 6 swaps atomically. No stop-the-world.

### 4.5 mmap and larger-than-RAM

Step 2 of recovery deliberately keeps the base as an **archived view over
mmap** rather than deserializing to owned structures. Consequences:

- Warm start is O(name-index build), not O(graph bytes) — the OS pages data
  in on demand.
- Cold regions of the graph never occupy RAM; beam search locality (walks stay
  near seed nodes) means the resident set ≈ the hot neighborhood, not the
  whole graph. This is the "larger-than-RAM graphs via OS paging" capability.
- Requires `ArchivedGraph` to implement `GraphAccessor` — **already exists**
  (accessor.rs supports owned + archived). The only new code is the mmap open
  path and crc check.
- Platform note: file must not be truncated/replaced in place while mapped —
  guaranteed by the rename-only discipline above (the old mapping keeps the
  old inode alive until dropped).

### 4.6 Python API

```python
import orpheusgraph

g = orpheusgraph.open("./graph_dir", create=True)   # PersistentGraph

g.apply([
    {"op": "upsert_node", "name": "x_membership", "kind": "model",
     "base_weight": 0.6, "noise_penalty": 0.0, "metadata": {}},
    {"op": "add_edge", "from": "x_membership", "to": "res.partner",
     "kind": "relates_to", "field_name": "partner_id", "base_weight": 0.8},
    {"op": "remove_edge", "from": "sale.order", "to": "legacy.model",
     "kind": "relates_to"},
])            # one WAL frame; all-or-nothing; raises on invalid op

g.flush()     # fsync WAL (no-op under EveryBatch policy)
g.compact()   # manual compaction; also happens automatically by threshold
g.seq         # current state sequence number (replaces cache "generation")

# Read API — unchanged, same as OrpheusGraph:
g.beam_traverse("sale.order", k=5, depth=3, ctx=ctx)
g.find_path(...); g.contextual_subgraph(...)

g.close()     # flush + release mmap; deterministic like today
```

Compatibility: `build_graph` / `to_rkyv` / `from_rkyv` remain untouched — the
ephemeral in-memory workflow (current Orpheus L1/L2 path) keeps working. The
persistent store is additive.

---

## 5. Failure-Mode Pass (7 categories)

1. **Security** — no shell/SQL surface. Path handling: `open()` canonicalizes
   `graph_dir` and refuses paths containing symlinked parents outside the
   provided root? — no; that is caller policy. In-scope: never `format!` paths
   from node names (snapshot names are seq-derived only); node names are
   opaque bytes in postcard, no injection surface.
2. **Idempotency** — WAL replay after any crash re-applies whole batches;
   upsert + tombstone semantics make re-application of the same batch a no-op.
   Torn tail is truncated once, with a warning; a second open is a no-op.
3. **Atomicity** — batch = one frame (all-or-nothing on disk); snapshot and
   MANIFEST updates via tmp-fsync-rename; interrupted compaction leaves the
   previous state fully valid (§4.4). `apply()` validates the entire batch
   before touching the delta — a mid-batch validation error mutates nothing.
4. **State transitions** — remove→re-add symmetric for nodes and edges
   (tombstone cleared); compact→open round-trips to an identical logical
   graph (property test, §6). `close()` after `flush()` is a no-op sequence.
5. **Deterministic ordering** — `IndexMap`/`Vec` for delta, side indices
   preserve insertion order, WAL replay order == apply order; therefore
   (seq, ctx) fully determines traversal output. Explicit test: shuffle
   HashMap-iteration-dependent code must not exist in delta paths.
6. **Silent success** — banned outcomes: skipped WAL frame without warning,
   `add_edge` to a missing node silently dropped, crc-mismatch snapshot
   silently rebuilt. All are either hard errors or logged warnings with
   counts. `apply()` returns the new `seq`; errors raise, never return
   status dicts.
7. **Caller contract** — corruption (seq gap, bad snapshot crc, unknown
   format_version) → **raise**, never auto-heal by discarding data; the one
   sanctioned auto-heal is torn-WAL-tail truncation (expected crash artifact),
   and it is logged. `create=False` + missing dir → raise (no silent empty
   graph over a typo'd path).

---

## 6. Testing Plan

- **Unit**: delta resolution rules table (§3.2) — each row a test; tombstone
  masking of incident edges; upsert-after-remove; batch validation rejects
  mid-batch without side effects.
- **Property (proptest)**: for random op sequences —
  `traverse(base+delta) == traverse(build_graph(materialized))` (delta layer
  is semantically invisible); `open(dir)` after `apply*` round-trips; compact
  preserves the logical graph.
- **Crash harness** (the load-bearing one): a child process applies batches
  in a loop with per-batch markers; the parent `kill -9`s it at random
  intervals, reopens the dir, and asserts: (a) open succeeds, (b) state equals
  a prefix of applied batches, (c) **nothing is missing under any fsync
  policy** — process death must not lose acknowledged batches (§4.2: frames
  are written through to the OS per batch). Run ×100 in CI (release build,
  tmpfs). Power-loss semantics (losing the OS page cache) are NOT covered by
  kill -9; validating them needs `dm-flakey` or a VM harness — documented as
  out of scope for v1, revisit if a durability-critical consumer appears.
- **Concurrency**: N reader threads traversing in a loop while the writer
  applies + compacts; assert no torn reads (every traversal sees a valid
  `seq`) under `cargo test` + a `loom`-lite smoke or `ThreadSanitizer` job.
- **Bench additions**: `apply(batch of 100)`, `open()` warm (mmap) vs current
  `from_rkyv`, traversal overhead of DeltaAccessor vs plain accessor at delta
  sizes 0 / 1% / 10% of base (target: ≤ 1.3× at 10%).

---

## 7. Rollout

| Phase | Deliverable | Est. |
|---|---|---|
| 1 | `GraphDelta` + `DeltaAccessor` + mutation API + unit/property tests | ~1 week |
| 2a | WAL append/replay + MANIFEST + recovery + crash harness | ~1 week |
| 2b | Compaction + auto-threshold + mmap base + GC + benches | ~1 week |
| — | Docs: README section, CHANGELOG, migration note (none needed — additive) | 1–2 days |

Out of scope, unlocked next (format headroom already reserved): temporal
validity on edges (`valid_from`/`valid_to` + `as_of` in ctx), edge-level ACL
tags filtered via ctx, SQLite-style cross-process readers (mmap snapshot +
WAL tail follow), thin gRPC/MCP facade.

**Cheap unlock — WAL retention (event-sourced history).** The WAL already IS
a full event-sourced history of the graph; compaction merely destroys it by
truncating. A `retain_wal: bool` config makes compaction rotate frames into
`wal-archive/wal-{seq}.log` instead of deleting them, which enables — as pure
readers of the existing format, no write-path changes:

- `history(node_name)` — every op that ever touched a node (audit trail;
  actor/reason can ride in batch-level metadata);
- `open_at(seq)` — time travel: reconstruct the graph exactly as it was.

Note the bi-temporal distinction: WAL retention answers *system time* ("when
was the graph changed"), temporal validity answers *valid time* ("when was
the fact true in the modeled world"). They are independent, compatible
features.

**Explicitly rejected: interactive transactions.** Atomic multi-op batches +
snapshot-isolated readers (ArcSwap) already cover the useful subset of ACID
for this workload. Interactive `BEGIN → read → decide → write → COMMIT` with
write-conflict detection only pays off with multiple writers reading inside
transactions — that is ledger territory (a different system), and the added
machinery would reduce reliability, not add it. Reliability strategy here is
SQLite's: few hard invariants (batch atomicity, monotonic seq, rename+dir
fsync, flock) beaten by a brutal test harness — not more mechanisms.
