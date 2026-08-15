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
| `get_node(name)` | in `added_nodes` → that (a *shadow* wins over base, see §3.3 rule 1); else tombstoned → `None`; else base node |
| `outgoing_neighbors(n)` | **if `n` is tombstoned → empty** (a removed node exposes no edges in either direction); else: base edges, minus edges whose target is tombstoned, minus tombstoned edges; **then** delta edges from `n` (skipping tombstoned targets), in insertion order |
| `incoming_neighbors(n)` | symmetric — **empty if `n` is tombstoned**, else base in-edges minus edges whose *source* is tombstoned, minus tombstoned edges, then delta in-edges |
| `node_count()` | `base − |tombstones∩base| + |added_nodes keys ∉ base|` (maintained counters, O(1)). A shadow — an `added_nodes` key that IS in base — changes neither term. |
| `edge_count()` | same approach |

**Shadow vs tombstone — the load-bearing distinction.** A base node has two
independent delta states: *shadowed* (data replaced, key in `added_nodes`,
base edges intact, count unchanged) and *tombstoned* (masked, key in
`removed_nodes`, base edges masked, count −1). They must never both be set
for the same name; the invariant is `added_nodes.keys() ∩ removed_nodes = ∅`
(every mutation that sets one clears the other). This is what makes a
metadata-only upsert of a hub node non-destructive.

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

1. **Upsert = shadow, last-write-wins.** `UpsertNode` puts the node into
   `added_nodes` (replacing any prior entry) **and clears any tombstone** for
   that name. It never sets a tombstone. Upserting a *base* node therefore
   shadows it — data replaced, base edges kept, count unchanged (§3.2). This
   is the fix for the earlier "tombstone-in-base" formulation, which
   contradicted the tombstone-masks-edges rule and silently disconnected hub
   nodes on a metadata-only edit.
2. **`RemoveNode` masks the node and all incident base edges.** Sets the
   tombstone **and removes any `added_nodes` shadow** for that name (upholding
   the disjointness invariant). Incident *delta* edges are marked dead in place,
   **not physically removed** from the `added_edges` Vec: `added_edges` entries
   are append-only and carry a `dead: bool` flag, so the `u32` positions cached
   in `out_index`/`in_index` stay valid (a `Vec::remove`/`swap_remove` would
   shift other entries' positions and silently corrupt the side indices).
   Dead entries are skipped on read and physically reclaimed only at compaction.
   `RemoveEdge` on a delta edge sets the same `dead` flag.
3. **State transitions are symmetric.** `UpsertNode` after `RemoveNode` clears
   the tombstone and re-shadows; `RemoveNode` after `UpsertNode` drops the
   shadow and tombstones. Same for edges. No name is ever both shadowed and
   tombstoned.
4. **Batch validation runs against sequentially-simulated intra-batch state**,
   then the whole batch applies all-or-nothing (§4.2 framing). Precisely:
   - Validation folds ops left-to-right over a scratch view of the current
     state, so `[UpsertNode(x), AddEdge(x→base)]` in one batch is **valid** —
     `x` exists by the time the edge is checked (this is the canonical §4.6
     ingestion pattern; validating against pre-batch state would wrongly
     reject it).
   - `AddEdge` whose endpoint, in that simulated state, does not exist **or is
     tombstoned** is a hard error — never a silent skip (bans the
     `[RemoveNode(A), AddEdge(B→A)]` → hidden-edge outcome).
   - Op order within a batch is caller-significant and preserved in the WAL
     frame. **Replay does NOT re-run validation.** A frame in the WAL was
     already accepted at original `apply()` time; recovery replays its ops
     directly (structural/crc checks only), applying them in order. This is
     deliberate: validation *semantics* are not frozen by `format_version`
     (only the op-tag enum is, §4.1), so a newer version that tightens a
     validation rule must never reject a batch that a prior version legally
     committed and made durable — re-validating on replay would brick a valid
     store after an upgrade. Validity is decided once, at apply; durability is
     permanent.
   - A batch that references the same name in incompatible ways
     (`[UpsertNode(A), RemoveNode(A)]`) is applied in order — final state is
     the last op — and is legal; the disjointness invariant holds because
     each op maintains it.
5. **Weight semantics unchanged:** delta nodes carry `base_weight` /
   `noise_penalty` exactly like build-time nodes; normalization is NOT applied
   to delta values (documented: caller provides values in [0,1]).
6. **Optimistic CAS for read-modify-write.** `apply(ops, expected_seq=None)`:
   when `expected_seq` is given, the writer — atomically with the apply, under
   the same writer lock — rejects the batch with `ConflictError` if
   `current_seq != expected_seq`. This closes the read→apply race for callers
   whose writes depend on reads (two callers read seq=N, both compute, second
   apply is based on a stale read — the writer mutex serializes applies but
   not read→write windows). On conflict the caller re-reads and retries.
   Granularity is deliberately whole-graph (any concurrent write conflicts):
   false conflicts cost a µs-scale retry, which is fine at target write
   rates; per-node versioning is the known next step if contention ever
   demands it, and is out of scope v1. Blind ingestion (write does not depend
   on a read) omits `expected_seq` and keeps last-write-wins semantics.

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
  "format_version": 2,
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
- `crc32` covers **`len` concatenated with `payload`** (not payload alone): a
  torn write in the 4-byte `len` prefix (a realistic power-loss artifact)
  otherwise yields an arbitrary length up to 4 GiB. `seq` strictly increases
  by 1 per frame.
- **Never allocate from an unvalidated `len`.** Recovery reads `len`, checks
  `len ≤ bytes_remaining_in_file` FIRST, then reads exactly that many bytes and
  verifies the crc over `len‖payload`. A `len` larger than the remaining file
  is treated as a torn tail (truncate, §4.3) — recovery must not
  `Vec::with_capacity(len)` before the bounds check, or a torn `len` becomes
  the same uncatchable allocation abort fixed in traversal.
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
- **Append-failure → writer poisoning (no mid-WAL torn frames).** Any append
  error — short write (ENOSPC/EINTR partial), `write()`/`fsync()` failure —
  transitions the writer to a **poisoned, read-only** state: the current
  `apply()` raises, and every subsequent `apply()` raises `PoisonedError`
  until the store is reopened. Reads keep serving the last published state.
  This is what guarantees the §4.3 assumption that a torn/partial frame can
  only ever be the **last** frame in the WAL: nothing is appended after a
  failed write, so recovery's tail-truncation can never discard an
  acknowledged frame in the middle. On the next `open()`, recovery truncates
  the WAL to the last complete, crc-valid frame boundary before any replay.
- **seq is the durable commit counter, not an in-memory one.** The next seq is
  derived from the last durably-appended frame, and the in-memory
  `GraphState.seq` is advanced **only after** the frame is written (and
  fsynced under `EveryBatch`). Therefore a panic between append and the
  ArcSwap publish cannot re-mint a seq: the frame owns seq N+1 durably, and
  the process either sees it via the failed op's own bookkeeping or
  reconstructs it on the next `open()`. Append is the commit point; publish is
  a best-effort in-memory reflection of an already-committed fact.

### 4.3 Recovery (open)

```
open(dir):
  1. read MANIFEST.json; verify format_version
  2. load snapshot:
       mmap file → verify crc32 → rkyv::access (zero-copy ArchivedGraph)
       + build name→index side map (existing from_bytes path, ~5.6 ms @50K)
  0. read a `clean_shutdown` flag from MANIFEST (set only by close(), cleared
     on the first apply after open) — its absence means a crash/power-loss
  1. read MANIFEST.json; verify format_version
  2. load snapshot (as above)
  3. scan wal.log frames in order:
       - frame.seq ≤ snapshot_seq → **already folded into the snapshot; skip
         it** (this is the *normal* leftover from a compaction that renamed the
         MANIFEST before truncating the WAL — §4.4 crash window, expressly
         recoverable). NOT an error.
       - frame.seq == expected_next (== last_applied + 1, starting at
         snapshot_seq+1) → replay its ops directly (no re-validation, §3.3
         rule 4)
       - crc mismatch or truncated/over-long frame → torn tail from a crash
         (guaranteed to be the LAST surviving frame by writer poisoning, §4.2):
         TRUNCATE wal at that boundary, log a WARNING with dropped-frame count,
         stop replay
       - a real seq GAP (expected_next < frame.seq, i.e. a frame is missing
         before a later one) → corruption → hard error (do not guess)
  4. mint a fresh random `epoch` and rewrite MANIFEST **if EITHER** the WAL tail
     was truncated in step 3 **OR** `clean_shutdown` was absent in step 0
     (a clean power-loss can drop whole un-fsynced frames that ended on a valid
     boundary — no torn frame to detect — yet still forked the timeline; the
     shutdown flag is what catches that case)
  5. publish GraphState { base: Archived(mmap), delta: replayed, seq: last,
     epoch }; clear the persisted `clean_shutdown` flag
  6. GC: delete snapshot-*.og not referenced by MANIFEST (orphans from
     interrupted compactions), log what was removed
```

**seq is unique only within an `epoch`.** Under `OnFlush`, a power-loss
rewind means new applies re-issue seq numbers that previously named different
states — an ABA trap for anyone persisting seq (external cache keys, derived
artifact versions, `open_at(seq)` over retained WAL archives). Fix: MANIFEST
carries a random `epoch` id, re-minted on every recovery that could have
forked the timeline — i.e. a truncated WAL tail **or** a missing
`clean_shutdown` flag (step 0/4 above; the latter catches clean un-fsynced
tail loss where no torn frame exists to detect). The durable
identity of a state is the pair `(epoch, seq)`. In-process CAS
(`expected_seq`, §3.3) stays raw-seq — it is only valid within one
incarnation, and a reopen invalidates in-flight CAS tokens by definition.
**Rule:** never persist a bare seq across process boundaries; persist
`(epoch, seq)`.

`open()` on an empty/missing dir creates it and starts with an empty graph at
`seq = 0` and a fresh `epoch` (explicit `create: bool` flag to catch typo'd
paths — opening a nonexistent dir without `create=True` is an error, not a
silent empty graph).

### 4.4 Compaction

Trigger: manual `compact()` or automatic when
`delta_ops > max(1000, 10% of base node count)` (config).

```
compact():
  0. if the delta is empty AND the base is already an on-disk snapshot at the
     current seq → NO-OP, return. (Prevents new_seq == old_seq path collision:
     an empty-delta compaction would otherwise fold to the same seq, and step 3
     would rename over the live snapshot while step 7 deletes it — bricking the
     store. Compaction only runs when there is delta to fold.)
  1. materialize: iterate base+delta through DeltaAccessor →
     Vec<NodeInput>, Vec<EdgeInput>
  2. build_graph(...)            # recomputes normalization + PageRank (~48 ms)
  3. to_rkyv (CSR layout, §4.4b) → write snapshot-{new_seq}-{cid}.og.tmp
     → fsync → rename → fsync(dir)
  4. write MANIFEST.json.tmp (pointing at new snapshot) → fsync → rename
     → fsync(dir)
  5. rotate WAL: truncate wal.log (frames ≤ new_seq are now in the snapshot)
  6. publish GraphState { base: new, delta: empty, seq: new_seq }
  7. delete the previous snapshot (only if its path ≠ the new snapshot path —
     defensive against any future same-seq case; see GC note below)
```

The snapshot filename carries a monotonic **compaction id `cid`** in addition
to `seq` (`snapshot-{seq}-{cid}.og`), so two snapshots never collide on a path
even in the (now guarded) equal-seq case, and step 7's delete can always name
a distinct file.

Crash-safety by rename atomicity:

- Crash after 3, before 4 → MANIFEST still points at the old snapshot; the new
  file is an orphan → removed by GC on next open. State intact.
- Crash after 4, before 5 → new snapshot is live; WAL frames with
  `seq ≤ snapshot_seq` are skipped by the replay filter. State intact.
- Crash after 6, before 7 → old snapshot leaks as an orphan → GC on next open.

Compaction runs on the writer; readers keep serving the old `GraphState`
until step 6 swaps atomically. No stop-the-world.

**Snapshot GC — not only at open().** Step 7 deletes the superseded snapshot
inline, because the flagship workload is a process that `open()`s once and
runs for months auto-compacting: relying on open()-time GC alone would leak
one multi-MB snapshot per compaction until the disk fills (→ the next WAL
append short-writes → writer poisons). Deleting a still-mmap'd old snapshot is
safe on **Linux**: `unlink` keeps the inode alive for existing readers until
they drop the mapping (same argument as the rename discipline in §4.5). On
**Windows** deleting a mapped file fails — there, step 7 defers to a
retry-on-open sweep, and this platform limit is called out rather than left
implicit. open()-time GC remains as the backstop for orphans from crashes
between steps.

### 4.4b Snapshot layout: CSR adjacency (required for mmap traversal)

**Motivating defect:** the current `ArchivedGraphView::outgoing_neighbors`
finds neighbors by a linear scan over ALL edges (O(E) per expansion instead
of O(degree)). Today this is masked because consumers traverse an owned
petgraph (L1) and use the archived view only as transport; traversing
directly over mmap on this layout would be orders slower than owned.

Fix, baked into `format_version: 1` from day one: the snapshot stores edges
**sorted by `from_idx` plus a per-node offset array (CSR)**, and a mirror
index sorted by `to_idx` for incoming edges. Archived neighbor lookup becomes
an O(degree) slice read, same asymptotics as petgraph — and the index lives
inside the mmap'd file, so `open()` builds nothing beyond the name index.

**Compatibility — this is a NEW on-disk struct, and that is intentional.**
The persistent-store snapshot is `SerializableGraphV1` (CSR: sorted edges +
`node_offsets: Vec<u32>` + `in_mirror: Vec<u32>`), distinct from today's flat
`SerializableGraph` (unsorted `Vec<SerializableEdge>`). The earlier claims
that the snapshot is "the existing format", that "`ArchivedGraph` already
exists — only new code is the mmap open path", and that "`to_rkyv`/`from_rkyv`
remain untouched" apply **only to the legacy ephemeral Redis path** (which
stays exactly as-is, tagged `format_version: 0`). The persistent store adds
the archived-CSR-accessor as genuinely new code. Both formats coexist behind
the version tag; V0 is never mmap-traversed (it lacks the CSR index). This
supersedes the "untouched / already exists" wording in §4.1, §4.5, §4.6.

> **Amendment (implemented as V2 — see `persistence_impl_log.md` §6):** the
> shipped CSR format additionally persists the `name -> idx` index itself
> (`name_buckets`: FNV-1a open-addressing table, zero-copy probed), so `open()`
> builds NOTHING — the eager O(N) name-index build this section still assumed
> was measured to dominate warm open and was removed. `format_version` is `2`;
> `open()`/`compact()` read and write V2 only, and pre-release legacy tags
> `0`/`1` are rejected at open (recreate the store to migrate). Everything else
> in this section (CSR layout, sorted edges, offsets, `in_mirror`) is unchanged
> and carried into V2.

### 4.5 mmap and larger-than-RAM

Step 2 of recovery deliberately keeps the base as an **archived view over
mmap** rather than deserializing to owned structures. Consequences:

- Warm start avoids *deserialization* into owned structures, but it is **not**
  O(name-index) unconditionally: verifying `snapshot_crc32` reads every byte,
  and `rkyv::access` with full rancor validation walks the whole archive — so
  a validating open is O(graph bytes) (the earlier "~5.6 ms warm start" figure
  holds only at 50K scale, and only relative to full deserialization, not as
  an absolute for larger-than-RAM). Open validation is therefore a config:
  `validate = "full"` (crc + rkyv structural check, default),
  `validate = "crc"` (crc only — trusts rkyv layout), or `validate = "none"`
  (trusted local file, O(1) open — the true fast path, for a snapshot the
  process itself just wrote). Untrusted snapshots (shared/remote store) must
  use `"full"`; see §5.1 trust boundary.
- Cold regions of the graph never occupy RAM; beam search locality (walks stay
  near seed nodes) means the resident set ≈ the hot neighborhood, not the
  whole graph. This is the "larger-than-RAM graphs via OS paging" capability.
- **Honest latency caveat.** "Cold page faults once" holds only when the graph
  fits in the page cache. On a genuinely larger-than-RAM graph under memory
  pressure, hot pages can be evicted and re-faulted arbitrarily often, so
  hot-path P99 is **not** bounded the way the in-RAM µs figures suggest —
  larger-than-RAM trades a hard RAM ceiling for an unbounded-tail-latency
  risk, and that is the correct trade only when the working set is much
  smaller than the graph. Not a free lunch; documented as such.
- **SIGBUS.** A lazily-faulted mmap page backed by an I/O error (bad sector,
  thin-provisioned volume hitting its limit, a network filesystem) delivers
  SIGBUS, which by default kills the process mid-traversal with no Rust error
  path. v1 mitigation: **only mmap files on local block storage** (documented
  precondition), and for the larger-than-RAM/network case fall back to
  `mode="owned"` (read+validate up front, no lazy faulting). A SIGBUS handler
  that converts the fault into a recoverable error is possible but out of
  scope v1 (signal-handling complexity across the PyO3 boundary).
- Requires `ArchivedGraph` to implement `GraphAccessor` — **already exists**
  (accessor.rs supports owned + archived). The only new code is the mmap open
  path and crc check.
- Platform note: file must not be truncated/replaced in place while mapped —
  guaranteed by the rename-only discipline above (the old mapping keeps the
  old inode alive until dropped).
- **Performance knobs.** Warm mmap reads are page-cache reads — RAM speed;
  only cold pages pay a fault (~µs on NVMe, once). `open()` options:
  `prefault=True` (madvise WILLNEED — warm everything up front, matching
  today's load-all behavior but without deserialization) and
  `mode="mmap"|"owned"` — owned materializes a petgraph at open for
  small graphs that want the exact current hot path, while keeping WAL
  persistence; mmap is the default and the only option for
  larger-than-RAM.
- **Empty-delta fast path.** When the delta is empty (e.g. right after
  compaction), `DeltaAccessor` must delegate straight to the base with zero
  merge overhead — readers of a quiescent graph pay nothing for mutability
  existing.

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

# Read-modify-write with a serializability guarantee (optimistic CAS, §3.3.6):
seq = g.seq                                   # pin the state you read from
node = g.get_node("account.A")                # ... read, decide ...
g.apply(ops, expected_seq=seq)                # ConflictError if state moved →
                                              # re-read, recompute, retry

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
   **§5.1 Untrusted-snapshot trust boundary (also a live code bug today).**
   `rkyv::access` structurally validates a buffer but does **not** check that
   edge `from_idx`/`to_idx` are in range for `nodes.len()`. The current
   zero-copy path (`ArchivedGraphView::outgoing_neighbors`,
   serialization.rs:175) indexes `nodes[to_idx]` directly, so a snapshot with
   an out-of-range edge index passes `from_rkyv`/open and then **panics on the
   first traversal** — while the sibling `from_rkyv_rebuild` path
   (builder.rs) already guards the same indices. Any snapshot from a shared or
   remote store (Redis, the persistence dir on a network mount) is untrusted
   input. **Required under `validate="full"` (reject with an error, never panic
   on query):**
   - every edge `from_idx`/`to_idx` < `nodes.len()`;
   - **the CSR arrays introduced by §4.4b** — `node_offsets` must be
     monotonic non-decreasing with `node_offsets[nodes.len()] == edges.len()`
     and every entry ≤ `edges.len()`; the `in_mirror` array must be a
     permutation of `0..edges.len()` (each entry < `edges.len()`). These are
     consumed as slice bounds during neighbor lookup, so a snapshot with valid
     endpoints but a corrupt/non-monotonic offset array (or an out-of-range
     mirror entry) would otherwise slice out of bounds and panic — the same
     class of bug as the endpoint check, one layer down.
   `validate="none"` is permitted only for a file the process itself wrote this
   run. This closes the audit's serialization.rs finding and is a precondition
   for the `validate` modes in §4.5.
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
  sizes 0 / 1% / 10% of base (targets: 0 measurable overhead at empty delta,
  ≤ 1.3× at 10%), and `beam_traverse` over warm mmap (CSR layout) vs owned
  petgraph (target: ≤ 1.5×; the current O(E) archived scan must show as fixed
  here).

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

**Concurrent sources ≠ multi-writer.** "Two sources writing facts at the same
time" decomposes into two problems, neither needing MVCC:

- *Physical concurrency*: callers serialize on the writer mutex (§3.5) — with
  µs-scale `apply()`, a queue of even thousands of batches/sec drains
  instantly (the SQLite/Redis single-writer precedent). Multiple threads/tasks
  may call `apply()` concurrently today; multiple *processes* route through
  the graph-owning process (flock enforces this). True multi-writer MVCC only
  pays off when writers read inside long transactions — not a fact-ingestion
  workload.
- *Semantic conflict* (source A: "owner is X", source B: "owner is Y"): MVCC
  would not help — both commits succeed physically and last-write-wins by seq
  order, same as here. The real fix is **provenance** (edge metadata: source,
  confidence, observed_at — store both claims with attribution, never
  silently overwrite) plus **temporal validity** (invalidate the older fact
  via valid_to), with contradiction resolution as domain logic above the
  store. This is why provenance/temporal rank high on the roadmap.

**Explicitly rejected: interactive transactions.** Atomic multi-op batches +
snapshot-isolated readers (ArcSwap) already cover the useful subset of ACID
for this workload, and read-modify-write callers get a serializability
guarantee via optimistic CAS (`expected_seq`, §3.3.6) — the Redis
WATCH/MULTI / etcd-revision pattern — at the cost of a retry loop instead of
MVCC machinery. What stays rejected is held-open interactive transactions
(`BEGIN → read → decide → write → COMMIT` as a server-side session) with
write-conflict detection at row granularity: that only pays off with many
writers holding long transactions — ledger territory (a different system),
and the added machinery would reduce reliability, not add it. Reliability strategy here is
SQLite's: few hard invariants (batch atomicity, monotonic seq, rename+dir
fsync, flock) beaten by a brutal test harness — not more mechanisms.
