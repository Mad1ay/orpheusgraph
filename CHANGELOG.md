# Changelog

All notable changes to orpheusgraph.

## [Unreleased]

Three deferred persistence features delivered; on-disk `format_version` bumped **2 → 3**
(the COMMIT sidecar is the addition; the CSR byte layout is unchanged). Legacy stores 0/1/2
are rejected at `open()` with `CorruptError` — pre-release, no migration; recreate the store.

### Durability — per-frame COMMIT marker + honest `apply()` contract
- Recovery now trusts an authoritative fsync-gated **COMMIT** high-water marker (two-slot
  torn-safe ping-pong sidecar) instead of "whatever crc-valid WAL frames survived". Recovered
  state is EXACTLY the committed prefix; crc-valid frames beyond the marker are truncated and
  counted. **Behavior change:** an `OnFlush` write-through tail that survives a process crash but
  was never fsync-acknowledged is now discarded (it was never `apply`-`Ok`-durable).
- **`apply()` durability contract, stated honestly** (a single fsync cannot distinguish
  fails-but-persists from success): `apply`-`Ok` ⇒ durable; `apply`-`Err` ⇒ **indeterminate** —
  the batch may or may not be visible on reopen, but recovery ALWAYS yields a consistent
  committed prefix with no `Ok`-acked loss. Reconcile an `Err` by reading `committed_seq()` /
  `recovery_report()` after reopen, never by blind retry (double-apply risk on non-idempotent ops).
- New Rust surface: `RecoveryReport { uncommitted_tail_frames, torn_tail_frames }` (re-exported),
  `PersistentGraph::committed_seq()`, `PersistentGraph::recovery_report()`. *(Not yet exposed to
  Python — a Python caller reconciles via `seq` after reopen, which equals `committed_seq`.)*

### Durability testing — deterministic power-loss harness (`Vfs`)
- New public `Vfs` / `VfsFile` / `RealVfs` seam over the persist layer's write-and-durability
  ops (a RocksDB-`Env`-style extension point; prod uses `RealVfs`, a zero-cost `std::fs`
  passthrough — the hot read path never touches it). `PersistentGraph::create_with_vfs` /
  `open_with_vfs` (`#[doc(hidden)]`) accept a custom `Vfs`.
- A deterministic userspace fault-injection FS (`FaultVfs`) + `tests/powerloss_harness.rs`
  replace the spec's `dm-flakey`/VM plan — seeded, CI-able, models torn writes and the
  fsync-fails-but-persists ambiguity. 1600+ randomized power cuts found no marker/recovery bug.

### Query — temporal validity + edge-level ACL (query-time filtering)
- Edges gain `valid_from` / `valid_to` (`Optional[int]`, half-open `[from, to)`, `None` = unbounded)
  and `acl` (`list[str]`, empty = public, sorted+deduped at ingest). `DynamicContext` gains
  `as_of` (`None` = no temporal filter) and `principals` (fail-closed: a tagged edge is visible
  iff its `acl` intersects `principals`; empty `principals` sees only public edges).
- Filtering is **query-time only** and applied by all four ctx-taking ops (`beam_traverse`,
  `find_path`, `contextual_subgraph`, `multi_beam_intersection`); PageRank and all base metrics
  still run over ALL edges. The raw `outgoing_edges`/`incoming_edges` inspectors are UNFILTERED by
  design (documented) — do not surface them under an access-control assumption.
- Python: `add_edge` op dicts and `build_graph` edge dicts accept `valid_from`/`valid_to`/`acl`;
  `DynamicContext(as_of=None, principals=None)`; `EdgeResult` exposes the three fields. Fully
  backward-compatible (missing keys default to unbounded/public). A `valid_from > valid_to` edge is
  skipped by `build_graph` but hard-rejects a `PersistentGraph.apply` batch (`ValueError`).

## [0.1.0] — 2026-03-08

### Sprint 7 — Python Persistence API (Phase 3)
- `orpheusgraph.open(dir, create=False, mmap=True, validate="full", prefault=False) -> PersistentGraph` — open or initialize the durable store; missing store + `create=False` raises `FileNotFoundError`
- `orpheusgraph.create_persistent(dir, nodes, edges) -> PersistentGraph` — one-pass base seeding from `build_graph`-style dicts, instead of replaying through the delta; refuses to clobber an existing store
- `PersistentGraph.apply(ops, expected_seq=None) -> int` — atomic batch (one WAL frame); op dicts: `upsert_node`, `remove_node`, `add_edge`, `remove_edge`; weights must be pre-normalized to [0,1] (`ValueError` otherwise)
- Optimistic CAS via `expected_seq` — `orpheusgraph.ConflictError` on a stale sequence, nothing written
- `.seq` / `.epoch` properties — durable commit seq (CAS token / cache generation) and incarnation id
- Read API identical to `OrpheusGraph` over the live base+delta view: `.get_node`, `.outgoing_edges`, `.incoming_edges`, `.beam_traverse`, `.find_path`, `.contextual_subgraph`, `.multi_beam_intersection`; GIL released during traversal and during `apply`/`flush`/`compact`/`close`
- `.flush()`, `.compact()` (also auto-fires by threshold), `.set_fsync_policy()`, `.set_auto_compact_threshold()`, `.close()` (idempotent) + context-manager support (`with orpheusgraph.open(...) as g:`)
- New exceptions: `orpheusgraph.ConflictError`, `orpheusgraph.CorruptError`
- `to_rkyv`/`from_rkyv` unchanged, but `build_graph` now sorts input nodes by name
  before petgraph insertion so PageRank's summation order is canonical regardless of
  input order (previously order-dependent, since f32 addition is non-associative) —
  this applies to the ephemeral in-memory workflow too, so a caller relying on the
  prior `to_rkyv()` byte layout or exact `pagerank_weight` bits for identical inputs
  will see different (now-deterministic) output — a bugfix, but a behavior change
- `PersistentGraph` on-disk snapshot format bumped to **V2**: a persisted open-addressing
  `name -> idx` index (`name_buckets`, FNV-1a) built once at write time, giving O(1)
  warm-open (no more eager per-open index build) and O(1) lookup while preserving mmap
  larger-than-RAM lazy paging
- Legacy snapshot formats 0 and 1 are now rejected at `open()` (`CorruptError`) instead of
  being read — pre-release, no in-place migration; recreate the store
- `DeltaAccessor` (internal delta-overlay reader): fixed an allocation regression in
  `outgoing_neighbors`/`incoming_neighbors` — a per-base-edge 3-`String` tuple
  tombstone-probe key was built even when the delta had no removals; now masks the base
  edge list in place and skips the tombstone check entirely when there's nothing to mask
  (`get_node`'s empty-set short-circuits are a readability win, not an allocation fix —
  `HashMap::get`/`HashSet::contains` never allocated)
- Added `benches/bench_persist.rs`: `apply`, warm-`open`, delta-traversal-overhead, and
  CSR-vs-petgraph-beam benchmarks for the persistent store (spec §6)

### Sprint 1 — Core Types & Builder
- `NodeData`, `EdgeData`, `DynamicContext`, `PathStep` types
- `build_graph()` with base_weight normalization and PageRank computation
- Immutable `OrpheusGraphInner` wrapper with edge inspection
- `Cargo.toml`: petgraph, serde, rayon, jemalloc

### Sprint 2 — Scoring & Overlay
- Multiplicative noise formula: `raw × (1.0 - noise_penalty)`
- Domain-aware `noise_tags` filtering
- Virtual overlay iterator with `max_fan_out` cutoff + pagerank bypass
- Tenant isolation via ephemeral overlays

### Sprint 3 — Traversal
- `beam_traverse()` — Top-K pruned BFS
- `find_path()` — Weighted Dijkstra with direction tracking
- `contextual_subgraph()` — Extract context-relevant subgraph
- `.explain_score()` for debugging score components
- Criterion benchmarks at 50K nodes

### Sprint 4 — Serialization & PyO3
- rkyv serialization (`to_rkyv`, `from_rkyv`) with validation
- `ArchivedGraphView` — zero-copy graph reads via `GraphAccessor` trait
- PyO3 bindings: `OrpheusGraph`, `DynamicContext`, `NodeResult`, `EdgeResult`, `PathStep`, `SubGraph`
- GIL release (`py.allow_threads`) on all traversal methods
- `.close()` + `Drop` trait for deterministic memory release
- Python type stubs (`.pyi`)

### Sprint 5 — Caching
- 3-tier cache: L1 in-process → L2 Redis (lz4) → L3 source rebuild
- Generation counter for L1 staleness detection across workers
- BLPOP coordination (thundering herd protection)
- Lock renewal watchdog (SIGKILL safety)
- Error marker on build failure
- Schema version in Redis key
- LRU eviction (maxsize=3) for L1
- `format_for_llm()` — Markdown rendering of a subgraph for LLM context

### Sprint 6 — Hardening
- CI: `rust-check` job (clippy, cargo test, bench compile)
- CI: `bench-gate` job (critcmp 10% regression threshold on PRs)
- Risk #3: `semantic_boosts` cap at 200 before FFI transfer
- Risk #14: Overlay cache per `overlay_cache_key` in Rust
- Documentation: README.md, CONTRIBUTING.md, CHANGELOG.md
- All 20 spec risks addressed
