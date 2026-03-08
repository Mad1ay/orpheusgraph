# Changelog

All notable changes to orpheusgraph.

## [0.1.0] — 2026-03-08

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

### Sprint 5 — Cache & OSDS Integration
- 3-tier cache: L1 in-process → L2 Redis (lz4) → L3 PostgreSQL rebuild
- Generation counter for L1 staleness detection across workers
- BLPOP coordination (thundering herd protection)
- Lock renewal watchdog (SIGKILL safety)
- Error marker on build failure
- Schema version + architecture in Redis key
- LRU eviction (maxsize=3) for L1
- `format_for_llm()` — Markdown output for LLM context
- `traverse_erp_graph()` tool for LangGraph agents
- `init_pipeline_graph()` — graph generation snapshot per pipeline
- L1 warmup in server lifespan

### Sprint 6 — Hardening
- CI: `rust-check` job (clippy, cargo test, bench compile)
- CI: `bench-gate` job (critcmp 10% regression threshold on PRs)
- Risk #3: `semantic_boosts` cap at 200 before FFI transfer
- Risk #14: Overlay cache per `overlay_cache_key` in Rust
- Documentation: README.md, CONTRIBUTING.md, CHANGELOG.md
- All 20 spec risks addressed
