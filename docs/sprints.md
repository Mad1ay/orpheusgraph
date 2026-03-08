# orpheusgraph — Sprint Breakdown

> Global spec: [orpheusgraph_lib_spec.md](./orpheusgraph_lib_spec.md)
> Diagrams: [orpheusgraph_diagrams.md](./orpheusgraph_diagrams.md)

## Sprint Overview

| Sprint | Название | Фокус | Deliverable |
|---|---|---|---|
| **S1** | Core Types & Builder | Rust crate scaffold, types, builder, tests | `build_graph()` работает |
| **S2** | Scoring & Overlay | Scoring formula, DynamicContext, overlay iterator | Контекстный скоринг |
| **S3** | Traversal | beam_traverse, find_path, contextual_subgraph | Все 3 алгоритма |
| **S4** | Serialization & PyO3 | rkyv, lz4, Python bindings, .close(), EdgeResult | `pip install` работает |
| **S5** | Cache & OSDS Integration | 3-tier cache, BLPOP, generation, format_for_llm | OSDS connector |
| **S6** | Hardening | Benchmarks, CI/CD, all 20 risks addressed | Production-ready |

---

## S1 — Core Types & Builder (3-4 дня)

### Цель
Работающий Rust crate: типы данных + сборка графа из raw data.

### Задачи
- [x] `types.rs` — `NodeData`, `EdgeData`, `PathStep`, `DynamicContext`
- [x] `builder.rs` — `build_graph(nodes, edges) → DiGraph`
  - Нормализация `base_weight` к [0.0, 1.0]
  - Compute `pagerank_weight` для God Object detection
- [x] `graph.rs` — immutable DiGraph wrapper
  - `.node_count()`, `.edge_count()`, `.get_node()`
  - `.outgoing_edges()`, `.incoming_edges()` → `Vec<EdgeResult>`
- [x] `tests/test_builder.rs` — round-trip: build → verify structure
- [x] `Cargo.toml` — petgraph, serde, rayon, jemalloc

### Definition of Done
```rust
let graph = build_graph(nodes, edges);
assert_eq!(graph.node_count(), 50000);
assert!(graph.get_node("sale.order").is_some());
```

---

## S2 — Scoring & Overlay (3-4 дня)

### Цель
Вычисление весов с учётом контекста. Виртуальные ноды.

### Задачи
- [x] `scoring.rs` — multiplicative formula:
  ```
  W = (w_base * base + w_semantic * sem + w_override * ovr) * (1.0 - noise)
  ```
- [x] `overlay.rs` — `neighbors_with_overlay()` — zero-alloc chain iterator
  - `max_fan_out` cutoff с pagerank bypass
  - `noise_tags` domain filtering
- [x] `types.rs` — add `w_base`, `w_semantic`, `w_noise`, `w_override`, `noise_tags`, `max_fan_out`
- [x] `tests/test_scoring.rs` — verify noise=0.9 kills node
- [x] `tests/test_overlay.rs` — overlay nodes visible, base untouched

### Definition of Done
```rust
let ctx = DynamicContext { noise_tags: {"technical"}, max_fan_out: Some(50), .. };
let score = compute_score(&node, &ctx);
assert!(score < 0.1);  // noise_penalty=0.9 → 10% of base
```

---

## S3 — Traversal (3-4 дня)

### Цель
Три режима обхода: Beam Search, Dijkstra, Subgraph extraction.

### Задачи
- [x] `traversal.rs` — `beam_traverse(start, k, depth, ctx) → Vec<NodeResult>`
  - Level-by-level Top-K pruning
  - Lazy scoring (only visited nodes)
- [x] `traversal.rs` — `find_path(start, end, ctx) → Vec<PathStep>`
  - Weighted Dijkstra с `direction` field
- [x] `traversal.rs` — `contextual_subgraph(ctx, k) → SubGraph`
- [x] `NodeResult` — add `.explain_score()` method
- [x] `tests/test_traversal.rs` — 50K node graph, verify Top-K correctness
- [x] `benches/bench_traversal.rs` — Criterion benchmark

### Definition of Done
```rust
let results = beam_traverse("sale.order", 5, 3, &ctx);
assert_eq!(results.len(), 15);  // 5 * 3 levels
assert!(results[0].weight > results[14].weight);  // sorted
```

---

## S4 — Serialization & PyO3 (4-5 дней)

### Цель
Python bindings. rkyv zero-copy. Публикуемый wheel.

### Задачи
- [x] `serialization.rs` — `.to_rkyv()`, `.from_rkyv()` with validation
- [x] `lib.rs` — PyO3 module: `#[pymodule]`
  - `OrpheusGraph` — `#[pyclass]` with `Option<DiGraph>` for `.close()`
  - `DynamicContext` — `#[pyclass]` with all fields
  - `NodeResult`, `EdgeResult` — `#[pyclass]` lightweight
- [x] `py.allow_threads()` on all traversal methods
- [x] `_pinned_bytes: Py<PyBytes>` — rkyv memory safety (Risk #1)
- [x] Drop ordering: `inner` before `_pinned_bytes` (Risk #18)
- [x] `python/orpheusgraph.pyi` — type stubs
- [x] `pyproject.toml` — maturin config
- [x] Test: `maturin develop` → `import orpheusgraph` → `build_graph()` → `beam_traverse()`

### Definition of Done
```python
import orpheusgraph
graph = orpheusgraph.build_graph(nodes=[...], edges=[...])
results = graph.beam_traverse("sale.order", k=5, depth=3, ctx=ctx)
graph.close()  # no segfault
```

---

## S5 — Cache & OSDS Integration (4-5 дней)

### Цель
3-tier cache в OSDS. format_for_llm. Pipeline snapshot.

### Задачи
- [x] `server/app/core/graph_cache.py` — get_graph() с L1/L2/L3 cascade
  - Generation counter, BLPOP coordination
  - lz4 compression, schema version + arch in key
  - Lock renewal watchdog (Risk #16)
  - Error marker on build failure (Risk #9)
  - LRU eviction for L1 (Risk #20)
- [x] `ai/utils/graph_format.py` — format_for_llm() с Markdown arrows
  - OUTGOING / INCOMING sections
  - `.explain_score()` debug info
- [x] `ai/tools.py` — `traverse_erp_graph()` с DynamicContext
- [x] `ai/agents/graph.py` — `init_pipeline()` graph generation snapshot
- [x] L1 warmup in `server/app/main.py` (lifespan)
- [x] Integration test: pipeline end-to-end

### Definition of Done
```python
graph = await get_graph("odoo", "18.0", db)
results = graph.beam_traverse("sale.order", k=5, depth=3, ctx=ctx)
md = format_for_llm(results, graph)
assert "## [MODEL] sale.order" in md
assert "--[relates_to]-->" in md
```

---

## S6 — Hardening (3-4 дня)

### Цель
Production-ready. Benchmarks. CI/CD.

### Задачи
- [x] Criterion benchmarks: все targets из Performance Targets
  - `build_graph` <10ms, `from_rkyv` <0.1ms, `beam_traverse` <1ms
- [x] GitHub Actions: CI (test + lint + bench) + Release (multi-platform wheels)
- [x] `README.md` — Quick Start, API docs, examples
- [x] `CONTRIBUTING.md` — Dev setup, PR process
- [x] `CHANGELOG.md` — Sprint log
- [x] Verify all 20 risks addressed
- [x] `git subtree split` prep for standalone repo

### Definition of Done
```bash
cargo bench  # all targets met
maturin build --release  # clean wheel
pytest tests/ -v  # all green
```
