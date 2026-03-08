# orpheusgraph — Knowledge Graph Traversal Engine

> **Rust library with Python bindings for context-aware weighted graph traversal.**
> Open-source. Domain-agnostic. Built for RAG pipelines that need deterministic structure, not probabilistic guessing.

## What It Does

LLMs hallucinate relationships. `orpheusgraph` doesn't. It provides:

1. **Build** weighted knowledge graphs from any structured data
2. **Cache** via 3-tier: L1 in-process memory → L2 Redis → L3 source DB
3. **Traverse** with context-aware Beam Search — returns Top-K relevant nodes instead of exponential BFS explosion
4. **Isolate** tenants via virtual overlay — zero shared mutable state

```
Your data (DB, API, files)
    │
    ▼
[Python] ─── raw rows ───→ [Rust: orpheusgraph]
                              │
                              ├── build_graph() → petgraph DiGraph
                               ├── to_rkyv() → Redis (immutable, zero-copy)
                              ├── beam_traverse(ctx) → Top-K nodes
                              ├── find_path(A, B, ctx) → shortest weighted path
                              └── contextual_subgraph(topic, ctx) → focused subgraph
```

---

## Core Concepts

### Nodes & Edges

```rust
// Generic — no ERP/CRM/domain assumptions
pub struct NodeData {
    pub name: String,           // unique key: "sale.order", "users_table", "AccountEntity"
    pub kind: String,           // "model", "field", "module", "doc", "table", etc.
    pub metadata: HashMap<String, String>,  // arbitrary key-value pairs
    pub base_weight: f32,       // static weight (usage frequency, computed offline)
    pub noise_penalty: f32,     // static penalty for system/technical nodes
}

pub struct EdgeData {
    pub kind: String,           // "inherits", "relates_to", "depends_on", "contains", etc.
    pub field_name: Option<String>,  // how nodes are linked (e.g. "partner_id")
    pub base_weight: f32,       // static edge strength
}
```

### Dynamic Context (Flyweight Pattern)

The graph itself is **immutable after build**. All per-request customization goes through `DynamicContext`:

```rust
pub struct DynamicContext {
    /// Semantic boosts: node_name → multiplier (from embedding similarity)
    pub semantic_boosts: HashMap<String, f32>,

    /// Virtual overlay: temporary nodes visible only during this traversal
    pub overlay_nodes: Vec<NodeData>,
    pub overlay_edges: Vec<(String, String, EdgeData)>,  // (from, to, edge)

    /// Per-request weight overrides (e.g. project-specific usage stats)
    pub weight_overrides: HashMap<String, f32>,

    /// Scoring coefficients — configurable for A/B testing without Rust recompile
    pub w_base: f32,      // default 1.0
    pub w_semantic: f32,  // default 1.5
    pub w_noise: f32,     // default 1.0
    pub w_override: f32,  // default 1.0

    /// Domain-aware noise filter (e.g. "technical", "audit", "messaging")
    /// Nodes tagged with these domains get boosted noise_penalty
    pub noise_tags: HashSet<String>,  // e.g. {"technical"} → create_uid penalized

    /// Degree cutoff for "God Object" nodes (e.g. res.partner with 1000+ edges)
    pub max_fan_out: Option<usize>,  // default None = unlimited. Set to 50 to cap
}
```

> [!IMPORTANT]
> `DynamicContext` is **ephemeral** — created per function call, never stored.
> The base graph in Redis/memory is **never mutated** by traversal operations.

### Scoring Formula

$$W_{total} = (w_{base} \cdot base\_weight) + (w_{semantic} \cdot semantic\_boost) - (w_{noise} \cdot noise\_penalty) + (w_{override} \cdot weight\_override)$$

All `w_*` coefficients live in `DynamicContext` — tunable per-request from Python.

> [!WARNING]
> **All base values MUST be normalized to [0.0, 1.0]** at graph build time. If `base_weight` is raw counts (e.g. 5000 field usages) and `semantic_boost` is cosine similarity (0.0-1.0), the boost will be invisible. Normalize in `builder.rs`.

---

## GraphRAG: Documentation Nodes

Граф без документации — чисто структурный. Агент знает ЧТО есть (`stock.picking`), но не знает КАК это работает (*"Odoo supports backorders via picking split"*). Для полного GraphRAG нужны `doc` ноды.

### Node Types

```
Structural (kind)          Documentation (kind)
─────────────────          ─────────────────────
"model"  sale.order        "doc"  "Backorder processing"
"field"  partner_id        "doc"  "POS workflow overview"
"module" stock             "doc"  "Multi-warehouse routing"
```

### Edge: `describes`

```
DocChunk("Backorder processing")  ──describes──→  stock.picking
DocChunk("Backorder processing")  ──describes──→  stock.backorder.confirmation
DocChunk("POS workflow")          ──describes──→  pos.order
DocChunk("POS workflow")          ──describes──→  pos.session
```

### Как создавать `describes` edges (при парсинге KB)

```python
# server/app/services/parsers/ — при загрузке документации

def extract_doc_edges(doc_chunk: ERPDocChunk, known_models: set[str]) -> list[dict]:
    """Extract model references from doc text → describes edges."""
    edges = []
    
    # 1. Regex: ищем model names в тексте ("stock.picking", "sale.order")
    for model in re.findall(r'[a-z]+\.[a-z_.]+', doc_chunk.content):
        if model in known_models:
            edges.append({"from": doc_chunk.id, "to": model, "kind": "describes"})
    
    # 2. NER (optional): извлекает бизнес-сущности ("Sales Order" → sale.order)
    # 3. Embedding proximity: cosine_sim(doc_emb, model_emb) > threshold
    
    return edges

# Threshold — configurable, NOT hardcoded (varies by embedding model)
DOC_EDGE_SIMILARITY_THRESHOLD = float(os.getenv("DOC_EDGE_THRESHOLD", "0.85"))
# Regex match is preferred. Embedding fallback only at high threshold.
# OpenAI text-embedding-3: ~0.85 | BGE/E5: ~0.75 | Cohere: ~0.80
```

### Проблемы и решения

| Проблема | Суть | Решение |
|---|---|---|
| **Graph bloat** | 50K doc chunks → граф раздуется до 100K+ нод | **Representative docs**: группируем chunks по source file, берём 1 ноду на файл (не chunk). Chunks хранятся в metadata |
| **Fuzzy edges** | Не все доки упоминают конкретные модели. "Odoo architecture overview" даст шум | **Confidence threshold**: edge создаётся только если ≥2 model references в тексте ИЛИ embedding similarity > 0.7 |
| **Dual storage** | Doc chunks уже в pgvector. Дублировать в графе? | **Не дублируем контент**. В графе только `{name, kind: "doc", metadata: {source_file, chunk_count}}`. Полный текст — из pgvector по UUID при необходимости |
| **Stale docs** | Доки обновляются реже моделей — граф может отстать | **Invalidation**: при `parse-docs` → пересоздаём doc edges, bump generation |

### Как это меняет Beam Search

```
Запрос: "нужны частичные отгрузки"
  │
  ├─ Vector search (pgvector) → top-5 doc chunks по embedding
  │    └─ DocChunk("Backorders in Odoo") → score 0.92
  │
  ├─ Graph: doc node → describes → stock.picking
  │    └─ beam_traverse("stock.picking", ctx={semantic_boost: {stock.*: high}})
  │         → stock.picking → stock.move → stock.backorder.confirmation
  │
  └─ LLM получает:
     • БИЗНЕС-контекст: "Odoo supports backorders via picking split" (из doc)
     • СТРУКТУРНЫЙ: stock.picking.fields, relations, inheritance (из graph)
     → точный gap-analysis: "стандарт покрывает, модуль stock"
```

---

## Traversal Modes

### 1. `beam_traverse(start, k, depth, ctx) → Vec<NodeResult>`

Top-K pruned BFS. On each level, keeps only the `k` highest-scored neighbors.

```
beam_traverse("sale.order", k=5, depth=3, ctx)
→ 15 nodes (5×3) instead of ~500
→ ~2-3K tokens for LLM context instead of ~15K
```

### 2. `find_path(start, end, ctx) → Vec<PathStep>`

Weighted Dijkstra. Returns the full relationship chain with edge metadata.

```rust
pub struct PathStep {
    pub node: String,       // "sale.order"
    pub edge_kind: String,  // "relates_to"
    pub field_name: String, // "partner_id"
    pub direction: String,  // "outgoing" | "incoming"
}
```

```
find_path("res.partner", "stock.picking", ctx)
→ [
    PathStep { node: "res.partner", edge: "relates_to", field: "partner_id", dir: "incoming" },
    PathStep { node: "sale.order",  edge: "relates_to", field: "origin",     dir: "outgoing" },
    PathStep { node: "stock.picking", ... }
  ]
```

### 3. `contextual_subgraph(ctx, k) → SubGraph`

Extract a compact subgraph of `k` nodes most relevant to the context (by semantic_boost), plus their depth-1 neighbors.

---

## Architecture: 3-Tier Cache

```
┌─────────────────────────────────────────────────────────┐
│ L3: Source DB (PostgreSQL)                              │
│ ERPKnowledge* tables — source of truth, always durable  │
└────────────────────────┬────────────────────────────────┘
                         │ rebuild <10ms (on L1+L2 miss)
┌────────────────────────▼────────────────────────────────┐
│ L2: Redis (shared cache)                                │
│ msgpack bytes, immutable, TTL 24h                       │
│ One key per graph version: "graph:{system}:{version}"   │
│ Thundering herd protection via distributed lock         │
└──────────┬─────────────┬─────────────┬──────────────────┘
           │ ~5ms        │             │
    ┌──────▼──────┐ ┌────▼─────┐ ┌────▼─────┐
    │ L1: Worker A│ │L1: W-B   │ │L1: W-C   │
    │ In-process  │ │In-process│ │In-process│
    │ Rust memory │ │Rust mem  │ │Rust mem  │
    │ <0.1ms read │ │          │ │          │
    │             │ │          │ │          │
    │ + ctx:{...} │ │+ ctx:{…} │ │+ ctx:{…} │
    │   (ephemeral│ │          │ │          │
    │    overlay) │ │          │ │          │
    └─────────────┘ └──────────┘ └──────────┘
```

**Lookup flow**: L1 hit → 0.1ms done. L1 miss → L2 Redis → deserialize → warm L1. L2 miss → L3 PG → `build_graph()` → warm L2 + L1.

**Invalidation**: on KB parse → delete L2 key → signal L1 (workers lazy-reload on next request).

---

## Crate Structure

```
orpheusgraph/
├── Cargo.toml
├── src/
│   ├── lib.rs             # PyO3 module + public API
│   ├── graph.rs            # petgraph DiGraph (immutable after build)
│   ├── builder.rs          # build_graph(rows) → DiGraph
│   ├── overlay.rs          # Virtual overlay from DynamicContext
│   ├── scoring.rs          # Lazy W_total: base + ctx, never mutates
│   ├── traversal.rs        # beam_traverse, find_path, contextual_subgraph
│   ├── serialization.rs    # rkyv zero-copy ↔ petgraph
│   └── types.rs            # NodeData, EdgeData, PathStep, DynamicContext
├── python/
│   └── orpheusgraph.pyi          # Python type stubs
├── tests/
│   ├── test_builder.rs
│   ├── test_scoring.rs
│   ├── test_traversal.rs
│   └── test_overlay.rs     # Tenant isolation verification
├── benches/
│   └── bench_traversal.rs  # Criterion benchmarks
├── LICENSE                  # MIT or Apache-2.0
└── README.md
```

### Dependencies

```toml
[dependencies]
petgraph = "0.6"
pyo3 = { version = "0.21", features = ["extension-module"] }
rkyv = { version = "0.8", features = ["validation"] }  # zero-copy deserialization
serde = { version = "1", features = ["derive"] }
rayon = "1.8"

[dev-dependencies]
criterion = { version = "0.5", features = ["html_reports"] }

[profile.release]
lto = true

[target.'cfg(not(target_env = "msvc"))'.dependencies]
jemallocator = "0.5"  # better alloc for long-lived graph structs
```

> [!TIP]
> **`rkyv` instead of `rmp-serde`**: zero-copy deserialization maps Redis bytes directly into memory without allocating individual structs. L2→L1 load goes from ~10ms to **microseconds**.

---

## Rust Implementation Notes

### 1. Virtual Overlay Iterator (zero-alloc)

Base `DiGraph` is immutable — no way to "add" overlay nodes to petgraph. Solution: custom iterator that chains base + overlay:

```rust
// overlay.rs
pub fn neighbors_with_overlay<'a>(
    graph: &'a DiGraph<NodeData, EdgeData>,
    overlay: &'a DynamicContext,
    node: &str,
) -> impl Iterator<Item = (&'a str, &'a EdgeData)> + 'a {
    let base = graph.neighbors(node_idx)
        .map(|idx| (graph[idx].name.as_str(), &graph[edge_idx]));
    let extra = overlay.overlay_edges.iter()
        .filter(move |(from, _, _)| from == node)
        .map(|(_, to, edge)| (to.as_str(), edge));
    base.chain(extra)  // zero allocation, just iterator composition
}
```

### 2. DynamicContext via PyO3 `#[pyclass]`

Avoid PyDict → HashMap conversion on every call — use native PyO3 type:

```rust
#[pyclass]
pub struct DynamicContext {
    #[pyo3(get, set)]
    pub semantic_boosts: HashMap<String, f32>,
    #[pyo3(get, set)]
    pub overlay_nodes: Vec<NodeData>,
    #[pyo3(get, set)]
    pub overlay_edges: Vec<(String, String, EdgeData)>,
    #[pyo3(get, set)]
    pub weight_overrides: HashMap<String, f32>,
}
```

### 3. GIL Release + Explicit `.close()`

Release Python's GIL during pure Rust computation. Implement `.close()` for deterministic memory release (don't rely on GC):

```rust
#[pyclass]
pub struct OrpheusGraph {
    inner: Option<DiGraph<NodeData, EdgeData>>,  // Option for take()
}

#[pymethods]
impl OrpheusGraph {
    fn beam_traverse(&self, py: Python, start: &str, k: usize, depth: usize, ctx: &DynamicContext) -> PyResult<Vec<NodeResult>> {
        let graph = self.inner.as_ref().ok_or(PyValueError::new_err("Graph closed"))?;
        Ok(py.allow_threads(|| {
            inner_beam_traverse(graph, start, k, depth, ctx)
        }))
    }

    /// Deterministic drop — don't wait for Python GC
    fn close(&mut self) {
        self.inner.take();  // drops DiGraph, frees Rust memory immediately
    }
}
```

Python usage:
```python
# Context manager for safety:
async with get_graph("odoo", "18.0", db) as graph:
    results = graph.beam_traverse(...)
# graph.close() called automatically on exit

# Or explicit in invalidate():
old = _L1_CACHE.pop(key, None)
if old:
    old.close()  # Rust memory freed NOW
```

> [!TIP]
> `py.allow_threads()` is critical for Celery prefork workers — without it, long `contextual_subgraph` calls would block the event loop.

---

## Python API

```python
import orpheusgraph

# ── Build ────────────────────────────────────────
# Pass raw data from your DB/API — orpheusgraph builds petgraph internally
graph = orpheusgraph.build_graph(
    nodes=[
        {"name": "sale.order", "kind": "model", "base_weight": 0.7, "noise_penalty": 0.0},
        {"name": "res.partner", "kind": "model", "base_weight": 0.8, "noise_penalty": 0.0},
        {"name": "create_uid", "kind": "field", "base_weight": 0.1, "noise_penalty": 0.9},
    ],
    edges=[
        {"from": "sale.order", "to": "res.partner", "kind": "relates_to", "field": "partner_id"},
    ],
)

# ── Cache ────────────────────────────────────────
data = graph.to_rkyv()               # bytes → store in Redis
graph = orpheusgraph.from_rkyv(data)  # restore (zero-copy, immutable)

# ── Traverse ─────────────────────────────────────
ctx = orpheusgraph.DynamicContext(
    semantic_boosts={"stock.picking": 2.5, "stock.move": 2.0},
    overlay_nodes=[{"name": "x_custom", "kind": "model", "base_weight": 0.5}],
    overlay_edges=[{"from": "stock.picking", "to": "x_custom", "kind": "relates_to"}],
    weight_overrides={"sale.order": 0.9},
)

results = graph.beam_traverse("sale.order", k=5, depth=3, ctx=ctx)
path = graph.find_path("res.partner", "stock.picking", ctx=ctx)
subgraph = graph.contextual_subgraph(ctx, k=30)

# ── Inspect ────────────────────────────────────────────
print(graph.node_count())   # 50000
print(graph.edge_count())   # 180000
print(graph.get_node("sale.order"))  # NodeResult

# ── Directional Edge Inspection ───────────────────────
out = graph.outgoing_edges("sale.order")   # [EdgeResult(target="res.partner", kind="relates_to", field="partner_id")]
in_ = graph.incoming_edges("sale.order")   # [EdgeResult(source="stock.picking", kind="relates_to", field="sale_id")]
```

---

## Build & Publish

```rust
// EdgeResult — lightweight, returned to Python
#[pyclass]
pub struct EdgeResult {
    #[pyo3(get)] pub source: String,
    #[pyo3(get)] pub target: String,
    #[pyo3(get)] pub kind: String,
    #[pyo3(get)] pub field_name: Option<String>,
    #[pyo3(get)] pub weight: f32,
}
```

---

## Build & Publish

```bash
# Development
cd orpheusgraph
maturin develop          # Install locally for testing

# Release
maturin build --release  # → orpheusgraph-0.1.0-cp311-*.whl
twine upload dist/*      # → PyPI

# Docker (multi-stage)
FROM rust:1.77 AS rust-builder
WORKDIR /orpheusgraph
COPY orpheusgraph/ .
RUN pip install maturin && maturin build --release

FROM python:3.11-slim
COPY --from=rust-builder /orpheusgraph/target/wheels/*.whl /tmp/
RUN pip install /tmp/orpheusgraph-*.whl
```

---

## OSDS Integration Example

How the generic `orpheusgraph` maps to OSDS ERP domain:

```python
# server/app/core/graph_cache.py — 3-tier cache with generation counter

_L1_CACHE: dict[str, orpheusgraph.Graph] = {}  # in-process, per-worker
_L1_GEN: dict[str, int] = {}             # tracks generation per key

async def get_graph(
    erp: str, ver: str, db: AsyncSession,
    pin_generation: int | None = None,     # pipeline snapshot support
) -> orpheusgraph.Graph:
    key = f"graph:{erp}:{ver}"
    gen_key = f"graph_gen:{erp}:{ver}"

    # Check generation — stale L1 across workers
    current_gen = int(await redis.get(gen_key) or 0)
    if pin_generation and current_gen != pin_generation:
        current_gen = pin_generation  # pipeline uses its own snapshot

    # L1: in-process Rust memory (<0.1ms)
    if key in _L1_CACHE and _L1_GEN.get(key) == current_gen:
        return _L1_CACHE[key]

    # L2: Redis (shared, lz4-compressed rkyv)
    cached = await redis.get(key)
    if cached:
        raw = lz4.frame.decompress(cached)
        _L1_CACHE[key] = orpheusgraph.OrpheusGraph.from_rkyv(raw)
        _L1_GEN[key] = current_gen
        return _L1_CACHE[key]

    # L3: rebuild from PG — BLPOP coordination (not polling)
    lock = await redis.set(f"lock:{key}", "1", nx=True, ex=30)
    if not lock:
        result = await redis.blpop(f"graph_signal:{key}", timeout=30)
        if result:
            cached = await redis.get(key)
            if cached:
                raw = lz4.frame.decompress(cached)
                _L1_CACHE[key] = orpheusgraph.OrpheusGraph.from_rkyv(raw)
                _L1_GEN[key] = current_gen
                return _L1_CACHE[key]
        await redis.delete(f"lock:{key}")  # timeout — force rebuild

    nodes, edges = await _load_from_db(db, erp, ver)
    graph = orpheusgraph.build_graph(nodes=nodes, edges=edges)
    compressed = lz4.frame.compress(graph.to_rkyv())
    await redis.set(key, compressed, ex=86400)
    await redis.lpush(f"graph_signal:{key}", "ready")  # wake waiters
    await redis.delete(f"lock:{key}")
    _L1_CACHE[key] = graph
    _L1_GEN[key] = current_gen
    return graph

async def invalidate(erp: str, ver: str):
    """Bumps generation → all L1 caches go stale."""
    key = f"graph:{erp}:{ver}"
    await redis.delete(key)
    await redis.incr(f"graph_gen:{erp}:{ver}")
    old = _L1_CACHE.pop(key, None)
    _L1_GEN.pop(key, None)
    if old:
        old.close()  # deterministic Rust memory release
```

```python
# ai/tools.py

@tool
async def traverse_erp_graph(entity_name: str, version: str = "18.0") -> str:
    """Graph traversal with project-aware context."""
    # 1. Get graph (L1 → L2 → L3 cascade, pinned to pipeline generation)
    graph = await get_graph(
        "odoo", version, db,
        pin_generation=state.get("graph_gen"),  # snapshot: consistent within pipeline
    )

    # 2. Build ephemeral context (never stored)
    ctx = orpheusgraph.DynamicContext(
        semantic_boosts=await compute_semantic_boosts(state["transcript"]),
        overlay_nodes=await get_custom_fields(state["project_id"]),
        weight_overrides=await get_project_usage_stats(state["project_id"]),
    )

    # 3. Beam Search (Rust, <1ms, immutable graph)
    results = graph.beam_traverse(entity_name, k=5, depth=3, ctx=ctx)
    return format_for_llm(results, graph)
```

```python
# ai/utils/graph_format.py — LLM-optimized Markdown output

def format_for_llm(results: list, graph) -> str:
    lines = ["# GRAPH CONTEXT (Top-K relevant nodes)\n"]

    for node in results:
        lines.append(f"## [{node.kind.upper()}] {node.name}")
        lines.append(f"**Relevance:** {node.weight:.1f}")
        score = node.explain_score()
        lines.append(f"*base={score['base']:.1f} sem={score['semantic']:.1f} noise={score['noise']:.1f}*")

        # Outgoing
        out = graph.outgoing_edges(node.name)
        if out:
            lines.append("\n### OUTGOING:")
            for e in out:
                field = f" (field: {e.field_name})" if e.field_name else ""
                lines.append(f"- `{node.name}` --[{e.kind}]--> `{e.target}`{field}")

        # Incoming
        inc = graph.incoming_edges(node.name)
        if inc:
            lines.append("\n### INCOMING:")
            for e in inc:
                field = f" (field: {e.field_name})" if e.field_name else ""
                lines.append(f"- `{e.source}` --[{e.kind}]--> `{node.name}`{field}")

        lines.append("\n---")

    return "\n".join(lines)
```

```python
# ai/agents/graph.py — pipeline startup: pin graph generation

async def init_pipeline(state: AgentState) -> AgentState:
    """Snapshot graph generation at pipeline start for consistency."""
    erp = state.get("erp_system", "odoo")
    ver = state.get("erp_version", "18.0")
    state["graph_gen"] = int(await redis.get(f"graph_gen:{erp}:{ver}") or 0)
    return state
```
```

---

## Performance Targets

| Operation | Target | Graph size | Note |
|---|---|---|---|
| `build_graph` (from rows) | <10ms | 50K nodes | |
| `to_rkyv` | <1ms | 50K nodes | |
| `from_rkyv` (zero-copy) | **<0.1ms** | 50K nodes | Memory-mapped, no allocation |
| `beam_traverse(k=5, d=3)` | <1ms | 50K nodes | |
| `find_path` | <0.5ms | 50K nodes | |
| `contextual_subgraph(k=30)` | <2ms | 50K nodes | |
| Memory (base graph) | ~15MB | 50K nodes | |
| Memory (DynamicContext) | ~50KB | typical overlay | semantic_boosts capped at top-200 |

---

## Risks & Mitigations

### 1. 🔴 rkyv Segfault — Python GC vs Rust pointers

`rkyv` zero-copy casts pointers directly onto raw bytes. If Python GC collects the `bytes` object from `redis.get()` while Rust still references it → **segfault**.

**Fix**: Pin the byte buffer inside `OrpheusGraph`:
```rust
#[pyclass]
pub struct OrpheusGraph {
    _pinned_bytes: Py<PyBytes>,  // prevent Python GC from collecting
    inner: Option<&'static ArchivedGraph>,  // points into _pinned_bytes
}

#[pymethods]
impl OrpheusGraph {
    #[staticmethod]
    fn from_rkyv(py: Python, data: &PyBytes) -> PyResult<Self> {
        let pinned: Py<PyBytes> = data.into_py(py);  // prevent GC
        let archived = unsafe { rkyv::archived_root::<Graph>(data.as_bytes()) };
        Ok(Self { _pinned_bytes: pinned, inner: Some(archived) })
    }
}
```

### 2. Thundering Herd — BLPOP over Pub/Sub

Redis Pub/Sub is fire-and-forget — network blip = missed event = worker sleeps forever. **Fix**: `BLPOP` with timeout:
```python
# Builder: push signal when done
await redis.lpush(f"graph_signal:{erp}:{ver}", "ready")

# Waiters: blocking pop with timeout
result = await redis.blpop(f"graph_signal:{erp}:{ver}", timeout=30)
if not result:
    # Timeout — force rebuild
    ...
```

### 3. semantic_boosts — cap at top-N

`MAX_SEMANTIC_BOOSTS = 200` before passing to Rust via FFI.

### 4. L1 Memory — Drop trait as safety net

`.close()` + context manager = happy path. But if `.close()` is forgotten, Rust `Drop` trait ensures cleanup:
```rust
impl Drop for OrpheusGraph {
    fn drop(&mut self) {
        self.inner.take();  // frees graph even if .close() was never called
    }
}
```

### 5. FFI Return Overhead

Return lightweight `#[pyclass]` structs, not dicts. Heavy metadata fetched lazily from PG:
```rust
#[pyclass]
pub struct NodeResult {
    #[pyo3(get)] pub name: String,
    #[pyo3(get)] pub kind: String,
    #[pyo3(get)] pub weight: f32,
    // metadata: fetch by UUID from PG only for nodes that go to prompt
}
```

### 6. Domain-aware noise_penalty

`noise_tags` in DynamicContext — `{"technical"}` for sales, `set()` for audit.

### 7. L2 Network — graph size ~15-20MB

Multiple graph versions × frequent cold starts = network bottleneck. **Fix**: `lz4` compression before storing in Redis:
```python
# Store: lz4 compress (~4x ratio) → ~4MB over wire
await redis.set(key, lz4.frame.compress(graph.to_rkyv()), ex=86400)

# Load:
raw = lz4.frame.decompress(await redis.get(key))
graph = orpheusgraph.OrpheusGraph.from_rkyv(raw)
```

### 8. God Objects — degree cutoff

`res.partner`, `ir.attachment`, `mail.message` — сверхсвязанные ноды с 1000+ edges. Beam Search потратит весь `k` на оценку их рёбер. **Fix**: `max_fan_out` в DynamicContext:
```python
ctx = DynamicContext(max_fan_out=50)  # skip nodes with degree > 50 unless boosted
```

Rust: в `neighbors_with_overlay()` — если `degree(node) > max_fan_out && !semantic_boosts.contains(node)` → skip.

### 9. L3 Build Error — error marker

Если `_load_from_db` или `build_graph` падает с exception, лок протухнет через 30s → все waiters пойдут rebuild → штурм PG. **Fix**: error marker:
```python
try:
    nodes, edges = await _load_from_db(db, erp, ver)
    graph = orpheusgraph.build_graph(nodes=nodes, edges=edges)
except Exception as e:
    await redis.set(f"build_error:{key}", str(e), ex=60)  # block retries for 1 min
    await redis.delete(f"lock:{key}")
    raise
```

### 10. `.explain_score()` — debug скоринга

Балансировка `w_*` коэффициентов — отдельный квест. **Fix**: метод `.explain_score()` на `NodeResult`:
```rust
#[pymethods]
impl NodeResult {
    fn explain_score(&self) -> HashMap<String, f32> {
        HashMap::from([
            ("base".into(), self.base_component),
            ("semantic".into(), self.semantic_component),
            ("noise".into(), self.noise_component),
            ("override".into(), self.override_component),
            ("total".into(), self.weight),
        ])
    }
}
```

### 11. L1 Warmup — прогрев при старте

Первый запрос после cold start = полный L2→L1 overhead. **Fix**: warmup task при старте worker:
```python
# server/app/main.py (lifespan)
async def lifespan(app):
    # Warmup main graph versions
    for ver in ["18.0", "17.0"]:
        try:
            await get_graph("odoo", ver, db)
            logger.info(f"L1 warmed: odoo:{ver}")
        except Exception:
            pass  # non-fatal, will load on first request
    yield
```

### 12. rkyv Schema Versioning

Layout change between orpheusgraph versions + old Redis blob → segfault. Cross-architecture (ARM vs x86) rkyv bytes are **incompatible**. **Fix**: schema version + arch in key:
```python
import platform
GRAPH_SCHEMA_VERSION = "v1"  # bump on NodeData/EdgeData struct change
key = f"graph:{GRAPH_SCHEMA_VERSION}:{platform.machine()}:{erp}:{ver}"
```

### 13. God Objects — pagerank fallback

`max_fan_out` can break the only path between nodes. **Fix**: `pagerank_weight: f32` in `NodeData` (static, computed at build). High-pagerank God Objects pass through cutoff even without semantic boost.

### 14. Heavy Overlay Cache

2000+ custom fields = FFI overhead per call. **Fix**: cache overlay Rust-side per `project_id`:
```python
ctx = DynamicContext(overlay_cache_key=f"project:{project_id}")
# Rust: if overlay_cache_key matches previous call → reuse, skip FFI transfer
```

### 15. Partial Invalidation (Future)

50K nodes → full rebuild <10ms, fine. Millions → not fine. **Note for future**: subgraph versioning by module. Currently overkill, foundation exists via generation counter.

### 16. SIGKILL during L3 build (**must-have**)

`except Exception` won't catch OOM kill / Rust panic. **Fix**: lock renewal watchdog — background `asyncio.Task` extends lock TTL every 3s while build runs. If process dies, 30s TTL expires naturally, next worker takes over.

### 17. Multiplicative noise (**v1, not future**)

Linear subtraction is unsafe — high `semantic_boost` can override noise. **v1 formula**:
```
W = (w_base * base_weight + w_semantic * semantic_boost + w_override * override) * (1.0 - noise_penalty)
```
With `noise_penalty = 0.9`, even boosted nodes get 10% of their score → never in top-K.

### 18. Drop ordering with unsafe rkyv

`_pinned_bytes` MUST outlive `inner` (which points into it). Rust drops fields in declaration order, so `inner: Option<...>` must be declared **before** `_pinned_bytes: Py<PyBytes>` to ensure correct drop order.

### 19. PageRank is mandatory for God Object cutoff

Confirming: risk #8 (`max_fan_out`) is **unsafe without** risk #13 (`pagerank_weight`). Both must ship together.

### 20. L1 LRU Eviction

Long-lived pipelines + frequent graph updates → `_L1_CACHE` accumulates stale generations. **Fix**: `functools.lru_cache` or manual LRU with `maxsize=3` (current + 2 previous versions per erp:ver key).

