# orpheusgraph

> Rust library with Python bindings for context-aware weighted graph traversal.
> Source-available. Domain-agnostic. Built for RAG pipelines that need deterministic structure.

## What It Does

1. **Build** weighted knowledge graphs from any structured data
2. **Cache** via 3-tier: L1 in-process → L2 Redis → L3 source DB
3. **Traverse** with context-aware Beam Search — Top-K relevant nodes
4. **Isolate** tenants via virtual overlay — zero shared mutable state

## Quick Start

```bash
# Prerequisites: Rust toolchain, Python 3.11+, maturin
pip install maturin

# Development build
cd orpheusgraph
maturin develop

# Verify
python -c "import orpheusgraph; print('OK')"
```

## Usage

```python
import orpheusgraph

# Build
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

# Traverse
ctx = orpheusgraph.DynamicContext(
    semantic_boosts={"sale.order": 2.0},
    weight_overrides={"sale.order": 0.9},
)
results = graph.beam_traverse("sale.order", k=5, depth=3, ctx=ctx)
path = graph.find_path("sale.order", "res.partner", ctx=ctx)
subgraph = graph.contextual_subgraph(ctx, k=30)

# Inspect
print(graph.node_count(), graph.edge_count())
print(graph.outgoing_edges("sale.order"))

# Serialize (for Redis cache)
data = graph.to_rkyv()
graph2 = orpheusgraph.from_rkyv(data)

# Cleanup
graph.close()
```

## API Reference

### `build_graph(nodes, edges) → OrpheusGraph`
Build an immutable graph. Normalizes weights and computes PageRank.

### `OrpheusGraph`
| Method | Description |
|---|---|
| `.beam_traverse(start, k, depth, ctx)` | Top-K pruned BFS |
| `.find_path(start, end, ctx)` | Weighted Dijkstra shortest path |
| `.contextual_subgraph(ctx, k)` | Extract k most relevant nodes + neighbors |
| `.node_count()` / `.edge_count()` | Graph size |
| `.get_node(name)` | Look up a node |
| `.outgoing_edges(name)` / `.incoming_edges(name)` | Edge inspection |
| `.to_rkyv()` | Serialize to bytes (for Redis) |
| `.close()` | Deterministic memory release |

### `DynamicContext`
Ephemeral per-request context. Never stored. All parameters optional:

| Parameter | Default | Description |
|---|---|---|
| `semantic_boosts` | `{}` | node → multiplier (from embeddings) |
| `weight_overrides` | `{}` | node → weight override |
| `noise_tags` | `{}` | domain tags to penalize (e.g. `{"technical"}`) |
| `max_fan_out` | `None` | degree cutoff for God Objects |
| `w_base` | `1.0` | base weight coefficient |
| `w_semantic` | `1.5` | semantic boost coefficient |
| `w_noise` | `1.0` | noise penalty coefficient |
| `w_override` | `1.0` | weight override coefficient |
| `overlay_nodes` | `[]` | virtual tenant-specific nodes |
| `overlay_edges` | `[]` | virtual tenant-specific edges |
| `overlay_cache_key` | `None` | cache key for overlay data (per project) |

### Scoring Formula

```
raw = (w_base × base_weight) + (w_semantic × semantic_boost) + (w_override × override)
W_total = raw × (1.0 - noise_penalty)
```

## Benchmarks

Measured with [criterion](https://github.com/bheisler/criterion.rs) on a synthetic graph of
**50K nodes / ~150K edges** ([benches/bench_traversal.rs](benches/bench_traversal.rs)).
Hardware: Intel Core i7-13650HX, 16 GB RAM, Windows 11, rustc 1.94, release build with LTO.

**Hot path** — runs on every request:

| Operation | Median time |
|---|---|
| `beam_traverse(k=5, depth=3)` | **6.8 µs** |
| `contextual_subgraph(k=30)` | **47.6 µs** |
| `find_path` (weighted Dijkstra) | 40.4 ms ¹ |

**Cold path** — runs once per cache fill:

| Operation | Median time |
|---|---|
| `build_graph` (incl. PageRank) | 48.1 ms |
| `to_rkyv` | 28.0 ms |
| `from_rkyv` (zero-copy view + name index) | 5.6 ms |

¹ `find_path` currently keeps its Dijkstra frontier in string-keyed hash maps, and the
benchmark topology (long-range shortcut edges) forces a wide frontier — worst case, not
typical. Moving the frontier to node indices is on the roadmap.

Reproduce with `cargo bench`.

## Architecture

```
src/
├── types.rs          # NodeData, EdgeData, DynamicContext, NodeResult, PathStep
├── builder.rs        # build_graph() + PageRank computation
├── graph.rs          # Immutable DiGraph wrapper
├── accessor.rs       # GraphAccessor trait (owned + archived)
├── scoring.rs        # Multiplicative noise scoring formula
├── overlay.rs        # Virtual overlay iterator + max_fan_out
├── traversal.rs      # beam_traverse, find_path, contextual_subgraph
├── serialization.rs  # rkyv zero-copy serialization
├── pybridge.rs       # PyO3 Python bindings + overlay cache
└── lib.rs            # Module registration
```

## License

`orpheusgraph` is **source-available** under the
[PolyForm Noncommercial License 1.0.0](LICENSE.md):

- ✅ **Free** for any **noncommercial** use — personal projects, research,
  education, evaluation, and noncommercial organizations.
- 💼 **Commercial use requires a commercial license.** If you use
  `orpheusgraph` in a product, service, or for-profit operation, see
  [COMMERCIAL.md](COMMERCIAL.md) or contact <kseniabezobiuk@gmail.com>.

Contributions are accepted under the [Contributor License Agreement](CLA.md).
