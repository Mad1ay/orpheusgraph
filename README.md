# orpheusgraph

[![CI](https://github.com/Mad1ay/orpheusgraph/actions/workflows/ci.yml/badge.svg)](https://github.com/Mad1ay/orpheusgraph/actions/workflows/ci.yml)
![License: PolyForm NC 1.0.0](https://img.shields.io/badge/license-PolyForm%20NC%201.0.0-blue)
![Status: v0.1.0](https://img.shields.io/badge/status-v0.1.0%20(early)-orange)

> Rust library with optional Python bindings for context-aware weighted graph traversal.
> Source-available. Domain-agnostic. Built for RAG pipelines that need deterministic structure.

**Status:** v0.1.0 — early and source-available; the API may still change.

## Why

Most RAG pipelines retrieve by vector similarity alone, which ignores how entities
actually relate. orpheusgraph adds the missing structural half: it walks a weighted
knowledge graph so retrieval follows real relationships, penalizes noisy low-signal
nodes, and stays deterministic and sub-millisecond on the hot path. Unlike a general
graph library (networkx), it is built for per-request scoring and multi-tenant isolation
rather than one-off analysis; unlike a graph database (neo4j), it is an embeddable
in-process library with no server to run — it sits inside your retrieval pipeline and
returns the Top-K relevant nodes in microseconds.

The mechanism worth knowing before anything else: **the caller supplies the objective
function, per request, without writing to the store.** A `DynamicContext` is an ordinary
function argument, never persisted, and it can add a per-request bonus to any node
(`weight_bonuses`), add an embedding-derived bonus (`semantic_boosts`), suppress a whole
class of nodes by metadata
domain (`noise_tags`), cut hubs (`max_fan_out`), inject nodes and edges that do not exist in
the store (`overlay_nodes`/`overlay_edges`), view the graph at a past instant (`as_of`) and
restrict it to what the caller may see (`principals`). The score is computed *inside* the
expansion loop and decides which nodes are expanded next, so it steers the walk rather than
re-ranking its output. The next request sees none of it — which is why tenant isolation here
is a consequence of the API shape, not a feature bolted on.

The flip side is in [Retrieval quality](#retrieval-quality--what-the-pruning-costs): the
engine is only as good as the objective you hand it.

## What It Does

1. **Build** weighted knowledge graphs from any structured data
2. **Cache** via 3-tier: L1 in-process → L2 Redis → L3 source DB
3. **Traverse** with context-aware Beam Search — Top-K relevant nodes
4. **Isolate** tenants via virtual overlay — zero shared mutable state

## Quick Start

The Python bindings are **opt-in** behind the `python` feature (default off), so a
pure-Rust consumer uses orpheusgraph as a plain rlib — no PyO3 compiled, no Python
toolchain required.

### Rust

```toml
[dependencies]
orpheusgraph = { git = "https://github.com/Mad1ay/orpheusgraph" }
```

```bash
cargo build --lib   # zero pyo3 crates
cargo test
```

### Python

```bash
# Prerequisites: Rust toolchain, CPython 3.10-3.13, maturin
pip install maturin

# Development build - maturin passes --features python via pyproject.toml
cd orpheusgraph
maturin develop

# Verify
python -c "import orpheusgraph; print('OK')"
```

PyO3 0.23 supports CPython up to 3.13. On a newer interpreter the build additionally
needs `PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1` and is untested.

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
    weight_bonuses={"sale.order": 0.9},
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

> The example uses ERP model names (`sale.order`, `res.partner`) for familiarity, but
> nodes and edges are arbitrary — orpheusgraph treats them as opaque strings and is fully
> domain-agnostic.

## Python API — Persistent Store

For workloads that need a durable, crash-safe graph instead of an ephemeral per-request
build, `orpheusgraph` also exposes `PersistentGraph`: an immutable base plus a WAL-backed
mutable delta, with optimistic CAS, a CSR mmap snapshot, and compaction. The read API
(`beam_traverse`, `find_path`, `contextual_subgraph`, …) is identical to `OrpheusGraph`.

```python
import orpheusgraph as og

# Open or create a durable store on disk
g = og.open("./graph_dir", create=True)

# Mutate: one call = one atomic WAL frame (all-or-nothing)
seq = g.apply([
    {"op": "upsert_node", "name": "sale.order", "kind": "model", "base_weight": 0.6},
    {"op": "upsert_node", "name": "res.partner", "kind": "model", "base_weight": 0.9},
    {"op": "add_edge", "from": "sale.order", "to": "res.partner",
     "kind": "relates_to", "field": "partner_id", "base_weight": 0.8},
])

# Optimistic CAS for read-modify-write: pass expected_seq, retry on conflict
try:
    g.apply([{"op": "remove_node", "name": "res.partner"}], expected_seq=0)  # stale
except og.ConflictError:
    g.apply([{"op": "remove_node", "name": "res.partner"}], expected_seq=g.seq)  # re-read + retry

# Traverse — same API as OrpheusGraph, over the live base+delta view
ctx = og.DynamicContext()
results = g.beam_traverse("sale.order", k=5, depth=3, ctx=ctx)

# Durability + compaction
g.flush()      # fsync WAL per the fsync policy
g.compact()    # fold delta into a new base snapshot (also auto-fires by threshold)
g.close()      # flush + mark clean shutdown; idempotent

# Context-manager form
with og.open("./graph_dir") as g:
    g.apply([{"op": "upsert_node", "name": "extra", "base_weight": 0.3}])

# Seed a new store in one pass (instead of replaying through the delta)
with og.create_persistent(
    "./graph_dir_seed",
    nodes=[{"name": "a", "base_weight": 0.5}, {"name": "b", "base_weight": 0.5}],
    edges=[{"from": "a", "to": "b", "kind": "relates_to", "base_weight": 1.0}],
) as seeded:
    print(seeded.node_count(), seeded.edge_count())
```

> `orpheusgraph.open(dir, create=False, ...)` raises `FileNotFoundError` for a missing
> store; `create_persistent` refuses to clobber an existing one. Weights passed to `apply`
> must already be normalized to `[0.0, 1.0]` — an out-of-range value raises `ValueError`.

> The on-disk store is `format_version` **3** (a CSR snapshot with a persisted `name -> idx`
> index for O(1) warm-open, plus a fsync-gated COMMIT marker). Stores written by an older
> build (0/1/2) are not auto-migrated — `open()` raises `CorruptError`; recreate the store.

> **Durability contract.** `apply()` returning normally ⇒ the batch is durable. `apply()`
> raising ⇒ durability is **indeterminate**: the batch may or may not survive a reopen, but
> recovery always yields a consistent committed prefix and never loses an already-acknowledged
> batch. Reconcile a failed `apply` by reopening and reading `.seq` (the recovered committed
> prefix) — do **not** blindly retry a non-idempotent batch. `OnFlush` (the default fsync
> policy) trades post-`flush()` tail durability for speed; use `EveryBatch` for per-batch fsync.

### Temporal validity & edge ACL (query-time)

Edges may carry `valid_from`/`valid_to` (half-open `[from, to)`, `None` = unbounded) and `acl`
(a list of tags; empty = public). A `DynamicContext` may carry `as_of` (an instant; `None` =
no temporal filter) and `principals` (tags the caller holds). A traversal shows an edge only if
it is temporally valid at `as_of` **and** ACL-visible (public, or `acl ∩ principals ≠ ∅`; empty
`principals` sees only public edges — fail-closed). Filtering is query-time only: PageRank and
all base metrics are computed over every edge, and the raw `.outgoing_edges`/`.incoming_edges`
inspectors return unfiltered adjacency (do not surface them under an access-control assumption).

```python
edges = [{"from": "a", "to": "b", "kind": "relates_to", "base_weight": 1.0,
          "valid_from": 100, "valid_to": 200, "acl": ["finance"]}]
ctx = og.DynamicContext(as_of=150, principals=["finance"])   # sees the edge
ctx = og.DynamicContext(as_of=250)                            # expired -> hidden
ctx = og.DynamicContext(as_of=150)                            # no principals -> hidden (tagged)
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

### `orpheusgraph.open(dir, create=False, mmap=True, validate="full", prefault=False) → PersistentGraph`
Open a durable store, or (`create=True`) initialize an empty one. `create=False` on a
missing store raises `FileNotFoundError`. `validate` ∈ `"full"` (default, mandatory for
untrusted/shared stores), `"crc"`, `"none"`.

### `orpheusgraph.create_persistent(dir, nodes, edges) → PersistentGraph`
Create a new store, seeding its immutable base directly from `build_graph`-style node/edge
dicts in one pass (instead of replaying them through the delta). Refuses to clobber an
existing store.

### `PersistentGraph`
Durable delta store: an immutable base + a crash-safe mutable delta. Read methods mirror
`OrpheusGraph` and run over the live base+delta view; the GIL is released during traversal
and during `apply`/`flush`/`compact`/`close`.

| Member | Description |
|---|---|
| `.seq` | Durable commit sequence — the CAS token / cache generation |
| `.epoch` | Incarnation id (changes across an unclean reopen) |
| `.apply(ops, expected_seq=None)` | Atomic batch = one WAL frame; returns the new `seq`. Ops are dicts keyed by `"op"`: `upsert_node`, `remove_node`, `add_edge`, `remove_edge`. Weights must be pre-normalized to `[0,1]` (`ValueError` otherwise) |
| `.flush()` | Durability point (fsync WAL per policy) |
| `.compact()` | Fold delta into a new base snapshot; also auto-fires by threshold |
| `.set_fsync_policy(policy, every_n=None)` | `on_flush` \| `every_batch` \| `every_n` |
| `.set_auto_compact_threshold(n)` | Override the auto-compaction op threshold |
| `.close()` | Flush + mark clean shutdown; idempotent |
| context manager | `with orpheusgraph.open(...) as g:` calls `.close()` on exit |

Optimistic **CAS**: pass `expected_seq` to `apply()`; if the store advanced past it,
`orpheusgraph.ConflictError` is raised and nothing is written — re-read `.seq` and retry.

### Exceptions

| Exception | Raised when |
|---|---|
| `orpheusgraph.ConflictError` | `apply(expected_seq=...)` loses the CAS race |
| `orpheusgraph.CorruptError` | Structural corruption or an unsupported on-disk format version |
| `FileNotFoundError` | `open(dir, create=False)` on a missing store |

### `DynamicContext`
Ephemeral per-request context. Never stored. All parameters optional:

| Parameter | Default | Description |
|---|---|---|
| `semantic_boosts` | `{}` | node → **additive** bonus, entering the score as `w_semantic × value` (not a multiplier) |
| `weight_bonuses` | `{}` | node → **additive** bonus, entering as `w_bonus × value`. It does **not** replace the node's base term: the node keeps `w_base × base_weight` and gains the bonus on top (note `base_weight` is the build-time *normalized* value, not the number you passed to `build_graph`) |
| `noise_tags` | `{}` | domain tags to penalize. Matched against `metadata["domain"]` **only** — that key is hardcoded, so tagging under any other key is a silent no-op |
| `max_fan_out` | `None` | degree cutoff for God Objects. Not a hard bound — see below |
| `fan_out_pagerank_bypass` | `0.5` | `pagerank_weight` above which a node escapes `max_fan_out`; `None` removes the escape |
| `w_base` | `1.0` | base weight coefficient |
| `w_semantic` | `1.5` | semantic boost coefficient |
| `w_noise` | `1.0` | noise penalty coefficient |
| `w_bonus` | `1.0` | weight override coefficient |
| `overlay_nodes` | `[]` | virtual tenant-specific nodes |
| `overlay_edges` | `[]` | virtual tenant-specific edges |

### Scoring Formula

```
raw             = (w_base × base_weight)
                + (w_semantic × semantic_boost)
                + (w_bonus × override)

effective_noise = clamp(w_noise × noise_penalty, 0.0, 1.0)
                  # raised to at least 0.9 when metadata["domain"] ∈ noise_tags

W_total         = raw × (1.0 - effective_noise)
```

All three numerator terms are **added**; none replaces `base_weight`. `w_noise` scales the
penalty before clamping, so it only drops out of the formula at its default of `1.0`.

**`max_fan_out` is not a bound on expanded degree.** A node escapes the cutoff if it carries
a *positive* `semantic_boosts` entry or if its `pagerank_weight` exceeds
`fan_out_pagerank_bypass`, and overlay edges are never subject to it at all — the caller
injected those for this request, so they are not discarded. Set
`fan_out_pagerank_bypass=None` to remove the PageRank escape. Note the threshold is absolute
while PageRank's spread is graph-dependent: on a flat-degree graph nothing clears `0.5` and
the escape never fires, while on a hub-heavy graph many nodes clear it and `max_fan_out` does
nothing. Pick it from your own graph's distribution, the same way you pick `k`.

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

### Retrieval quality — what the pruning costs

`beam_traverse(k, depth)` keeps only the k best-scoring nodes per level, so a node that
would score high at depth 3 is lost when its depth-1 ancestor misses the cut. Measured
against exhaustive traversal (`k = node_count`, which prunes nothing) scored by the same
formula — synthetic social graph, 10K nodes / 113K edges, ~1,280 nodes reachable at
depth 3, 24 seeds, ties broken deterministically by name. Recall is a set overlap and so
hardware-independent; the cost ratios quoted below were measured on a different machine
than the latency table above, so compare them with each other, not with those absolutes:

| k | recall@5, score correlates with structure | recall@5, score uncorrelated |
|---|---|---|
| **5** | **0.48** | **0.09** |
| 10 | 0.72 | 0.13 |
| 20 | 0.87 | 0.21 |
| 50 | 0.98 | 0.48 |
| exhaustive | 1.00 | 1.00 |

Two consequences, both more important than the latency table above.

**Pruning only pays when your score signal correlates with graph locality.** The left column
scores nodes by follower count, which is structural by construction; the right column scores
by an attribute unrelated to the edges. With an uncorrelated signal the beam has nothing to
follow, reaching the same recall needs k ≈ 500 — about 86% of the cost of just traversing
exhaustively — and randomly placed `semantic_boosts` measure slightly *worse* than no boosts
at all. If your boosts come from embeddings and the relevant nodes are scattered rather than
clustered, prefer a large k or the exhaustive path, and do not assume a small beam is a free
optimisation.

**Choose k as a fraction of the reachable set, not as an absolute.** k does not transfer
between graphs: k=100 is 7.8% of the reachable set on the graph above and 1.9% on a
100K-node / 2.4M-edge one. With a structural signal the economics improve with scale — on
the larger graph recall@10 reaches 1.00 at k=100, 4.6x faster than exhaustive. The `k=5` used
in this README's examples is deliberately minimal for readability and sits far inside the
lossy regime; it is not a recommended default.

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
├── pybridge.rs       # PyO3 Python bindings (feature `python`)
└── lib.rs            # Module registration
```

## What It Is Not

orpheusgraph is an index, not a database, and the difference is load-bearing:

- **No query language and no planner.** Traversals are function calls.
- **No aggregation.** There is no `count`/`avg`/`GROUP BY`/`ORDER BY` over the graph.
- **No node enumeration.** The read surface is `get_node(name)`, `outgoing_edges(name)`,
  `incoming_edges(name)` and traversals from a seed. Nothing scans all nodes of a kind, so
  queries shaped like "every Person, grouped by city" are out of scope by construction.
- **Not the source of truth.** It indexes data owned elsewhere; rebuild it, do not migrate it.
- **Not a ledger.** No invariants, no cross-record constraints, no replication.

If you need those, you need a graph database. This library trades them for a microsecond
hot path, per-request scoring and determinism.

## License

`orpheusgraph` is **source-available** under the
[PolyForm Noncommercial License 1.0.0](LICENSE.md):

- ✅ **Free** for any **noncommercial** use — personal projects, research,
  education, evaluation, and noncommercial organizations.
- 💼 **Commercial use requires a commercial license.** If you use
  `orpheusgraph` in a product, service, or for-profit operation, see
  [COMMERCIAL.md](COMMERCIAL.md) or contact <kseniabezobiuk@gmail.com>.

Contributions are accepted under the [Contributor License Agreement](CLA.md).
