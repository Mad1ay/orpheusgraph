//! Persistence-path benchmarks (§6 of the persistence spec).
//!
//! These measure the perf claims the persistent store was designed around,
//! which were previously unverified:
//!
//!   * `apply(batch of 100)` — the durable write hot path.
//!   * warm `open()` over an mmap'd V2 CSR snapshot vs the legacy V0
//!     `from_rkyv` zero-copy view.
//!   * `DeltaAccessor` traversal overhead at delta fill 0 / 1% / 10% of base
//!     (targets: ~0 overhead at empty delta, <= 1.3x at 10%).
//!   * `beam_traverse` over the warm mmap CSR base vs the owned petgraph
//!     (target: <= 1.5x) and vs the legacy V0 archived view (the O(E) scan the
//!     CSR layout was meant to fix — expected to be the slow one here).
//!
//! Run: `cargo bench --bench bench_persist` (release). Read the ratios against
//! the targets above; criterion prints each group side by side.

use std::collections::HashMap;

use criterion::{criterion_group, criterion_main, BatchSize, Criterion};

use orpheusgraph::beam_traverse;
use orpheusgraph::builder::build_graph;
use orpheusgraph::graph::OrpheusGraphInner;
use orpheusgraph::serialization::{to_rkyv, ArchivedGraphView};
use orpheusgraph::types::{DynamicContext, EdgeData, EdgeInput, NodeData, NodeInput};
use orpheusgraph::{BaseMode, DeltaAccessor, GraphDelta, Op, PersistentGraph, Validate};

const N: usize = 50_000;
/// Traversal start node — a mid-graph vertex with a non-trivial neighborhood.
const START: &str = "node_25000";

// ---------------------------------------------------------------------------
// Shared graph construction (mirrors bench_traversal's topology: a chain plus
// two back-edges per node, ~3 edges/node).
// ---------------------------------------------------------------------------

fn build_inputs(n: usize) -> (Vec<NodeInput>, Vec<EdgeInput>) {
    let mut nodes: Vec<NodeInput> = Vec::with_capacity(n);
    let mut edges: Vec<EdgeInput> = Vec::with_capacity(n * 3);

    for i in 0..n {
        nodes.push(NodeInput {
            name: format!("node_{i}"),
            kind: "model".to_string(),
            metadata: HashMap::new(),
            base_weight: ((i % 100) as f32) + 1.0,
            noise_penalty: 0.0,
        });
        if i > 0 {
            edges.push(EdgeInput {
                from: format!("node_{}", i - 1),
                to: format!("node_{i}"),
                kind: "relates_to".to_string(),
                field_name: None,
                base_weight: 1.0,
                valid_from: None,
                valid_to: None,
                acl: Vec::new(),
            });
        }
        if i > 10 {
            edges.push(EdgeInput {
                from: format!("node_{i}"),
                to: format!("node_{}", i - 10),
                kind: "relates_to".to_string(),
                field_name: None,
                base_weight: 0.5,
                valid_from: None,
                valid_to: None,
                acl: Vec::new(),
            });
        }
        if i > 100 {
            edges.push(EdgeInput {
                from: format!("node_{i}"),
                to: format!("node_{}", i - 100),
                kind: "depends_on".to_string(),
                field_name: None,
                base_weight: 0.3,
                valid_from: None,
                valid_to: None,
                acl: Vec::new(),
            });
        }
    }

    (nodes, edges)
}

fn build_owned(n: usize) -> OrpheusGraphInner {
    let (nodes, edges) = build_inputs(n);
    let (g, m) = build_graph(nodes, edges);
    OrpheusGraphInner::new(g, m)
}

/// Build a delta holding `count` new nodes, each linked FROM an existing base
/// node spread across the whole id range (so some land on the traversal path).
fn build_delta(base: &OrpheusGraphInner, count: usize) -> GraphDelta {
    let mut delta = GraphDelta::new();
    if count == 0 {
        return delta;
    }
    let step = (N / count).max(1);
    let mut ops: Vec<Op> = Vec::with_capacity(count * 2);
    for i in 0..count {
        let dname = format!("delta_{i}");
        ops.push(Op::UpsertNode(NodeData {
            name: dname.clone(),
            kind: "model".to_string(),
            metadata: HashMap::new(),
            base_weight: 0.5,
            noise_penalty: 0.0,
            pagerank_weight: 0.0,
        }));
        ops.push(Op::AddEdge {
            from: format!("node_{}", (i * step) % N),
            to: dname,
            edge: EdgeData {
                kind: "relates_to".to_string(),
                field_name: None,
                base_weight: 0.7,
                valid_from: None,
                valid_to: None,
                acl: Vec::new(),
            },
        });
    }
    delta.apply(base, ops).expect("delta batch valid");
    delta
}

// ---------------------------------------------------------------------------
// 1. apply(batch of 100) — durable write hot path
// ---------------------------------------------------------------------------

fn bench_apply_batch(c: &mut Criterion) {
    let dir = tempfile::tempdir().expect("tempdir");
    let pg = PersistentGraph::create(dir.path(), build_owned(N)).expect("create");
    // Disable auto-compaction so we time apply, not the occasional fold.
    //
    // CAVEAT: nothing in the timed `b.iter()` closure below ever compacts or
    // truncates the WAL either, so across a full criterion run (thousands of
    // iterations, each appending a 100-op batch) the WAL grows ~linearly and
    // is never reset. The reported number is therefore an average over a
    // file that keeps growing across the run (hundreds of MB by the end for
    // 10^5+ iterations), not a steady-state per-apply cost at a fixed WAL
    // size — read it as an amortized figure, not steady-state.
    pg.set_auto_compact_threshold(Some(usize::MAX));

    // 100 upserts to fixed names => in-place updates, bounded delta.
    let batch: Vec<Op> = (0..100)
        .map(|i| {
            Op::UpsertNode(NodeData {
                name: format!("hot_{i}"),
                kind: "model".to_string(),
                metadata: HashMap::new(),
                base_weight: 0.5,
                noise_penalty: 0.0,
                pagerank_weight: 0.0,
            })
        })
        .collect();

    c.bench_function("apply batch=100 (50K base)", |b| {
        // iter_batched: the 100-op Vec clone (Strings + maps) runs in the setup
        // closure, OUTSIDE the timed routine — the measurement is the durable
        // write path only, not clone overhead.
        b.iter_batched(
            || batch.clone(),
            |batch| pg.apply(batch, None).expect("apply"),
            BatchSize::SmallInput,
        )
    });
}

// ---------------------------------------------------------------------------
// 2. Warm open: mmap V2 CSR vs legacy V0 from_rkyv
// ---------------------------------------------------------------------------

/// Warm-open decomposition at a given base size. Runs V2 mmap open under the
/// crc and full-validate modes against the legacy V0 zero-copy view, so the
/// cost breakdown (crc vs semantic sweep vs V0 bytecheck) is visible, and — by
/// running at two sizes — whether the V2/V0 gap is a fixed overhead or a
/// per-size penalty. (V2 opens do NO name-index build: the `name_buckets`
/// table is persisted in the snapshot and probed zero-copy.)
///
/// `Validate::None` is deliberately not benched here: `PersistentGraph::open_with`
/// unconditionally upgrades `None` to `Crc` for every public open (a soundness
/// guard — `None` reaches `rkyv::access_unchecked`, which is UB on untrusted
/// bytes), so a "none" row through this API would run byte-identical code to
/// the "crc" row, not an isolated index/setup-only measurement.
fn warm_open_group(c: &mut Criterion, n: usize) {
    let dir = tempfile::tempdir().expect("tempdir");
    PersistentGraph::create(dir.path(), build_owned(n))
        .expect("create")
        .close()
        .expect("close");
    let path = dir.path();

    let mut group = c.benchmark_group(format!("warm_open ({}K)", n / 1000));

    // crc32 over the whole snapshot file (the self-written fast path): mmap +
    // structural access + manifest + flock + WAL scan + crc, but NO semantic
    // sweep and no name-index build (the index is persisted, probed zero-copy).
    group.bench_function("open mmap V2 (crc)", |b| {
        b.iter(|| {
            let pg = PersistentGraph::open_with(path, false, BaseMode::Mmap, Validate::Crc, false)
                .unwrap();
            pg.close().unwrap();
        })
    });

    // + the O(N+E) semantic sweep (mandatory for untrusted/shared snapshots).
    group.bench_function("open mmap V2 (full validate)", |b| {
        b.iter(|| {
            let pg = PersistentGraph::open_with(path, false, BaseMode::Mmap, Validate::Full, false)
                .unwrap();
            pg.close().unwrap();
        })
    });

    // Legacy V0 zero-copy view from rkyv bytes (the current ephemeral warm path;
    // also builds a name index but has no crc/manifest/flock/WAL machinery).
    let v0_bytes = to_rkyv(&build_owned(n));
    group.bench_function("from_rkyv V0 view", |b| {
        b.iter_batched(
            || v0_bytes.clone(),
            |data| ArchivedGraphView::from_bytes(data).unwrap(),
            BatchSize::LargeInput,
        )
    });

    group.finish();
}

fn bench_warm_open(c: &mut Criterion) {
    warm_open_group(c, N); // 50K
    warm_open_group(c, 200_000); // scaling trend: is the V2/V0 gap fixed or per-size?
}

// ---------------------------------------------------------------------------
// 3. DeltaAccessor traversal overhead at delta 0 / 1% / 10%
// ---------------------------------------------------------------------------

fn bench_delta_overhead(c: &mut Criterion) {
    let base = build_owned(N);
    let ctx = DynamicContext::default();

    let mut group = c.benchmark_group("beam over delta (50K base, k=5 d=3)");

    // Baseline: the plain owned accessor, no delta wrapper.
    group.bench_function("owned (no delta layer)", |b| {
        b.iter(|| beam_traverse(&base, &ctx, START, 5, 3))
    });

    for (label, count) in [
        ("delta 0%", 0usize),
        ("delta 1%", N / 100),
        ("delta 10%", N / 10),
    ] {
        let delta = build_delta(&base, count);
        let acc = DeltaAccessor::new(&base, &delta);
        group.bench_function(label, |b| b.iter(|| beam_traverse(&acc, &ctx, START, 5, 3)));
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// 4. beam over warm mmap CSR vs owned petgraph vs legacy V0 archived view
// ---------------------------------------------------------------------------

fn bench_csr_vs_petgraph(c: &mut Criterion) {
    let ctx = DynamicContext::default();

    // Owned petgraph (baseline).
    let owned = build_owned(N);

    // V2 CSR mmap base, opened from disk.
    let dir = tempfile::tempdir().expect("tempdir");
    PersistentGraph::create(dir.path(), build_owned(N))
        .expect("create")
        .close()
        .expect("close");
    let pg = PersistentGraph::open_with(dir.path(), false, BaseMode::Mmap, Validate::Crc, false)
        .unwrap();
    let state = pg.snapshot();
    let csr_acc = state.base.as_accessor(); // &ArchivedCsrView (empty delta path)

    // Legacy V0 archived view (the O(E)-per-expansion scan the CSR fixes).
    let v0_bytes = to_rkyv(&build_owned(N));
    let v0_view = ArchivedGraphView::from_bytes(v0_bytes).unwrap();

    let mut group = c.benchmark_group("beam by base repr (50K, k=5 d=3)");
    group.bench_function("owned petgraph", |b| {
        b.iter(|| beam_traverse(&owned, &ctx, START, 5, 3))
    });
    group.bench_function("mmap CSR V2", |b| {
        b.iter(|| beam_traverse(csr_acc, &ctx, START, 5, 3))
    });
    group.bench_function("archived V0 (O(E) scan)", |b| {
        b.iter(|| beam_traverse(&v0_view, &ctx, START, 5, 3))
    });
    group.finish();
}

criterion_group!(
    benches,
    bench_apply_batch,
    bench_warm_open,
    bench_delta_overhead,
    bench_csr_vs_petgraph
);
criterion_main!(benches);
