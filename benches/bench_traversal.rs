use std::collections::HashMap;

use criterion::{criterion_group, criterion_main, Criterion};

use orpheusgraph::builder::build_graph;
use orpheusgraph::graph::OrpheusGraphInner;
use orpheusgraph::serialization::{to_rkyv, ArchivedGraphView};
use orpheusgraph::traversal::{beam_traverse, contextual_subgraph, find_path};
use orpheusgraph::types::{DynamicContext, EdgeInput, NodeInput};

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
            edges.push(orpheusgraph::types::EdgeInput {
                from: format!("node_{}", i - 1),
                to: format!("node_{i}"),
                kind: "relates_to".to_string(),
                field_name: None,
                base_weight: 1.0,
            });
        }
        if i > 10 {
            edges.push(orpheusgraph::types::EdgeInput {
                from: format!("node_{i}"),
                to: format!("node_{}", i - 10),
                kind: "relates_to".to_string(),
                field_name: None,
                base_weight: 0.5,
            });
        }
        if i > 100 {
            edges.push(orpheusgraph::types::EdgeInput {
                from: format!("node_{i}"),
                to: format!("node_{}", i - 100),
                kind: "depends_on".to_string(),
                field_name: None,
                base_weight: 0.3,
            });
        }
    }

    (nodes, edges)
}

fn build_large_graph(n: usize) -> OrpheusGraphInner {
    let (nodes, edges) = build_inputs(n);
    let (g, m) = build_graph(nodes, edges);
    OrpheusGraphInner::new(g, m)
}

fn bench_build(c: &mut Criterion) {
    c.bench_function("build_graph (50K nodes)", |b| {
        b.iter_batched(
            || build_inputs(50_000),
            |(nodes, edges)| {
                let (g, m) = build_graph(nodes, edges);
                OrpheusGraphInner::new(g, m)
            },
            criterion::BatchSize::LargeInput,
        )
    });
}

fn bench_serialization(c: &mut Criterion) {
    let graph = build_large_graph(50_000);

    c.bench_function("to_rkyv (50K nodes)", |b| b.iter(|| to_rkyv(&graph)));

    let bytes = to_rkyv(&graph);
    c.bench_function("from_rkyv zero-copy view (50K nodes)", |b| {
        b.iter_batched(
            || bytes.clone(),
            |data| ArchivedGraphView::from_bytes(data).unwrap(),
            criterion::BatchSize::LargeInput,
        )
    });
}

fn bench_beam(c: &mut Criterion) {
    let graph = build_large_graph(50_000);
    let ctx = DynamicContext::default();

    c.bench_function("beam_traverse k=5 d=3 (50K nodes)", |b| {
        b.iter(|| beam_traverse(&graph, &ctx, "node_25000", 5, 3))
    });
}

fn bench_find_path(c: &mut Criterion) {
    let graph = build_large_graph(50_000);
    let ctx = DynamicContext::default();

    c.bench_function("find_path (50K nodes)", |b| {
        b.iter(|| find_path(&graph, &ctx, "node_25000", "node_25050"))
    });
}

fn bench_subgraph(c: &mut Criterion) {
    let graph = build_large_graph(50_000);
    let mut ctx = DynamicContext::default();
    for i in 0..30 {
        ctx.semantic_boosts
            .insert(format!("node_{}", i * 1000), 2.0 - (i as f32 * 0.05));
    }

    c.bench_function("contextual_subgraph k=30 (50K nodes)", |b| {
        b.iter(|| contextual_subgraph(&graph, &ctx, 30))
    });
}

criterion_group!(
    benches,
    bench_build,
    bench_beam,
    bench_find_path,
    bench_subgraph,
    bench_serialization
);
criterion_main!(benches);
