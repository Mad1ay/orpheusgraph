use std::collections::HashMap;

use petgraph::graph::{DiGraph, NodeIndex};

use crate::types::{EdgeData, EdgeInput, NodeData, NodeInput};
use crate::graph::OrpheusGraphInner;

/// Rebuild a graph from already-serialized data (nodes with pagerank, indexed edges).
/// No normalization or PageRank recomputation — data is already processed.
pub fn rebuild_from_serialized(
    nodes: Vec<NodeData>,
    edges: Vec<(usize, usize, EdgeData)>,
) -> OrpheusGraphInner {
    let mut graph = DiGraph::new();
    let mut index_map: HashMap<String, NodeIndex> = HashMap::with_capacity(nodes.len());
    let mut idx_vec: Vec<NodeIndex> = Vec::with_capacity(nodes.len());

    for node in nodes {
        let name = node.name.clone();
        let idx = graph.add_node(node);
        index_map.insert(name, idx);
        idx_vec.push(idx);
    }

    for (from_idx, to_idx, edge_data) in edges {
        if from_idx < idx_vec.len() && to_idx < idx_vec.len() {
            graph.add_edge(idx_vec[from_idx], idx_vec[to_idx], edge_data);
        }
    }

    OrpheusGraphInner::new(graph, index_map)
}

/// Build an immutable directed graph from raw node and edge inputs.
///
/// Performs:
/// 1. Node insertion with `base_weight` normalization to [0.0, 1.0]
/// 2. Edge insertion (skips edges referencing unknown nodes)
/// 3. PageRank computation (power iteration, damping=0.85, 20 iterations)
pub fn build_graph(
    nodes: Vec<NodeInput>,
    edges: Vec<EdgeInput>,
) -> (DiGraph<NodeData, EdgeData>, HashMap<String, NodeIndex>) {
    build_graph_inner(nodes, edges, true)
}

/// Like [`build_graph`] but WITHOUT re-normalizing `base_weight` — the caller
/// guarantees inputs are already in `[0, 1]`.
///
/// Used by compaction: the base's weights were normalized once at the original
/// build and delta weights are supplied in `[0, 1]`, so re-normalizing (divide
/// by the surviving max) would silently rescale surviving nodes whenever the
/// prior max node was removed — changing scoring output at an IDENTICAL seq and
/// breaking the "same (state seq, ctx) -> same output" guarantee. Non-finite /
/// negative inputs are still sanitized to keep the `[0, 1]` invariant; PageRank
/// is still recomputed (documented drift, §3.4).
pub fn build_graph_prenormalized(
    nodes: Vec<NodeInput>,
    edges: Vec<EdgeInput>,
) -> (DiGraph<NodeData, EdgeData>, HashMap<String, NodeIndex>) {
    build_graph_inner(nodes, edges, false)
}

fn build_graph_inner(
    nodes: Vec<NodeInput>,
    edges: Vec<EdgeInput>,
    normalize: bool,
) -> (DiGraph<NodeData, EdgeData>, HashMap<String, NodeIndex>) {
    let mut graph = DiGraph::new();
    let mut index_map: HashMap<String, NodeIndex> = HashMap::with_capacity(nodes.len());

    // --- Phase 1: Insert nodes ---
    // Dedup inputs by name (last-wins, matching upsert semantics) so a repeated
    // name produces exactly one node instead of an unreachable orphan whose
    // presence would skew normalization and PageRank.
    let mut deduped: Vec<&NodeInput> = Vec::with_capacity(nodes.len());
    let mut name_pos: HashMap<&str, usize> = HashMap::with_capacity(nodes.len());
    for input in &nodes {
        match name_pos.get(input.name.as_str()) {
            Some(&pos) => deduped[pos] = input, // last occurrence wins
            None => {
                name_pos.insert(input.name.as_str(), deduped.len());
                deduped.push(input);
            }
        }
    }

    // Sanitize weights before normalization: a single non-finite base_weight
    // would otherwise make the fold-max non-finite and divide every node to
    // zero/NaN. Non-finite -> 0.0, and negative finite -> 0.0 so a negative
    // base_weight can't produce a normalized value outside the documented
    // [0,1] range (which would invert ranking via a negative base_component).
    let sanitize = |w: f32| if w.is_finite() { w.max(0.0) } else { 0.0 };

    // Normalization divisor: divide base_weights by the max so they land in
    // [0,1]. Skipped entirely (divisor 1.0) when `normalize` is false — the
    // compaction path, where inputs are already normalized and re-scaling would
    // change scores at an unchanged seq. Also skipped on a non-finite / non-
    // positive max rather than poisoning every node.
    let norm_divisor = if normalize {
        let max_weight = deduped
            .iter()
            .map(|n| sanitize(n.base_weight))
            .fold(0.0_f32, f32::max);
        if max_weight.is_finite() && max_weight > f32::EPSILON {
            max_weight
        } else {
            1.0
        }
    } else {
        1.0
    };

    for input in &deduped {
        let node_data = NodeData {
            name: input.name.clone(),
            kind: input.kind.clone(),
            metadata: input.metadata.clone(),
            base_weight: sanitize(input.base_weight) / norm_divisor,
            noise_penalty: sanitize(input.noise_penalty).clamp(0.0, 1.0),
            pagerank_weight: 0.0, // computed in phase 3
        };
        let idx = graph.add_node(node_data);
        index_map.insert(input.name.clone(), idx);
    }

    // --- Phase 2: Insert edges ---
    for edge in &edges {
        let from_idx = match index_map.get(&edge.from) {
            Some(idx) => *idx,
            None => continue, // skip edges referencing unknown nodes
        };
        let to_idx = match index_map.get(&edge.to) {
            Some(idx) => *idx,
            None => continue,
        };
        let edge_data = EdgeData {
            kind: edge.kind.clone(),
            field_name: edge.field_name.clone(),
            base_weight: edge.base_weight,
        };
        graph.add_edge(from_idx, to_idx, edge_data);
    }

    // --- Phase 3: PageRank ---
    compute_pagerank(&mut graph, 0.85, 20);

    (graph, index_map)
}

/// Iterative PageRank computation (power iteration).
///
/// Results are normalized to [0.0, 1.0] and stored in `node.pagerank_weight`.
fn compute_pagerank(graph: &mut DiGraph<NodeData, EdgeData>, damping: f32, iterations: usize) {
    let n = graph.node_count();
    if n == 0 {
        return;
    }

    let n_f32 = n as f32;
    let initial = 1.0 / n_f32;

    // Initialize scores
    let mut scores: Vec<f32> = vec![initial; n];
    let mut new_scores: Vec<f32> = vec![0.0; n];

    // Out-degree is fixed across iterations — compute it once (the old code
    // recounted it every iteration).
    let out_degree: Vec<usize> = graph
        .node_indices()
        .map(|idx| {
            graph
                .neighbors_directed(idx, petgraph::Direction::Outgoing)
                .count()
        })
        .collect();

    let teleport = (1.0 - damping) / n_f32;

    for _ in 0..iterations {
        // Dangling mass is distributed UNIFORMLY, so it is the same scalar
        // added to every node — compute it once as a sum instead of looping
        // over all n nodes per dangling node (the old O(dangling*n) hot path).
        // Each node's dangling contribution is damping * (Σ dangling scores)/n,
        // identical to the old per-dangling accumulation.
        let dangling_sum: f32 = (0..n)
            .filter(|&i| out_degree[i] == 0)
            .map(|i| scores[i])
            .sum();
        let base = teleport + damping * dangling_sum / n_f32;
        for s in new_scores.iter_mut() {
            *s = base;
        }

        // Distribute scores through edges (dangling nodes already handled above)
        for node_idx in graph.node_indices() {
            let od = out_degree[node_idx.index()];
            if od == 0 {
                continue;
            }
            let share = damping * scores[node_idx.index()] / od as f32;
            for neighbor in graph.neighbors_directed(node_idx, petgraph::Direction::Outgoing) {
                new_scores[neighbor.index()] += share;
            }
        }

        std::mem::swap(&mut scores, &mut new_scores);
    }

    // Normalize to [0.0, 1.0]
    let max_score = scores.iter().copied().fold(0.0_f32, f32::max);
    let norm = if max_score > f32::EPSILON { max_score } else { 1.0 };

    for node_idx in graph.node_indices() {
        graph[node_idx].pagerank_weight = scores[node_idx.index()] / norm;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_node(name: &str, kind: &str, weight: f32) -> NodeInput {
        NodeInput {
            name: name.to_string(),
            kind: kind.to_string(),
            metadata: HashMap::new(),
            base_weight: weight,
            noise_penalty: 0.0,
        }
    }

    fn make_edge(from: &str, to: &str, kind: &str) -> EdgeInput {
        EdgeInput {
            from: from.to_string(),
            to: to.to_string(),
            kind: kind.to_string(),
            field_name: None,
            base_weight: 1.0,
        }
    }

    // Leaf-heavy graph: `hubs` interconnected models + many dangling "field"
    // nodes, each with one incoming edge from a hub. This is the schema shape
    // where dangling-node PageRank redistribution is the hot path.
    fn leaf_heavy(hubs: usize, leaves: usize) -> (Vec<NodeInput>, Vec<EdgeInput>) {
        let mut nodes = Vec::with_capacity(hubs + leaves);
        let mut edges = Vec::new();
        for h in 0..hubs {
            nodes.push(make_node(&format!("m{h}"), "model", 0.5));
        }
        for h in 0..hubs {
            edges.push(make_edge(&format!("m{h}"), &format!("m{}", (h + 1) % hubs), "relates_to"));
        }
        for l in 0..leaves {
            let name = format!("f{l}");
            nodes.push(make_node(&name, "field", 0.1));
            edges.push(make_edge(&format!("m{}", l % hubs), &name, "contains"));
        }
        (nodes, edges)
    }

    #[test]
    #[ignore] // timing bench: cargo test --release pagerank_leaf_heavy -- --ignored --nocapture
    fn bench_pagerank_leaf_heavy() {
        use std::time::Instant;
        let (nodes, edges) = leaf_heavy(200, 14_800);
        // warm + measure a few builds
        let mut best = std::f64::MAX;
        for _ in 0..3 {
            let (n2, e2) = (nodes.clone(), edges.clone());
            let t = Instant::now();
            let (g, _) = build_graph(n2, e2);
            let dt = t.elapsed().as_secs_f64() * 1000.0;
            std::hint::black_box(&g);
            if dt < best {
                best = dt;
            }
        }
        println!(
            "build_graph (200 hubs + 14800 dangling leaves): {best:.1} ms (best of 3)"
        );
    }

    #[test]
    fn test_pagerank_dangling_redistribution() {
        // A,B -> H (a well-cited hub); L is a dangling leaf. The optimized
        // scalar dangling handling must still: keep the hub highest-ranked,
        // normalize to [0,1] with max == 1, and give every node positive rank
        // (dangling mass is redistributed, not lost).
        let nodes = vec![
            make_node("A", "model", 0.5),
            make_node("B", "model", 0.5),
            make_node("H", "model", 0.5),
            make_node("L", "field", 0.1),
        ];
        let edges = vec![
            make_edge("A", "H", "relates_to"),
            make_edge("B", "H", "relates_to"),
            make_edge("H", "L", "contains"),
        ];
        let (graph, idx) = build_graph(nodes, edges);
        let pr = |name: &str| graph[idx[name]].pagerank_weight;
        // Normalization sets the max node to exactly 1.0.
        assert!((pr("H").max(pr("A")).max(pr("B")).max(pr("L")) - 1.0).abs() < 1e-6);
        // Nodes that receive edge mass (H from A,B; L as the sink H feeds)
        // outrank the pure sources A,B which only get teleport + dangling mass.
        assert!(pr("H") > pr("A") && pr("H") > pr("B"), "cited hub outranks its sources");
        assert!(pr("L") > pr("A"), "sink accumulating the hub's mass outranks a bare source");
        // Dangling mass + teleport is redistributed, never lost — all positive.
        for n in ["A", "B", "H", "L"] {
            assert!(pr(n) > 0.0, "{n} should get positive rank from teleport+dangling");
        }
    }

    #[test]
    fn test_empty_graph() {
        let (graph, index_map) = build_graph(vec![], vec![]);
        assert_eq!(graph.node_count(), 0);
        assert_eq!(graph.edge_count(), 0);
        assert!(index_map.is_empty());
    }

    #[test]
    fn test_build_basic() {
        let nodes = vec![
            make_node("sale.order", "model", 100.0),
            make_node("res.partner", "model", 200.0),
            make_node("stock.picking", "model", 50.0),
        ];
        let edges = vec![make_edge("sale.order", "res.partner", "relates_to")];

        let (graph, index_map) = build_graph(nodes, edges);
        assert_eq!(graph.node_count(), 3);
        assert_eq!(graph.edge_count(), 1);
        assert!(index_map.contains_key("sale.order"));
        assert!(index_map.contains_key("res.partner"));
        assert!(index_map.contains_key("stock.picking"));
    }

    #[test]
    fn test_weight_normalization() {
        let nodes = vec![
            make_node("a", "model", 100.0),
            make_node("b", "model", 200.0),
            make_node("c", "model", 500.0),
        ];
        let (graph, index_map) = build_graph(nodes, vec![]);

        let a = &graph[index_map["a"]];
        let b = &graph[index_map["b"]];
        let c = &graph[index_map["c"]];

        assert!((a.base_weight - 0.2).abs() < 0.001);
        assert!((b.base_weight - 0.4).abs() < 0.001);
        assert!((c.base_weight - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_pagerank_hub_node() {
        // Create a hub: many nodes point to "hub"
        let mut nodes = vec![make_node("hub", "model", 1.0)];
        let mut edges = vec![];
        for i in 0..10 {
            let name = format!("leaf_{i}");
            nodes.push(make_node(&name, "field", 1.0));
            edges.push(make_edge(&name, "hub", "relates_to"));
        }

        let (graph, index_map) = build_graph(nodes, edges);
        let hub_pr = graph[index_map["hub"]].pagerank_weight;
        let leaf_pr = graph[index_map["leaf_0"]].pagerank_weight;

        // Hub should have highest pagerank (normalized to 1.0)
        assert!(
            hub_pr > leaf_pr,
            "Hub PR ({hub_pr}) should be > leaf PR ({leaf_pr})"
        );
        assert!((hub_pr - 1.0).abs() < 0.001, "Hub should be normalized to 1.0");
    }

    #[test]
    fn test_skip_unknown_edges() {
        let nodes = vec![make_node("a", "model", 1.0)];
        let edges = vec![make_edge("a", "nonexistent", "relates_to")];
        let (graph, _) = build_graph(nodes, edges);
        assert_eq!(graph.edge_count(), 0); // edge skipped
    }

    #[test]
    fn test_non_finite_base_weight_does_not_poison_others() {
        // A single +inf base_weight must not collapse normalization for the
        // rest of the nodes.
        let nodes = vec![
            make_node("a", "model", 100.0),
            make_node("bad", "model", f32::INFINITY),
            make_node("c", "model", 200.0),
        ];
        let (graph, index_map) = build_graph(nodes, vec![]);

        let a = &graph[index_map["a"]];
        let bad = &graph[index_map["bad"]];
        let c = &graph[index_map["c"]];

        // +inf is sanitized to 0.0, so max finite weight (200.0) drives norm.
        assert!(a.base_weight.is_finite());
        assert!(bad.base_weight.is_finite());
        assert!(c.base_weight.is_finite());
        assert!((a.base_weight - 0.5).abs() < 0.001);
        assert!((bad.base_weight - 0.0).abs() < 0.001);
        assert!((c.base_weight - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_duplicate_names_deduped() {
        // Duplicate node names collapse to a single node (last-wins), so no
        // orphan is left behind.
        let nodes = vec![
            make_node("dup", "model", 100.0),
            make_node("other", "model", 50.0),
            make_node("dup", "model", 300.0), // overrides the first "dup"
        ];
        let (graph, index_map) = build_graph(nodes, vec![]);

        // Exactly one node per distinct name.
        assert_eq!(graph.node_count(), 2);
        assert_eq!(index_map.len(), 2);
        // Last write wins: dup's base_weight came from 300.0 (the max), so 1.0.
        assert!((graph[index_map["dup"]].base_weight - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_noise_penalty_clamped() {
        let nodes = vec![NodeInput {
            name: "noisy".to_string(),
            kind: "field".to_string(),
            metadata: HashMap::new(),
            base_weight: 1.0,
            noise_penalty: 1.5, // exceeds [0, 1]
        }];
        let (graph, index_map) = build_graph(nodes, vec![]);
        assert!((graph[index_map["noisy"]].noise_penalty - 1.0).abs() < 0.001);
    }
}
