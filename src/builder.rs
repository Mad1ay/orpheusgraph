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
    let mut graph = DiGraph::new();
    let mut index_map: HashMap<String, NodeIndex> = HashMap::with_capacity(nodes.len());

    // --- Phase 1: Insert nodes ---
    // Find max base_weight for normalization
    let max_weight = nodes
        .iter()
        .map(|n| n.base_weight)
        .fold(0.0_f32, f32::max);
    let norm_divisor = if max_weight > f32::EPSILON { max_weight } else { 1.0 };

    for input in &nodes {
        let node_data = NodeData {
            name: input.name.clone(),
            kind: input.kind.clone(),
            metadata: input.metadata.clone(),
            base_weight: input.base_weight / norm_divisor,
            noise_penalty: input.noise_penalty.clamp(0.0, 1.0),
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

    for _ in 0..iterations {
        // Reset
        for s in new_scores.iter_mut() {
            *s = (1.0 - damping) / n_f32;
        }

        // Distribute scores through edges
        for node_idx in graph.node_indices() {
            let out_degree = graph
                .neighbors_directed(node_idx, petgraph::Direction::Outgoing)
                .count();
            if out_degree == 0 {
                // Dangling node: distribute evenly to all nodes
                let share = damping * scores[node_idx.index()] / n_f32;
                for s in new_scores.iter_mut() {
                    *s += share;
                }
            } else {
                let share = damping * scores[node_idx.index()] / out_degree as f32;
                for neighbor in
                    graph.neighbors_directed(node_idx, petgraph::Direction::Outgoing)
                {
                    new_scores[neighbor.index()] += share;
                }
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
