use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet};

use crate::accessor::GraphAccessor;
use crate::overlay::{neighbors_with_overlay, resolve_overlay_node};
use crate::scoring::compute_score;
use crate::types::{
    DynamicContext, EdgeResult, NodeResult, PathStep, SubGraph,
};

// ---------------------------------------------------------------------------
// 1. Beam Traverse — Top-K pruned BFS
// ---------------------------------------------------------------------------

/// Top-K pruned BFS traversal.
///
/// On each level keeps only the `k` highest-scored neighbors.
/// Returns at most `k * depth` nodes, sorted by weight descending.
/// Nodes are never duplicated across levels (visited set).
pub fn beam_traverse(
    graph: &dyn GraphAccessor,
    ctx: &DynamicContext,
    start: &str,
    k: usize,
    depth: usize,
) -> Vec<NodeResult> {
    let mut visited: HashSet<String> = HashSet::new();
    visited.insert(start.to_string());

    let mut all_results: Vec<NodeResult> = Vec::with_capacity(k * depth);
    let mut frontier: Vec<String> = vec![start.to_string()];

    for _ in 0..depth {
        let mut level_candidates: Vec<NodeResult> = Vec::new();

        for node_name in &frontier {
            let neighbors = neighbors_with_overlay(graph, ctx, node_name);

            for neighbor in neighbors {
                if visited.contains(&neighbor.name) {
                    continue;
                }

                // Score the neighbor node
                let result = if let Some(node_view) = graph.get_node(&neighbor.name) {
                    compute_score(&node_view, ctx)
                } else if let Some(overlay_view) = resolve_overlay_node(&neighbor.name, ctx) {
                    compute_score(&overlay_view, ctx)
                } else {
                    continue;
                };

                level_candidates.push(result);
            }
        }

        // Sort by weight descending, take Top-K
        level_candidates.sort_by(|a, b| b.weight.partial_cmp(&a.weight).unwrap_or(Ordering::Equal));
        level_candidates.truncate(k);

        // Build next frontier from Top-K, mark visited
        frontier = Vec::with_capacity(level_candidates.len());
        for result in &level_candidates {
            if visited.insert(result.name.clone()) {
                frontier.push(result.name.clone());
            }
        }

        all_results.extend(level_candidates);
    }

    // Final sort: all results by weight descending
    all_results.sort_by(|a, b| b.weight.partial_cmp(&a.weight).unwrap_or(Ordering::Equal));
    all_results
}

// ---------------------------------------------------------------------------
// 2. Find Path — Weighted Dijkstra
// ---------------------------------------------------------------------------

#[derive(Debug)]
struct DijkstraEntry {
    cost: f32,
    node: String,
}

impl PartialEq for DijkstraEntry {
    fn eq(&self, other: &Self) -> bool {
        self.cost == other.cost
    }
}

impl Eq for DijkstraEntry {}

impl PartialOrd for DijkstraEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for DijkstraEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .cost
            .partial_cmp(&self.cost)
            .unwrap_or(Ordering::Equal)
    }
}

struct ParentInfo {
    from: String,
    edge_kind: String,
    field_name: Option<String>,
}

/// Weighted Dijkstra shortest path.
///
/// Edge cost = `1.0 / score(target)` — higher scored nodes are "closer".
/// Returns `None` if no path exists between start and end.
pub fn find_path(
    graph: &dyn GraphAccessor,
    ctx: &DynamicContext,
    start: &str,
    end: &str,
) -> Option<Vec<PathStep>> {
    if start == end {
        return Some(vec![PathStep {
            node: start.to_string(),
            edge_kind: String::new(),
            field_name: String::new(),
            direction: String::new(),
        }]);
    }

    let mut dist: HashMap<String, f32> = HashMap::new();
    let mut parent: HashMap<String, ParentInfo> = HashMap::new();
    let mut heap = BinaryHeap::new();

    dist.insert(start.to_string(), 0.0);
    heap.push(DijkstraEntry {
        cost: 0.0,
        node: start.to_string(),
    });

    while let Some(DijkstraEntry { cost, node }) = heap.pop() {
        if node == end {
            return Some(reconstruct_path(start, end, &parent));
        }

        if let Some(&best) = dist.get(&node) {
            if cost > best {
                continue;
            }
        }

        let neighbors = neighbors_with_overlay(graph, ctx, &node);

        for neighbor in neighbors {
            let target_score = if let Some(node_view) = graph.get_node(&neighbor.name) {
                compute_score(&node_view, ctx).weight
            } else if let Some(overlay_view) = resolve_overlay_node(&neighbor.name, ctx) {
                compute_score(&overlay_view, ctx).weight
            } else {
                0.001
            };

            let edge_cost = 1.0 / target_score.max(0.001);
            let new_cost = cost + edge_cost;

            let is_better = dist
                .get(&neighbor.name)
                .is_none_or(|&existing| new_cost < existing);

            if is_better {
                dist.insert(neighbor.name.clone(), new_cost);
                parent.insert(
                    neighbor.name.clone(),
                    ParentInfo {
                        from: node.clone(),
                        edge_kind: neighbor.edge_kind.clone(),
                        field_name: neighbor.field_name.clone(),
                    },
                );
                heap.push(DijkstraEntry {
                    cost: new_cost,
                    node: neighbor.name.clone(),
                });
            }
        }
    }

    None
}

fn reconstruct_path(start: &str, end: &str, parent: &HashMap<String, ParentInfo>) -> Vec<PathStep> {
    let mut path = Vec::new();
    let mut current = end.to_string();

    while current != start {
        if let Some(info) = parent.get(&current) {
            path.push(PathStep {
                node: current.clone(),
                edge_kind: info.edge_kind.clone(),
                field_name: info.field_name.clone().unwrap_or_default(),
                direction: "outgoing".to_string(),
            });
            current = info.from.clone();
        } else {
            break;
        }
    }

    path.push(PathStep {
        node: start.to_string(),
        edge_kind: String::new(),
        field_name: String::new(),
        direction: String::new(),
    });

    path.reverse();
    path
}

// ---------------------------------------------------------------------------
// 3. Contextual Subgraph
// ---------------------------------------------------------------------------

/// Extract a compact subgraph of `k` nodes most relevant to the context.
pub fn contextual_subgraph(
    graph: &dyn GraphAccessor,
    ctx: &DynamicContext,
    k: usize,
) -> SubGraph {
    let mut boost_entries: Vec<(&String, &f32)> = ctx.semantic_boosts.iter().collect();
    boost_entries.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap_or(Ordering::Equal));
    let seeds: Vec<&str> = boost_entries.iter().take(k).map(|(name, _)| name.as_str()).collect();

    let mut node_set: HashSet<String> = HashSet::new();
    let mut nodes: Vec<NodeResult> = Vec::new();
    let mut edges: Vec<EdgeResult> = Vec::new();

    for seed_name in &seeds {
        if node_set.contains(*seed_name) {
            continue;
        }

        let result = if let Some(node_view) = graph.get_node(seed_name) {
            compute_score(&node_view, ctx)
        } else if let Some(overlay_view) = resolve_overlay_node(seed_name, ctx) {
            compute_score(&overlay_view, ctx)
        } else {
            continue;
        };

        node_set.insert(seed_name.to_string());
        nodes.push(result);

        let neighbors = neighbors_with_overlay(graph, ctx, seed_name);
        for neighbor in neighbors {
            edges.push(EdgeResult {
                source: seed_name.to_string(),
                target: neighbor.name.clone(),
                kind: neighbor.edge_kind.clone(),
                field_name: neighbor.field_name.clone(),
                weight: neighbor.edge_weight,
            });

            if node_set.insert(neighbor.name.clone()) {
                let neighbor_result = if let Some(nv) = graph.get_node(&neighbor.name) {
                    compute_score(&nv, ctx)
                } else if let Some(ov) = resolve_overlay_node(&neighbor.name, ctx) {
                    compute_score(&ov, ctx)
                } else {
                    continue;
                };
                nodes.push(neighbor_result);
            }
        }
    }

    nodes.sort_by(|a, b| b.weight.partial_cmp(&a.weight).unwrap_or(Ordering::Equal));

    SubGraph { nodes, edges }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builder::build_graph;
    use crate::graph::OrpheusGraphInner;
    use crate::types::{EdgeData, EdgeInput, NodeData, NodeInput};

    fn make_node(name: &str, weight: f32) -> NodeInput {
        NodeInput {
            name: name.to_string(),
            kind: "model".to_string(),
            metadata: HashMap::new(),
            base_weight: weight,
            noise_penalty: 0.0,
        }
    }

    fn make_edge(from: &str, to: &str, kind: &str, field: Option<&str>) -> EdgeInput {
        EdgeInput {
            from: from.to_string(),
            to: to.to_string(),
            kind: kind.to_string(),
            field_name: field.map(|s| s.to_string()),
            base_weight: 1.0,
        }
    }

    fn build_chain_graph() -> OrpheusGraphInner {
        let nodes = vec![
            make_node("A", 0.5),
            make_node("B", 0.8),
            make_node("C", 0.3),
            make_node("D", 0.9),
            make_node("E", 0.6),
        ];
        let edges = vec![
            make_edge("A", "B", "relates_to", Some("partner_id")),
            make_edge("B", "C", "relates_to", Some("origin")),
            make_edge("C", "D", "relates_to", Some("move_id")),
            make_edge("D", "E", "relates_to", Some("lot_id")),
        ];
        let (g, m) = build_graph(nodes, edges);
        OrpheusGraphInner::new(g, m)
    }

    #[test]
    fn test_beam_basic() {
        let graph = build_chain_graph();
        let ctx = DynamicContext::default();
        let results = beam_traverse(&graph, &ctx, "A", 5, 1);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].name, "B");
    }

    #[test]
    fn test_beam_depth() {
        let graph = build_chain_graph();
        let ctx = DynamicContext::default();
        let results = beam_traverse(&graph, &ctx, "A", 5, 3);
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn test_beam_sorted() {
        let graph = build_chain_graph();
        let ctx = DynamicContext::default();
        let results = beam_traverse(&graph, &ctx, "A", 5, 4);
        for i in 0..results.len() - 1 {
            assert!(
                results[i].weight >= results[i + 1].weight,
                "Results not sorted: {} ({}) before {} ({})",
                results[i].name, results[i].weight,
                results[i + 1].name, results[i + 1].weight,
            );
        }
    }

    #[test]
    fn test_beam_dedup() {
        let graph = build_chain_graph();
        let ctx = DynamicContext::default();
        let results = beam_traverse(&graph, &ctx, "A", 5, 4);
        let names: HashSet<&str> = results.iter().map(|r| r.name.as_str()).collect();
        assert_eq!(names.len(), results.len());
    }

    #[test]
    fn test_beam_with_overlay() {
        let graph = build_chain_graph();
        let mut ctx = DynamicContext::default();
        ctx.overlay_nodes.push(NodeData {
            name: "X_CUSTOM".to_string(),
            kind: "model".to_string(),
            metadata: HashMap::new(),
            base_weight: 1.0,
            noise_penalty: 0.0,
            pagerank_weight: 0.0,
        });
        ctx.overlay_edges.push((
            "A".to_string(),
            "X_CUSTOM".to_string(),
            EdgeData {
                kind: "relates_to".to_string(),
                field_name: None,
                base_weight: 1.0,
            },
        ));

        let results = beam_traverse(&graph, &ctx, "A", 5, 1);
        let names: Vec<&str> = results.iter().map(|r| r.name.as_str()).collect();
        assert!(names.contains(&"X_CUSTOM"), "Overlay node should appear: {names:?}");
    }

    #[test]
    fn test_find_path_basic() {
        let graph = build_chain_graph();
        let ctx = DynamicContext::default();
        let path = find_path(&graph, &ctx, "A", "D");
        assert!(path.is_some());
        let path = path.unwrap();
        assert_eq!(path[0].node, "A");
        assert_eq!(*path.last().unwrap().node, *"D");
    }

    #[test]
    fn test_find_path_unreachable() {
        let nodes = vec![make_node("X", 1.0), make_node("Y", 1.0)];
        let (g, m) = build_graph(nodes, vec![]);
        let graph = OrpheusGraphInner::new(g, m);

        let ctx = DynamicContext::default();
        let path = find_path(&graph, &ctx, "X", "Y");
        assert!(path.is_none());
    }

    #[test]
    fn test_find_path_steps() {
        let graph = build_chain_graph();
        let ctx = DynamicContext::default();
        let path = find_path(&graph, &ctx, "A", "C").unwrap();

        assert_eq!(path.len(), 3);
        assert_eq!(path[0].node, "A");
        assert_eq!(path[1].node, "B");
        assert_eq!(path[1].edge_kind, "relates_to");
        assert_eq!(path[1].field_name, "partner_id");
        assert_eq!(path[1].direction, "outgoing");
        assert_eq!(path[2].node, "C");
        assert_eq!(path[2].field_name, "origin");
    }

    #[test]
    fn test_contextual_subgraph() {
        let graph = build_chain_graph();
        let mut ctx = DynamicContext::default();
        ctx.semantic_boosts.insert("B".to_string(), 2.0);
        ctx.semantic_boosts.insert("D".to_string(), 1.5);

        let sg = contextual_subgraph(&graph, &ctx, 2);

        let node_names: HashSet<&str> = sg.nodes.iter().map(|n| n.name.as_str()).collect();
        assert!(node_names.contains("B"), "Seed B missing");
        assert!(node_names.contains("D"), "Seed D missing");
        assert!(node_names.contains("C"), "B's neighbor C missing");
        assert!(node_names.contains("E"), "D's neighbor E missing");
        assert!(!sg.edges.is_empty());
    }
}
