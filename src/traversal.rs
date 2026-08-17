use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet};

use crate::accessor::GraphAccessor;
use crate::overlay::{neighbors_with_overlay, resolve_overlay_node};
use crate::scoring::compute_score;
use crate::types::{DynamicContext, EdgeResult, NodeResult, PathStep, SubGraph};

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

    // `k * depth` is only a capacity hint; it can overflow or request an
    // impossible allocation for hostile k/depth (Vec::with_capacity then
    // aborts the process — uncatchable from Python). Results can never exceed
    // the node count anyway, so cap the hint. Real work is bounded by the
    // graph, not by k/depth.
    let cap_hint = k.saturating_mul(depth).min(graph.node_count());
    let mut all_results: Vec<NodeResult> = Vec::with_capacity(cap_hint);
    let mut frontier: Vec<String> = vec![start.to_string()];

    for _ in 0..depth {
        // Nothing left to expand — stop early instead of spinning `depth`
        // times over an empty frontier (hostile depth = wasted CPU otherwise).
        if frontier.is_empty() {
            break;
        }
        // Dedup candidates within a level by node name as they are collected:
        // a node reachable from two frontier nodes must yield a single entry
        // (keep the highest score), otherwise duplicates survive truncate(k).
        let mut level_map: HashMap<String, NodeResult> = HashMap::new();

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

                level_map
                    .entry(neighbor.name.clone())
                    .and_modify(|existing| {
                        if result.weight > existing.weight {
                            *existing = result.clone();
                        }
                    })
                    .or_insert(result);
            }
        }

        let mut level_candidates: Vec<NodeResult> = level_map.into_values().collect();

        // Sort by weight descending, take Top-K. `total_cmp` gives a total
        // order even with NaN weights; the name tiebreak makes Top-K
        // deterministic despite the non-deterministic HashMap iteration order.
        level_candidates.sort_by(|a, b| {
            b.weight
                .total_cmp(&a.weight)
                .then_with(|| a.name.cmp(&b.name))
        });
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

    // Final sort: all results by weight descending, name tiebreak for a
    // total deterministic order (total_cmp is NaN-safe).
    all_results.sort_by(|a, b| {
        b.weight
            .total_cmp(&a.weight)
            .then_with(|| a.name.cmp(&b.name))
    });
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
        // total_cmp is NaN-safe → a total order, so the heap stays deterministic.
        other.cost.total_cmp(&self.cost)
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
pub fn contextual_subgraph(graph: &dyn GraphAccessor, ctx: &DynamicContext, k: usize) -> SubGraph {
    // Seed order must be deterministic: HashMap iteration order + a stable
    // sort on tied boosts would otherwise pick different seeds across runs.
    // Sort by (boost desc, name asc) so ties break by name (total_cmp is
    // NaN-safe).
    let mut boost_entries: Vec<(&String, &f32)> = ctx.semantic_boosts.iter().collect();
    boost_entries.sort_by(|a, b| b.1.total_cmp(a.1).then_with(|| a.0.cmp(b.0)));
    let seeds: Vec<&str> = boost_entries
        .iter()
        .take(k)
        .map(|(name, _)| name.as_str())
        .collect();

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
                valid_from: neighbor.valid_from,
                valid_to: neighbor.valid_to,
                acl: neighbor.acl.clone(),
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

    // total_cmp + name tiebreak → total deterministic order (NaN-safe).
    nodes.sort_by(|a, b| {
        b.weight
            .total_cmp(&a.weight)
            .then_with(|| a.name.cmp(&b.name))
    });

    // Drop dangling edges whose endpoint was never materialized as a node
    // (e.g. an overlay edge to a nonexistent target), then sort for a
    // deterministic edge order — mirrors the guard multi_beam_intersection has.
    let present: HashSet<&str> = nodes.iter().map(|n| n.name.as_str()).collect();
    edges.retain(|e| present.contains(e.source.as_str()) && present.contains(e.target.as_str()));
    edges.sort_by(|a, b| {
        a.source
            .cmp(&b.source)
            .then_with(|| a.target.cmp(&b.target))
            .then_with(|| a.kind.cmp(&b.kind))
    });

    SubGraph { nodes, edges }
}

// ---------------------------------------------------------------------------
// 4. Multi-Beam Intersection — Heatmap / Threshold
// ---------------------------------------------------------------------------

/// Multi-source heatmap intersection.
///
/// Launches an independent `beam_traverse` from each start node, then
/// accumulates weighted scores into a shared heatmap. Only nodes whose
/// **hit count ≥ threshold** (or that appear in `start_nodes`) survive.
///
/// Edges are reconstructed from the base graph + overlay, keeping only
/// those whose both endpoints are in the filtered set.
pub fn multi_beam_intersection(
    graph: &dyn GraphAccessor,
    ctx: &DynamicContext,
    start_nodes: &[String],
    k: usize,
    depth: usize,
    threshold: usize,
) -> SubGraph {
    use rayon::prelude::*;

    // ── 1. Multi-beam launch ──────────────────────────────────────────
    // Each beam returns Vec<NodeResult>. Collect them all.
    let beam_results: Vec<Vec<NodeResult>> = if start_nodes.len() > 2 {
        // Parallel: GraphAccessor is Send+Sync
        start_nodes
            .par_iter()
            .map(|start| beam_traverse(graph, ctx, start, k, depth))
            .collect()
    } else {
        start_nodes
            .iter()
            .map(|start| beam_traverse(graph, ctx, start, k, depth))
            .collect()
    };

    // ── 2. Accumulate heatmap ─────────────────────────────────────────
    // hit_count: how many beams visited this node
    // weight_sum: sum of W_total across beams (Variant B — weighted)
    let mut hit_count: HashMap<String, usize> = HashMap::new();
    let mut weight_sum: HashMap<String, f32> = HashMap::new();
    let mut best_result: HashMap<String, NodeResult> = HashMap::new();

    for results in &beam_results {
        // Track which nodes this particular beam visited (dedup per beam)
        let mut seen_this_beam: HashSet<String> = HashSet::new();

        for nr in results {
            if seen_this_beam.insert(nr.name.clone()) {
                *hit_count.entry(nr.name.clone()).or_insert(0) += 1;
            }
            *weight_sum.entry(nr.name.clone()).or_insert(0.0) += nr.weight;

            // Keep the highest-scoring NodeResult for each node
            let entry = best_result.entry(nr.name.clone());
            entry
                .and_modify(|existing| {
                    if nr.weight > existing.weight {
                        *existing = nr.clone();
                    }
                })
                .or_insert_with(|| nr.clone());
        }
    }

    // ── 3. Filter by threshold ────────────────────────────────────────
    let start_set: HashSet<&str> = start_nodes.iter().map(|s| s.as_str()).collect();

    let mut filtered_nodes: Vec<NodeResult> = Vec::new();
    let mut filtered_set: HashSet<String> = HashSet::new();

    // Always include start nodes (score them if they exist)
    for start in start_nodes {
        if filtered_set.insert(start.clone()) {
            if let Some(nr) = best_result.remove(start) {
                filtered_nodes.push(nr);
            } else {
                // Score the start node itself
                let nr = if let Some(nv) = graph.get_node(start) {
                    compute_score(&nv, ctx)
                } else if let Some(ov) = resolve_overlay_node(start, ctx) {
                    compute_score(&ov, ctx)
                } else {
                    continue;
                };
                filtered_nodes.push(nr);
            }
        }
    }

    // Include nodes above threshold
    for (name, count) in &hit_count {
        if *count >= threshold && !start_set.contains(name.as_str()) {
            if let Some(nr) = best_result.remove(name) {
                if filtered_set.insert(name.clone()) {
                    filtered_nodes.push(nr);
                }
            }
        }
    }

    // Sort by accumulated weight descending. total_cmp is NaN-safe and the
    // name tiebreak makes the order total and deterministic.
    filtered_nodes.sort_by(|a, b| {
        let wa = weight_sum.get(&a.name).copied().unwrap_or(0.0);
        let wb = weight_sum.get(&b.name).copied().unwrap_or(0.0);
        wb.total_cmp(&wa).then_with(|| a.name.cmp(&b.name))
    });

    // ── 4. Edge reconstruction ────────────────────────────────────────
    let mut edges: Vec<EdgeResult> = Vec::new();
    let mut seen_edges: HashSet<(String, String, String)> = HashSet::new();

    for node_name in &filtered_set {
        // Base graph outgoing edges. This bypasses `neighbors_with_overlay`
        // (the beam-launch phase already went through it), so temporal/ACL
        // visibility must be re-applied here directly via the shared helper —
        // this is a SEPARATE expansion/read site from beam_traverse's.
        for neighbor in graph.outgoing_neighbors(node_name) {
            if !ctx.is_edge_visible(neighbor.valid_from, neighbor.valid_to, &neighbor.acl) {
                continue;
            }
            if filtered_set.contains(&neighbor.target_name) {
                let key = (
                    node_name.clone(),
                    neighbor.target_name.clone(),
                    neighbor.edge_kind.clone(),
                );
                if seen_edges.insert(key) {
                    edges.push(EdgeResult {
                        source: node_name.clone(),
                        target: neighbor.target_name,
                        kind: neighbor.edge_kind,
                        field_name: neighbor.field_name,
                        weight: neighbor.edge_weight,
                        valid_from: neighbor.valid_from,
                        valid_to: neighbor.valid_to,
                        acl: neighbor.acl,
                    });
                }
            }
        }

        // Overlay edges
        for (from, to, edge) in &ctx.overlay_edges {
            if !ctx.is_edge_visible(edge.valid_from, edge.valid_to, &edge.acl) {
                continue;
            }
            if from == node_name && filtered_set.contains(to) {
                let key = (from.clone(), to.clone(), edge.kind.clone());
                if seen_edges.insert(key) {
                    edges.push(EdgeResult {
                        source: from.clone(),
                        target: to.clone(),
                        kind: edge.kind.clone(),
                        field_name: edge.field_name.clone(),
                        weight: edge.base_weight,
                        valid_from: edge.valid_from,
                        valid_to: edge.valid_to,
                        acl: edge.acl.clone(),
                    });
                }
            }
        }
    }

    // Edge reconstruction iterates `filtered_set` (a HashSet, random order),
    // so sort for a deterministic edge order — nodes are already deterministic.
    edges.sort_by(|a, b| {
        a.source
            .cmp(&b.source)
            .then_with(|| a.target.cmp(&b.target))
            .then_with(|| a.kind.cmp(&b.kind))
    });

    SubGraph {
        nodes: filtered_nodes,
        edges,
    }
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
            valid_from: None,
            valid_to: None,
            acl: Vec::new(),
        }
    }

    /// Like `make_edge` but with caller-supplied `valid_from`/`valid_to`/`acl`.
    fn make_edge_ext(
        from: &str,
        to: &str,
        kind: &str,
        valid_from: Option<u64>,
        valid_to: Option<u64>,
        acl: Vec<&str>,
    ) -> EdgeInput {
        EdgeInput {
            from: from.to_string(),
            to: to.to_string(),
            kind: kind.to_string(),
            field_name: None,
            base_weight: 1.0,
            valid_from,
            valid_to,
            acl: acl.into_iter().map(String::from).collect(),
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
                results[i].name,
                results[i].weight,
                results[i + 1].name,
                results[i + 1].weight,
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
    fn test_beam_no_duplicate_diamond() {
        // A→B, A→C, B→D, C→D: D is reachable from both B and C on the same
        // level. Without per-level dedup it would be pushed twice and survive
        // truncate(k). Beam from A, depth 2 must yield exactly one D.
        let graph = build_diamond_graph();
        let ctx = DynamicContext::default();
        let results = beam_traverse(&graph, &ctx, "A", 5, 2);

        let names: Vec<&str> = results.iter().map(|r| r.name.as_str()).collect();
        let unique: HashSet<&str> = names.iter().copied().collect();
        assert_eq!(
            unique.len(),
            names.len(),
            "Duplicate node in beam: {names:?}"
        );
        let d_count = names.iter().filter(|n| **n == "D").count();
        assert_eq!(d_count, 1, "D must appear exactly once: {names:?}");
    }

    #[test]
    fn test_beam_deterministic_with_ties() {
        // All-equal weights → sort ties resolved by name, not by the
        // non-deterministic HashMap iteration order. Repeated runs identical.
        let nodes = vec![
            make_node("A", 0.5),
            make_node("B", 0.5),
            make_node("C", 0.5),
            make_node("D", 0.5),
            make_node("E", 0.5),
        ];
        let edges = vec![
            make_edge("A", "B", "relates_to", None),
            make_edge("A", "C", "relates_to", None),
            make_edge("A", "D", "relates_to", None),
            make_edge("A", "E", "relates_to", None),
        ];
        let (g, m) = build_graph(nodes, edges);
        let graph = OrpheusGraphInner::new(g, m);
        let ctx = DynamicContext::default();

        let baseline: Vec<String> = beam_traverse(&graph, &ctx, "A", 2, 1)
            .iter()
            .map(|r| r.name.clone())
            .collect();
        for _ in 0..25 {
            let run: Vec<String> = beam_traverse(&graph, &ctx, "A", 2, 1)
                .iter()
                .map(|r| r.name.clone())
                .collect();
            assert_eq!(run, baseline, "Beam Top-K nondeterministic across runs");
        }
    }

    #[test]
    fn test_beam_deterministic_with_nan() {
        // A NaN weight makes partial_cmp non-total; total_cmp keeps a stable
        // deterministic order. build_graph sanitizes non-finite base_weight, so
        // the NaN must be injected via a route that ISN'T sanitized — a NaN
        // `semantic_boost` from the caller, which reaches compute_score and
        // produces a NaN node weight that actually hits the beam sort.
        let nodes = vec![
            make_node("A", 0.5),
            make_node("B", 0.6),
            make_node("C", 0.8),
            make_node("D", 0.4),
            make_node("E", 0.3),
        ];
        let edges = vec![
            make_edge("A", "B", "relates_to", None),
            make_edge("A", "C", "relates_to", None),
            make_edge("A", "D", "relates_to", None),
            make_edge("A", "E", "relates_to", None),
        ];
        let (g, m) = build_graph(nodes, edges);
        let graph = OrpheusGraphInner::new(g, m);
        let mut ctx = DynamicContext::default();
        // NaN boosts on two nodes → their computed weight is NaN at the sort.
        ctx.semantic_boosts.insert("B".to_string(), f32::NAN);
        ctx.semantic_boosts.insert("D".to_string(), f32::NAN);

        let baseline: Vec<String> = beam_traverse(&graph, &ctx, "A", 5, 1)
            .iter()
            .map(|r| r.name.clone())
            .collect();
        // Sanity: a NaN weight really did reach the results (else test is vacuous).
        assert!(
            beam_traverse(&graph, &ctx, "A", 5, 1)
                .iter()
                .any(|r| r.weight.is_nan()),
            "expected a NaN-weighted node to reach the beam sort"
        );
        for _ in 0..25 {
            let run: Vec<String> = beam_traverse(&graph, &ctx, "A", 5, 1)
                .iter()
                .map(|r| r.name.clone())
                .collect();
            assert_eq!(
                run, baseline,
                "Beam order nondeterministic with NaN weights"
            );
        }
    }

    #[test]
    fn test_contextual_subgraph_deterministic_tied_boosts() {
        // Tied boosts must break by name, not HashMap iteration order.
        let graph = build_chain_graph();
        let mut ctx = DynamicContext::default();
        ctx.semantic_boosts.insert("B".to_string(), 1.0);
        ctx.semantic_boosts.insert("C".to_string(), 1.0);
        ctx.semantic_boosts.insert("D".to_string(), 1.0);
        ctx.semantic_boosts.insert("E".to_string(), 1.0);

        let baseline: Vec<String> = contextual_subgraph(&graph, &ctx, 2)
            .nodes
            .iter()
            .map(|n| n.name.clone())
            .collect();
        for _ in 0..25 {
            let run: Vec<String> = contextual_subgraph(&graph, &ctx, 2)
                .nodes
                .iter()
                .map(|n| n.name.clone())
                .collect();
            assert_eq!(
                run, baseline,
                "Seed selection nondeterministic on tied boosts"
            );
        }
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
                valid_from: None,
                valid_to: None,
                acl: Vec::new(),
            },
        ));

        let results = beam_traverse(&graph, &ctx, "A", 5, 1);
        let names: Vec<&str> = results.iter().map(|r| r.name.as_str()).collect();
        assert!(
            names.contains(&"X_CUSTOM"),
            "Overlay node should appear: {names:?}"
        );
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

    // ── Multi-Beam Intersection Tests ────────────────────────────────

    fn build_diamond_graph() -> OrpheusGraphInner {
        // Diamond: A→B, A→C, B→D, C→D, D→E
        let nodes = vec![
            make_node("A", 0.7),
            make_node("B", 0.8),
            make_node("C", 0.6),
            make_node("D", 0.9),
            make_node("E", 0.5),
        ];
        let edges = vec![
            make_edge("A", "B", "relates_to", Some("partner_id")),
            make_edge("A", "C", "relates_to", Some("order_id")),
            make_edge("B", "D", "relates_to", Some("move_id")),
            make_edge("C", "D", "relates_to", Some("picking_id")),
            make_edge("D", "E", "relates_to", Some("lot_id")),
        ];
        let (g, m) = build_graph(nodes, edges);
        OrpheusGraphInner::new(g, m)
    }

    #[test]
    fn test_multi_beam_basic() {
        // Start from B and C; D is the shared intersection node
        let graph = build_diamond_graph();
        let ctx = DynamicContext::default();
        let starts = vec!["B".to_string(), "C".to_string()];
        let sg = multi_beam_intersection(&graph, &ctx, &starts, 5, 2, 2);

        let node_names: HashSet<&str> = sg.nodes.iter().map(|n| n.name.as_str()).collect();
        assert!(node_names.contains("B"), "Start B missing");
        assert!(node_names.contains("C"), "Start C missing");
        assert!(
            node_names.contains("D"),
            "Shared node D missing from intersection"
        );
    }

    #[test]
    fn test_multi_beam_threshold_filters() {
        // threshold=2: only nodes hit by BOTH beams survive
        let graph = build_diamond_graph();
        let ctx = DynamicContext::default();
        let starts = vec!["B".to_string(), "C".to_string()];
        let sg = multi_beam_intersection(&graph, &ctx, &starts, 5, 2, 2);

        let node_names: HashSet<&str> = sg.nodes.iter().map(|n| n.name.as_str()).collect();
        // E is only reachable from D which is hit by both, but E itself
        // might only be hit by beams that traverse D→E.
        // With threshold=2, E should appear only if reached by both beams.
        // Both B→D→E and C→D→E exist, so E should be hit by both.
        assert!(node_names.contains("D"), "D should be in intersection");
        assert!(
            node_names.contains("E"),
            "E reachable from both beams via D"
        );
    }

    #[test]
    fn test_multi_beam_start_nodes_always_kept() {
        let graph = build_chain_graph(); // A→B→C→D→E
        let ctx = DynamicContext::default();
        // Start from A and E — they share few nodes
        let starts = vec!["A".to_string(), "E".to_string()];
        let sg = multi_beam_intersection(&graph, &ctx, &starts, 5, 4, 2);

        let node_names: HashSet<&str> = sg.nodes.iter().map(|n| n.name.as_str()).collect();
        assert!(node_names.contains("A"), "Start A always kept");
        assert!(node_names.contains("E"), "Start E always kept");
    }

    #[test]
    fn test_multi_beam_edge_reconstruction() {
        let graph = build_diamond_graph();
        let ctx = DynamicContext::default();
        let starts = vec!["B".to_string(), "C".to_string()];
        let sg = multi_beam_intersection(&graph, &ctx, &starts, 5, 2, 2);

        let filtered_names: HashSet<&str> = sg.nodes.iter().map(|n| n.name.as_str()).collect();

        // Every edge in the result must connect two filtered nodes
        for edge in &sg.edges {
            assert!(
                filtered_names.contains(edge.source.as_str()),
                "Edge source '{}' not in filtered set",
                edge.source
            );
            assert!(
                filtered_names.contains(edge.target.as_str()),
                "Edge target '{}' not in filtered set",
                edge.target
            );
        }

        // At minimum we expect B→D and C→D edges
        assert!(!sg.edges.is_empty(), "Should have reconstructed edges");
    }

    // =======================================================================
    // Temporal validity + ACL visibility — per-op exclusion tests.
    //
    // NOTE on direction: none of the four query ops (beam_traverse, find_path,
    // contextual_subgraph, multi_beam_intersection) ever call
    // `GraphAccessor::incoming_neighbors` — they are outgoing-only forward
    // traversals (verified by grep: the only `incoming_neighbors` call sites in
    // the whole crate are the ctx-free inspection APIs in `pybridge`/`graph.rs`
    // and persist tests). So "incoming direction included where the op uses
    // it" is vacuously satisfied for all four; incoming-direction visibility
    // is instead covered directly at the accessor/delta level (see
    // `delta.rs::add_edge_temporal_and_acl_fields_carried_through_both_directions`
    // and the V2 CSR round-trip tests in `persist::snapshot`).
    // =======================================================================

    fn build_two_edge_graph(
        second_valid_from: Option<u64>,
        second_valid_to: Option<u64>,
        second_acl: Vec<&str>,
    ) -> OrpheusGraphInner {
        let nodes = vec![
            make_node("A", 0.5),
            make_node("B", 0.5),
            make_node("C", 0.5),
        ];
        let edges = vec![
            make_edge("A", "B", "relates_to", None),
            make_edge_ext(
                "B",
                "C",
                "relates_to",
                second_valid_from,
                second_valid_to,
                second_acl,
            ),
        ];
        let (g, m) = build_graph(nodes, edges);
        OrpheusGraphInner::new(g, m)
    }

    #[test]
    fn beam_traverse_excludes_temporally_invisible_edge() {
        let graph = build_two_edge_graph(None, Some(100), vec![]);

        let ctx_after = DynamicContext {
            as_of: Some(200),
            ..DynamicContext::default()
        };
        let results = beam_traverse(&graph, &ctx_after, "A", 5, 2);
        assert!(
            !results.iter().any(|r| r.name == "C"),
            "B->C expired at valid_to=100, as_of=200 -> C unreachable"
        );

        let ctx_before = DynamicContext {
            as_of: Some(50),
            ..DynamicContext::default()
        };
        let results = beam_traverse(&graph, &ctx_before, "A", 5, 2);
        assert!(
            results.iter().any(|r| r.name == "C"),
            "as_of=50 < valid_to=100 -> visible"
        );
    }

    #[test]
    fn beam_traverse_excludes_acl_invisible_edge() {
        let graph = build_two_edge_graph(None, None, vec!["secret"]);

        let ctx_none = DynamicContext::default();
        let results = beam_traverse(&graph, &ctx_none, "A", 5, 2);
        assert!(
            !results.iter().any(|r| r.name == "C"),
            "no principals -> fail closed for a tagged edge"
        );

        let ctx_match = DynamicContext {
            principals: vec!["secret".into()],
            ..DynamicContext::default()
        };
        let results = beam_traverse(&graph, &ctx_match, "A", 5, 2);
        assert!(results.iter().any(|r| r.name == "C"));
    }

    #[test]
    fn find_path_excludes_temporally_invisible_edge() {
        let graph = build_two_edge_graph(None, Some(100), vec![]);
        let ctx_after = DynamicContext {
            as_of: Some(200),
            ..DynamicContext::default()
        };
        assert!(find_path(&graph, &ctx_after, "A", "C").is_none());

        let ctx_before = DynamicContext {
            as_of: Some(50),
            ..DynamicContext::default()
        };
        assert!(find_path(&graph, &ctx_before, "A", "C").is_some());
    }

    #[test]
    fn find_path_excludes_acl_invisible_edge() {
        let graph = build_two_edge_graph(None, None, vec!["secret"]);
        let ctx_none = DynamicContext::default();
        assert!(find_path(&graph, &ctx_none, "A", "C").is_none());

        let ctx_match = DynamicContext {
            principals: vec!["secret".into()],
            ..DynamicContext::default()
        };
        assert!(find_path(&graph, &ctx_match, "A", "C").is_some());
    }

    #[test]
    fn contextual_subgraph_excludes_temporally_invisible_edge() {
        let graph = build_two_edge_graph(None, Some(100), vec![]);

        let mut ctx_after = DynamicContext {
            as_of: Some(200),
            ..DynamicContext::default()
        };
        ctx_after.semantic_boosts.insert("B".to_string(), 1.0);
        let sg = contextual_subgraph(&graph, &ctx_after, 1);
        assert!(!sg.nodes.iter().any(|n| n.name == "C"));

        let mut ctx_before = DynamicContext {
            as_of: Some(50),
            ..DynamicContext::default()
        };
        ctx_before.semantic_boosts.insert("B".to_string(), 1.0);
        let sg = contextual_subgraph(&graph, &ctx_before, 1);
        assert!(sg.nodes.iter().any(|n| n.name == "C"));
    }

    #[test]
    fn contextual_subgraph_excludes_acl_invisible_edge() {
        let graph = build_two_edge_graph(None, None, vec!["secret"]);

        let mut ctx_none = DynamicContext::default();
        ctx_none.semantic_boosts.insert("B".to_string(), 1.0);
        let sg = contextual_subgraph(&graph, &ctx_none, 1);
        assert!(!sg.nodes.iter().any(|n| n.name == "C"));

        let mut ctx_match = DynamicContext {
            principals: vec!["secret".into()],
            ..DynamicContext::default()
        };
        ctx_match.semantic_boosts.insert("B".to_string(), 1.0);
        let sg = contextual_subgraph(&graph, &ctx_match, 1);
        assert!(sg.nodes.iter().any(|n| n.name == "C"));
    }

    #[test]
    fn multi_beam_intersection_excludes_temporally_invisible_edge() {
        let graph = build_two_edge_graph(None, Some(100), vec![]);
        let starts = vec!["A".to_string()];

        let ctx_after = DynamicContext {
            as_of: Some(200),
            ..DynamicContext::default()
        };
        let sg = multi_beam_intersection(&graph, &ctx_after, &starts, 5, 2, 1);
        assert!(!sg.nodes.iter().any(|n| n.name == "C"), "node reachability");
        assert!(
            !sg.edges.iter().any(|e| e.target == "C"),
            "edge reconstruction"
        );

        let ctx_before = DynamicContext {
            as_of: Some(50),
            ..DynamicContext::default()
        };
        let sg = multi_beam_intersection(&graph, &ctx_before, &starts, 5, 2, 1);
        assert!(sg.nodes.iter().any(|n| n.name == "C"));
        assert!(sg.edges.iter().any(|e| e.target == "C"));
    }

    #[test]
    fn multi_beam_intersection_excludes_acl_invisible_edge() {
        let graph = build_two_edge_graph(None, None, vec!["secret"]);
        let starts = vec!["A".to_string()];

        let ctx_none = DynamicContext::default();
        let sg = multi_beam_intersection(&graph, &ctx_none, &starts, 5, 2, 1);
        assert!(!sg.nodes.iter().any(|n| n.name == "C"));
        assert!(!sg.edges.iter().any(|e| e.target == "C"));

        let ctx_match = DynamicContext {
            principals: vec!["secret".into()],
            ..DynamicContext::default()
        };
        let sg = multi_beam_intersection(&graph, &ctx_match, &starts, 5, 2, 1);
        assert!(sg.nodes.iter().any(|n| n.name == "C"));
        assert!(sg.edges.iter().any(|e| e.target == "C"));
    }

    // ---- PROPTEST: ctx-filtered traversal == traversal over a pre-filtered
    //      graph (matches the delta.rs `prop_delta_topology_equals_rebuild`
    //      style: build via `build_graph`, compare across two constructions of
    //      "the same logical visible graph"). Node scoring (`compute_score`)
    //      depends only on `base_weight`/`semantic_boosts`/etc — never on
    //      `pagerank_weight` or edge presence — so identical node inputs give
    //      bit-identical scores regardless of which edges are attached,
    //      making this an exact (not approximate) equivalence. ------------

    use proptest::prelude::*;

    fn tag_strat() -> impl Strategy<Value = String> {
        prop::sample::select(vec!["a", "b"]).prop_map(String::from)
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(128))]
        #[test]
        fn prop_ctx_filter_equals_prefiltered_graph(
            raw_edges in prop::collection::vec(
                (
                    0usize..5,
                    0usize..5,
                    prop::option::of(0u64..20),
                    prop::option::of(0u64..20),
                    prop::collection::vec(tag_strat(), 0..=2),
                ),
                0..12,
            ),
            as_of in prop::option::of(0u64..20),
            principals in prop::collection::vec(tag_strat(), 0..=2),
        ) {
            let names = ["n0", "n1", "n2", "n3", "n4"];
            let nodes: Vec<NodeInput> = names.iter().map(|n| make_node(n, 0.5)).collect();

            let mut edges: Vec<EdgeInput> = Vec::new();
            for (fi, ti, vf, vt, acl) in raw_edges {
                if fi == ti {
                    continue; // self-loops are irrelevant to this property
                }
                if let (Some(a), Some(b)) = (vf, vt) {
                    if a > b {
                        continue; // build_graph silently skips these; exclude from
                                  // both sides so the compared edge sets match
                    }
                }
                edges.push(make_edge_ext(
                    names[fi],
                    names[ti],
                    "rel",
                    vf,
                    vt,
                    acl.iter().map(String::as_str).collect(),
                ));
            }

            let ctx = DynamicContext {
                as_of,
                principals: principals.clone(),
                ..DynamicContext::default()
            };
            let ctx_none = DynamicContext::default();

            // Edges that pass `ctx`'s visibility check, with their
            // valid_from/valid_to/acl STRIPPED (cleared to unbounded/public).
            // ACL filtering (unlike temporal) is never "disabled" by ctx_none
            // — an empty-principals ctx still fails-closed on ANY non-empty
            // acl — so simply keeping the ORIGINAL acl on a kept edge would
            // make ctx_none re-reject it downstream, which is not what this
            // property is testing (it tests reachability equivalence, not
            // "the filtered graph is also a legal persisted graph").
            let visible_edges: Vec<EdgeInput> = edges
                .iter()
                .filter(|e| ctx.is_edge_visible(e.valid_from, e.valid_to, &e.acl))
                .cloned()
                .map(|mut e| {
                    e.valid_from = None;
                    e.valid_to = None;
                    e.acl = Vec::new();
                    e
                })
                .collect();

            let (g_full, m_full) = build_graph(nodes.clone(), edges);
            let full = OrpheusGraphInner::new(g_full, m_full);
            let (g_vis, m_vis) = build_graph(nodes, visible_edges);
            let visible = OrpheusGraphInner::new(g_vis, m_vis);

            for start in names {
                let a: Vec<(String, u32)> = beam_traverse(&full, &ctx, start, 5, 3)
                    .into_iter()
                    .map(|r| (r.name, r.weight.to_bits()))
                    .collect();
                let b: Vec<(String, u32)> = beam_traverse(&visible, &ctx_none, start, 5, 3)
                    .into_iter()
                    .map(|r| (r.name, r.weight.to_bits()))
                    .collect();
                prop_assert_eq!(a, b, "beam_traverse mismatch from {}", start);
            }

            for start in names {
                for end in names {
                    let pa = find_path(&full, &ctx, start, end)
                        .map(|p| p.into_iter().map(|s| s.node).collect::<Vec<_>>());
                    let pb = find_path(&visible, &ctx_none, start, end)
                        .map(|p| p.into_iter().map(|s| s.node).collect::<Vec<_>>());
                    prop_assert_eq!(pa, pb, "find_path mismatch {} -> {}", start, end);
                }
            }
        }
    }
}
