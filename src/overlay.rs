use std::collections::HashMap;

use crate::accessor::{GraphAccessor, NodeView};
use crate::types::{DynamicContext, EdgeData, NodeData};

/// Neighbor entry returned by `neighbors_with_overlay`.
#[derive(Debug, Clone)]
pub struct NeighborEntry {
    /// Name of the neighbor node
    pub name: String,
    /// Edge kind
    pub edge_kind: String,
    /// Edge field name
    pub field_name: Option<String>,
    /// Edge weight
    pub edge_weight: f32,
    /// Whether this came from the base graph or the overlay
    pub is_overlay: bool,
    pub valid_from: Option<u64>,
    pub valid_to: Option<u64>,
    pub acl: Vec<String>,
}

/// Per-traversal index over a context's overlay, so a traversal does not rescan the
/// whole overlay for every node it expands.
///
/// `overlay_edges`/`overlay_nodes` are flat `Vec`s on the context (the ergonomic shape
/// for a caller assembling a request), but a beam expanding N nodes used to walk the
/// entire edge list N times — the one part of the hot path that scaled with overlay
/// size rather than with the graph. Build this once at the traversal entry point.
pub struct OverlayIndex<'a> {
    edges: HashMap<&'a str, Vec<&'a (String, String, EdgeData)>>,
    nodes: HashMap<&'a str, &'a NodeData>,
}

impl<'a> OverlayIndex<'a> {
    pub fn build(ctx: &'a DynamicContext) -> Self {
        let mut edges: HashMap<&str, Vec<&(String, String, EdgeData)>> = HashMap::new();
        for e in &ctx.overlay_edges {
            edges.entry(e.0.as_str()).or_default().push(e);
        }
        // Last definition wins, matching the linear scan's `find` semantics only when
        // names are unique; duplicates were already ambiguous.
        let nodes = ctx.overlay_nodes.iter().map(|n| (n.name.as_str(), n)).collect();
        Self { edges, nodes }
    }

    fn edges_from(&self, node_name: &str) -> &[&'a (String, String, EdgeData)] {
        self.edges.get(node_name).map_or(&[][..], |v| v.as_slice())
    }

    fn node(&self, name: &str) -> Option<&'a NodeData> {
        self.nodes.get(name).copied()
    }
}

/// Get all outgoing neighbors of a node, combining base graph edges with
/// overlay edges, THEN applying query-time edge visibility
/// (`DynamicContext::is_edge_visible`: temporal `as_of` + ACL `principals`).
///
/// This is the ONE choke point `beam_traverse`/`find_path`/
/// `contextual_subgraph` (and, transitively via `beam_traverse`, the beam-launch
/// phase of `multi_beam_intersection`) expand through, so filtering here alone
/// covers all of them — see module docs on why filtering lives in the
/// traversal/query layer and not in `GraphAccessor` impls.
pub fn neighbors_with_overlay(
    graph: &dyn GraphAccessor,
    ctx: &DynamicContext,
    node_name: &str,
) -> Vec<NeighborEntry> {
    let index = OverlayIndex::build(ctx);
    neighbors_with_overlay_indexed(graph, ctx, &index, node_name)
}

/// Same as [`neighbors_with_overlay`], but reusing an [`OverlayIndex`] built once per
/// traversal instead of per expanded node.
pub fn neighbors_with_overlay_indexed(
    graph: &dyn GraphAccessor,
    ctx: &DynamicContext,
    index: &OverlayIndex<'_>,
    node_name: &str,
) -> Vec<NeighborEntry> {
    neighbors_with_overlay_unfiltered(graph, ctx, index, node_name)
        .into_iter()
        .filter(|n| ctx.is_edge_visible(n.valid_from, n.valid_to, &n.acl))
        .collect()
}

/// Implements **max_fan_out cutoff** (Risk #8) with two bypass conditions:
/// - Node has a semantic boost in the context (Risk #8)
/// - Node has high pagerank_weight (Risk #13)
///
/// Returns the RAW combined base+overlay neighbor list, unfiltered by
/// temporal/ACL visibility — `neighbors_with_overlay` applies that filter
/// once, uniformly, over the result of this function.
fn neighbors_with_overlay_unfiltered(
    graph: &dyn GraphAccessor,
    ctx: &DynamicContext,
    index: &OverlayIndex<'_>,
    node_name: &str,
) -> Vec<NeighborEntry> {
    let mut result = Vec::new();

    // Check max_fan_out cutoff
    if let Some(node_view) = graph.get_node(node_name) {
        let base_neighbors = graph.outgoing_neighbors(node_name);

        if let Some(max_fan_out) = ctx.max_fan_out {
            if base_neighbors.len() > max_fan_out {
                // A boost keeps the hub only if it actually raises the node. Testing
                // mere presence let `semantic_boosts[hub] = 0.0` — which means "this
                // node is irrelevant to this request" — read as "keep this hub",
                // inverting the caller's intent.
                let has_semantic_boost = ctx
                    .semantic_boosts
                    .get(node_name)
                    .is_some_and(|v| *v > 0.0);
                let has_high_pagerank = ctx
                    .fan_out_pagerank_bypass
                    .is_some_and(|t| node_view.pagerank_weight > t);

                if !has_semantic_boost && !has_high_pagerank {
                    // God Object cutoff: skip base edges, but still include overlay
                    return collect_overlay_edges(index, node_name, result);
                }
            }
        }

        // Collect base edges
        for neighbor in base_neighbors {
            result.push(NeighborEntry {
                name: neighbor.target_name,
                edge_kind: neighbor.edge_kind,
                field_name: neighbor.field_name,
                edge_weight: neighbor.edge_weight,
                is_overlay: false,
                valid_from: neighbor.valid_from,
                valid_to: neighbor.valid_to,
                acl: neighbor.acl,
            });
        }
    }

    // Add overlay edges. Deliberately not subject to max_fan_out: the caller injected
    // them for this one request, so cutting them would discard what was explicitly asked
    // for. Documented, because it means max_fan_out does not bound total expanded degree.
    collect_overlay_edges(index, node_name, result)
}

/// Append overlay edges originating from `node_name` to the result vec.
fn collect_overlay_edges(
    index: &OverlayIndex<'_>,
    node_name: &str,
    mut result: Vec<NeighborEntry>,
) -> Vec<NeighborEntry> {
    for (_from, to, edge) in index.edges_from(node_name) {
        {
            result.push(NeighborEntry {
                name: to.clone(),
                edge_kind: edge.kind.clone(),
                field_name: edge.field_name.clone(),
                edge_weight: edge.base_weight,
                is_overlay: true,
                valid_from: edge.valid_from,
                valid_to: edge.valid_to,
                acl: edge.acl.clone(),
            });
        }
    }
    result
}

/// Look up an overlay node by name, returning a NodeView.
pub fn resolve_overlay_node(name: &str, ctx: &DynamicContext) -> Option<NodeView> {
    ctx.overlay_nodes
        .iter()
        .find(|n| n.name == name)
        .map(NodeView::from)
}

/// Same as [`resolve_overlay_node`], over a prebuilt index.
pub fn resolve_overlay_node_indexed(
    name: &str,
    index: &OverlayIndex<'_>,
) -> Option<NodeView> {
    index.node(name).map(NodeView::from)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builder::build_graph;
    use crate::graph::OrpheusGraphInner;
    use crate::types::{EdgeData, EdgeInput, NodeData, NodeInput};
    use std::collections::HashMap;

    /// 1 hub with 100 outgoing edges plus a cycle, so the hub carries real pagerank.
    fn hub_graph() -> (OrpheusGraphInner, ()) {
        let mut nodes = vec![make_node("hub", 1.0), make_node("other", 1.0)];
        let mut edges = vec![];
        for i in 0..100 {
            let name = format!("target_{i}");
            nodes.push(make_node(&name, 1.0));
            edges.push(make_edge("hub", &name));
            edges.push(make_edge(&name, "hub"));
        }
        let (g, m) = build_graph(nodes, edges);
        (OrpheusGraphInner::new(g, m), ())
    }

    fn make_overlay_node(name: &str) -> NodeData {
        NodeData {
            name: name.to_string(),
            kind: "virtual".to_string(),
            metadata: HashMap::new(),
            base_weight: 0.5,
            noise_penalty: 0.0,
            pagerank_weight: 0.0,
        }
    }

    fn make_overlay_edge() -> EdgeData {
        EdgeData {
            kind: "relates_to".to_string(),
            field_name: None,
            base_weight: 1.0,
            valid_from: None,
            valid_to: None,
            acl: Vec::new(),
        }
    }

    fn make_node(name: &str, weight: f32) -> NodeInput {
        NodeInput {
            name: name.to_string(),
            kind: "model".to_string(),
            metadata: HashMap::new(),
            base_weight: weight,
            noise_penalty: 0.0,
        }
    }

    fn make_edge(from: &str, to: &str) -> EdgeInput {
        EdgeInput {
            from: from.to_string(),
            to: to.to_string(),
            kind: "relates_to".to_string(),
            field_name: None,
            base_weight: 1.0,
            valid_from: None,
            valid_to: None,
            acl: Vec::new(),
        }
    }

    fn overlay_edge_data(kind: &str) -> EdgeData {
        EdgeData {
            kind: kind.to_string(),
            field_name: None,
            base_weight: 1.0,
            valid_from: None,
            valid_to: None,
            acl: Vec::new(),
        }
    }

    fn build_simple_graph() -> OrpheusGraphInner {
        let nodes = vec![
            make_node("sale.order", 1.0),
            make_node("res.partner", 1.0),
            make_node("stock.picking", 1.0),
        ];
        let edges = vec![
            make_edge("sale.order", "res.partner"),
            make_edge("sale.order", "stock.picking"),
        ];
        let (g, m) = build_graph(nodes, edges);
        OrpheusGraphInner::new(g, m)
    }

    #[test]
    fn test_base_neighbors_only() {
        let graph = build_simple_graph();
        let ctx = DynamicContext::default();
        let neighbors = neighbors_with_overlay(&graph, &ctx, "sale.order");
        assert_eq!(neighbors.len(), 2);
        let names: Vec<&str> = neighbors.iter().map(|n| n.name.as_str()).collect();
        assert!(names.contains(&"res.partner"));
        assert!(names.contains(&"stock.picking"));
        assert!(neighbors.iter().all(|n| !n.is_overlay));
    }

    #[test]
    fn test_overlay_edge_visible() {
        let graph = build_simple_graph();
        let mut ctx = DynamicContext::default();
        ctx.overlay_edges.push((
            "sale.order".to_string(),
            "x_custom".to_string(),
            overlay_edge_data("relates_to"),
        ));

        let neighbors = neighbors_with_overlay(&graph, &ctx, "sale.order");
        assert_eq!(neighbors.len(), 3);
        let overlay_n: Vec<&NeighborEntry> = neighbors.iter().filter(|n| n.is_overlay).collect();
        assert_eq!(overlay_n.len(), 1);
        assert_eq!(overlay_n[0].name, "x_custom");
    }

    #[test]
    fn test_base_graph_untouched_after_overlay() {
        let graph = build_simple_graph();
        let mut ctx = DynamicContext::default();
        ctx.overlay_edges.push((
            "sale.order".to_string(),
            "x_custom".to_string(),
            overlay_edge_data("relates_to"),
        ));

        let _ = neighbors_with_overlay(&graph, &ctx, "sale.order");
        assert_eq!(graph.edge_count(), 2);
        assert_eq!(graph.node_count(), 3);
    }

    #[test]
    fn test_max_fan_out_cutoff() {
        let mut nodes = vec![make_node("hub", 1.0)];
        let mut edges = vec![];
        for i in 0..100 {
            let name = format!("target_{i}");
            nodes.push(make_node(&name, 1.0));
            edges.push(make_edge("hub", &name));
            if i > 0 {
                edges.push(make_edge(&format!("target_{}", i - 1), &name));
            }
        }
        edges.push(make_edge("target_99", "target_0"));

        let (g, m) = build_graph(nodes, edges);
        let graph = OrpheusGraphInner::new(g, m);

        let hub = graph.get_node("hub").unwrap();
        assert!(hub.pagerank_weight <= 0.5);

        let ctx = DynamicContext {
            max_fan_out: Some(50),
            ..DynamicContext::default()
        };

        let neighbors = neighbors_with_overlay(&graph, &ctx, "hub");
        assert_eq!(neighbors.len(), 0);
    }

    #[test]
    fn test_max_fan_out_semantic_bypass() {
        let mut nodes = vec![make_node("hub", 1.0)];
        let mut edges = vec![];
        for i in 0..100 {
            let name = format!("target_{i}");
            nodes.push(make_node(&name, 1.0));
            edges.push(make_edge("hub", &name));
        }
        let (g, m) = build_graph(nodes, edges);
        let graph = OrpheusGraphInner::new(g, m);

        let ctx = DynamicContext {
            max_fan_out: Some(50),
            semantic_boosts: HashMap::from([("hub".to_string(), 2.0)]),
            ..DynamicContext::default()
        };

        let neighbors = neighbors_with_overlay(&graph, &ctx, "hub");
        assert_eq!(neighbors.len(), 100);
    }

    #[test]
    fn test_max_fan_out_pagerank_bypass() {
        let mut nodes = vec![make_node("hub", 1.0)];
        let mut edges_in = vec![];
        let mut edges_out = vec![];
        for i in 0..100 {
            let name = format!("source_{i}");
            nodes.push(make_node(&name, 1.0));
            edges_in.push(make_edge(&name, "hub"));
            edges_out.push(make_edge("hub", &name));
        }
        let mut all_edges = edges_in;
        all_edges.extend(edges_out);
        let (g, m) = build_graph(nodes, all_edges);
        let graph = OrpheusGraphInner::new(g, m);

        let hub = graph.get_node("hub").unwrap();
        assert!(hub.pagerank_weight > 0.5);

        let ctx = DynamicContext {
            max_fan_out: Some(50),
            ..DynamicContext::default()
        };

        let neighbors = neighbors_with_overlay(&graph, &ctx, "hub");
        assert_eq!(neighbors.len(), 100);
    }

    #[test]
    fn test_tenant_isolation() {
        let graph = build_simple_graph();

        let mut ctx_a = DynamicContext::default();
        ctx_a.overlay_edges.push((
            "sale.order".to_string(),
            "x_warehouse".to_string(),
            overlay_edge_data("relates_to"),
        ));

        let mut ctx_b = DynamicContext::default();
        ctx_b.overlay_edges.push((
            "sale.order".to_string(),
            "x_hr_skill".to_string(),
            overlay_edge_data("relates_to"),
        ));

        let names_a: Vec<String> = neighbors_with_overlay(&graph, &ctx_a, "sale.order")
            .iter()
            .map(|n| n.name.clone())
            .collect();
        let names_b: Vec<String> = neighbors_with_overlay(&graph, &ctx_b, "sale.order")
            .iter()
            .map(|n| n.name.clone())
            .collect();

        assert!(names_a.contains(&"x_warehouse".to_string()));
        assert!(!names_a.contains(&"x_hr_skill".to_string()));
        assert!(names_b.contains(&"x_hr_skill".to_string()));
        assert!(!names_b.contains(&"x_warehouse".to_string()));
    }

    #[test]
    fn zero_semantic_boost_does_not_rescue_a_hub_from_max_fan_out() {
        // A boost of 0.0 says "irrelevant to this request". Testing mere presence used to
        // read that as "keep this hub", inverting the caller's intent.
        let (graph, _) = hub_graph();
        let mut ctx = DynamicContext {
            max_fan_out: Some(10),
            fan_out_pagerank_bypass: None,
            ..Default::default()
        };
        ctx.semantic_boosts.insert("hub".to_string(), 0.0);
        assert!(
            neighbors_with_overlay(&graph, &ctx, "hub").is_empty(),
            "a 0.0 boost must not bypass the cutoff"
        );

        ctx.semantic_boosts.insert("hub".to_string(), 0.1);
        assert!(
            !neighbors_with_overlay(&graph, &ctx, "hub").is_empty(),
            "a positive boost must still bypass the cutoff"
        );
    }

    #[test]
    fn fan_out_pagerank_bypass_is_configurable_and_disablable() {
        let (graph, _) = hub_graph();
        let pr = graph.get_node("hub").unwrap().pagerank_weight;

        // None => no escape, so the cutoff actually bounds base expansion.
        let strict = DynamicContext {
            max_fan_out: Some(10),
            fan_out_pagerank_bypass: None,
            ..Default::default()
        };
        assert!(neighbors_with_overlay(&graph, &strict, "hub").is_empty());

        // A threshold below the node's pagerank lets it through.
        let loose = DynamicContext {
            max_fan_out: Some(10),
            fan_out_pagerank_bypass: Some((pr - 0.01).max(0.0)),
            ..Default::default()
        };
        assert!(!neighbors_with_overlay(&graph, &loose, "hub").is_empty());

        // A threshold above it does not.
        let tight = DynamicContext {
            max_fan_out: Some(10),
            fan_out_pagerank_bypass: Some(pr + 0.01),
            ..Default::default()
        };
        assert!(neighbors_with_overlay(&graph, &tight, "hub").is_empty());
    }

    #[test]
    fn overlay_index_matches_the_linear_scan() {
        // The index is a performance change only; it must not alter what is returned.
        let (graph, _) = hub_graph();
        let ctx = DynamicContext {
            overlay_nodes: vec![make_overlay_node("virt")],
            overlay_edges: vec![
                ("hub".to_string(), "virt".to_string(), make_overlay_edge()),
                ("hub".to_string(), "target_0".to_string(), make_overlay_edge()),
                ("other".to_string(), "virt".to_string(), make_overlay_edge()),
            ],
            ..Default::default()
        };
        let index = OverlayIndex::build(&ctx);
        for node in ["hub", "other", "target_0", "absent"] {
            let via_wrapper: Vec<String> = neighbors_with_overlay(&graph, &ctx, node)
                .into_iter()
                .map(|n| n.name)
                .collect();
            let via_index: Vec<String> =
                neighbors_with_overlay_indexed(&graph, &ctx, &index, node)
                    .into_iter()
                    .map(|n| n.name)
                    .collect();
            assert_eq!(via_wrapper, via_index, "mismatch on {node}");
        }
        assert_eq!(
            resolve_overlay_node("virt", &ctx).map(|v| v.name),
            resolve_overlay_node_indexed("virt", &index).map(|v| v.name)
        );
        assert!(resolve_overlay_node_indexed("absent", &index).is_none());
    }

    #[test]
    fn test_resolve_overlay_node() {
        let ctx = DynamicContext {
            overlay_nodes: vec![NodeData {
                name: "x_custom".to_string(),
                kind: "model".to_string(),
                metadata: HashMap::new(),
                base_weight: 0.5,
                noise_penalty: 0.0,
                pagerank_weight: 0.0,
            }],
            ..DynamicContext::default()
        };

        assert!(resolve_overlay_node("x_custom", &ctx).is_some());
        assert!(resolve_overlay_node("nonexistent", &ctx).is_none());
    }

    #[test]
    fn test_empty_overlay_base_only() {
        let graph = build_simple_graph();
        let ctx = DynamicContext::default();
        let neighbors = neighbors_with_overlay(&graph, &ctx, "sale.order");
        assert_eq!(neighbors.len(), 2);
        assert!(neighbors.iter().all(|n| !n.is_overlay));
    }

    // ---- temporal + ACL visibility filtering (the shared choke point) ---

    fn make_edge_full(from: &str, to: &str, edge: EdgeData) -> EdgeInput {
        EdgeInput {
            from: from.to_string(),
            to: to.to_string(),
            kind: edge.kind,
            field_name: edge.field_name,
            base_weight: edge.base_weight,
            valid_from: edge.valid_from,
            valid_to: edge.valid_to,
            acl: edge.acl,
        }
    }

    #[test]
    fn neighbors_with_overlay_filters_temporal_base_edge() {
        let mut expired = overlay_edge_data("relates_to");
        expired.valid_to = Some(100);
        let nodes = vec![make_node("a", 1.0), make_node("b", 1.0)];
        let edges = vec![make_edge_full("a", "b", expired)];
        let (g, m) = build_graph(nodes, edges);
        let graph = OrpheusGraphInner::new(g, m);

        let ctx = DynamicContext {
            as_of: Some(200), // past valid_to=100 -> invisible
            ..DynamicContext::default()
        };
        assert!(neighbors_with_overlay(&graph, &ctx, "a").is_empty());

        let ctx_within = DynamicContext {
            as_of: Some(50), // before valid_to=100 -> visible
            ..DynamicContext::default()
        };
        assert_eq!(neighbors_with_overlay(&graph, &ctx_within, "a").len(), 1);

        // No as_of at all -> no temporal filtering, always visible.
        let ctx_unfiltered = DynamicContext::default();
        assert_eq!(
            neighbors_with_overlay(&graph, &ctx_unfiltered, "a").len(),
            1
        );
    }

    #[test]
    fn neighbors_with_overlay_filters_acl_base_edge() {
        let mut tagged = overlay_edge_data("relates_to");
        tagged.acl = vec!["team-x".into()];
        let nodes = vec![make_node("a", 1.0), make_node("b", 1.0)];
        let edges = vec![make_edge_full("a", "b", tagged)];
        let (g, m) = build_graph(nodes, edges);
        let graph = OrpheusGraphInner::new(g, m);

        // No principals -> fail closed for a tagged edge.
        let ctx_none = DynamicContext::default();
        assert!(neighbors_with_overlay(&graph, &ctx_none, "a").is_empty());

        // Wrong principal -> still invisible.
        let ctx_wrong = DynamicContext {
            principals: vec!["team-y".into()],
            ..DynamicContext::default()
        };
        assert!(neighbors_with_overlay(&graph, &ctx_wrong, "a").is_empty());

        // Matching principal -> visible.
        let ctx_match = DynamicContext {
            principals: vec!["team-x".into()],
            ..DynamicContext::default()
        };
        assert_eq!(neighbors_with_overlay(&graph, &ctx_match, "a").len(), 1);
    }

    #[test]
    fn neighbors_with_overlay_filters_overlay_edge_acl() {
        // Overlay (virtual) edges are `EdgeData` too — same visibility rule.
        let graph = build_simple_graph();
        let mut tagged = overlay_edge_data("relates_to");
        tagged.acl = vec!["secret".into()];
        let mut ctx = DynamicContext::default();
        ctx.overlay_edges
            .push(("sale.order".to_string(), "x_custom".to_string(), tagged));

        let names: Vec<String> = neighbors_with_overlay(&graph, &ctx, "sale.order")
            .iter()
            .map(|n| n.name.clone())
            .collect();
        assert!(
            !names.contains(&"x_custom".to_string()),
            "untagged principal must not see the ACL-tagged overlay edge"
        );

        ctx.principals.push("secret".to_string());
        let names: Vec<String> = neighbors_with_overlay(&graph, &ctx, "sale.order")
            .iter()
            .map(|n| n.name.clone())
            .collect();
        assert!(names.contains(&"x_custom".to_string()));
    }
}
