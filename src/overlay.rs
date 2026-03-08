use crate::accessor::{GraphAccessor, NodeView};
use crate::types::DynamicContext;

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
}

/// Get all outgoing neighbors of a node, combining base graph edges with overlay edges.
///
/// Implements **max_fan_out cutoff** (Risk #8) with two bypass conditions:
/// - Node has a semantic boost in the context (Risk #8)
/// - Node has high pagerank_weight (Risk #13)
pub fn neighbors_with_overlay(
    graph: &dyn GraphAccessor,
    ctx: &DynamicContext,
    node_name: &str,
) -> Vec<NeighborEntry> {
    let mut result = Vec::new();

    // Check max_fan_out cutoff
    if let Some(node_view) = graph.get_node(node_name) {
        let base_neighbors = graph.outgoing_neighbors(node_name);

        if let Some(max_fan_out) = ctx.max_fan_out {
            if base_neighbors.len() > max_fan_out {
                let has_semantic_boost = ctx.semantic_boosts.contains_key(node_name);
                let has_high_pagerank = node_view.pagerank_weight > 0.5;

                if !has_semantic_boost && !has_high_pagerank {
                    // God Object cutoff: skip base edges, but still include overlay
                    return collect_overlay_edges(ctx, node_name, result);
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
            });
        }
    }

    // Add overlay edges
    collect_overlay_edges(ctx, node_name, result)
}

/// Append overlay edges originating from `node_name` to the result vec.
fn collect_overlay_edges(
    ctx: &DynamicContext,
    node_name: &str,
    mut result: Vec<NeighborEntry>,
) -> Vec<NeighborEntry> {
    for (from, to, edge) in &ctx.overlay_edges {
        if from == node_name {
            result.push(NeighborEntry {
                name: to.clone(),
                edge_kind: edge.kind.clone(),
                field_name: edge.field_name.clone(),
                edge_weight: edge.base_weight,
                is_overlay: true,
            });
        }
    }
    result
}

/// Look up an overlay node by name, returning a NodeView.
pub fn resolve_overlay_node(name: &str, ctx: &DynamicContext) -> Option<NodeView> {
    ctx.overlay_nodes.iter().find(|n| n.name == name).map(NodeView::from)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builder::build_graph;
    use crate::graph::OrpheusGraphInner;
    use crate::types::{EdgeData, EdgeInput, NodeData, NodeInput};
    use std::collections::HashMap;

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
            EdgeData {
                kind: "relates_to".to_string(),
                field_name: None,
                base_weight: 1.0,
            },
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
            EdgeData {
                kind: "relates_to".to_string(),
                field_name: None,
                base_weight: 1.0,
            },
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
            EdgeData { kind: "relates_to".to_string(), field_name: None, base_weight: 1.0 },
        ));

        let mut ctx_b = DynamicContext::default();
        ctx_b.overlay_edges.push((
            "sale.order".to_string(),
            "x_hr_skill".to_string(),
            EdgeData { kind: "relates_to".to_string(), field_name: None, base_weight: 1.0 },
        ));

        let names_a: Vec<String> = neighbors_with_overlay(&graph, &ctx_a, "sale.order")
            .iter().map(|n| n.name.clone()).collect();
        let names_b: Vec<String> = neighbors_with_overlay(&graph, &ctx_b, "sale.order")
            .iter().map(|n| n.name.clone()).collect();

        assert!(names_a.contains(&"x_warehouse".to_string()));
        assert!(!names_a.contains(&"x_hr_skill".to_string()));
        assert!(names_b.contains(&"x_hr_skill".to_string()));
        assert!(!names_b.contains(&"x_warehouse".to_string()));
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
}
