use std::collections::HashMap;

use orpheusgraph::builder::build_graph;
use orpheusgraph::graph::OrpheusGraphInner;
use orpheusgraph::types::{EdgeInput, NodeInput};

fn make_node(name: &str, kind: &str, weight: f32) -> NodeInput {
    NodeInput {
        name: name.to_string(),
        kind: kind.to_string(),
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

/// Test: basic graph construction and inspection via OrpheusGraphInner
#[test]
fn test_graph_wrapper_basic() {
    let nodes = vec![
        make_node("sale.order", "model", 0.7),
        make_node("res.partner", "model", 0.8),
        make_node("stock.picking", "model", 0.5),
    ];
    let edges = vec![
        make_edge("sale.order", "res.partner", "relates_to", Some("partner_id")),
        make_edge("sale.order", "stock.picking", "relates_to", Some("origin")),
    ];

    let (digraph, index_map) = build_graph(nodes, edges);
    let graph = OrpheusGraphInner::new(digraph, index_map);

    assert_eq!(graph.node_count(), 3);
    assert_eq!(graph.edge_count(), 2);
}

/// Test: get_node returns correct data
#[test]
fn test_get_node() {
    let nodes = vec![make_node("sale.order", "model", 0.7)];
    let (digraph, index_map) = build_graph(nodes, vec![]);
    let graph = OrpheusGraphInner::new(digraph, index_map);

    let node = graph.get_node("sale.order").expect("node should exist");
    assert_eq!(node.name, "sale.order");
    assert_eq!(node.kind, "model");
    assert!((node.base_weight - 1.0).abs() < 0.001); // single node → normalized to 1.0
}

/// Test: get_node returns None for nonexistent node
#[test]
fn test_get_node_missing() {
    let (digraph, index_map) = build_graph(vec![], vec![]);
    let graph = OrpheusGraphInner::new(digraph, index_map);
    assert!(graph.get_node("nonexistent").is_none());
}

/// Test: outgoing edges
#[test]
fn test_outgoing_edges() {
    let nodes = vec![
        make_node("sale.order", "model", 1.0),
        make_node("res.partner", "model", 1.0),
        make_node("stock.picking", "model", 1.0),
    ];
    let edges = vec![
        make_edge("sale.order", "res.partner", "relates_to", Some("partner_id")),
        make_edge("sale.order", "stock.picking", "relates_to", Some("origin")),
    ];

    let (digraph, index_map) = build_graph(nodes, edges);
    let graph = OrpheusGraphInner::new(digraph, index_map);

    let out = graph.outgoing_edges("sale.order");
    assert_eq!(out.len(), 2);

    // Verify edge content
    let targets: Vec<&str> = out.iter().map(|e| e.target.as_str()).collect();
    assert!(targets.contains(&"res.partner"));
    assert!(targets.contains(&"stock.picking"));

    // All edges from sale.order
    for e in &out {
        assert_eq!(e.source, "sale.order");
        assert_eq!(e.kind, "relates_to");
    }
}

/// Test: incoming edges
#[test]
fn test_incoming_edges() {
    let nodes = vec![
        make_node("sale.order", "model", 1.0),
        make_node("res.partner", "model", 1.0),
    ];
    let edges = vec![make_edge(
        "sale.order",
        "res.partner",
        "relates_to",
        Some("partner_id"),
    )];

    let (digraph, index_map) = build_graph(nodes, edges);
    let graph = OrpheusGraphInner::new(digraph, index_map);

    let inc = graph.incoming_edges("res.partner");
    assert_eq!(inc.len(), 1);
    assert_eq!(inc[0].source, "sale.order");
    assert_eq!(inc[0].target, "res.partner");
    assert_eq!(inc[0].field_name, Some("partner_id".to_string()));
}

/// Test: edges for nonexistent node return empty
#[test]
fn test_edges_missing_node() {
    let (digraph, index_map) = build_graph(vec![], vec![]);
    let graph = OrpheusGraphInner::new(digraph, index_map);

    assert!(graph.outgoing_edges("nope").is_empty());
    assert!(graph.incoming_edges("nope").is_empty());
}

/// Test: large graph builds without panic
#[test]
fn test_large_graph() {
    let n = 50_000;
    let mut nodes: Vec<NodeInput> = Vec::with_capacity(n);
    let mut edges: Vec<EdgeInput> = Vec::with_capacity(n);

    for i in 0..n {
        nodes.push(NodeInput {
            name: format!("node_{i}"),
            kind: "model".to_string(),
            metadata: HashMap::new(),
            base_weight: (i as f32) + 1.0,
            noise_penalty: 0.0,
        });
        if i > 0 {
            edges.push(EdgeInput {
                from: format!("node_{}", i - 1),
                to: format!("node_{i}"),
                kind: "relates_to".to_string(),
                field_name: None,
                base_weight: 1.0,
            });
        }
    }

    let (digraph, index_map) = build_graph(nodes, edges);
    let graph = OrpheusGraphInner::new(digraph, index_map);

    assert_eq!(graph.node_count(), n);
    assert_eq!(graph.edge_count(), n - 1);
    assert!(graph.get_node("node_0").is_some());
    assert!(graph.get_node("node_49999").is_some());
}
