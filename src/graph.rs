use std::collections::HashMap;

use petgraph::graph::{DiGraph, NodeIndex};
use petgraph::visit::EdgeRef;

use crate::types::{EdgeData, EdgeResult, NodeData};

/// Immutable wrapper around a `petgraph::DiGraph`.
///
/// The graph cannot be mutated after construction. All per-request
/// customization is done through `DynamicContext` (ephemeral overlays).
pub struct OrpheusGraphInner {
    graph: DiGraph<NodeData, EdgeData>,
    index_map: HashMap<String, NodeIndex>,
}

impl OrpheusGraphInner {
    /// Create a new wrapper from a built graph and its index map.
    pub fn new(
        graph: DiGraph<NodeData, EdgeData>,
        index_map: HashMap<String, NodeIndex>,
    ) -> Self {
        Self { graph, index_map }
    }

    /// Total number of nodes in the graph.
    pub fn node_count(&self) -> usize {
        self.graph.node_count()
    }

    /// Total number of edges in the graph.
    pub fn edge_count(&self) -> usize {
        self.graph.edge_count()
    }

    /// Look up a node by its unique name.
    pub fn get_node(&self, name: &str) -> Option<&NodeData> {
        self.index_map.get(name).map(|idx| &self.graph[*idx])
    }

    /// Get the `NodeIndex` for a given node name.
    pub fn get_index(&self, name: &str) -> Option<NodeIndex> {
        self.index_map.get(name).copied()
    }

    /// Return all outgoing edges from the given node.
    pub fn outgoing_edges(&self, name: &str) -> Vec<EdgeResult> {
        let idx = match self.index_map.get(name) {
            Some(idx) => *idx,
            None => return vec![],
        };
        let source_name = &self.graph[idx].name;

        self.graph
            .edges_directed(idx, petgraph::Direction::Outgoing)
            .map(|edge_ref| {
                let target_idx = edge_ref.target();
                let edge_data = edge_ref.weight();
                EdgeResult {
                    source: source_name.clone(),
                    target: self.graph[target_idx].name.clone(),
                    kind: edge_data.kind.clone(),
                    field_name: edge_data.field_name.clone(),
                    weight: edge_data.base_weight,
                }
            })
            .collect()
    }

    /// Return all incoming edges to the given node.
    pub fn incoming_edges(&self, name: &str) -> Vec<EdgeResult> {
        let idx = match self.index_map.get(name) {
            Some(idx) => *idx,
            None => return vec![],
        };
        let target_name = &self.graph[idx].name;

        self.graph
            .edges_directed(idx, petgraph::Direction::Incoming)
            .map(|edge_ref| {
                let source_idx = edge_ref.source();
                let edge_data = edge_ref.weight();
                EdgeResult {
                    source: self.graph[source_idx].name.clone(),
                    target: target_name.clone(),
                    kind: edge_data.kind.clone(),
                    field_name: edge_data.field_name.clone(),
                    weight: edge_data.base_weight,
                }
            })
            .collect()
    }

    /// Borrow the underlying petgraph `DiGraph` (for traversal algorithms).
    pub fn inner_graph(&self) -> &DiGraph<NodeData, EdgeData> {
        &self.graph
    }

    /// Borrow the name → NodeIndex mapping.
    pub fn index_map(&self) -> &HashMap<String, NodeIndex> {
        &self.index_map
    }
}
