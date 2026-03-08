use std::collections::HashMap;

use petgraph::graph::DiGraph;
use petgraph::visit::EdgeRef;

use crate::types::{EdgeData, NodeData};

/// Lightweight view of a node — works for both owned `NodeData` and archived `ArchivedNodeData`.
#[derive(Debug, Clone)]
pub struct NodeView {
    pub name: String,
    pub kind: String,
    pub base_weight: f32,
    pub noise_penalty: f32,
    pub pagerank_weight: f32,
    pub metadata: HashMap<String, String>,
}

impl From<&NodeData> for NodeView {
    fn from(n: &NodeData) -> Self {
        Self {
            name: n.name.clone(),
            kind: n.kind.clone(),
            base_weight: n.base_weight,
            noise_penalty: n.noise_penalty,
            pagerank_weight: n.pagerank_weight,
            metadata: n.metadata.clone(),
        }
    }
}

/// Lightweight view of a neighbor edge + target.
#[derive(Debug, Clone)]
pub struct NeighborView {
    pub target_name: String,
    pub edge_kind: String,
    pub field_name: Option<String>,
    pub edge_weight: f32,
}

/// Unified read-only graph access for both owned and archived graphs.
pub trait GraphAccessor: Send + Sync {
    fn node_count(&self) -> usize;
    fn edge_count(&self) -> usize;
    fn get_node(&self, name: &str) -> Option<NodeView>;
    fn outgoing_neighbors(&self, name: &str) -> Vec<NeighborView>;
    fn incoming_neighbors(&self, name: &str) -> Vec<NeighborView>;
}

// ---------------------------------------------------------------------------
// Impl for owned OrpheusGraphInner
// ---------------------------------------------------------------------------

use crate::graph::OrpheusGraphInner;

impl GraphAccessor for OrpheusGraphInner {
    fn node_count(&self) -> usize {
        self.inner_graph().node_count()
    }

    fn edge_count(&self) -> usize {
        self.inner_graph().edge_count()
    }

    fn get_node(&self, name: &str) -> Option<NodeView> {
        let idx = self.get_index(name)?;
        Some(NodeView::from(&self.inner_graph()[idx]))
    }

    fn outgoing_neighbors(&self, name: &str) -> Vec<NeighborView> {
        graph_neighbors(self.inner_graph(), self.index_map(), name, petgraph::Direction::Outgoing)
    }

    fn incoming_neighbors(&self, name: &str) -> Vec<NeighborView> {
        graph_neighbors(self.inner_graph(), self.index_map(), name, petgraph::Direction::Incoming)
    }
}

fn graph_neighbors(
    graph: &DiGraph<NodeData, EdgeData>,
    index_map: &HashMap<String, petgraph::graph::NodeIndex>,
    name: &str,
    direction: petgraph::Direction,
) -> Vec<NeighborView> {
    let idx = match index_map.get(name) {
        Some(&idx) => idx,
        None => return vec![],
    };

    graph
        .edges_directed(idx, direction)
        .map(|edge_ref| {
            let other_idx = match direction {
                petgraph::Direction::Outgoing => edge_ref.target(),
                petgraph::Direction::Incoming => edge_ref.source(),
            };
            let other_node = &graph[other_idx];
            let edge_data = edge_ref.weight();
            NeighborView {
                target_name: other_node.name.clone(),
                edge_kind: edge_data.kind.clone(),
                field_name: edge_data.field_name.clone(),
                edge_weight: edge_data.base_weight,
            }
        })
        .collect()
}
