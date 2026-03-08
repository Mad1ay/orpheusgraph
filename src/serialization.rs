use crate::accessor::{GraphAccessor, NeighborView, NodeView};
use crate::builder::rebuild_from_serialized;
use crate::graph::OrpheusGraphInner;
use crate::types::{EdgeData, NodeData};

/// Intermediate structure for serialization.
/// petgraph `DiGraph` doesn't implement rkyv — we extract nodes + indexed edges.
#[derive(Debug, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
#[rkyv(derive(Debug))]
pub struct SerializableGraph {
    pub nodes: Vec<NodeData>,
    pub edges: Vec<SerializableEdge>,
}

/// Edge stored by node indices for compact serialization.
#[derive(Debug, Clone, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
#[rkyv(derive(Debug))]
pub struct SerializableEdge {
    pub from_idx: u32,
    pub to_idx: u32,
    pub kind: String,
    pub field_name: Option<String>,
    pub base_weight: f32,
}

/// Serialize an owned graph to rkyv bytes.
pub fn to_rkyv(graph: &OrpheusGraphInner) -> Vec<u8> {
    let inner = graph.inner_graph();

    // Extract nodes in index order
    let mut nodes: Vec<NodeData> = Vec::with_capacity(inner.node_count());
    let mut name_to_idx: std::collections::HashMap<&str, u32> = std::collections::HashMap::new();

    for node_idx in inner.node_indices() {
        let node = &inner[node_idx];
        name_to_idx.insert(&node.name, nodes.len() as u32);
        nodes.push(node.clone());
    }

    // Extract edges
    let mut edges: Vec<SerializableEdge> = Vec::with_capacity(inner.edge_count());
    for edge_idx in inner.edge_indices() {
        let (from, to) = inner.edge_endpoints(edge_idx).unwrap();
        let edge_data = &inner[edge_idx];
        let from_name = &inner[from].name;
        let to_name = &inner[to].name;
        edges.push(SerializableEdge {
            from_idx: name_to_idx[from_name.as_str()],
            to_idx: name_to_idx[to_name.as_str()],
            kind: edge_data.kind.clone(),
            field_name: edge_data.field_name.clone(),
            base_weight: edge_data.base_weight,
        });
    }

    let sg = SerializableGraph { nodes, edges };
    rkyv::to_bytes::<rkyv::rancor::Error>(&sg)
        .expect("rkyv serialization failed")
        .to_vec()
}

/// Deserialize from rkyv bytes, rebuilding the petgraph DiGraph.
///
/// This does NOT use zero-copy for the graph itself (petgraph needs owned nodes).
/// Use `ArchivedGraphView` for true zero-copy reads without rebuild.
pub fn from_rkyv_rebuild(data: &[u8]) -> Result<OrpheusGraphInner, String> {
    let sg = rkyv::from_bytes::<SerializableGraph, rkyv::rancor::Error>(data)
        .map_err(|e| format!("rkyv deserialization failed: {e}"))?;

    let edges: Vec<(usize, usize, EdgeData)> = sg
        .edges
        .into_iter()
        .map(|e| {
            (
                e.from_idx as usize,
                e.to_idx as usize,
                EdgeData {
                    kind: e.kind,
                    field_name: e.field_name,
                    base_weight: e.base_weight,
                },
            )
        })
        .collect();

    Ok(rebuild_from_serialized(sg.nodes, edges))
}

// ---------------------------------------------------------------------------
// Zero-copy archived view
// ---------------------------------------------------------------------------

/// Wrapper around archived bytes providing zero-copy graph access.
pub struct ArchivedGraphView {
    // SAFETY: _data must outlive any references to archived.
    _data: Vec<u8>,
    archived: *const ArchivedSerializableGraph,
    // Pre-built index for O(1) name lookups
    name_index: std::collections::HashMap<String, usize>,
}

// SAFETY: The archived data is immutable and the pointer is stable.
unsafe impl Send for ArchivedGraphView {}
unsafe impl Sync for ArchivedGraphView {}

impl ArchivedGraphView {
    /// Create a zero-copy view from owned bytes.
    pub fn from_bytes(data: Vec<u8>) -> Result<Self, String> {
        // Validate the archived bytes
        let archived = rkyv::access::<ArchivedSerializableGraph, rkyv::rancor::Error>(&data)
            .map_err(|e| format!("rkyv validation failed: {e}"))?;
        let ptr = archived as *const ArchivedSerializableGraph;

        // Build name index
        let mut name_index = std::collections::HashMap::new();
        for (i, node) in archived.nodes.iter().enumerate() {
            name_index.insert(node.name.to_string(), i);
        }

        Ok(Self {
            _data: data,
            archived: ptr,
            name_index,
        })
    }

    fn archived(&self) -> &ArchivedSerializableGraph {
        // SAFETY: _data is alive as long as self, pointer is stable
        unsafe { &*self.archived }
    }
}

impl GraphAccessor for ArchivedGraphView {
    fn node_count(&self) -> usize {
        self.archived().nodes.len()
    }

    fn edge_count(&self) -> usize {
        self.archived().edges.len()
    }

    fn get_node(&self, name: &str) -> Option<NodeView> {
        let &idx = self.name_index.get(name)?;
        let node = &self.archived().nodes[idx];
        Some(NodeView {
            name: node.name.to_string(),
            kind: node.kind.to_string(),
            base_weight: node.base_weight.into(),
            noise_penalty: node.noise_penalty.into(),
            pagerank_weight: node.pagerank_weight.into(),
            metadata: node
                .metadata
                .iter()
                .map(|(k, v)| (k.to_string(), v.to_string()))
                .collect(),
        })
    }

    fn outgoing_neighbors(&self, name: &str) -> Vec<NeighborView> {
        let Some(&from_idx) = self.name_index.get(name) else {
            return vec![];
        };
        let archived = self.archived();
        let from_idx_u32 = from_idx as u32;

        archived
            .edges
            .iter()
            .filter(|e| {
                let idx: u32 = e.from_idx.into();
                idx == from_idx_u32
            })
            .map(|e| {
                let to_idx: u32 = e.to_idx.into();
                let target = &archived.nodes[to_idx as usize];
                NeighborView {
                    target_name: target.name.to_string(),
                    edge_kind: e.kind.to_string(),
                    field_name: e.field_name.as_ref().map(|s| s.to_string()),
                    edge_weight: e.base_weight.into(),
                }
            })
            .collect()
    }

    fn incoming_neighbors(&self, name: &str) -> Vec<NeighborView> {
        let Some(&to_idx) = self.name_index.get(name) else {
            return vec![];
        };
        let archived = self.archived();
        let to_idx_u32 = to_idx as u32;

        archived
            .edges
            .iter()
            .filter(|e| {
                let idx: u32 = e.to_idx.into();
                idx == to_idx_u32
            })
            .map(|e| {
                let from_idx: u32 = e.from_idx.into();
                let source = &archived.nodes[from_idx as usize];
                NeighborView {
                    target_name: source.name.to_string(),
                    edge_kind: e.kind.to_string(),
                    field_name: e.field_name.as_ref().map(|s| s.to_string()),
                    edge_weight: e.base_weight.into(),
                }
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builder::build_graph;
    use crate::types::{EdgeInput, NodeInput};
    use std::collections::HashMap;

    fn make_graph() -> OrpheusGraphInner {
        let nodes = vec![
            NodeInput {
                name: "A".into(), kind: "model".into(), metadata: HashMap::new(),
                base_weight: 0.5, noise_penalty: 0.1,
            },
            NodeInput {
                name: "B".into(), kind: "model".into(), metadata: HashMap::new(),
                base_weight: 0.8, noise_penalty: 0.0,
            },
            NodeInput {
                name: "C".into(), kind: "field".into(), metadata: HashMap::new(),
                base_weight: 0.3, noise_penalty: 0.5,
            },
        ];
        let edges = vec![
            EdgeInput {
                from: "A".into(), to: "B".into(), kind: "relates_to".into(),
                field_name: Some("partner_id".into()), base_weight: 1.0,
            },
            EdgeInput {
                from: "B".into(), to: "C".into(), kind: "contains".into(),
                field_name: None, base_weight: 0.5,
            },
        ];
        let (g, m) = build_graph(nodes, edges);
        OrpheusGraphInner::new(g, m)
    }

    #[test]
    fn test_roundtrip_rebuild() {
        let graph = make_graph();
        let bytes = to_rkyv(&graph);

        let graph2 = from_rkyv_rebuild(&bytes).unwrap();
        assert_eq!(graph2.node_count(), 3);
        assert_eq!(graph2.edge_count(), 2);
        let a = graph2.get_node("A").unwrap();
        assert!(a.base_weight > 0.0);
    }

    #[test]
    fn test_roundtrip_zero_copy() {
        let graph = make_graph();
        let bytes = to_rkyv(&graph);

        let view = ArchivedGraphView::from_bytes(bytes).unwrap();
        assert_eq!(view.node_count(), 3);
        assert_eq!(view.edge_count(), 2);

        let a = view.get_node("A").unwrap();
        assert_eq!(a.kind, "model");
        assert!(a.base_weight > 0.0);

        let neighbors = view.outgoing_neighbors("A");
        assert_eq!(neighbors.len(), 1);
        assert_eq!(neighbors[0].target_name, "B");
    }

    #[test]
    fn test_archived_accessor_trait() {
        let graph = make_graph();
        let bytes = to_rkyv(&graph);
        let view = ArchivedGraphView::from_bytes(bytes).unwrap();

        let accessor: &dyn GraphAccessor = &view;
        assert_eq!(accessor.node_count(), 3);
        assert!(accessor.get_node("B").is_some());
        assert!(accessor.get_node("nonexistent").is_none());
    }
}
