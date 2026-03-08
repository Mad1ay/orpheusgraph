use std::collections::{HashMap, HashSet};

/// Data stored in each graph node.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize,
         rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
#[rkyv(derive(Debug))]
pub struct NodeData {
    /// Unique key: "sale.order", "users_table", "AccountEntity"
    pub name: String,
    /// Node type: "model", "field", "module", "doc", "table", etc.
    pub kind: String,
    /// Arbitrary key-value metadata
    pub metadata: HashMap<String, String>,
    /// Static weight (usage frequency, computed offline). Normalized to [0.0, 1.0].
    pub base_weight: f32,
    /// Static penalty for system/technical nodes. Range [0.0, 1.0].
    pub noise_penalty: f32,
    /// PageRank-based weight for God Object detection. Computed at build time.
    pub pagerank_weight: f32,
}

/// Data stored on each graph edge.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize,
         rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
#[rkyv(derive(Debug))]
pub struct EdgeData {
    /// Edge type: "inherits", "relates_to", "depends_on", "contains", "describes", etc.
    pub kind: String,
    /// How nodes are linked (e.g. "partner_id")
    pub field_name: Option<String>,
    /// Static edge strength. Normalized to [0.0, 1.0].
    pub base_weight: f32,
}

/// Lightweight edge result returned by inspection API.
#[derive(Debug, Clone)]
pub struct EdgeResult {
    pub source: String,
    pub target: String,
    pub kind: String,
    pub field_name: Option<String>,
    pub weight: f32,
}

/// Result of scoring a node. Carries total weight + breakdown per component.
#[derive(Debug, Clone)]
pub struct NodeResult {
    pub name: String,
    pub kind: String,
    /// Total computed weight: W_total
    pub weight: f32,
    /// w_base * base_weight
    pub base_component: f32,
    /// w_semantic * semantic_boost
    pub semantic_component: f32,
    /// Effective noise factor applied (multiplicative)
    pub noise_component: f32,
    /// w_override * weight_override
    pub override_component: f32,
}

impl NodeResult {
    /// Return a breakdown of all score components for debugging.
    pub fn explain_score(&self) -> HashMap<String, f32> {
        HashMap::from([
            ("base".into(), self.base_component),
            ("semantic".into(), self.semantic_component),
            ("noise".into(), self.noise_component),
            ("override".into(), self.override_component),
            ("total".into(), self.weight),
        ])
    }
}

/// A step in a path returned by `find_path`.
#[derive(Debug, Clone)]
pub struct PathStep {
    pub node: String,
    pub edge_kind: String,
    pub field_name: String,
    pub direction: String, // "outgoing" | "incoming"
}

/// Per-request traversal context. Ephemeral — created per call, never stored.
///
/// The base graph is immutable; all per-request customization goes through this struct.
#[derive(Debug, Clone)]
pub struct DynamicContext {
    /// Semantic boosts: node_name → multiplier (from embedding similarity)
    pub semantic_boosts: HashMap<String, f32>,

    /// Virtual overlay: temporary nodes visible only during this traversal
    pub overlay_nodes: Vec<NodeData>,
    pub overlay_edges: Vec<(String, String, EdgeData)>, // (from, to, edge)

    /// Per-request weight overrides (e.g. project-specific usage stats)
    pub weight_overrides: HashMap<String, f32>,

    /// Scoring coefficients — configurable for A/B testing without Rust recompile
    pub w_base: f32,     // default 1.0
    pub w_semantic: f32, // default 1.5
    pub w_noise: f32,    // default 1.0
    pub w_override: f32, // default 1.0

    /// Domain-aware noise filter (e.g. "technical", "audit", "messaging")
    /// Nodes tagged with these domains get boosted noise_penalty
    pub noise_tags: HashSet<String>,

    /// Degree cutoff for "God Object" nodes (e.g. res.partner with 1000+ edges)
    pub max_fan_out: Option<usize>,
}

impl Default for DynamicContext {
    fn default() -> Self {
        Self {
            semantic_boosts: HashMap::new(),
            overlay_nodes: Vec::new(),
            overlay_edges: Vec::new(),
            weight_overrides: HashMap::new(),
            w_base: 1.0,
            w_semantic: 1.5,
            w_noise: 1.0,
            w_override: 1.0,
            noise_tags: HashSet::new(),
            max_fan_out: None,
        }
    }
}

/// Compact subgraph extracted by `contextual_subgraph`.
#[derive(Debug, Clone)]
pub struct SubGraph {
    pub nodes: Vec<NodeResult>,
    pub edges: Vec<EdgeResult>,
}

/// Input format for building nodes (accepted by `build_graph`).
#[derive(Debug, Clone)]
pub struct NodeInput {
    pub name: String,
    pub kind: String,
    pub metadata: HashMap<String, String>,
    pub base_weight: f32,
    pub noise_penalty: f32,
}

/// Input format for building edges (accepted by `build_graph`).
#[derive(Debug, Clone)]
pub struct EdgeInput {
    pub from: String,
    pub to: String,
    pub kind: String,
    pub field_name: Option<String>,
    pub base_weight: f32,
}
