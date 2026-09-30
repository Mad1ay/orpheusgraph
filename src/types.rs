use std::collections::{HashMap, HashSet};

/// Data stored in each graph node.
#[derive(
    Debug,
    Clone,
    serde::Serialize,
    serde::Deserialize,
    rkyv::Archive,
    rkyv::Serialize,
    rkyv::Deserialize,
)]
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
#[derive(
    Debug,
    Clone,
    serde::Serialize,
    serde::Deserialize,
    rkyv::Archive,
    rkyv::Serialize,
    rkyv::Deserialize,
)]
#[rkyv(derive(Debug))]
pub struct EdgeData {
    /// Edge type: "inherits", "relates_to", "depends_on", "contains", "describes", etc.
    pub kind: String,
    /// How nodes are linked (e.g. "partner_id")
    pub field_name: Option<String>,
    /// Static edge strength. Normalized to [0.0, 1.0].
    pub base_weight: f32,
    /// Opaque caller-defined valid-time instant: the edge is valid from this
    /// point on (inclusive). `None` = unbounded on this side. Half-open
    /// interval together with `valid_to`: `[valid_from, valid_to)`. Query-time
    /// only — never affects PageRank/base metrics, only `DynamicContext`-aware
    /// traversal (`DynamicContext::is_edge_visible`).
    pub valid_from: Option<u64>,
    /// Opaque caller-defined valid-time instant: the edge stops being valid at
    /// this point (exclusive). `None` = unbounded on this side.
    pub valid_to: Option<u64>,
    /// ACL tags. Empty = public (always visible). Non-empty = visible only to
    /// a `DynamicContext` whose `principals` intersect this set. Normalized
    /// (sorted + deduped) at every ingestion boundary — see [`normalize_acl`].
    pub acl: Vec<String>,
}

/// Canonicalize an edge ACL tag list: sort + dedup in place.
///
/// Required at every edge ingestion boundary (builder + persistent `apply`) so
/// that two logically-identical edges whose caller supplied `acl` in a
/// different order produce byte-identical snapshots (mirrors why `CsrNode`
/// stores metadata as a sorted `Vec` instead of archiving the `HashMap`
/// as-is). Idempotent, so re-normalizing already-canonical data (e.g. on WAL
/// replay of a previously-applied op) is a cheap no-op.
pub fn normalize_acl(acl: &mut Vec<String>) {
    acl.sort();
    acl.dedup();
}

/// Lightweight edge result returned by inspection API.
#[derive(Debug, Clone)]
pub struct EdgeResult {
    pub source: String,
    pub target: String,
    pub kind: String,
    pub field_name: Option<String>,
    pub weight: f32,
    pub valid_from: Option<u64>,
    pub valid_to: Option<u64>,
    pub acl: Vec<String>,
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
    /// Semantic bonus per node (typically from embedding similarity).
    ///
    /// ADDITIVE, not a multiplier: the value enters the score as
    /// `w_semantic * value`, added to the base term. A node with
    /// `base_weight == 0.0` is therefore still liftable, which a multiplier
    /// could not do.
    pub semantic_boosts: HashMap<String, f32>,

    /// Virtual overlay: temporary nodes visible only during this traversal
    pub overlay_nodes: Vec<NodeData>,
    pub overlay_edges: Vec<(String, String, EdgeData)>, // (from, to, edge)

    /// Per-request weight bonus (e.g. project-specific usage stats).
    ///
    /// ADDITIVE despite the name: the value enters as `w_override * value`,
    /// added alongside `base_weight` — it does NOT replace it. A node with
    /// `base_weight = 0.3` and an entry of `0.5` scores `0.8`, not `0.5`.
    pub weight_overrides: HashMap<String, f32>,

    /// Scoring coefficients — configurable for A/B testing without Rust recompile
    pub w_base: f32, // default 1.0
    pub w_semantic: f32, // default 1.5
    pub w_noise: f32,    // default 1.0
    pub w_override: f32, // default 1.0

    /// Domain-aware noise filter (e.g. "technical", "audit", "messaging")
    /// Nodes tagged with these domains get boosted noise_penalty
    pub noise_tags: HashSet<String>,

    /// Degree cutoff for "God Object" nodes (e.g. res.partner with 1000+ edges).
    ///
    /// NOT a hard bound on expanded degree: a node escapes the cutoff if it
    /// carries a positive `semantic_boosts` entry or clears
    /// `fan_out_pagerank_bypass`, and overlay edges are never subject to it.
    /// Set `fan_out_pagerank_bypass: None` to remove the PageRank escape.
    pub max_fan_out: Option<usize>,

    /// `pagerank_weight` above which a node escapes the `max_fan_out` cutoff,
    /// on the grounds that a structurally central node is worth expanding even
    /// when it is wide. `None` disables the escape, which is what makes
    /// `max_fan_out` an actual bound on base-edge expansion.
    ///
    /// The default of `Some(0.5)` preserves historical behaviour, but note it
    /// is an ABSOLUTE threshold against a graph-dependent distribution: on a
    /// flat-degree graph nothing clears it and the escape never fires, while on
    /// a hub-heavy graph many nodes clear it and `max_fan_out` does nothing.
    /// Pick it from your own graph's PageRank spread rather than inheriting it.
    pub fan_out_pagerank_bypass: Option<f32>,

    /// Valid-time instant for temporal filtering. `None` = no temporal
    /// filtering (every edge passes regardless of `valid_from`/`valid_to`).
    /// `Some(t)` filters to edges whose `[valid_from, valid_to)` window
    /// contains `t` (see [`DynamicContext::is_edge_visible`]).
    pub as_of: Option<u64>,

    /// ACL principals held by the caller. An edge with a non-empty `acl` is
    /// visible only if it intersects this set; an empty `acl` is always
    /// public. An empty `principals` therefore fails CLOSED for any
    /// ACL-tagged edge (no tag can ever match). Callers should normalize
    /// (sort + dedup) at construction; [`crate::pybridge`]'s FFI boundary does
    /// this for Python callers.
    pub principals: Vec<String>,
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
            fan_out_pagerank_bypass: Some(0.5),
            as_of: None,
            principals: Vec::new(),
        }
    }
}

impl DynamicContext {
    /// Query-time edge visibility: temporal validity AND edge-level ACL. Both
    /// checks must pass. This is the ONE shared helper every expansion/read
    /// site in the traversal/query layer calls — accessors themselves stay
    /// ctx-free (see `crate::accessor` / `crate::persist::snapshot` module docs).
    ///
    /// **Temporal** (`[valid_from, valid_to)`, half-open): `ctx.as_of == None`
    /// disables temporal filtering entirely (every edge passes). `Some(t)`
    /// passes iff `(valid_from.is_none() || t >= valid_from) &&
    /// (valid_to.is_none() || t < valid_to)` — `t == valid_from` passes,
    /// `t == valid_to` fails (exclusive upper bound).
    ///
    /// **ACL**: an edge with an empty `acl` is public and always passes.
    /// A non-empty `acl` passes iff it intersects `ctx.principals`. An empty
    /// `ctx.principals` therefore fails CLOSED for any tagged edge — this is
    /// deliberate (unauthenticated/anonymous context sees no tagged data,
    /// never "sees everything because nothing was checked").
    ///
    /// Base metrics (PageRank, node/edge counts, etc.) are computed over the
    /// FULL untouched graph — this helper is consulted ONLY by the query
    /// layer (`beam_traverse`, `find_path`, `contextual_subgraph`,
    /// `multi_beam_intersection`), never by scoring or storage.
    pub fn is_edge_visible(
        &self,
        valid_from: Option<u64>,
        valid_to: Option<u64>,
        acl: &[String],
    ) -> bool {
        let temporal_ok = match self.as_of {
            None => true,
            Some(t) => valid_from.is_none_or(|vf| t >= vf) && valid_to.is_none_or(|vt| t < vt),
        };
        if !temporal_ok {
            return false;
        }
        acl.is_empty()
            || acl
                .iter()
                .any(|tag| self.principals.iter().any(|p| p == tag))
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
    pub valid_from: Option<u64>,
    pub valid_to: Option<u64>,
    pub acl: Vec<String>,
}
