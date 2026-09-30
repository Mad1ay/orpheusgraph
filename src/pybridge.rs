use std::collections::{HashMap, HashSet};

use pyo3::exceptions::{PyFileNotFoundError, PyKeyError, PyOSError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList};

use crate::types::{normalize_acl, EdgeData, NodeData};

use crate::accessor::GraphAccessor;
use crate::builder::build_graph;
use crate::delta::{DeltaAccessor, Op};
use crate::graph::OrpheusGraphInner;
use crate::persist::{BaseMode, FsyncPolicy, PersistError, PersistentGraph, Validate};
use crate::scoring::compute_score;
use crate::serialization::{to_rkyv, ArchivedGraphView};
use crate::traversal;
use crate::types::{DynamicContext, EdgeInput, NodeInput};

// ---------------------------------------------------------------------------
// Persistence exceptions (surface PersistError variants Python code branches on)
// ---------------------------------------------------------------------------

pyo3::create_exception!(
    orpheusgraph,
    ConflictError,
    pyo3::exceptions::PyException,
    "Optimistic-CAS conflict: the store advanced past the caller's expected_seq."
);
pyo3::create_exception!(
    orpheusgraph,
    CorruptError,
    pyo3::exceptions::PyException,
    "The on-disk store is structurally corrupt or an unsupported format version."
);

/// Map a `PersistError` to the closest Python exception. Corruption and CAS
/// conflicts get dedicated types so callers can `except` them precisely; the
/// rest fall back to stdlib exceptions with the Rust message preserved.
fn map_persist_err(e: PersistError) -> PyErr {
    // Render the message BEFORE moving `e` into the match (borrow, then match on
    // discriminant only — never touch the message after the move).
    let msg = e.to_string();
    match e {
        PersistError::Conflict { .. } => ConflictError::new_err(msg),
        PersistError::NotFound(_) => PyFileNotFoundError::new_err(msg),
        PersistError::Delta(_) => PyValueError::new_err(msg),
        PersistError::Poisoned | PersistError::LockHeld(_) => PyRuntimeError::new_err(msg),
        PersistError::Corrupt(_) | PersistError::UnsupportedVersion { .. } => {
            CorruptError::new_err(msg)
        }
        PersistError::Io(_) => PyOSError::new_err(msg),
    }
}

// ---------------------------------------------------------------------------
// GraphInner — dual-mode: Owned or Archived zero-copy
// ---------------------------------------------------------------------------

enum GraphInner {
    Owned(OrpheusGraphInner),
    Archived(ArchivedGraphView),
}

impl GraphInner {
    fn as_accessor(&self) -> &dyn GraphAccessor {
        match self {
            GraphInner::Owned(g) => g,
            GraphInner::Archived(v) => v,
        }
    }
}

// ---------------------------------------------------------------------------
// PyOrpheusGraph
// ---------------------------------------------------------------------------

/// Python-facing graph wrapper.
#[pyclass(name = "OrpheusGraph")]
pub struct PyOrpheusGraph {
    inner: Option<GraphInner>,
}

impl Drop for PyOrpheusGraph {
    fn drop(&mut self) {
        self.inner.take(); // Safety net: frees graph even if .close() was never called
    }
}

impl PyOrpheusGraph {
    fn require_inner(&self) -> PyResult<&dyn GraphAccessor> {
        self.inner.as_ref().map(|g| g.as_accessor()).ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err(
                "Graph has been closed. Call build_graph() or from_rkyv() to create a new one.",
            )
        })
    }

    /// Resolve a PyDynamicContext into a fresh Rust DynamicContext.
    ///
    /// Overlays are re-parsed on every call so this stays `&self` and multiple
    /// readers can run concurrently under a shared borrow while the GIL is
    /// released. That concurrency property is worth keeping, but the parse is
    /// NOT cheap and this comment used to claim it was: it is O(overlay size)
    /// per traversal, and measured through the Python bindings a context
    /// carrying 20k overlay edges costs ~6.2 ms per `beam_traverse` against
    /// ~1.2 µs with no overlay — and that is with the traversal collecting none
    /// of those edges. It dominates the hot path the moment an overlay is
    /// non-trivial, and it dwarfs `OverlayIndex`, which only removes the
    /// per-expanded-node rescan *inside* the traversal.
    ///
    /// Fixing it means caching the parsed overlay on the context while keeping
    /// this path `&self` — so an interior-mutability cache plus explicit
    /// setters that invalidate it, since `overlay_nodes_raw`/`overlay_edges_raw`
    /// are settable from Python and a construction-time parse would go stale.
    fn resolve_context(&self, ctx: &PyDynamicContext) -> PyResult<DynamicContext> {
        resolve_dynamic_context(ctx)
    }
}

/// Shared context resolution used by both `OrpheusGraph` and `PersistentGraph`
/// (the read API is identical; only the backing store differs).
fn resolve_dynamic_context(ctx: &PyDynamicContext) -> PyResult<DynamicContext> {
    let (overlay_nodes, overlay_edges) = ctx.parse_overlays()?;

    // Normalize principals (sort + dedup) at this construction boundary — the
    // actual "construction" funnel for real (Python) callers building a
    // context from external input. Pure-Rust callers building `DynamicContext`
    // via struct-literal syntax are not funneled through any single point (no
    // other field is normalized either, e.g. `noise_tags` dedups structurally
    // via its `HashSet` type), so this is where the spec's "normalize:
    // sort+dedup at construction" is actually enforced.
    let mut principals = ctx.principals.clone();
    principals.sort();
    principals.dedup();

    Ok(DynamicContext {
        semantic_boosts: ctx.semantic_boosts.clone(),
        weight_bonuses: ctx.weight_bonuses.clone(),
        noise_tags: ctx.noise_tags.clone(),
        max_fan_out: ctx.max_fan_out,
        fan_out_pagerank_bypass: ctx.fan_out_pagerank_bypass,
        w_base: ctx.w_base,
        w_semantic: ctx.w_semantic,
        w_noise: ctx.w_noise,
        w_bonus: ctx.w_bonus,
        overlay_nodes,
        overlay_edges,
        as_of: ctx.as_of,
        principals,
    })
}

#[pymethods]
impl PyOrpheusGraph {
    fn node_count(&self) -> PyResult<usize> {
        Ok(self.require_inner()?.node_count())
    }

    fn edge_count(&self) -> PyResult<usize> {
        Ok(self.require_inner()?.edge_count())
    }

    fn get_node(&self, name: &str) -> PyResult<Option<PyNodeResult>> {
        let graph = self.require_inner()?;
        Ok(graph.get_node(name).map(|nv| {
            let ctx = DynamicContext::default();
            let nr = compute_score(&nv, &ctx);
            PyNodeResult::from_node_result(nr)
        }))
    }

    fn outgoing_edges(&self, name: &str) -> PyResult<Vec<PyEdgeResult>> {
        let graph = self.require_inner()?;
        Ok(graph
            .outgoing_neighbors(name)
            .iter()
            .map(|n| PyEdgeResult {
                source: name.to_string(),
                target: n.target_name.clone(),
                kind: n.edge_kind.clone(),
                field_name: n.field_name.clone(),
                weight: n.edge_weight,
                valid_from: n.valid_from,
                valid_to: n.valid_to,
                acl: n.acl.clone(),
            })
            .collect())
    }

    fn incoming_edges(&self, name: &str) -> PyResult<Vec<PyEdgeResult>> {
        let graph = self.require_inner()?;
        Ok(graph
            .incoming_neighbors(name)
            .iter()
            .map(|n| PyEdgeResult {
                source: n.target_name.clone(),
                target: name.to_string(),
                kind: n.edge_kind.clone(),
                field_name: n.field_name.clone(),
                weight: n.edge_weight,
                valid_from: n.valid_from,
                valid_to: n.valid_to,
                acl: n.acl.clone(),
            })
            .collect())
    }

    /// Top-K pruned BFS. GIL released during traversal.
    fn beam_traverse(
        &self,
        py: Python<'_>,
        start: &str,
        k: usize,
        depth: usize,
        ctx: &PyDynamicContext,
    ) -> PyResult<Vec<PyNodeResult>> {
        let rust_ctx = self.resolve_context(ctx)?;
        let graph = self.require_inner()?;
        let start_owned = start.to_string();

        let results =
            py.allow_threads(|| traversal::beam_traverse(graph, &rust_ctx, &start_owned, k, depth));

        Ok(results
            .into_iter()
            .map(PyNodeResult::from_node_result)
            .collect())
    }

    /// Weighted Dijkstra. GIL released during traversal.
    fn find_path(
        &self,
        py: Python<'_>,
        start: &str,
        end: &str,
        ctx: &PyDynamicContext,
    ) -> PyResult<Option<Vec<PyPathStep>>> {
        let rust_ctx = self.resolve_context(ctx)?;
        let graph = self.require_inner()?;
        let start_owned = start.to_string();
        let end_owned = end.to_string();

        let path =
            py.allow_threads(|| traversal::find_path(graph, &rust_ctx, &start_owned, &end_owned));

        Ok(path.map(|steps| steps.into_iter().map(PyPathStep::from_path_step).collect()))
    }

    /// Contextual subgraph extraction. GIL released.
    fn contextual_subgraph(
        &self,
        py: Python<'_>,
        ctx: &PyDynamicContext,
        k: usize,
    ) -> PyResult<PySubGraph> {
        let rust_ctx = self.resolve_context(ctx)?;
        let graph = self.require_inner()?;

        let sg = py.allow_threads(|| traversal::contextual_subgraph(graph, &rust_ctx, k));

        Ok(PySubGraph {
            nodes: sg
                .nodes
                .into_iter()
                .map(PyNodeResult::from_node_result)
                .collect(),
            edges: sg
                .edges
                .into_iter()
                .map(PyEdgeResult::from_edge_result)
                .collect(),
        })
    }

    /// Multi-source heatmap intersection. GIL released during computation.
    ///
    /// Launches beam_traverse from each start node, accumulates a weighted
    /// heatmap, then keeps only nodes hit by `threshold` or more beams.
    /// Start nodes are always preserved.
    #[pyo3(signature = (start_nodes, k, depth, ctx, threshold = None))]
    fn multi_beam_intersection(
        &self,
        py: Python<'_>,
        start_nodes: Vec<String>,
        k: usize,
        depth: usize,
        ctx: &PyDynamicContext,
        threshold: Option<usize>,
    ) -> PyResult<PySubGraph> {
        let rust_ctx = self.resolve_context(ctx)?;
        let graph = self.require_inner()?;
        // Default to a true intersection: a node must be hit by >= 2 beams
        // (min(len, max(2, len-1))). The old len-1 default degraded to a union
        // for 2 seeds (threshold 1).
        let t = threshold.unwrap_or_else(|| {
            let len = start_nodes.len();
            len.min(len.saturating_sub(1).max(2))
        });

        let sg = py.allow_threads(|| {
            traversal::multi_beam_intersection(graph, &rust_ctx, &start_nodes, k, depth, t)
        });

        Ok(PySubGraph {
            nodes: sg
                .nodes
                .into_iter()
                .map(PyNodeResult::from_node_result)
                .collect(),
            edges: sg
                .edges
                .into_iter()
                .map(PyEdgeResult::from_edge_result)
                .collect(),
        })
    }

    /// Serialize to rkyv bytes. Only works on Owned graphs.
    fn to_rkyv<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        let inner = self
            .inner
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("Graph has been closed"))?;

        match inner {
            GraphInner::Owned(graph) => {
                let bytes = to_rkyv(graph);
                Ok(PyBytes::new(py, &bytes))
            }
            GraphInner::Archived(_) => Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Cannot serialize an archived graph. Use build_graph() first.",
            )),
        }
    }

    /// Deterministic memory release.
    fn close(&mut self) {
        self.inner.take();
    }

    fn __repr__(&self) -> String {
        match &self.inner {
            Some(g) => {
                let acc = g.as_accessor();
                format!(
                    "OrpheusGraph(nodes={}, edges={})",
                    acc.node_count(),
                    acc.edge_count()
                )
            }
            None => "OrpheusGraph(closed)".to_string(),
        }
    }
}

// ---------------------------------------------------------------------------
// PyDynamicContext
// ---------------------------------------------------------------------------

#[pyclass(name = "DynamicContext")]
#[derive(Clone)]
pub struct PyDynamicContext {
    #[pyo3(get, set)]
    pub semantic_boosts: HashMap<String, f32>,
    #[pyo3(get, set)]
    pub weight_bonuses: HashMap<String, f32>,
    #[pyo3(get, set)]
    pub noise_tags: HashSet<String>,
    #[pyo3(get, set)]
    pub max_fan_out: Option<usize>,
    /// `pagerank_weight` above which a node escapes the `max_fan_out` cutoff.
    /// `None` removes the escape, making `max_fan_out` an actual bound on
    /// base-edge expansion. Defaults to 0.5 (historical behaviour), which is an
    /// absolute threshold against a graph-dependent distribution — pick it from
    /// your own graph's PageRank spread.
    #[pyo3(get, set)]
    pub fan_out_pagerank_bypass: Option<f32>,
    #[pyo3(get, set)]
    pub w_base: f32,
    #[pyo3(get, set)]
    pub w_semantic: f32,
    #[pyo3(get, set)]
    pub w_noise: f32,
    #[pyo3(get, set)]
    pub w_bonus: f32,
    /// Valid-time instant for temporal edge filtering. `None` = no temporal
    /// filtering.
    #[pyo3(get, set)]
    pub as_of: Option<u64>,
    /// ACL principals held by the caller. Normalized (sorted + deduped) when
    /// resolved into a Rust `DynamicContext` (see `resolve_dynamic_context`).
    #[pyo3(get, set)]
    pub principals: Vec<String>,
    /// Virtual overlay nodes (per-tenant customization).
    pub overlay_nodes_raw: Vec<HashMap<String, String>>,
    /// Virtual overlay edges: [{"from": ..., "to": ..., "kind": ...}].
    ///
    /// NOTE: overlay edges do NOT currently support `valid_from`/`valid_to`/
    /// `acl` via this dict API — `overlay_edges_raw` is `Vec<HashMap<String,
    /// String>>` (every value is a string, e.g. `"base_weight": "1.0"`), and
    /// `acl` is a list, not a scalar string, so it does not fit this schema
    /// without a breaking change to it. Overlay edges therefore always
    /// resolve to `valid_from=None, valid_to=None, acl=[]` (unbounded/public)
    /// — a safe default, never silently over- or under-filtered. Rust-native
    /// callers constructing `EdgeData` directly for `ctx.overlay_edges` DO
    /// get full support (see `overlay.rs` tests).
    pub overlay_edges_raw: Vec<HashMap<String, String>>,
}

#[pymethods]
impl PyDynamicContext {
    #[new]
    #[pyo3(signature = (
        semantic_boosts = None,
        weight_bonuses = None,
        noise_tags = None,
        max_fan_out = None,
        fan_out_pagerank_bypass = Some(0.5),
        w_base = 1.0,
        w_semantic = 1.5,
        w_noise = 1.0,
        w_bonus = 1.0,
        overlay_nodes = None,
        overlay_edges = None,
        as_of = None,
        principals = None
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        semantic_boosts: Option<HashMap<String, f32>>,
        weight_bonuses: Option<HashMap<String, f32>>,
        noise_tags: Option<HashSet<String>>,
        max_fan_out: Option<usize>,
        fan_out_pagerank_bypass: Option<f32>,
        w_base: f32,
        w_semantic: f32,
        w_noise: f32,
        w_bonus: f32,
        overlay_nodes: Option<Vec<HashMap<String, String>>>,
        overlay_edges: Option<Vec<HashMap<String, String>>>,
        as_of: Option<u64>,
        principals: Option<Vec<String>>,
    ) -> Self {
        Self {
            semantic_boosts: semantic_boosts.unwrap_or_default(),
            weight_bonuses: weight_bonuses.unwrap_or_default(),
            noise_tags: noise_tags.unwrap_or_default(),
            max_fan_out,
            fan_out_pagerank_bypass,
            w_base,
            w_semantic,
            w_noise,
            w_bonus,
            as_of,
            principals: principals.unwrap_or_default(),
            overlay_nodes_raw: overlay_nodes.unwrap_or_default(),
            overlay_edges_raw: overlay_edges.unwrap_or_default(),
        }
    }

    /// Insert/update a semantic boost. The `semantic_boosts` getter returns a
    /// copy, so mutating that copy in-place is a no-op — use this instead.
    fn add_boost(&mut self, name: String, val: f32) {
        self.semantic_boosts.insert(name, val);
    }

    /// Insert/update a weight bonus. See `add_boost` for why the getter copy
    /// cannot be mutated in place.
    fn add_bonus(&mut self, name: String, val: f32) {
        self.weight_bonuses.insert(name, val);
    }

    /// Add a noise tag. The `noise_tags` getter returns a copy, so mutating that
    /// copy in-place is a no-op — use this instead.
    fn add_noise_tag(&mut self, tag: String) {
        self.noise_tags.insert(tag);
    }

    fn __repr__(&self) -> String {
        format!(
            "DynamicContext(boosts={}, bonuses={}, noise_tags={}, max_fan_out={:?}, \
             as_of={:?}, principals={})",
            self.semantic_boosts.len(),
            self.weight_bonuses.len(),
            self.noise_tags.len(),
            self.max_fan_out,
            self.as_of,
            self.principals.len(),
        )
    }
}

/// Overlay parse result: (overlay nodes, overlay edges as `(from, to, edge)`).
type ParsedOverlays = (Vec<NodeData>, Vec<(String, String, EdgeData)>);

impl PyDynamicContext {
    /// Parse overlay raw dicts into Rust types.
    ///
    /// Required fields are hard errors — a missing/typo'd key must never be
    /// silently defaulted, as that would inject phantom ''-named nodes or
    /// mis-weighted edges into the traversal.
    fn parse_overlays(&self) -> PyResult<ParsedOverlays> {
        let mut overlay_nodes = Vec::with_capacity(self.overlay_nodes_raw.len());
        for m in &self.overlay_nodes_raw {
            let name = m.get("name").cloned().unwrap_or_default();
            if name.is_empty() {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "overlay node is missing required non-empty 'name'",
                ));
            }
            overlay_nodes.push(NodeData {
                kind: m.get("kind").cloned().unwrap_or_default(),
                metadata: HashMap::new(),
                base_weight: parse_overlay_f32(m, "base_weight", 0.5, &name)?,
                noise_penalty: parse_overlay_f32(m, "noise_penalty", 0.0, &name)?,
                pagerank_weight: 0.0,
                name,
            });
        }

        let mut overlay_edges = Vec::with_capacity(self.overlay_edges_raw.len());
        for m in &self.overlay_edges_raw {
            let from = require_overlay_field(m, "from", "edge")?;
            let to = require_overlay_field(m, "to", "edge")?;
            let kind = require_overlay_field(m, "kind", "edge")?;
            let base_weight = parse_overlay_f32(m, "base_weight", 1.0, &format!("{from}->{to}"))?;
            // valid_from/valid_to/acl are not exposed via this string-only
            // dict schema (see the `overlay_edges_raw` doc comment) — always
            // unbounded/public, never silently mis-filtered.
            let edge = EdgeData {
                kind,
                field_name: m.get("field").cloned(),
                base_weight,
                valid_from: None,
                valid_to: None,
                acl: Vec::new(),
            };
            overlay_edges.push((from, to, edge));
        }

        Ok((overlay_nodes, overlay_edges))
    }
}

/// Extract a required non-empty overlay field or raise a PyErr.
fn require_overlay_field(m: &HashMap<String, String>, field: &str, what: &str) -> PyResult<String> {
    match m.get(field) {
        Some(v) if !v.is_empty() => Ok(v.clone()),
        _ => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "overlay {what} is missing required non-empty '{field}'"
        ))),
    }
}

/// Parse an optional numeric overlay field; a present-but-invalid value is a
/// hard error rather than a silent default.
fn parse_overlay_f32(
    m: &HashMap<String, String>,
    field: &str,
    default: f32,
    ctx: &str,
) -> PyResult<f32> {
    match m.get(field) {
        Some(v) => v.parse::<f32>().map_err(|_| {
            pyo3::exceptions::PyValueError::new_err(format!(
                "overlay '{ctx}' has invalid '{field}': {v:?}"
            ))
        }),
        None => Ok(default),
    }
}

// ---------------------------------------------------------------------------
// Lightweight PyO3 result types
// ---------------------------------------------------------------------------

#[pyclass(name = "NodeResult")]
#[derive(Clone)]
pub struct PyNodeResult {
    #[pyo3(get)]
    pub name: String,
    #[pyo3(get)]
    pub kind: String,
    #[pyo3(get)]
    pub weight: f32,
    #[pyo3(get)]
    pub base_component: f32,
    #[pyo3(get)]
    pub semantic_component: f32,
    #[pyo3(get)]
    pub noise_component: f32,
    #[pyo3(get)]
    pub bonus_component: f32,
}

impl PyNodeResult {
    fn from_node_result(nr: crate::types::NodeResult) -> Self {
        Self {
            name: nr.name,
            kind: nr.kind,
            weight: nr.weight,
            base_component: nr.base_component,
            semantic_component: nr.semantic_component,
            noise_component: nr.noise_component,
            bonus_component: nr.bonus_component,
        }
    }
}

#[pymethods]
impl PyNodeResult {
    fn explain_score(&self) -> HashMap<String, f32> {
        let mut m = HashMap::new();
        m.insert("base".to_string(), self.base_component);
        m.insert("semantic".to_string(), self.semantic_component);
        m.insert("noise".to_string(), self.noise_component);
        m.insert("bonus".to_string(), self.bonus_component);
        m.insert("total".to_string(), self.weight);
        m
    }

    fn __repr__(&self) -> String {
        format!(
            "NodeResult(name={:?}, weight={:.4})",
            self.name, self.weight
        )
    }
}

#[pyclass(name = "EdgeResult")]
#[derive(Clone)]
pub struct PyEdgeResult {
    #[pyo3(get)]
    pub source: String,
    #[pyo3(get)]
    pub target: String,
    #[pyo3(get)]
    pub kind: String,
    #[pyo3(get)]
    pub field_name: Option<String>,
    #[pyo3(get)]
    pub weight: f32,
    #[pyo3(get)]
    pub valid_from: Option<u64>,
    #[pyo3(get)]
    pub valid_to: Option<u64>,
    #[pyo3(get)]
    pub acl: Vec<String>,
}

impl PyEdgeResult {
    fn from_edge_result(er: crate::types::EdgeResult) -> Self {
        Self {
            source: er.source,
            target: er.target,
            kind: er.kind,
            field_name: er.field_name,
            weight: er.weight,
            valid_from: er.valid_from,
            valid_to: er.valid_to,
            acl: er.acl,
        }
    }
}

#[pymethods]
impl PyEdgeResult {
    fn __repr__(&self) -> String {
        format!(
            "EdgeResult({:?} -> {:?}, kind={:?}, field={:?})",
            self.source, self.target, self.kind, self.field_name
        )
    }
}

#[pyclass(name = "PathStep")]
#[derive(Clone)]
pub struct PyPathStep {
    #[pyo3(get)]
    pub node: String,
    #[pyo3(get)]
    pub edge_kind: String,
    #[pyo3(get)]
    pub field_name: String,
    #[pyo3(get)]
    pub direction: String,
}

impl PyPathStep {
    fn from_path_step(ps: crate::types::PathStep) -> Self {
        Self {
            node: ps.node,
            edge_kind: ps.edge_kind,
            field_name: ps.field_name,
            direction: ps.direction,
        }
    }
}

#[pymethods]
impl PyPathStep {
    fn __repr__(&self) -> String {
        format!(
            "PathStep(node={:?}, edge={:?}, field={:?}, dir={:?})",
            self.node, self.edge_kind, self.field_name, self.direction
        )
    }
}

#[pyclass(name = "SubGraph")]
pub struct PySubGraph {
    #[pyo3(get)]
    pub nodes: Vec<PyNodeResult>,
    #[pyo3(get)]
    pub edges: Vec<PyEdgeResult>,
}

#[pymethods]
impl PySubGraph {
    fn __repr__(&self) -> String {
        format!(
            "SubGraph(nodes={}, edges={})",
            self.nodes.len(),
            self.edges.len()
        )
    }
}

// ---------------------------------------------------------------------------
// Module-level functions
// ---------------------------------------------------------------------------

/// Parse a Python list of node dicts into `NodeInput`s (shared by `build_graph`
/// and `create_persistent`).
fn parse_node_inputs(nodes: &Bound<'_, PyList>) -> PyResult<Vec<NodeInput>> {
    let mut rust_nodes: Vec<NodeInput> = Vec::with_capacity(nodes.len());
    for item in nodes.iter() {
        let dict = item.downcast::<PyDict>()?;
        rust_nodes.push(NodeInput {
            name: dict
                .get_item("name")?
                .ok_or_else(|| PyKeyError::new_err("name"))?
                .extract()?,
            kind: dict
                .get_item("kind")?
                .ok_or_else(|| PyKeyError::new_err("kind"))?
                .extract()?,
            metadata: dict
                .get_item("metadata")?
                .map(|v| v.extract())
                .transpose()?
                .unwrap_or_default(),
            base_weight: dict
                .get_item("base_weight")?
                .ok_or_else(|| PyKeyError::new_err("base_weight"))?
                .extract()?,
            noise_penalty: dict
                .get_item("noise_penalty")?
                .map(|v| v.extract::<f32>())
                .transpose()?
                .unwrap_or(0.0),
        });
    }
    Ok(rust_nodes)
}

/// Optional u64 dict field (absent -> None). A present-but-non-u64 value is a
/// hard error rather than a silent default.
fn optional_dict_u64(d: &Bound<'_, PyDict>, key: &str) -> PyResult<Option<u64>> {
    d.get_item(key)?.map(|v| v.extract::<u64>()).transpose()
}

/// Optional list[str] dict field; absent -> empty `Vec`. Normalized (sorted +
/// deduped) here — see `normalize_acl` — so `acl` is canonical from the
/// moment it enters Rust, matching the ingest-time-normalization contract.
fn optional_dict_acl(d: &Bound<'_, PyDict>, key: &str) -> PyResult<Vec<String>> {
    let mut acl: Vec<String> = match d.get_item(key)? {
        Some(v) => v.extract()?,
        None => Vec::new(),
    };
    normalize_acl(&mut acl);
    Ok(acl)
}

/// Parse a Python list of edge dicts into `EdgeInput`s.
fn parse_edge_inputs(edges: &Bound<'_, PyList>) -> PyResult<Vec<EdgeInput>> {
    let mut rust_edges: Vec<EdgeInput> = Vec::with_capacity(edges.len());
    for item in edges.iter() {
        let dict = item.downcast::<PyDict>()?;
        rust_edges.push(EdgeInput {
            from: dict
                .get_item("from")?
                .ok_or_else(|| PyKeyError::new_err("from"))?
                .extract()?,
            to: dict
                .get_item("to")?
                .ok_or_else(|| PyKeyError::new_err("to"))?
                .extract()?,
            kind: dict
                .get_item("kind")?
                .ok_or_else(|| PyKeyError::new_err("kind"))?
                .extract()?,
            field_name: dict.get_item("field")?.map(|v| v.extract()).transpose()?,
            base_weight: dict
                .get_item("base_weight")?
                .map(|v| v.extract::<f32>())
                .transpose()?
                .unwrap_or(1.0),
            // Missing keys default to unbounded/public (backward-compatible
            // with pre-existing callers that never pass these). A malformed
            // valid_from > valid_to range is NOT rejected here — build_graph's
            // established error style for bad edge input is a silent skip
            // (matches "skip edges referencing unknown nodes"); see
            // `builder.rs::build_graph_inner`.
            valid_from: optional_dict_u64(dict, "valid_from")?,
            valid_to: optional_dict_u64(dict, "valid_to")?,
            acl: optional_dict_acl(dict, "acl")?,
        });
    }
    Ok(rust_edges)
}

/// Build a graph from Python dicts.
#[pyfunction]
#[pyo3(name = "build_graph")]
pub fn py_build_graph(
    nodes: &Bound<'_, PyList>,
    edges: &Bound<'_, PyList>,
) -> PyResult<PyOrpheusGraph> {
    let rust_nodes = parse_node_inputs(nodes)?;
    let rust_edges = parse_edge_inputs(edges)?;

    let (g, m) = build_graph(rust_nodes, rust_edges);
    let graph_inner = OrpheusGraphInner::new(g, m);

    Ok(PyOrpheusGraph {
        inner: Some(GraphInner::Owned(graph_inner)),
    })
}

/// Load a graph from rkyv bytes (zero-copy).
#[pyfunction]
#[pyo3(name = "from_rkyv")]
pub fn py_from_rkyv(data: &Bound<'_, PyBytes>) -> PyResult<PyOrpheusGraph> {
    let bytes = data.as_bytes().to_vec();
    let view =
        ArchivedGraphView::from_bytes(bytes).map_err(pyo3::exceptions::PyValueError::new_err)?;

    Ok(PyOrpheusGraph {
        inner: Some(GraphInner::Archived(view)),
    })
}

// ===========================================================================
// PersistentGraph — Python bindings for the durable delta store (§4.6)
// ===========================================================================

/// Require a present, non-empty string field or raise. Node/edge endpoints and
/// the discriminant must never be silently defaulted — an empty name injects a
/// phantom node into traversal (same contract as the overlay parser).
fn require_op_str(d: &Bound<'_, PyDict>, key: &str) -> PyResult<String> {
    match d.get_item(key)? {
        Some(v) => {
            let s: String = v.extract()?;
            if s.is_empty() {
                return Err(PyValueError::new_err(format!(
                    "op field '{key}' must be a non-empty string"
                )));
            }
            Ok(s)
        }
        None => Err(PyKeyError::new_err(key.to_string())),
    }
}

/// Optional string field (absent → None).
fn optional_op_str(d: &Bound<'_, PyDict>, key: &str) -> PyResult<Option<String>> {
    d.get_item(key)?.map(|v| v.extract::<String>()).transpose()
}

/// Optional f32 field with a default.
fn op_f32(d: &Bound<'_, PyDict>, key: &str, default: f32) -> PyResult<f32> {
    Ok(d.get_item(key)?
        .map(|v| v.extract::<f32>())
        .transpose()?
        .unwrap_or(default))
}

/// Enforce the §3.3-rule-5 contract at the FFI boundary: weights must already be
/// normalized to [0,1]. Rejecting here (rather than clamping) keeps a caller
/// mistake loud and keeps out-of-range values off the delta `upsert` path. The
/// delta layer itself no longer asserts this (it stores weights verbatim for the
/// audit-#6 determinism guarantee), so this boundary check is where the [0,1]
/// contract is actually enforced for Python callers.
fn require_unit(v: f32, field: &str, ctx: &str) -> PyResult<()> {
    if !(0.0..=1.0).contains(&v) {
        return Err(PyValueError::new_err(format!(
            "'{ctx}': {field}={v} is out of range [0.0, 1.0] (weights must be pre-normalized)"
        )));
    }
    Ok(())
}

/// Parse a Python list of op dicts into `Vec<Op>`. Every op is validated up
/// front; a bad op in the middle raises before any WAL frame is written, so an
/// `apply()` is all-or-nothing (§3.3 rule 4).
fn parse_ops(ops: &Bound<'_, PyList>) -> PyResult<Vec<Op>> {
    let mut out: Vec<Op> = Vec::with_capacity(ops.len());
    for item in ops.iter() {
        let d = item.downcast::<PyDict>()?;
        let op = require_op_str(d, "op")?;
        match op.as_str() {
            "upsert_node" => {
                let name = require_op_str(d, "name")?;
                let base_weight = op_f32(d, "base_weight", 0.5)?;
                let noise_penalty = op_f32(d, "noise_penalty", 0.0)?;
                require_unit(base_weight, "base_weight", &name)?;
                require_unit(noise_penalty, "noise_penalty", &name)?;
                let kind = optional_op_str(d, "kind")?.unwrap_or_default();
                let metadata = d
                    .get_item("metadata")?
                    .map(|v| v.extract())
                    .transpose()?
                    .unwrap_or_default();
                out.push(Op::UpsertNode(NodeData {
                    name,
                    kind,
                    metadata,
                    base_weight,
                    noise_penalty,
                    pagerank_weight: 0.0,
                }));
            }
            "remove_node" => {
                let name = require_op_str(d, "name")?;
                out.push(Op::RemoveNode { name });
            }
            "add_edge" => {
                let from = require_op_str(d, "from")?;
                let to = require_op_str(d, "to")?;
                let kind = require_op_str(d, "kind")?;
                let base_weight = op_f32(d, "base_weight", 1.0)?;
                require_unit(base_weight, "base_weight", &format!("{from}->{to}"))?;
                let field_name = optional_op_str(d, "field")?;
                // Missing valid_from/valid_to/acl default to unbounded/public
                // (backward-compatible). A malformed valid_from > valid_to
                // range is NOT rejected here — it is deferred to
                // `GraphDelta::apply`'s PASS-1 validation
                // (`DeltaError::InvalidEdgeValidity`, mapped to `PyValueError`
                // by `map_persist_err`), matching how AddEdge endpoint
                // existence is ALSO deferred there rather than checked in this
                // parser — apply's established error style, not the FFI
                // parse-time style `require_unit` uses for weights.
                let valid_from = optional_dict_u64(d, "valid_from")?;
                let valid_to = optional_dict_u64(d, "valid_to")?;
                let acl = optional_dict_acl(d, "acl")?;
                out.push(Op::AddEdge {
                    from,
                    to,
                    edge: EdgeData {
                        kind,
                        field_name,
                        base_weight,
                        valid_from,
                        valid_to,
                        acl,
                    },
                });
            }
            "remove_edge" => {
                let from = require_op_str(d, "from")?;
                let to = require_op_str(d, "to")?;
                let kind = require_op_str(d, "kind")?;
                out.push(Op::RemoveEdge { from, to, kind });
            }
            other => {
                return Err(PyValueError::new_err(format!(
                    "unknown op {other:?}; expected one of \
                     upsert_node | remove_node | add_edge | remove_edge"
                )))
            }
        }
    }
    Ok(out)
}

/// Python-facing durable graph store. Wraps a Rust `PersistentGraph`; the read
/// API mirrors `OrpheusGraph` (over a live base∘delta view), plus the mutating
/// `apply`/`flush`/`compact`/`close` durability surface (§4.6).
#[pyclass(name = "PersistentGraph")]
pub struct PyPersistentGraph {
    // `Option` so `close()`/`__exit__` can consume the store (its `close(self)`
    // takes ownership) while the Python object lives on. A dropped-without-close
    // store still releases its flock + mmap; it just is not marked clean, so the
    // next open re-mints the epoch (the process-death path, §4.3 — safe).
    pg: Option<PersistentGraph>,
}

impl PyPersistentGraph {
    fn require_pg(&self) -> PyResult<&PersistentGraph> {
        self.pg
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("PersistentGraph has been closed"))
    }
}

#[pymethods]
impl PyPersistentGraph {
    /// Current durable commit seq — the CAS token and the cache "generation".
    #[getter]
    fn seq(&self) -> PyResult<u64> {
        Ok(self.require_pg()?.seq())
    }

    /// Current incarnation epoch (changes across an unclean reopen).
    #[getter]
    fn epoch(&self) -> PyResult<u128> {
        Ok(self.require_pg()?.epoch())
    }

    fn node_count(&self) -> PyResult<usize> {
        let pg = self.require_pg()?;
        let s = pg.snapshot();
        let acc = DeltaAccessor::new(s.base.as_accessor(), s.delta.as_ref());
        Ok(acc.node_count())
    }

    fn edge_count(&self) -> PyResult<usize> {
        let pg = self.require_pg()?;
        let s = pg.snapshot();
        let acc = DeltaAccessor::new(s.base.as_accessor(), s.delta.as_ref());
        Ok(acc.edge_count())
    }

    /// Apply a batch atomically (one WAL frame). Returns the new seq.
    ///
    /// `expected_seq` gives optimistic CAS: if the store moved past it, a
    /// `ConflictError` is raised and nothing is written — re-read and retry.
    /// The GIL is released across the durable append (fsync may block).
    ///
    /// Durability: a returned seq (Ok) means the batch is durable (WAL frame +
    /// a covering COMMIT marker both fsync'd) — it survives process death and
    /// power loss. A raised error means durability is INDETERMINATE: a COMMIT
    /// fsync can fail while its slot still persists, so the batch MAY be visible
    /// or invisible after a reopen. Recovery always reopens to a consistent
    /// committed prefix and never loses an acked batch; reconcile an error by
    /// re-opening the store and reading `seq` (the recovered committed prefix),
    /// NOT by blindly retrying (a non-idempotent batch could double-apply).
    #[pyo3(signature = (ops, expected_seq = None))]
    fn apply(
        &self,
        py: Python<'_>,
        ops: &Bound<'_, PyList>,
        expected_seq: Option<u64>,
    ) -> PyResult<u64> {
        let parsed = parse_ops(ops)?;
        let pg = self.require_pg()?;
        py.allow_threads(|| pg.apply(parsed, expected_seq))
            .map_err(map_persist_err)
    }

    /// Force the WAL to stable storage (the durability point under OnFlush).
    fn flush(&self, py: Python<'_>) -> PyResult<()> {
        let pg = self.require_pg()?;
        py.allow_threads(|| pg.flush()).map_err(map_persist_err)
    }

    /// Fold the delta into a fresh base snapshot and empty the delta. Also fires
    /// automatically once the delta crosses its size threshold.
    fn compact(&self, py: Python<'_>) -> PyResult<()> {
        let pg = self.require_pg()?;
        py.allow_threads(|| pg.compact()).map_err(map_persist_err)
    }

    fn get_node(&self, name: &str) -> PyResult<Option<PyNodeResult>> {
        let pg = self.require_pg()?;
        let s = pg.snapshot();
        let acc = DeltaAccessor::new(s.base.as_accessor(), s.delta.as_ref());
        Ok(acc.get_node(name).map(|nv| {
            let ctx = DynamicContext::default();
            PyNodeResult::from_node_result(compute_score(&nv, &ctx))
        }))
    }

    fn outgoing_edges(&self, name: &str) -> PyResult<Vec<PyEdgeResult>> {
        let pg = self.require_pg()?;
        let s = pg.snapshot();
        let acc = DeltaAccessor::new(s.base.as_accessor(), s.delta.as_ref());
        Ok(acc
            .outgoing_neighbors(name)
            .iter()
            .map(|n| PyEdgeResult {
                source: name.to_string(),
                target: n.target_name.clone(),
                kind: n.edge_kind.clone(),
                field_name: n.field_name.clone(),
                weight: n.edge_weight,
                valid_from: n.valid_from,
                valid_to: n.valid_to,
                acl: n.acl.clone(),
            })
            .collect())
    }

    fn incoming_edges(&self, name: &str) -> PyResult<Vec<PyEdgeResult>> {
        let pg = self.require_pg()?;
        let s = pg.snapshot();
        let acc = DeltaAccessor::new(s.base.as_accessor(), s.delta.as_ref());
        Ok(acc
            .incoming_neighbors(name)
            .iter()
            .map(|n| PyEdgeResult {
                source: n.target_name.clone(),
                target: name.to_string(),
                kind: n.edge_kind.clone(),
                field_name: n.field_name.clone(),
                weight: n.edge_weight,
                valid_from: n.valid_from,
                valid_to: n.valid_to,
                acl: n.acl.clone(),
            })
            .collect())
    }

    /// Top-K pruned BFS over the live base∘delta view. GIL released.
    fn beam_traverse(
        &self,
        py: Python<'_>,
        start: &str,
        k: usize,
        depth: usize,
        ctx: &PyDynamicContext,
    ) -> PyResult<Vec<PyNodeResult>> {
        let rust_ctx = resolve_dynamic_context(ctx)?;
        let pg = self.require_pg()?;
        let s = pg.snapshot();
        let acc = DeltaAccessor::new(s.base.as_accessor(), s.delta.as_ref());
        let start_owned = start.to_string();

        let results =
            py.allow_threads(|| traversal::beam_traverse(&acc, &rust_ctx, &start_owned, k, depth));

        Ok(results
            .into_iter()
            .map(PyNodeResult::from_node_result)
            .collect())
    }

    /// Weighted Dijkstra over the live base∘delta view. GIL released.
    fn find_path(
        &self,
        py: Python<'_>,
        start: &str,
        end: &str,
        ctx: &PyDynamicContext,
    ) -> PyResult<Option<Vec<PyPathStep>>> {
        let rust_ctx = resolve_dynamic_context(ctx)?;
        let pg = self.require_pg()?;
        let s = pg.snapshot();
        let acc = DeltaAccessor::new(s.base.as_accessor(), s.delta.as_ref());
        let start_owned = start.to_string();
        let end_owned = end.to_string();

        let path =
            py.allow_threads(|| traversal::find_path(&acc, &rust_ctx, &start_owned, &end_owned));

        Ok(path.map(|steps| steps.into_iter().map(PyPathStep::from_path_step).collect()))
    }

    /// Contextual subgraph extraction over the live view. GIL released.
    fn contextual_subgraph(
        &self,
        py: Python<'_>,
        ctx: &PyDynamicContext,
        k: usize,
    ) -> PyResult<PySubGraph> {
        let rust_ctx = resolve_dynamic_context(ctx)?;
        let pg = self.require_pg()?;
        let s = pg.snapshot();
        let acc = DeltaAccessor::new(s.base.as_accessor(), s.delta.as_ref());

        let sg = py.allow_threads(|| traversal::contextual_subgraph(&acc, &rust_ctx, k));

        Ok(PySubGraph {
            nodes: sg
                .nodes
                .into_iter()
                .map(PyNodeResult::from_node_result)
                .collect(),
            edges: sg
                .edges
                .into_iter()
                .map(PyEdgeResult::from_edge_result)
                .collect(),
        })
    }

    /// Multi-source heatmap intersection over the live view. GIL released.
    #[pyo3(signature = (start_nodes, k, depth, ctx, threshold = None))]
    fn multi_beam_intersection(
        &self,
        py: Python<'_>,
        start_nodes: Vec<String>,
        k: usize,
        depth: usize,
        ctx: &PyDynamicContext,
        threshold: Option<usize>,
    ) -> PyResult<PySubGraph> {
        let rust_ctx = resolve_dynamic_context(ctx)?;
        let pg = self.require_pg()?;
        let s = pg.snapshot();
        let acc = DeltaAccessor::new(s.base.as_accessor(), s.delta.as_ref());
        let t = threshold.unwrap_or_else(|| {
            let len = start_nodes.len();
            len.min(len.saturating_sub(1).max(2))
        });

        let sg = py.allow_threads(|| {
            traversal::multi_beam_intersection(&acc, &rust_ctx, &start_nodes, k, depth, t)
        });

        Ok(PySubGraph {
            nodes: sg
                .nodes
                .into_iter()
                .map(PyNodeResult::from_node_result)
                .collect(),
            edges: sg
                .edges
                .into_iter()
                .map(PyEdgeResult::from_edge_result)
                .collect(),
        })
    }

    /// Set the WAL fsync policy: "on_flush" (default), "every_batch", or
    /// "every_n" (requires `every_n=N`).
    #[pyo3(signature = (policy, every_n = None))]
    fn set_fsync_policy(&self, policy: &str, every_n: Option<u32>) -> PyResult<()> {
        let p = match policy {
            "on_flush" => FsyncPolicy::OnFlush,
            "every_batch" => FsyncPolicy::EveryBatch,
            "every_n" => FsyncPolicy::EveryN(every_n.ok_or_else(|| {
                PyValueError::new_err("fsync policy 'every_n' requires every_n=N")
            })?),
            other => {
                return Err(PyValueError::new_err(format!(
                    "unknown fsync policy {other:?}; expected on_flush | every_batch | every_n"
                )))
            }
        };
        self.require_pg()?.set_fsync_policy(p);
        Ok(())
    }

    /// Override the auto-compaction op threshold (None restores the computed
    /// `max(1000, base_nodes/10)`).
    #[pyo3(signature = (threshold = None))]
    fn set_auto_compact_threshold(&self, threshold: Option<usize>) -> PyResult<()> {
        self.require_pg()?.set_auto_compact_threshold(threshold);
        Ok(())
    }

    /// Flush + mark a clean shutdown, then release the store. Idempotent: a
    /// second call (or use after `__exit__`) is a no-op.
    ///
    /// Takes `&mut self`, so under multithreading it can raise a transient
    /// `RuntimeError("Already borrowed")` if another thread is mid-traversal on
    /// the same object (PyO3's borrow check) — treat `close()` as retryable, or
    /// call it only once no read is in flight.
    fn close(&mut self, py: Python<'_>) -> PyResult<()> {
        if let Some(pg) = self.pg.take() {
            py.allow_threads(|| pg.close()).map_err(map_persist_err)?;
        }
        Ok(())
    }

    fn __enter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    #[pyo3(signature = (_exc_type, _exc_value, _traceback))]
    fn __exit__(
        &mut self,
        py: Python<'_>,
        _exc_type: &Bound<'_, PyAny>,
        _exc_value: &Bound<'_, PyAny>,
        _traceback: &Bound<'_, PyAny>,
    ) -> PyResult<bool> {
        self.close(py)?;
        Ok(false) // never suppress an in-`with` exception
    }

    fn __repr__(&self) -> String {
        match &self.pg {
            Some(pg) => format!("PersistentGraph(seq={})", pg.seq()),
            None => "PersistentGraph(closed)".to_string(),
        }
    }
}

/// Open (or create) a durable store at `dir` (§4.6).
///
/// `create=True` initializes an empty store when the directory has none;
/// `create=False` raises `FileNotFoundError`. `mmap=True` (default) maps the CSR
/// snapshot lazily (larger-than-RAM capable); `validate` is "full" (default,
/// mandatory for untrusted/shared stores), "crc", or "none".
#[pyfunction]
#[pyo3(name = "open", signature = (dir, create = false, mmap = true, validate = "full", prefault = false))]
pub fn py_open(
    dir: &str,
    create: bool,
    mmap: bool,
    validate: &str,
    prefault: bool,
) -> PyResult<PyPersistentGraph> {
    let mode = if mmap {
        BaseMode::Mmap
    } else {
        BaseMode::Owned
    };
    let v = match validate {
        "full" => Validate::Full,
        "crc" => Validate::Crc,
        "none" => Validate::None,
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown validate mode {other:?}; expected full | crc | none"
            )))
        }
    };
    let pg = PersistentGraph::open_with(dir, create, mode, v, prefault).map_err(map_persist_err)?;
    Ok(PyPersistentGraph { pg: Some(pg) })
}

/// Create a NEW durable store at `dir`, seeding its immutable base from
/// `nodes`/`edges` (same dict schema as `build_graph`). Refuses to clobber an
/// existing store. Use this to persist a freshly-built base graph in one pass
/// instead of replaying it through the delta.
#[pyfunction]
#[pyo3(name = "create_persistent")]
pub fn py_create_persistent(
    dir: &str,
    nodes: &Bound<'_, PyList>,
    edges: &Bound<'_, PyList>,
) -> PyResult<PyPersistentGraph> {
    let rust_nodes = parse_node_inputs(nodes)?;
    let rust_edges = parse_edge_inputs(edges)?;
    let (g, m) = build_graph(rust_nodes, rust_edges);
    let base = OrpheusGraphInner::new(g, m);
    let pg = PersistentGraph::create(dir, base).map_err(map_persist_err)?;
    Ok(PyPersistentGraph { pg: Some(pg) })
}
