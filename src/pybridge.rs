use std::collections::{HashMap, HashSet};

use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList};

use crate::types::{EdgeData, NodeData};

use crate::accessor::GraphAccessor;
use crate::builder::build_graph;
use crate::graph::OrpheusGraphInner;
use crate::scoring::compute_score;
use crate::serialization::{to_rkyv, ArchivedGraphView};
use crate::traversal;
use crate::types::{
    DynamicContext, EdgeInput, NodeInput,
};

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
    /// Risk #14: cache parsed overlay data per project_id to avoid FFI overhead.
    overlay_cache: HashMap<String, (Vec<NodeData>, Vec<(String, String, EdgeData)>)>,
}

impl Drop for PyOrpheusGraph {
    fn drop(&mut self) {
        self.inner.take(); // Safety net: frees graph even if .close() was never called
    }
}

impl PyOrpheusGraph {
    fn require_inner(&self) -> PyResult<&dyn GraphAccessor> {
        self.inner
            .as_ref()
            .map(|g| g.as_accessor())
            .ok_or_else(|| {
                pyo3::exceptions::PyRuntimeError::new_err(
                    "Graph has been closed. Call build_graph() or from_rkyv() to create a new one.",
                )
            })
    }

    /// Risk #14: resolve DynamicContext with overlay caching.
    /// If overlay_cache_key is set, reuse cached parsed overlay data.
    fn resolve_context(&mut self, ctx: &PyDynamicContext) -> DynamicContext {
        let (overlay_nodes, overlay_edges) = match &ctx.overlay_cache_key {
            Some(key) if !ctx.overlay_nodes_raw.is_empty() || !ctx.overlay_edges_raw.is_empty() => {
                if let Some(cached) = self.overlay_cache.get(key) {
                    cached.clone()
                } else {
                    let parsed = ctx.parse_overlays();
                    self.overlay_cache.insert(key.clone(), parsed.clone());
                    parsed
                }
            }
            _ => ctx.parse_overlays(),
        };

        DynamicContext {
            semantic_boosts: ctx.semantic_boosts.clone(),
            weight_overrides: ctx.weight_overrides.clone(),
            noise_tags: ctx.noise_tags.clone(),
            max_fan_out: ctx.max_fan_out,
            w_base: ctx.w_base,
            w_semantic: ctx.w_semantic,
            w_noise: ctx.w_noise,
            w_override: ctx.w_override,
            overlay_nodes,
            overlay_edges,
        }
    }
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
            })
            .collect())
    }

    /// Top-K pruned BFS. GIL released during traversal.
    fn beam_traverse(
        &mut self,
        py: Python<'_>,
        start: &str,
        k: usize,
        depth: usize,
        ctx: &PyDynamicContext,
    ) -> PyResult<Vec<PyNodeResult>> {
        let rust_ctx = self.resolve_context(ctx);
        let graph = self.require_inner()?;
        let start_owned = start.to_string();

        let results = py.allow_threads(|| {
            traversal::beam_traverse(graph, &rust_ctx, &start_owned, k, depth)
        });

        Ok(results.into_iter().map(PyNodeResult::from_node_result).collect())
    }

    /// Weighted Dijkstra. GIL released during traversal.
    fn find_path(
        &mut self,
        py: Python<'_>,
        start: &str,
        end: &str,
        ctx: &PyDynamicContext,
    ) -> PyResult<Option<Vec<PyPathStep>>> {
        let rust_ctx = self.resolve_context(ctx);
        let graph = self.require_inner()?;
        let start_owned = start.to_string();
        let end_owned = end.to_string();

        let path = py.allow_threads(|| {
            traversal::find_path(graph, &rust_ctx, &start_owned, &end_owned)
        });

        Ok(path.map(|steps| steps.into_iter().map(PyPathStep::from_path_step).collect()))
    }

    /// Contextual subgraph extraction. GIL released.
    fn contextual_subgraph(
        &mut self,
        py: Python<'_>,
        ctx: &PyDynamicContext,
        k: usize,
    ) -> PyResult<PySubGraph> {
        let rust_ctx = self.resolve_context(ctx);
        let graph = self.require_inner()?;

        let sg = py.allow_threads(|| {
            traversal::contextual_subgraph(graph, &rust_ctx, k)
        });

        Ok(PySubGraph {
            nodes: sg.nodes.into_iter().map(PyNodeResult::from_node_result).collect(),
            edges: sg.edges.into_iter().map(PyEdgeResult::from_edge_result).collect(),
        })
    }

    /// Multi-source heatmap intersection. GIL released during computation.
    ///
    /// Launches beam_traverse from each start node, accumulates a weighted
    /// heatmap, then keeps only nodes hit by `threshold` or more beams.
    /// Start nodes are always preserved.
    #[pyo3(signature = (start_nodes, k, depth, ctx, threshold = None))]
    fn multi_beam_intersection(
        &mut self,
        py: Python<'_>,
        start_nodes: Vec<String>,
        k: usize,
        depth: usize,
        ctx: &PyDynamicContext,
        threshold: Option<usize>,
    ) -> PyResult<PySubGraph> {
        let rust_ctx = self.resolve_context(ctx);
        let graph = self.require_inner()?;
        let t = threshold.unwrap_or_else(|| start_nodes.len().saturating_sub(1).max(1));

        let sg = py.allow_threads(|| {
            traversal::multi_beam_intersection(graph, &rust_ctx, &start_nodes, k, depth, t)
        });

        Ok(PySubGraph {
            nodes: sg.nodes.into_iter().map(PyNodeResult::from_node_result).collect(),
            edges: sg.edges.into_iter().map(PyEdgeResult::from_edge_result).collect(),
        })
    }

    /// Serialize to rkyv bytes. Only works on Owned graphs.
    fn to_rkyv<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        let inner = self.inner.as_ref().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("Graph has been closed")
        })?;

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
    pub weight_overrides: HashMap<String, f32>,
    #[pyo3(get, set)]
    pub noise_tags: HashSet<String>,
    #[pyo3(get, set)]
    pub max_fan_out: Option<usize>,
    #[pyo3(get, set)]
    pub w_base: f32,
    #[pyo3(get, set)]
    pub w_semantic: f32,
    #[pyo3(get, set)]
    pub w_noise: f32,
    #[pyo3(get, set)]
    pub w_override: f32,
    /// Virtual overlay nodes (per-tenant customization).
    pub overlay_nodes_raw: Vec<HashMap<String, String>>,
    /// Virtual overlay edges: [{"from": ..., "to": ..., "kind": ...}].
    pub overlay_edges_raw: Vec<HashMap<String, String>>,
    /// Risk #14: cache key for overlay data per project.
    /// If set & matches previous call, reuse cached parsed overlay.
    #[pyo3(get, set)]
    pub overlay_cache_key: Option<String>,
}

#[pymethods]
impl PyDynamicContext {
    #[new]
    #[pyo3(signature = (
        semantic_boosts = None,
        weight_overrides = None,
        noise_tags = None,
        max_fan_out = None,
        w_base = 1.0,
        w_semantic = 1.5,
        w_noise = 1.0,
        w_override = 1.0,
        overlay_nodes = None,
        overlay_edges = None,
        overlay_cache_key = None
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        semantic_boosts: Option<HashMap<String, f32>>,
        weight_overrides: Option<HashMap<String, f32>>,
        noise_tags: Option<HashSet<String>>,
        max_fan_out: Option<usize>,
        w_base: f32,
        w_semantic: f32,
        w_noise: f32,
        w_override: f32,
        overlay_nodes: Option<Vec<HashMap<String, String>>>,
        overlay_edges: Option<Vec<HashMap<String, String>>>,
        overlay_cache_key: Option<String>,
    ) -> Self {
        Self {
            semantic_boosts: semantic_boosts.unwrap_or_default(),
            weight_overrides: weight_overrides.unwrap_or_default(),
            noise_tags: noise_tags.unwrap_or_default(),
            max_fan_out,
            w_base,
            w_semantic,
            w_noise,
            w_override,
            overlay_nodes_raw: overlay_nodes.unwrap_or_default(),
            overlay_edges_raw: overlay_edges.unwrap_or_default(),
            overlay_cache_key,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "DynamicContext(boosts={}, overrides={}, noise_tags={}, max_fan_out={:?})",
            self.semantic_boosts.len(),
            self.weight_overrides.len(),
            self.noise_tags.len(),
            self.max_fan_out,
        )
    }
}

impl PyDynamicContext {
    /// Parse overlay raw dicts into Rust types.
    fn parse_overlays(&self) -> (Vec<NodeData>, Vec<(String, String, EdgeData)>) {
        let overlay_nodes = self
            .overlay_nodes_raw
            .iter()
            .map(|m| NodeData {
                name: m.get("name").cloned().unwrap_or_default(),
                kind: m.get("kind").cloned().unwrap_or_default(),
                metadata: HashMap::new(),
                base_weight: m.get("base_weight").and_then(|v| v.parse().ok()).unwrap_or(0.5),
                noise_penalty: m.get("noise_penalty").and_then(|v| v.parse().ok()).unwrap_or(0.0),
                pagerank_weight: 0.0,
            })
            .collect();

        let overlay_edges = self
            .overlay_edges_raw
            .iter()
            .map(|m| {
                let from = m.get("from").cloned().unwrap_or_default();
                let to = m.get("to").cloned().unwrap_or_default();
                let edge = EdgeData {
                    kind: m.get("kind").cloned().unwrap_or_else(|| "relates_to".to_string()),
                    field_name: m.get("field").cloned(),
                    base_weight: m.get("base_weight").and_then(|v| v.parse().ok()).unwrap_or(1.0),
                };
                (from, to, edge)
            })
            .collect();

        (overlay_nodes, overlay_edges)
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
    pub override_component: f32,
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
            override_component: nr.override_component,
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
        m.insert("override".to_string(), self.override_component);
        m.insert("total".to_string(), self.weight);
        m
    }

    fn __repr__(&self) -> String {
        format!("NodeResult(name={:?}, weight={:.4})", self.name, self.weight)
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
}

impl PyEdgeResult {
    fn from_edge_result(er: crate::types::EdgeResult) -> Self {
        Self {
            source: er.source,
            target: er.target,
            kind: er.kind,
            field_name: er.field_name,
            weight: er.weight,
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

/// Build a graph from Python dicts.
#[pyfunction]
#[pyo3(name = "build_graph")]
pub fn py_build_graph(nodes: &Bound<'_, PyList>, edges: &Bound<'_, PyList>) -> PyResult<PyOrpheusGraph> {
    let mut rust_nodes: Vec<NodeInput> = Vec::with_capacity(nodes.len());
    for item in nodes.iter() {
        let dict = item.downcast::<PyDict>()?;
        rust_nodes.push(NodeInput {
            name: dict.get_item("name")?.ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("name"))?.extract()?,
            kind: dict.get_item("kind")?.ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("kind"))?.extract()?,
            metadata: dict.get_item("metadata")?.map(|v| v.extract()).transpose()?.unwrap_or_default(),
            base_weight: dict.get_item("base_weight")?.ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("base_weight"))?.extract()?,
            noise_penalty: dict.get_item("noise_penalty")?.map(|v| v.extract::<f32>()).transpose()?.unwrap_or(0.0),
        });
    }

    let mut rust_edges: Vec<EdgeInput> = Vec::with_capacity(edges.len());
    for item in edges.iter() {
        let dict = item.downcast::<PyDict>()?;
        rust_edges.push(EdgeInput {
            from: dict.get_item("from")?.ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("from"))?.extract()?,
            to: dict.get_item("to")?.ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("to"))?.extract()?,
            kind: dict.get_item("kind")?.ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("kind"))?.extract()?,
            field_name: dict.get_item("field")?.map(|v| v.extract()).transpose()?,
            base_weight: dict.get_item("base_weight")?.map(|v| v.extract::<f32>()).transpose()?.unwrap_or(1.0),
        });
    }

    let (g, m) = build_graph(rust_nodes, rust_edges);
    let graph_inner = OrpheusGraphInner::new(g, m);

    Ok(PyOrpheusGraph {
        inner: Some(GraphInner::Owned(graph_inner)),
        overlay_cache: HashMap::new(),
    })
}

/// Load a graph from rkyv bytes (zero-copy).
#[pyfunction]
#[pyo3(name = "from_rkyv")]
pub fn py_from_rkyv(data: &Bound<'_, PyBytes>) -> PyResult<PyOrpheusGraph> {
    let bytes = data.as_bytes().to_vec();
    let view = ArchivedGraphView::from_bytes(bytes)
        .map_err(pyo3::exceptions::PyValueError::new_err)?;

    Ok(PyOrpheusGraph {
        inner: Some(GraphInner::Archived(view)),
        overlay_cache: HashMap::new(),
    })
}
