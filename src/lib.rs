//! # OrpheusGraph — Knowledge graph traversal engine
//!
//! A Rust crate for building, scoring, and traversing knowledge graphs
//! with support for dynamic overlays, zero-copy serialization (rkyv),
//! and Python bindings (PyO3).

pub mod accessor;
pub mod builder;
pub mod graph;
pub mod overlay;
pub mod pybridge;
pub mod scoring;
pub mod serialization;
pub mod traversal;
pub mod types;

// Re-exports for convenient access
pub use builder::build_graph;
pub use graph::OrpheusGraphInner;
pub use overlay::{neighbors_with_overlay, resolve_overlay_node, NeighborEntry};
pub use scoring::compute_score;
pub use traversal::{beam_traverse, contextual_subgraph, find_path};
pub use types::*;

use pyo3::prelude::*;

/// Python module entry point.
#[pymodule]
fn orpheusgraph(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<pybridge::PyOrpheusGraph>()?;
    m.add_class::<pybridge::PyDynamicContext>()?;
    m.add_class::<pybridge::PyNodeResult>()?;
    m.add_class::<pybridge::PyEdgeResult>()?;
    m.add_class::<pybridge::PyPathStep>()?;
    m.add_class::<pybridge::PySubGraph>()?;
    m.add_function(wrap_pyfunction!(pybridge::py_build_graph, m)?)?;
    m.add_function(wrap_pyfunction!(pybridge::py_from_rkyv, m)?)?;
    Ok(())
}
