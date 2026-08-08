//! # V1 CSR snapshot format + mmap-traversed archived accessor (Phase 2b)
//!
//! This module is the persistent-store-only snapshot path. It is deliberately
//! separate from [`crate::serialization`] (the legacy ephemeral/Redis V0 flat
//! path, `format_version` 0, never mmap-traversed):
//!
//! * V1 is a **CSR** (compressed-sparse-row) layout — `nodes`, `edges`,
//!   `node_offsets` (outgoing), plus `in_offsets` + `in_mirror` (incoming) —
//!   so neighbor expansion is `O(degree)` over a memory-mapped file with no
//!   per-open build cost beyond the `name → idx` index. This fixes the §4.4b
//!   defect where V0's [`crate::serialization::ArchivedGraphView`] does an
//!   `O(E)` `.filter` scan per expansion.
//! * It colocates the `memmap2` unsafe surface, the [`ArchivedCsrView`]
//!   self-referential pointer, the [`Validate`] trust-boundary modes and the
//!   §5.1 semantic sweep in one auditable place.
//!
//! ## Trust boundary (§5.1)
//! [`validate_v1`] with [`Validate::Full`] is MANDATORY for any untrusted
//! (shared/remote/network) snapshot: it checks the crc, runs rkyv structural
//! (`bytecheck`) validation, and then a linear `O(N+E)` semantic sweep that
//! proves every edge endpoint is in range, `node_offsets`/`in_offsets` are
//! monotonic + correctly sentinel-terminated and group-consistent, and
//! `in_mirror` is a genuine permutation of `0..E`. A hostile snapshot is
//! REJECTED with a typed error — it never reaches a query and panics. No
//! `unwrap`/`expect`/`access_unchecked` is used on any archived query path or
//! on any open over untrusted bytes.

use std::collections::HashMap;
use std::fs::File;
use std::path::Path;

use crate::accessor::{GraphAccessor, NeighborView, NodeView};
use crate::builder::rebuild_from_serialized;
use crate::graph::OrpheusGraphInner;
use crate::persist::error::PersistError;
use crate::types::{EdgeData, NodeData};

use super::BaseGraph;
use super::BaseMode;

// ---------------------------------------------------------------------------
// V1 CSR archive types
// ---------------------------------------------------------------------------

/// V1 CSR snapshot. Reuses [`NodeData`] for nodes and a dedicated [`CsrEdge`]
/// for edges. The offset arrays are the CSR machinery:
///
/// * `node_offsets` (len `N+1`): outgoing CSR keyed by `from_idx`. Node `i`'s
///   outgoing edges are `edges[node_offsets[i] .. node_offsets[i+1]]`.
/// * `in_offsets` (len `N+1`) + `in_mirror` (len `E`): incoming CSR keyed by
///   `to_idx`. Node `j`'s incoming edges are the `edges` at indices
///   `in_mirror[in_offsets[j] .. in_offsets[j+1]]`. `in_mirror` is a permutation
///   of `0..E`. `in_offsets` alone would force an `O(E)` scan to find a node's
///   incoming group, so it is validated with the same rules as `node_offsets`.
#[derive(Debug, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
#[rkyv(derive(Debug))]
pub struct SerializableGraphV1 {
    pub nodes: Vec<NodeData>,
    pub edges: Vec<CsrEdge>,
    pub node_offsets: Vec<u32>,
    pub in_offsets: Vec<u32>,
    pub in_mirror: Vec<u32>,
}

/// A CSR edge stored by node indices. Distinct from
/// [`crate::serialization::SerializableEdge`] so V1 stays self-contained.
#[derive(Debug, Clone, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
#[rkyv(derive(Debug))]
pub struct CsrEdge {
    pub from_idx: u32,
    pub to_idx: u32,
    pub kind: String,
    pub field_name: Option<String>,
    pub base_weight: f32,
}

// ---------------------------------------------------------------------------
// Serialization: OrpheusGraphInner -> V1 CSR bytes (deterministic)
// ---------------------------------------------------------------------------

/// Serialize an owned graph to V1 CSR rkyv bytes.
///
/// **Determinism** (spec: byte-identical for the same logical graph): nodes are
/// emitted in total-order by `name` (names are unique keys); edges are emitted
/// in total-order by `(from_idx, to_idx, kind, field_name, base_weight)`, which
/// also groups them by `from_idx` — exactly the outgoing-CSR order. `in_mirror`
/// is a stable-by-index sort of `0..E` by `to_idx`. Two serializations of the
/// same graph therefore produce identical bytes regardless of internal petgraph
/// index order.
pub fn to_rkyv_v1(graph: &OrpheusGraphInner) -> Vec<u8> {
    let inner = graph.inner_graph();

    // 1. Deterministic node order: total-order by unique name.
    let mut nodes: Vec<NodeData> = inner.node_indices().map(|i| inner[i].clone()).collect();
    nodes.sort_by(|a, b| a.name.cmp(&b.name));

    let n = nodes.len();
    let mut name_to_idx: HashMap<&str, u32> = HashMap::with_capacity(n);
    for (i, nd) in nodes.iter().enumerate() {
        name_to_idx.insert(nd.name.as_str(), i as u32);
    }

    // 2. Edges in the new (sorted) index space.
    let mut edges: Vec<CsrEdge> = Vec::with_capacity(inner.edge_count());
    for e_idx in inner.edge_indices() {
        let (from, to) = inner
            .edge_endpoints(e_idx)
            .expect("edge_endpoints on a live edge index");
        let ed = &inner[e_idx];
        edges.push(CsrEdge {
            from_idx: name_to_idx[inner[from].name.as_str()],
            to_idx: name_to_idx[inner[to].name.as_str()],
            kind: ed.kind.clone(),
            field_name: ed.field_name.clone(),
            base_weight: ed.base_weight,
        });
    }

    // 3. Total-order sort. Primary key `from_idx` groups outgoing CSR runs.
    edges.sort_by(|a, b| {
        a.from_idx
            .cmp(&b.from_idx)
            .then(a.to_idx.cmp(&b.to_idx))
            .then(a.kind.cmp(&b.kind))
            .then(a.field_name.cmp(&b.field_name))
            .then(a.base_weight.total_cmp(&b.base_weight))
    });
    let e = edges.len();

    // 4. Outgoing CSR offsets via a count + prefix-sum (deterministic).
    let mut node_offsets = vec![0u32; n + 1];
    for ed in &edges {
        node_offsets[ed.from_idx as usize + 1] += 1;
    }
    for i in 0..n {
        node_offsets[i + 1] += node_offsets[i];
    }

    // 5. Incoming CSR: in_mirror = 0..E sorted by to_idx (index tiebreak =>
    //    deterministic + stable); in_offsets via count + prefix-sum.
    let mut in_mirror: Vec<u32> = (0..e as u32).collect();
    in_mirror.sort_by(|&x, &y| {
        edges[x as usize]
            .to_idx
            .cmp(&edges[y as usize].to_idx)
            .then(x.cmp(&y))
    });
    let mut in_offsets = vec![0u32; n + 1];
    for ed in &edges {
        in_offsets[ed.to_idx as usize + 1] += 1;
    }
    for i in 0..n {
        in_offsets[i + 1] += in_offsets[i];
    }

    let sg = SerializableGraphV1 {
        nodes,
        edges,
        node_offsets,
        in_offsets,
        in_mirror,
    };
    rkyv::to_bytes::<rkyv::rancor::Error>(&sg)
        .expect("rkyv V1 serialization failed")
        .to_vec()
}

/// Rebuild an owned [`OrpheusGraphInner`] from V1 CSR bytes (for `mode=Owned`).
///
/// The CSR offset arrays are ignored on rebuild — the petgraph is reconstructed
/// from `nodes` + `edges(from_idx,to_idx)` alone, mirroring
/// [`crate::serialization::from_rkyv_rebuild`].
pub fn from_rkyv_rebuild_v1(data: &[u8]) -> Result<OrpheusGraphInner, String> {
    let sg = rkyv::from_bytes::<SerializableGraphV1, rkyv::rancor::Error>(data)
        .map_err(|e| format!("rkyv V1 deserialization failed: {e}"))?;

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
// Validation modes (§5.1 trust boundary)
// ---------------------------------------------------------------------------

/// How much a snapshot is checked at open. See module docs for the trust
/// contract. Defaults to [`Validate::Full`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Validate {
    /// crc32 + rkyv structural (`bytecheck`) + the `O(N+E)` §5.1 semantic sweep.
    /// MANDATORY for any untrusted / shared / remote snapshot.
    #[default]
    Full,
    /// crc32 + rkyv structural only; SKIP the semantic sweep. For a file THIS
    /// process wrote but whose RAM copy it no longer trusts bit-for-bit — a
    /// matching crc proves the bytes are exactly what we wrote, and we only ever
    /// write semantically-valid V1.
    Crc,
    /// No crc, no structural walk, `access_unchecked` — `O(1)`. Permitted ONLY
    /// for a file the process itself wrote this run (e.g. the post-compaction
    /// re-mmap). NEVER for untrusted input.
    None,
}

/// Validate V1 bytes per `mode`. Never panics — every failure is a typed
/// [`PersistError`]. `expected_crc` is the crc recorded in the MANIFEST.
pub fn validate_v1(bytes: &[u8], mode: Validate, expected_crc: u32) -> Result<(), PersistError> {
    match mode {
        Validate::None => Ok(()),
        Validate::Crc => {
            check_crc(bytes, expected_crc)?;
            access_checked(bytes)?;
            Ok(())
        }
        Validate::Full => {
            check_crc(bytes, expected_crc)?;
            let archived = access_checked(bytes)?;
            validate_csr(archived)?;
            Ok(())
        }
    }
}

fn check_crc(bytes: &[u8], expected_crc: u32) -> Result<(), PersistError> {
    let computed = crc32fast::hash(bytes);
    if computed != expected_crc {
        return Err(PersistError::Corrupt(format!(
            "V1 snapshot crc mismatch: computed {computed}, expected {expected_crc}"
        )));
    }
    Ok(())
}

fn access_checked(bytes: &[u8]) -> Result<&ArchivedSerializableGraphV1, PersistError> {
    rkyv::access::<ArchivedSerializableGraphV1, rkyv::rancor::Error>(bytes)
        .map_err(|e| PersistError::Corrupt(format!("V1 rkyv structural validation failed: {e}")))
}

/// The §5.1 semantic sweep. One linear pass each over edges and the two CSR
/// offset arrays; a single `seen` bitset proves `in_mirror` is a permutation.
/// Consumes ONLY `.get()`/`.to_native()` — never index-panics.
fn validate_csr(a: &ArchivedSerializableGraphV1) -> Result<(), PersistError> {
    let n = a.nodes.len();
    let e = a.edges.len();

    let corrupt = |msg: String| PersistError::Corrupt(format!("V1 CSR invalid: {msg}"));

    // --- edge endpoints in range ---
    for (i, edge) in a.edges.iter().enumerate() {
        let from = edge.from_idx.to_native() as usize;
        let to = edge.to_idx.to_native() as usize;
        if from >= n || to >= n {
            return Err(corrupt(format!(
                "edge {i} endpoint out of range (from={from}, to={to}, n={n})"
            )));
        }
    }

    // --- node_offsets: shape + monotonic + sentinel + group-consistency ---
    if a.node_offsets.len() != n + 1 {
        return Err(corrupt(format!(
            "node_offsets len {} != n+1 ({})",
            a.node_offsets.len(),
            n + 1
        )));
    }
    let off_at = |v: &rkyv::vec::ArchivedVec<rkyv::Archived<u32>>, i: usize| -> Option<u32> {
        v.get(i).map(|x| x.to_native())
    };
    // Canonical CSR base: offsets[0] MUST be 0, else edges [0, offsets[0]) fall
    // into no group and are silently invisible to outgoing traversal (they pass
    // the monotonic/sentinel checks otherwise). Required for the group-
    // consistency claim to cover ALL edges, not just [offsets[0], E).
    if off_at(&a.node_offsets, 0) != Some(0) {
        return Err(corrupt(
            "node_offsets[0] must be 0 (canonical CSR base)".into(),
        ));
    }
    let mut prev = 0u32;
    for i in 0..=n {
        let cur = off_at(&a.node_offsets, i).ok_or_else(|| corrupt("node_offsets short".into()))?;
        if cur < prev {
            return Err(corrupt(format!(
                "node_offsets not monotonic at {i}: {prev} -> {cur}"
            )));
        }
        if cur as usize > e {
            return Err(corrupt(format!("node_offsets[{i}]={cur} > E={e}")));
        }
        prev = cur;
    }
    if prev as usize != e {
        return Err(corrupt(format!("node_offsets sentinel {prev} != E={e}")));
    }
    // Strong form: edges[node_offsets[i]..node_offsets[i+1]].from_idx == i.
    // The two `off_at` reads are `?`-guarded (not unwrapped) so a hostile file
    // can never panic here even though the shape check above already proved len.
    for i in 0..n {
        let lo = off_at(&a.node_offsets, i)
            .ok_or_else(|| corrupt("node_offsets short (strong)".into()))?
            as usize;
        let hi = off_at(&a.node_offsets, i + 1)
            .ok_or_else(|| corrupt("node_offsets short (strong)".into()))?
            as usize;
        for k in lo..hi {
            let edge = a
                .edges
                .get(k)
                .ok_or_else(|| corrupt(format!("node_offsets slice {k} OOB")))?;
            if edge.from_idx.to_native() as usize != i {
                return Err(corrupt(format!(
                    "node_offsets group {i} holds edge {k} with from_idx {}",
                    edge.from_idx.to_native()
                )));
            }
        }
    }

    // --- in_offsets: shape + monotonic + sentinel ---
    if a.in_offsets.len() != n + 1 {
        return Err(corrupt(format!(
            "in_offsets len {} != n+1 ({})",
            a.in_offsets.len(),
            n + 1
        )));
    }
    if off_at(&a.in_offsets, 0) != Some(0) {
        return Err(corrupt(
            "in_offsets[0] must be 0 (canonical CSR base)".into(),
        ));
    }
    let mut prev = 0u32;
    for i in 0..=n {
        let cur = off_at(&a.in_offsets, i).ok_or_else(|| corrupt("in_offsets short".into()))?;
        if cur < prev {
            return Err(corrupt(format!(
                "in_offsets not monotonic at {i}: {prev} -> {cur}"
            )));
        }
        if cur as usize > e {
            return Err(corrupt(format!("in_offsets[{i}]={cur} > E={e}")));
        }
        prev = cur;
    }
    if prev as usize != e {
        return Err(corrupt(format!("in_offsets sentinel {prev} != E={e}")));
    }

    // --- in_mirror: a permutation of 0..E (len + <E + no dup via seen bitset) ---
    if a.in_mirror.len() != e {
        return Err(corrupt(format!(
            "in_mirror len {} != E={e}",
            a.in_mirror.len()
        )));
    }
    let mut seen = vec![false; e];
    for k in 0..e {
        let m = a
            .in_mirror
            .get(k)
            .ok_or_else(|| corrupt("in_mirror short".into()))?
            .to_native() as usize;
        if m >= e {
            return Err(corrupt(format!("in_mirror[{k}]={m} >= E={e}")));
        }
        if seen[m] {
            return Err(corrupt(format!("in_mirror duplicate entry {m}")));
        }
        seen[m] = true;
    }
    // Strong form: for each node j, edges[in_mirror[in_offsets[j]..j+1]].to_idx == j.
    // All reads `?`-guarded — no unwrap/panic on hostile bytes.
    for j in 0..n {
        let lo = off_at(&a.in_offsets, j)
            .ok_or_else(|| corrupt("in_offsets short (strong)".into()))? as usize;
        let hi = off_at(&a.in_offsets, j + 1)
            .ok_or_else(|| corrupt("in_offsets short (strong)".into()))? as usize;
        for k in lo..hi {
            let m = a
                .in_mirror
                .get(k)
                .ok_or_else(|| corrupt("in_mirror short (strong)".into()))?
                .to_native() as usize;
            let edge = a
                .edges
                .get(m)
                .ok_or_else(|| corrupt(format!("in_mirror target {m} OOB")))?;
            if edge.to_idx.to_native() as usize != j {
                return Err(corrupt(format!(
                    "in_offsets group {j} holds edge {m} with to_idx {}",
                    edge.to_idx.to_native()
                )));
            }
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// ArchivedCsrView — mmap-backed zero-copy accessor
// ---------------------------------------------------------------------------

/// Zero-copy `O(degree)` accessor over a memory-mapped V1 CSR snapshot.
///
/// # Soundness of the `unsafe` Send/Sync + self-referential pointer
/// * The mapping is **read-only** (no interior mutability, no writes ever), so
///   shared cross-thread access is a data-race-free read => `Sync` is sound;
///   the `Mmap` handle is owned => `Send` is sound.
/// * `archived` is derived once from a validated archive and is stable.
///   Crucially, [`memmap2::Mmap`]'s bytes live in an OS mapping whose virtual
///   address is INDEPENDENT of where the `Mmap` handle struct sits, so moving
///   the `ArchivedCsrView` moves only the small handle, never the mapped region
///   — the `*const` stays valid across moves (a stronger guarantee than the
///   `Vec<u8>`-backed [`crate::serialization::ArchivedGraphView`]).
pub struct ArchivedCsrView {
    // Owns the mapped bytes; must outlive `archived`. Kept first so it is
    // dropped last (though drop order does not affect the raw pointer here).
    _mmap: memmap2::Mmap,
    archived: *const ArchivedSerializableGraphV1,
    name_index: HashMap<String, u32>,
}

// SAFETY: read-only mapping, pointer stable across moves (see type docs).
unsafe impl Send for ArchivedCsrView {}
unsafe impl Sync for ArchivedCsrView {}

impl ArchivedCsrView {
    fn archived(&self) -> &ArchivedSerializableGraphV1 {
        // SAFETY: `_mmap` is alive as long as `self`; the pointer was derived
        // from a validated archive over that same mapping and is stable.
        unsafe { &*self.archived }
    }

    /// Live node names, in stored (name-sorted) order. Used by compaction to
    /// enumerate the base for `materialize`.
    pub fn node_names(&self) -> Vec<String> {
        self.archived()
            .nodes
            .iter()
            .map(|n| n.name.to_string())
            .collect()
    }

    #[inline]
    fn off(v: &rkyv::vec::ArchivedVec<rkyv::Archived<u32>>, i: usize) -> Option<usize> {
        v.get(i).map(|x| x.to_native() as usize)
    }
}

impl GraphAccessor for ArchivedCsrView {
    fn node_count(&self) -> usize {
        self.archived().nodes.len()
    }

    fn edge_count(&self) -> usize {
        self.archived().edges.len()
    }

    fn get_node(&self, name: &str) -> Option<NodeView> {
        let &idx = self.name_index.get(name)?;
        let node = self.archived().nodes.get(idx as usize)?;
        Some(NodeView {
            name: node.name.to_string(),
            kind: node.kind.to_string(),
            base_weight: node.base_weight.to_native(),
            noise_penalty: node.noise_penalty.to_native(),
            pagerank_weight: node.pagerank_weight.to_native(),
            metadata: node
                .metadata
                .iter()
                .map(|(k, v)| (k.to_string(), v.to_string()))
                .collect(),
        })
    }

    fn outgoing_neighbors(&self, name: &str) -> Vec<NeighborView> {
        let Some(&idx) = self.name_index.get(name) else {
            return vec![];
        };
        let a = self.archived();
        let idx = idx as usize;
        // `.get()` everywhere: even a `none`-validated hostile file cannot panic.
        let (Some(lo), Some(hi)) = (
            Self::off(&a.node_offsets, idx),
            Self::off(&a.node_offsets, idx + 1),
        ) else {
            return vec![];
        };
        let mut out = Vec::new();
        for k in lo..hi {
            let Some(edge) = a.edges.get(k) else { break };
            let Some(target) = a.nodes.get(edge.to_idx.to_native() as usize) else {
                continue;
            };
            out.push(NeighborView {
                target_name: target.name.to_string(),
                edge_kind: edge.kind.to_string(),
                field_name: edge.field_name.as_ref().map(|s| s.to_string()),
                edge_weight: edge.base_weight.to_native(),
            });
        }
        out
    }

    fn incoming_neighbors(&self, name: &str) -> Vec<NeighborView> {
        let Some(&idx) = self.name_index.get(name) else {
            return vec![];
        };
        let a = self.archived();
        let idx = idx as usize;
        let (Some(lo), Some(hi)) = (
            Self::off(&a.in_offsets, idx),
            Self::off(&a.in_offsets, idx + 1),
        ) else {
            return vec![];
        };
        let mut inc = Vec::new();
        for k in lo..hi {
            let Some(m) = a.in_mirror.get(k).map(|x| x.to_native() as usize) else {
                break;
            };
            let Some(edge) = a.edges.get(m) else { continue };
            let Some(source) = a.nodes.get(edge.from_idx.to_native() as usize) else {
                continue;
            };
            inc.push(NeighborView {
                target_name: source.name.to_string(),
                edge_kind: edge.kind.to_string(),
                field_name: edge.field_name.as_ref().map(|s| s.to_string()),
                edge_weight: edge.base_weight.to_native(),
            });
        }
        inc
    }
}

// ---------------------------------------------------------------------------
// open_snapshot — the mmap / owned open path
// ---------------------------------------------------------------------------

/// Open a V1 CSR snapshot at `path` and return a [`BaseGraph`].
///
/// * `mode = Owned`: read + validate the bytes up front and materialize an
///   owned petgraph (`from_rkyv_rebuild_v1`). No lazy faulting — safe for
///   network / larger-than-RAM-under-pressure callers (§4.5 SIGBUS precondition).
/// * `mode = Mmap`: map the file read-only, validate per `validate`, derive the
///   stable archive pointer, build the `name → idx` index, and (if `prefault`)
///   `madvise(WILLNEED)` to warm every page up front.
///
/// **mmap precondition (§4.5):** the file must live on local block storage and
/// must not be mutated in place while mapped — guaranteed here by the
/// rename-only + unlink-safe (Linux) snapshot discipline.
pub fn open_snapshot(
    path: &Path,
    mode: BaseMode,
    validate: Validate,
    prefault: bool,
    expected_crc: u32,
) -> Result<BaseGraph, PersistError> {
    match mode {
        BaseMode::Owned => {
            let bytes = std::fs::read(path)?;
            validate_v1(&bytes, validate, expected_crc)?;
            let inner = from_rkyv_rebuild_v1(&bytes).map_err(PersistError::Corrupt)?;
            Ok(BaseGraph::Owned(inner))
        }
        BaseMode::Mmap => {
            let file = File::open(path)?;
            // SAFETY: read-only map; the file is not mutated in place while
            // mapped (rename-only + unlink-safe discipline, §4.5).
            let mmap = unsafe { memmap2::Mmap::map(&file)? };

            validate_v1(&mmap[..], validate, expected_crc)?;

            // Obtain the archive pointer. full/crc => structural (checked);
            // none => access_unchecked (trusted self-written file only).
            let archived: *const ArchivedSerializableGraphV1 = match validate {
                Validate::None => {
                    // SAFETY: `none` is only ever passed for a file this process
                    // wrote this run (post-compaction re-mmap); its structure is
                    // known-valid, so `access_unchecked` has no UB.
                    let a =
                        unsafe { rkyv::access_unchecked::<ArchivedSerializableGraphV1>(&mmap[..]) };
                    a as *const _
                }
                _ => {
                    let a = access_checked(&mmap[..])?;
                    a as *const _
                }
            };

            // Build the name index (O(N); the only per-open build cost).
            // SAFETY: `mmap` is live for the duration of this block.
            let name_index = {
                let a = unsafe { &*archived };
                let mut map = HashMap::with_capacity(a.nodes.len());
                for (i, node) in a.nodes.iter().enumerate() {
                    map.insert(node.name.to_string(), i as u32);
                }
                map
            };

            if prefault {
                // madvise(WILLNEED): warm all pages up front (matches the V0
                // load-all without deserialization). Unix-only; a no-op knob
                // elsewhere. Best-effort — an advise failure is not fatal.
                #[cfg(unix)]
                {
                    let _ = mmap.advise(memmap2::Advice::WillNeed);
                }
            }

            Ok(BaseGraph::Archived(ArchivedCsrView {
                _mmap: mmap,
                archived,
                name_index,
            }))
        }
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::accessor::GraphAccessor;
    use crate::builder::build_graph;
    use crate::types::{EdgeInput, NodeInput};
    use std::collections::HashMap as Map;
    use std::io::Write;

    fn inner(nodes: Vec<(&str, &str)>, edges: Vec<(&str, &str, &str)>) -> OrpheusGraphInner {
        let n: Vec<NodeInput> = nodes
            .iter()
            .map(|(name, kind)| NodeInput {
                name: name.to_string(),
                kind: kind.to_string(),
                metadata: Map::new(),
                base_weight: 0.5,
                noise_penalty: 0.1,
            })
            .collect();
        let e: Vec<EdgeInput> = edges
            .iter()
            .map(|(f, t, k)| EdgeInput {
                from: f.to_string(),
                to: t.to_string(),
                kind: k.to_string(),
                field_name: Some(format!("{f}_{t}")),
                base_weight: 0.7,
            })
            .collect();
        let (g, m) = build_graph(n, e);
        OrpheusGraphInner::new(g, m)
    }

    fn sample() -> OrpheusGraphInner {
        inner(
            vec![
                ("A", "model"),
                ("B", "model"),
                ("C", "field"),
                ("D", "model"),
            ],
            vec![
                ("A", "B", "rel"),
                ("A", "C", "contains"),
                ("B", "C", "contains"),
                ("D", "A", "rel"),
                ("A", "A", "self"), // self-loop: outgoing AND incoming for A
            ],
        )
    }

    /// (target, kind, field, weight-bits) — sorted, order-insensitive.
    fn nbrs(acc: &dyn GraphAccessor, name: &str, out: bool) -> Vec<(String, String, String, u32)> {
        let list = if out {
            acc.outgoing_neighbors(name)
        } else {
            acc.incoming_neighbors(name)
        };
        let mut v: Vec<_> = list
            .into_iter()
            .map(|nb| {
                (
                    nb.target_name,
                    nb.edge_kind,
                    nb.field_name.unwrap_or_default(),
                    nb.edge_weight.to_bits(),
                )
            })
            .collect();
        v.sort();
        v
    }

    fn write_temp(bytes: &[u8]) -> (tempfile::TempDir, std::path::PathBuf) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("snapshot-test.og");
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(bytes).unwrap();
        f.sync_all().unwrap();
        (dir, path)
    }

    fn open_csr(bytes: &[u8], validate: Validate) -> Result<BaseGraph, PersistError> {
        let crc = crc32fast::hash(bytes);
        let (_dir, path) = write_temp(bytes);
        let base = open_snapshot(&path, BaseMode::Mmap, validate, false, crc);
        // Keep the tempdir alive for the duration by leaking it into the mmap's
        // lifetime is not possible; instead callers must finish before _dir drops.
        // We return only after mapping; the file stays until _dir drops at fn end.
        // To keep the mapping valid in tests, we forget the dir guard.
        std::mem::forget(_dir);
        base
    }

    // ---- round-trip vs owned -------------------------------------------

    #[test]
    fn v1_round_trip_matches_owned() {
        let owned = sample();
        let bytes = to_rkyv_v1(&owned);
        let base = open_csr(&bytes, Validate::Full).unwrap();
        let view = base.as_accessor();

        assert_eq!(view.node_count(), owned.node_count());
        assert_eq!(view.edge_count(), owned.edge_count());

        for name in ["A", "B", "C", "D"] {
            let a = view.get_node(name).unwrap();
            let o = owned.get_node(name).unwrap();
            assert_eq!(a.name, o.name);
            assert_eq!(a.kind, o.kind);
            assert_eq!(a.base_weight.to_bits(), o.base_weight.to_bits());
            assert_eq!(a.pagerank_weight.to_bits(), o.pagerank_weight.to_bits());
            assert_eq!(
                nbrs(view, name, true),
                nbrs(&owned, name, true),
                "out {name}"
            );
            assert_eq!(
                nbrs(view, name, false),
                nbrs(&owned, name, false),
                "in {name}"
            );
        }
        assert!(view.get_node("nonexistent").is_none());
        assert!(view.outgoing_neighbors("nonexistent").is_empty());
    }

    // ---- CSR structure correctness -------------------------------------

    #[test]
    fn csr_structure_is_well_formed() {
        let bytes = to_rkyv_v1(&sample());
        let a = access_checked(&bytes).unwrap();
        let n = a.nodes.len();
        let e = a.edges.len();
        // node_offsets: sentinel + monotonic
        assert_eq!(a.node_offsets.len(), n + 1);
        assert_eq!(a.node_offsets.get(n).unwrap().to_native() as usize, e);
        // in_offsets sentinel
        assert_eq!(a.in_offsets.get(n).unwrap().to_native() as usize, e);
        // in_mirror is a permutation of 0..E
        assert_eq!(a.in_mirror.len(), e);
        let mut seen = vec![false; e];
        for k in 0..e {
            let m = a.in_mirror.get(k).unwrap().to_native() as usize;
            assert!(!seen[m]);
            seen[m] = true;
        }
        assert!(seen.iter().all(|&b| b));
        // Full validation passes.
        validate_csr(a).unwrap();
    }

    // ---- determinism ----------------------------------------------------

    #[test]
    fn v1_serialization_is_byte_identical() {
        let g1 = sample();
        let g2 = sample();
        assert_eq!(to_rkyv_v1(&g1), to_rkyv_v1(&g2));
    }

    // ---- owned open path -----------------------------------------------

    #[test]
    fn owned_open_matches_mmap_open() {
        let owned = sample();
        let bytes = to_rkyv_v1(&owned);
        let crc = crc32fast::hash(&bytes);
        let (_dir, path) = write_temp(&bytes);

        let m = open_snapshot(&path, BaseMode::Mmap, Validate::Full, false, crc).unwrap();
        let o = open_snapshot(&path, BaseMode::Owned, Validate::Full, false, crc).unwrap();
        for name in ["A", "B", "C", "D"] {
            assert_eq!(
                nbrs(m.as_accessor(), name, true),
                nbrs(o.as_accessor(), name, true)
            );
            assert_eq!(
                nbrs(m.as_accessor(), name, false),
                nbrs(o.as_accessor(), name, false)
            );
        }
        assert_eq!(m.as_accessor().node_count(), o.as_accessor().node_count());
    }

    // ---- prefault ------------------------------------------------------

    #[test]
    fn prefault_open_is_transparent() {
        let owned = sample();
        let bytes = to_rkyv_v1(&owned);
        let crc = crc32fast::hash(&bytes);
        let (_dir, path) = write_temp(&bytes);
        std::mem::forget(_dir);
        let base = open_snapshot(&path, BaseMode::Mmap, Validate::Full, true, crc).unwrap();
        assert_eq!(base.as_accessor().node_count(), owned.node_count());
        assert_eq!(nbrs(base.as_accessor(), "A", true), nbrs(&owned, "A", true));
    }

    // ---- validate rejects hostile bytes (no panic) ---------------------

    fn hostile(mutate: impl FnOnce(&mut SerializableGraphV1)) -> Vec<u8> {
        // Start from a valid graph, then corrupt one invariant.
        let owned = inner(vec![("A", "m"), ("B", "m")], vec![("A", "B", "r")]);
        let good = to_rkyv_v1(&owned);
        let mut sg = rkyv::from_bytes::<SerializableGraphV1, rkyv::rancor::Error>(&good).unwrap();
        mutate(&mut sg);
        rkyv::to_bytes::<rkyv::rancor::Error>(&sg).unwrap().to_vec()
    }

    #[test]
    fn full_rejects_oob_edge_index() {
        let bytes = hostile(|sg| sg.edges[0].to_idx = 99);
        let crc = crc32fast::hash(&bytes);
        let err = validate_v1(&bytes, Validate::Full, crc).unwrap_err();
        assert!(matches!(err, PersistError::Corrupt(_)));
    }

    #[test]
    fn full_rejects_bad_node_offsets_nonmonotonic() {
        // Make node_offsets non-monotonic: [0, 5, 1, ...] style.
        let bytes = hostile(|sg| {
            // len stays n+1 but break monotonicity + sentinel.
            let n = sg.nodes.len();
            sg.node_offsets = vec![0; n + 1];
            sg.node_offsets[1] = 5;
            sg.node_offsets[n] = 1;
        });
        let crc = crc32fast::hash(&bytes);
        let err = validate_v1(&bytes, Validate::Full, crc).unwrap_err();
        assert!(matches!(err, PersistError::Corrupt(_)));
    }

    #[test]
    fn full_rejects_wrong_sentinel() {
        let bytes = hostile(|sg| {
            let n = sg.nodes.len();
            // Monotonic but sentinel != E.
            sg.node_offsets = vec![0; n + 1];
        });
        let crc = crc32fast::hash(&bytes);
        let err = validate_v1(&bytes, Validate::Full, crc).unwrap_err();
        assert!(matches!(err, PersistError::Corrupt(_)));
    }

    #[test]
    fn full_rejects_short_node_offsets() {
        let bytes = hostile(|sg| {
            sg.node_offsets.pop(); // now len n, not n+1
        });
        let crc = crc32fast::hash(&bytes);
        let err = validate_v1(&bytes, Validate::Full, crc).unwrap_err();
        assert!(matches!(err, PersistError::Corrupt(_)));
    }

    #[test]
    fn full_rejects_in_mirror_out_of_range() {
        let bytes = hostile(|sg| {
            if !sg.in_mirror.is_empty() {
                sg.in_mirror[0] = 42; // >= E
            }
        });
        let crc = crc32fast::hash(&bytes);
        let err = validate_v1(&bytes, Validate::Full, crc).unwrap_err();
        assert!(matches!(err, PersistError::Corrupt(_)));
    }

    #[test]
    fn full_rejects_nonzero_offsets_base() {
        // node_offsets[0] != 0 orphans edges [0, offsets[0]) (invisible to
        // outgoing traversal) while passing monotonic/sentinel — must be rejected.
        let bytes = hostile(|sg| sg.node_offsets[0] = 1);
        let crc = crc32fast::hash(&bytes);
        assert!(matches!(
            validate_v1(&bytes, Validate::Full, crc),
            Err(PersistError::Corrupt(_))
        ));
        // Symmetric for in_offsets (incoming traversal).
        let bytes2 = hostile(|sg| sg.in_offsets[0] = 1);
        let crc2 = crc32fast::hash(&bytes2);
        assert!(matches!(
            validate_v1(&bytes2, Validate::Full, crc2),
            Err(PersistError::Corrupt(_))
        ));
    }

    #[test]
    fn full_rejects_in_mirror_duplicate() {
        // Two-edge graph so a duplicate is possible.
        let owned = inner(
            vec![("A", "m"), ("B", "m")],
            vec![("A", "B", "r"), ("B", "A", "r")],
        );
        let good = to_rkyv_v1(&owned);
        let mut sg = rkyv::from_bytes::<SerializableGraphV1, rkyv::rancor::Error>(&good).unwrap();
        // Force a duplicate (0,0) — no longer a permutation.
        sg.in_mirror = vec![0, 0];
        let bytes = rkyv::to_bytes::<rkyv::rancor::Error>(&sg).unwrap().to_vec();
        let crc = crc32fast::hash(&bytes);
        let err = validate_v1(&bytes, Validate::Full, crc).unwrap_err();
        assert!(matches!(err, PersistError::Corrupt(_)));
    }

    // ---- crc vs none ----------------------------------------------------

    #[test]
    fn crc_rejects_bitflip_none_skips() {
        let owned = sample();
        let mut bytes = to_rkyv_v1(&owned);
        let good_crc = crc32fast::hash(&bytes);
        // Flip a byte: crc mode must reject.
        bytes[16] ^= 0xFF;
        assert!(matches!(
            validate_v1(&bytes, Validate::Crc, good_crc),
            Err(PersistError::Corrupt(_))
        ));
        // none mode skips all checks (documents the trust contract).
        assert!(validate_v1(&bytes, Validate::None, good_crc).is_ok());
    }

    #[test]
    fn none_loads_semantically_invalid_self_written_buffer() {
        // Structurally valid (so access_unchecked is sound) but semantically
        // invalid (OOB to_idx). `none` loads it without error; we only assert
        // node_count (never touch the poisoned edge path).
        let bytes = hostile(|sg| sg.edges[0].to_idx = 99);
        let _crc = crc32fast::hash(&bytes); // matching crc (self-written); unused under Validate::None
        let base = open_csr(&bytes, Validate::None).unwrap();
        assert_eq!(base.as_accessor().node_count(), 2);
    }
}
