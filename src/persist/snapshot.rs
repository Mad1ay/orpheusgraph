//! # V2 CSR snapshot format + mmap-traversed archived accessor (Phase 2b)
//!
//! This module is the persistent-store-only snapshot path. It is deliberately
//! separate from [`crate::serialization`] (the legacy ephemeral/Redis V0 flat
//! path, `format_version` 0, never mmap-traversed):
//!
//! * V2 is a **CSR** (compressed-sparse-row) layout — `nodes`, `edges`,
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
//! [`validate_v2`] with [`Validate::Full`] is MANDATORY for any untrusted
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

// Note: acl is normalized (sorted+deduped) at INGEST time (builder + delta
// `apply`), not here — by the time an OrpheusGraphInner reaches `to_rkyv_v2`
// every edge.acl is already canonical, so this module trusts (and does not
// re-derive) that invariant.

use super::BaseGraph;
use super::BaseMode;

// ---------------------------------------------------------------------------
// V2 CSR archive types
// ---------------------------------------------------------------------------

/// V2 CSR snapshot. Reuses [`NodeData`] for nodes and a dedicated [`CsrEdge`]
/// for edges. The offset arrays are the CSR machinery:
///
/// * `node_offsets` (len `N+1`): outgoing CSR keyed by `from_idx`. Node `i`'s
///   outgoing edges are `edges[node_offsets[i] .. node_offsets[i+1]]`.
/// * `in_offsets` (len `N+1`) + `in_mirror` (len `E`): incoming CSR keyed by
///   `to_idx`. Node `j`'s incoming edges are the `edges` at indices
///   `in_mirror[in_offsets[j] .. in_offsets[j+1]]`. `in_mirror` is a permutation
///   of `0..E`. `in_offsets` alone would force an `O(E)` scan to find a node's
///   incoming group, so it is validated with the same rules as `node_offsets`.
/// * `name_buckets` (len a power of two `> N`): a persisted open-addressing hash
///   index, `fnv1a(name) & (M-1)` with linear probing, each slot holding a node
///   index or `EMPTY_BUCKET`. It moves `name -> idx` resolution from an O(N)
///   per-open build (which faulted every node into RAM, defeating larger-than-
///   RAM mmap) to a zero-copy O(1) probe over the mapped bytes (§4.5). The fixed
///   FNV-1a hash + name-sorted node order make the built table byte-deterministic.
#[derive(Debug, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
#[rkyv(derive(Debug))]
pub struct SerializableGraphV2 {
    pub nodes: Vec<CsrNode>,
    pub edges: Vec<CsrEdge>,
    pub node_offsets: Vec<u32>,
    pub in_offsets: Vec<u32>,
    pub in_mirror: Vec<u32>,
    pub name_buckets: Vec<u32>,
}

/// A snapshot node. Mirrors [`NodeData`] but stores `metadata` as a
/// **key-sorted `Vec`** instead of a `HashMap`. rkyv archives a `HashMap` in its
/// (per-instance, `RandomState`-seeded) iteration order, so the same node's
/// metadata would serialize to different bytes each run — breaking the byte-
/// determinism guarantee. A sorted `Vec<(String, String)>` archives to a stable,
/// order-independent layout (and rkyv archives it without the `ArchiveContext`
/// bound a `BTreeMap` field would impose). Keys are unique (from a `HashMap`), so
/// the sort is a total order.
#[derive(Debug, Clone, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
#[rkyv(derive(Debug))]
pub struct CsrNode {
    pub name: String,
    pub kind: String,
    pub metadata: Vec<(String, String)>,
    pub base_weight: f32,
    pub noise_penalty: f32,
    pub pagerank_weight: f32,
}

impl CsrNode {
    /// Convert a runtime [`NodeData`] into a deterministic snapshot node
    /// (metadata sorted by key).
    fn from_node_data(nd: &NodeData) -> Self {
        let mut metadata: Vec<(String, String)> = nd
            .metadata
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();
        metadata.sort();
        Self {
            name: nd.name.clone(),
            kind: nd.kind.clone(),
            metadata,
            base_weight: nd.base_weight,
            noise_penalty: nd.noise_penalty,
            pagerank_weight: nd.pagerank_weight,
        }
    }

    /// Convert back to a runtime [`NodeData`] (used by the `Owned`-mode rebuild).
    fn into_node_data(self) -> NodeData {
        NodeData {
            name: self.name,
            kind: self.kind,
            metadata: self.metadata.into_iter().collect(),
            base_weight: self.base_weight,
            noise_penalty: self.noise_penalty,
            pagerank_weight: self.pagerank_weight,
        }
    }
}

/// Empty-slot sentinel in `name_buckets`. Node count is capped well below this
/// (indices are `u32`), so it never collides with a real index.
pub const EMPTY_BUCKET: u32 = u32::MAX;

/// Deterministic 64-bit FNV-1a over raw bytes. Pure integer arithmetic, so it is
/// identical across platforms and endianness (unlike a randomly-seeded SipHash),
/// which is what keeps the persisted `name_buckets` layout byte-reproducible.
#[inline]
pub fn fnv1a(bytes: &[u8]) -> u64 {
    const OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const PRIME: u64 = 0x0000_0100_0000_01b3;
    let mut h = OFFSET;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(PRIME);
    }
    h
}

/// Table size for `n` entries at a ~0.7 load factor: the next power of two
/// strictly greater than `n` and `>= n / 0.7`. Strictly greater guarantees at
/// least one `EMPTY_BUCKET`, so linear probing always terminates.
fn bucket_capacity(n: usize) -> usize {
    // n * 10 / 7 ≈ n / 0.7; +1 guarantees strictly-greater-than-n even when the
    // ratio rounds down. `next_power_of_two` of that; min 2 so `M-1` masks work.
    // Fully saturating so there is no overflow discontinuity (unreachable for the
    // u32-bounded node count, but keeps the function total for any usize).
    let want = (n.saturating_mul(10) / 7)
        .saturating_add(1)
        .max(n.saturating_add(1));
    want.checked_next_power_of_two().unwrap_or(want).max(2)
}

/// Build the open-addressing `name -> idx` table for name-sorted `nodes`.
/// Deterministic: fixed hash, fixed (sorted) insertion order, linear probing.
fn build_name_buckets(nodes: &[NodeData]) -> Vec<u32> {
    let m = bucket_capacity(nodes.len());
    let mask = (m - 1) as u64;
    let mut buckets = vec![EMPTY_BUCKET; m];
    for (i, node) in nodes.iter().enumerate() {
        let mut slot = (fnv1a(node.name.as_bytes()) & mask) as usize;
        while buckets[slot] != EMPTY_BUCKET {
            slot = (slot + 1) & (mask as usize);
        }
        buckets[slot] = i as u32;
    }
    buckets
}

/// A CSR edge stored by node indices. Distinct from
/// [`crate::serialization::SerializableEdge`] so V2 stays self-contained.
#[derive(Debug, Clone, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
#[rkyv(derive(Debug))]
pub struct CsrEdge {
    pub from_idx: u32,
    pub to_idx: u32,
    pub kind: String,
    pub field_name: Option<String>,
    pub base_weight: f32,
    pub valid_from: Option<u64>,
    pub valid_to: Option<u64>,
    pub acl: Vec<String>,
}

// ---------------------------------------------------------------------------
// Serialization: OrpheusGraphInner -> V2 CSR bytes (deterministic)
// ---------------------------------------------------------------------------

/// Serialize an owned graph to V2 CSR rkyv bytes.
///
/// **Determinism** (spec: byte-identical for the same logical graph): nodes are
/// emitted in total-order by `name` (names are unique keys); edges are emitted
/// in total-order by `(from_idx, to_idx, kind, field_name, base_weight)`, which
/// also groups them by `from_idx` — exactly the outgoing-CSR order. `in_mirror`
/// is a stable-by-index sort of `0..E` by `to_idx`. Two serializations of the
/// same graph therefore produce identical bytes regardless of internal petgraph
/// index order.
pub fn to_rkyv_v2(graph: &OrpheusGraphInner) -> Vec<u8> {
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
            valid_from: ed.valid_from,
            valid_to: ed.valid_to,
            acl: ed.acl.clone(),
        });
    }

    // 3. Total-order sort. Primary key `from_idx` groups outgoing CSR runs.
    // valid_from/valid_to/acl participate in the sort key too: `sort_by` is
    // STABLE, so without them two parallel edges tying on every other key
    // (from_idx, to_idx, kind, field_name, base_weight) but differing only in
    // acl/temporal validity would keep their PRE-sort (petgraph insertion)
    // relative order — reintroducing exactly the source-insertion-order
    // determinism bug `build_graph`'s node-name sort was fixed for (see
    // `v2_bytes_identical_across_source_insertion_order`). `Option<u64>` and
    // `Vec<String>` are both `Ord` (lexicographic for the Vec), so this stays
    // a total order.
    edges.sort_by(|a, b| {
        a.from_idx
            .cmp(&b.from_idx)
            .then(a.to_idx.cmp(&b.to_idx))
            .then(a.kind.cmp(&b.kind))
            .then(a.field_name.cmp(&b.field_name))
            .then(a.base_weight.total_cmp(&b.base_weight))
            .then(a.valid_from.cmp(&b.valid_from))
            .then(a.valid_to.cmp(&b.valid_to))
            .then(a.acl.cmp(&b.acl))
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

    // Persisted name -> idx index (built from the name-sorted nodes).
    let name_buckets = build_name_buckets(&nodes);

    // Snapshot nodes: metadata sorted for byte-determinism (see `CsrNode`).
    let csr_nodes: Vec<CsrNode> = nodes.iter().map(CsrNode::from_node_data).collect();

    let sg = SerializableGraphV2 {
        nodes: csr_nodes,
        edges,
        node_offsets,
        in_offsets,
        in_mirror,
        name_buckets,
    };
    rkyv::to_bytes::<rkyv::rancor::Error>(&sg)
        .expect("rkyv V2 serialization failed")
        .to_vec()
}

/// Rebuild an owned [`OrpheusGraphInner`] from V2 CSR bytes (for `mode=Owned`).
///
/// The CSR offset arrays are ignored on rebuild — the petgraph is reconstructed
/// from `nodes` + `edges(from_idx,to_idx)` alone, mirroring
/// [`crate::serialization::from_rkyv_rebuild`].
pub fn from_rkyv_rebuild_v2(data: &[u8]) -> Result<OrpheusGraphInner, String> {
    let sg = rkyv::from_bytes::<SerializableGraphV2, rkyv::rancor::Error>(data)
        .map_err(|e| format!("rkyv V2 deserialization failed: {e}"))?;

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
                    valid_from: e.valid_from,
                    valid_to: e.valid_to,
                    acl: e.acl,
                },
            )
        })
        .collect();

    let nodes: Vec<NodeData> = sg.nodes.into_iter().map(CsrNode::into_node_data).collect();
    Ok(rebuild_from_serialized(nodes, edges))
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
    /// write semantically-valid V2.
    Crc,
    /// No crc, no structural walk, `access_unchecked` — `O(1)`. Permitted ONLY
    /// for a file the process itself wrote this run (e.g. the post-compaction
    /// re-mmap). NEVER for untrusted input.
    None,
}

/// Validate V2 bytes per `mode`. Never panics — every failure is a typed
/// [`PersistError`]. `expected_crc` is the crc recorded in the MANIFEST.
pub fn validate_v2(bytes: &[u8], mode: Validate, expected_crc: u32) -> Result<(), PersistError> {
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
            "V2 snapshot crc mismatch: computed {computed}, expected {expected_crc}"
        )));
    }
    Ok(())
}

fn access_checked(bytes: &[u8]) -> Result<&ArchivedSerializableGraphV2, PersistError> {
    rkyv::access::<ArchivedSerializableGraphV2, rkyv::rancor::Error>(bytes)
        .map_err(|e| PersistError::Corrupt(format!("V2 rkyv structural validation failed: {e}")))
}

/// The §5.1 semantic sweep. One linear pass each over edges and the two CSR
/// offset arrays; a single `seen` bitset proves `in_mirror` is a permutation.
/// Consumes ONLY `.get()`/`.to_native()` — never index-panics.
fn validate_csr(a: &ArchivedSerializableGraphV2) -> Result<(), PersistError> {
    let n = a.nodes.len();
    let e = a.edges.len();

    let corrupt = |msg: String| PersistError::Corrupt(format!("V2 CSR invalid: {msg}"));

    // --- nodes strictly name-sorted (=> names unique) ---
    // `to_rkyv_v2` emits nodes in strict ascending name order. Enforcing it here
    // does double duty: it proves node NAMES ARE UNIQUE, which is what makes the
    // `name_buckets` reachability check below (which matches by index) equivalent
    // to `idx_of` (which matches by name). Without uniqueness a hostile snapshot
    // with two equally-named nodes would pass the table checks yet make one of
    // them unreachable by name.
    for i in 1..n {
        let prev = a
            .nodes
            .get(i - 1)
            .ok_or_else(|| corrupt("nodes short (sort check)".into()))?
            .name
            .as_str();
        let cur = a
            .nodes
            .get(i)
            .ok_or_else(|| corrupt("nodes short (sort check)".into()))?
            .name
            .as_str();
        if prev >= cur {
            return Err(corrupt(format!(
                "nodes not strictly name-sorted at {i}: {prev:?} >= {cur:?}"
            )));
        }
    }

    // --- name_buckets: the persisted name -> idx hash index ---
    // Lookups (`idx_of`) hash the query and linear-probe this table, confirming
    // the real node name on each hit. A hostile/bit-rotted table can never cause
    // UB (every access is `.get()`-bounded and probing is capped at `M`), but it
    // could make lookups silently MISS an existing node. The checks close that:
    //   (a) M is EXACTLY the canonical `bucket_capacity(N)` — a power of two > N
    //       (so the mask is valid and >= 1 EMPTY_BUCKET exists => probes
    //       terminate) AND bounded to Θ(N), so an attacker cannot inflate M to
    //       make the O(M) bijection scan a super-linear validation-time DoS;
    //   (b) occupied slots form a bijection with 0..N (each node index appears
    //       exactly once, in range) — no duplicates, no phantom indices;
    //   (c) reachability — `idx_of(name[i]) == Some(i)` for every node, checked
    //       via the linear-probe invariant (see below), NOT by re-probing.
    // (a)+(b) alone do NOT imply (c): a valid bijection can still strand a node
    // behind an empty slot. A table built by `build_name_buckets` satisfies all
    // three by construction. The whole sweep is a hard O(N+M): (a) is O(1), (b)
    // is one O(M) pass, (c) is O(1) per node over a prefix sum — pathological
    // FNV clustering (self-inflicted or adversarial) cannot inflate the cost,
    // so no probe budget is needed and every self-written table validates.
    let m = a.name_buckets.len();
    let expected_m = bucket_capacity(n);
    if m != expected_m {
        return Err(corrupt(format!(
            "name_buckets len {m} != canonical bucket_capacity({n})={expected_m}"
        )));
    }
    // Independent guard of the mask precondition: `& (M-1)` is only a modulo
    // when M is a power of two, and probe termination needs >= 1 EMPTY slot
    // (M > N). `bucket_capacity` guarantees both today; assert them directly so
    // a future change to its load-factor math cannot silently break the mask.
    if !m.is_power_of_two() || m <= n {
        return Err(corrupt(format!(
            "name_buckets len {m} is not a power of two > N={n}"
        )));
    }
    let mask = (m - 1) as u64;
    let bucket_at = |i: usize| a.name_buckets.get(i).map(|x| x.to_native());
    // (b) bijection — one pass that also records, for (c), where each node
    // index sits (`slot_of`) and a prefix count of EMPTY slots (`empties[s]` =
    // EMPTY slots among `buckets[0..s)`).
    let mut seen = vec![false; n];
    let mut slot_of = vec![0usize; n];
    let mut empties = vec![0usize; m + 1];
    let mut occupied = 0usize;
    for slot in 0..m {
        let b = bucket_at(slot).ok_or_else(|| corrupt("name_buckets short".into()))?;
        empties[slot + 1] = empties[slot] + usize::from(b == EMPTY_BUCKET);
        if b == EMPTY_BUCKET {
            continue;
        }
        let idx = b as usize;
        if idx >= n {
            return Err(corrupt(format!(
                "name_buckets slot {slot} holds idx {idx} >= N={n}"
            )));
        }
        if seen[idx] {
            return Err(corrupt(format!(
                "name_buckets maps idx {idx} more than once"
            )));
        }
        seen[idx] = true;
        slot_of[idx] = slot;
        occupied += 1;
    }
    if occupied != n {
        return Err(corrupt(format!(
            "name_buckets has {occupied} entries != node count {n}"
        )));
    }
    // (c) reachability: idx_of(name[i]) must resolve to i. For an
    // insertion-only linear-probe table this is EXACTLY "no EMPTY slot on the
    // cyclic path [home(i), slot(i))": the probe walks forward from
    // `home = fnv1a(name) & mask` through occupied slots and stops at the first
    // EMPTY, and no earlier slot can answer the lookup first because names are
    // unique (strict-sort check above) and every hit is name-confirmed.
    // `build_name_buckets` gives the property by construction — an insertion
    // lands on the first EMPTY after its home, and later insertions only fill
    // slots, never empty them — so ANY self-written table passes, including
    // pathologically FNV-clustered names (an earlier cumulative probe BUDGET
    // here rejected exactly such tables that the write path had just produced:
    // write/validate asymmetry). Checked in O(1) per node against the EMPTY
    // prefix sum, so hostile clustering cannot inflate validation cost either.
    // (FNV-1a is not collision-resistant; collisions only cost probe locality,
    // never correctness — matching §5.1's "structural safety, not
    // hash-flooding".)
    for (i, &slot) in slot_of.iter().enumerate() {
        let name = a
            .nodes
            .get(i)
            .ok_or_else(|| corrupt("nodes short (reachability)".into()))?
            .name
            .as_str();
        let home = (fnv1a(name.as_bytes()) & mask) as usize;
        let empties_on_path = if home <= slot {
            empties[slot] - empties[home]
        } else {
            // Wrapped probe: home..M-1, then 0..slot.
            (empties[m] - empties[home]) + empties[slot]
        };
        if empties_on_path != 0 {
            return Err(corrupt(format!(
                "node {i} ({name:?}) is unreachable by its own hash probe: \
                 {empties_on_path} EMPTY slot(s) on the path from its hash home"
            )));
        }
    }

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

/// Zero-copy `O(degree)` accessor over a memory-mapped V2 CSR snapshot.
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
    archived: *const ArchivedSerializableGraphV2,
}

// SAFETY: read-only mapping, pointer stable across moves (see type docs).
unsafe impl Send for ArchivedCsrView {}
unsafe impl Sync for ArchivedCsrView {}

impl ArchivedCsrView {
    fn archived(&self) -> &ArchivedSerializableGraphV2 {
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

    /// Resolve `name` to its node index via the persisted open-addressing table
    /// (`name_buckets`). Zero-copy `O(1)` expected over the mapped bytes: hash the
    /// query, probe from that slot, and confirm the real node name matches (so a
    /// hash collision resolves to the correct node, never a false hit). Touches
    /// only the probed bucket page and the one matching node — unlike an eager
    /// `name -> idx` HashMap, which at open faulted EVERY node into RAM and
    /// allocated a `String` per name, defeating the mmap's larger-than-RAM
    /// purpose. Probing is bounded to `M` steps so even a malformed (fully
    /// occupied) table cannot loop; every access is `.get()`-checked, never UB.
    fn idx_of(&self, name: &str) -> Option<usize> {
        let a = self.archived();
        let m = a.name_buckets.len();
        if m == 0 {
            return None;
        }
        let mask = (m - 1) as u64; // M is a power of two (checked at open)
        let mut slot = (fnv1a(name.as_bytes()) & mask) as usize;
        for _ in 0..m {
            let bucket = a.name_buckets.get(slot)?.to_native();
            if bucket == EMPTY_BUCKET {
                return None; // empty slot ends the probe: not present
            }
            let idx = bucket as usize;
            if a.nodes.get(idx)?.name.as_str() == name {
                return Some(idx);
            }
            slot = (slot + 1) & (mask as usize);
        }
        None
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
        let idx = self.idx_of(name)?;
        let node = self.archived().nodes.get(idx)?;
        Some(NodeView {
            name: node.name.to_string(),
            kind: node.kind.to_string(),
            base_weight: node.base_weight.to_native(),
            noise_penalty: node.noise_penalty.to_native(),
            pagerank_weight: node.pagerank_weight.to_native(),
            metadata: node
                .metadata
                .iter()
                .map(|entry| (entry.0.to_string(), entry.1.to_string()))
                .collect(),
        })
    }

    fn outgoing_neighbors(&self, name: &str) -> Vec<NeighborView> {
        let Some(idx) = self.idx_of(name) else {
            return vec![];
        };
        let a = self.archived();
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
                valid_from: edge.valid_from.as_ref().map(|v| v.to_native()),
                valid_to: edge.valid_to.as_ref().map(|v| v.to_native()),
                acl: edge.acl.iter().map(|s| s.to_string()).collect(),
            });
        }
        out
    }

    fn incoming_neighbors(&self, name: &str) -> Vec<NeighborView> {
        let Some(idx) = self.idx_of(name) else {
            return vec![];
        };
        let a = self.archived();
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
                valid_from: edge.valid_from.as_ref().map(|v| v.to_native()),
                valid_to: edge.valid_to.as_ref().map(|v| v.to_native()),
                acl: edge.acl.iter().map(|s| s.to_string()).collect(),
            });
        }
        inc
    }
}

// ---------------------------------------------------------------------------
// open_snapshot — the mmap / owned open path
// ---------------------------------------------------------------------------

/// Open a V2 CSR snapshot at `path` and return a [`BaseGraph`].
///
/// * `mode = Owned`: read + validate the bytes up front and materialize an
///   owned petgraph (`from_rkyv_rebuild_v2`). No lazy faulting — safe for
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
            validate_v2(&bytes, validate, expected_crc)?;
            let inner = from_rkyv_rebuild_v2(&bytes).map_err(PersistError::Corrupt)?;
            Ok(BaseGraph::Owned(inner))
        }
        BaseMode::Mmap => {
            let file = File::open(path)?;
            // SAFETY: read-only map; the file is not mutated in place while
            // mapped (rename-only + unlink-safe discipline, §4.5).
            let mmap = unsafe { memmap2::Mmap::map(&file)? };

            validate_v2(&mmap[..], validate, expected_crc)?;

            // Obtain the archive pointer. full/crc => structural (checked);
            // none => access_unchecked (trusted self-written file only).
            let archived: *const ArchivedSerializableGraphV2 = match validate {
                Validate::None => {
                    // SAFETY: `none` is only ever passed for a file this process
                    // wrote this run (post-compaction re-mmap); its structure is
                    // known-valid, so `access_unchecked` has no UB.
                    let a =
                        unsafe { rkyv::access_unchecked::<ArchivedSerializableGraphV2>(&mmap[..]) };
                    a as *const _
                }
                _ => {
                    let a = access_checked(&mmap[..])?;
                    a as *const _
                }
            };

            // No per-open name-index BUILD: name resolution probes the persisted
            // `name_buckets` hash table on demand (`ArchivedCsrView::idx_of`, an
            // FNV-1a + open-addressing probe), so open does not fault every node
            // into RAM to build an index — preserving the mmap's larger-than-RAM
            // lazy paging. (The index-build cost is O(1); a `Validate::Crc`/`Full`
            // open still pays an O(file) crc / O(N+E) sweep separately.)

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
                valid_from: None,
                valid_to: None,
                acl: Vec::new(),
            })
            .collect();
        let (g, m) = build_graph(n, e);
        OrpheusGraphInner::new(g, m)
    }

    /// Like `inner()` but lets the caller attach `valid_from`/`valid_to`/`acl`
    /// to every edge (same values on all edges — enough for the tests that
    /// need it).
    fn inner_with_edge_fields(
        nodes: Vec<(&str, &str)>,
        edges: Vec<(&str, &str, &str)>,
        valid_from: Option<u64>,
        valid_to: Option<u64>,
        acl: Vec<&str>,
    ) -> OrpheusGraphInner {
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
                valid_from,
                valid_to,
                acl: acl.iter().map(|s| s.to_string()).collect(),
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
        let bytes = to_rkyv_v2(&owned);
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
        let bytes = to_rkyv_v2(&sample());
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
        assert_eq!(to_rkyv_v2(&g1), to_rkyv_v2(&g2));
    }

    #[test]
    fn v2_bytes_identical_across_source_insertion_order() {
        // Same logical graph as `sample()`, but nodes and edges are handed to
        // `inner()`/`build_graph` in a different order. The persisted
        // `name_buckets` table is built from the name-SORTED node list with a
        // fixed hash, so ITS contribution to the byte stream does
        // canonicalize regardless of source insertion order (verified: only
        // `pagerank_weight` bytes differ between `forward`/`backward` below,
        // never `name_buckets`).
        //
        // This once exposed a real determinism bug (now FIXED): `compute_pagerank`
        // in `src/builder.rs` accumulated `new_scores[..] += share` in petgraph
        // insertion order, and non-associative `f32` addition made a different
        // input order yield a ~ULP-different `pagerank_weight` that leaked into
        // these bytes. `build_graph` now sorts nodes by name before insertion, so
        // the summation order is canonical for the same logical graph and these
        // two serializations are byte-identical.
        let forward = sample();
        let backward = inner(
            vec![
                ("D", "model"),
                ("C", "field"),
                ("B", "model"),
                ("A", "model"),
            ],
            vec![
                ("A", "A", "self"),
                ("D", "A", "rel"),
                ("B", "C", "contains"),
                ("A", "C", "contains"),
                ("A", "B", "rel"),
            ],
        );
        assert_eq!(to_rkyv_v2(&forward), to_rkyv_v2(&backward));
    }

    #[test]
    fn v2_bytes_identical_with_metadata_across_map_order() {
        // Node metadata is a HashMap (random-seeded iteration order). The snapshot
        // stores it as a key-sorted `CsrNode.metadata` Vec, so the same logical
        // metadata handed in two different HashMap insertion orders must still
        // serialize to identical bytes (rkyv would otherwise archive the HashMap
        // in its per-instance order -> non-deterministic bytes).
        let build = |pairs: &[(&str, &str)]| {
            let mut md = HashMap::new();
            for (k, v) in pairs {
                md.insert(k.to_string(), v.to_string());
            }
            let nodes = vec![crate::types::NodeInput {
                name: "n".into(),
                kind: "model".into(),
                metadata: md,
                base_weight: 1.0,
                noise_penalty: 0.0,
            }];
            let (g, m) = crate::builder::build_graph(nodes, vec![]);
            to_rkyv_v2(&OrpheusGraphInner::new(g, m))
        };
        let a = build(&[("z", "1"), ("a", "2"), ("m", "3")]);
        let b = build(&[("a", "2"), ("m", "3"), ("z", "1")]);
        assert_eq!(a, b, "metadata map order must not affect snapshot bytes");
    }

    #[test]
    fn v2_bytes_identical_with_acl_across_input_order() {
        // Same logical edge acl, supplied to EdgeInput in a different order —
        // normalize_acl() sorts+dedups at builder ingest, so both must produce
        // byte-identical snapshots regardless of caller-supplied tag order.
        let build = |acl: &[&str]| {
            let nodes = vec![("A", "m"), ("B", "m")]
                .into_iter()
                .map(|(name, kind)| crate::types::NodeInput {
                    name: name.into(),
                    kind: kind.into(),
                    metadata: HashMap::new(),
                    base_weight: 0.5,
                    noise_penalty: 0.0,
                })
                .collect();
            let edges = vec![crate::types::EdgeInput {
                from: "A".into(),
                to: "B".into(),
                kind: "rel".into(),
                field_name: None,
                base_weight: 1.0,
                valid_from: Some(1),
                valid_to: Some(9),
                acl: acl.iter().map(|s| s.to_string()).collect(),
            }];
            let (g, m) = crate::builder::build_graph(nodes, edges);
            to_rkyv_v2(&OrpheusGraphInner::new(g, m))
        };
        let a = build(&["z", "a", "m", "a"]); // has a duplicate too
        let b = build(&["m", "z", "a"]);
        assert_eq!(
            a, b,
            "acl input order (and duplicates) must not affect snapshot bytes"
        );
    }

    #[test]
    fn v2_bytes_identical_with_two_parallel_edges_differing_only_in_acl() {
        // Regression guard for the sort-key extension: two parallel A->B edges
        // that tie on (from_idx, to_idx, kind, field_name, base_weight) but
        // differ in acl must still sort into a canonical (insertion-order-
        // independent) position — otherwise Vec::sort_by's STABILITY would let
        // pre-sort (petgraph) insertion order leak into the byte stream.
        let build = |edges: Vec<crate::types::EdgeInput>| {
            let nodes = vec![("A", "m"), ("B", "m")]
                .into_iter()
                .map(|(name, kind)| crate::types::NodeInput {
                    name: name.into(),
                    kind: kind.into(),
                    metadata: HashMap::new(),
                    base_weight: 0.5,
                    noise_penalty: 0.0,
                })
                .collect();
            let (g, m) = crate::builder::build_graph(nodes, edges);
            to_rkyv_v2(&OrpheusGraphInner::new(g, m))
        };
        let mk = |acl: &str| crate::types::EdgeInput {
            from: "A".into(),
            to: "B".into(),
            kind: "rel".into(),
            field_name: None,
            base_weight: 1.0,
            valid_from: None,
            valid_to: None,
            acl: vec![acl.to_string()],
        };
        let forward = build(vec![mk("x"), mk("y")]);
        let backward = build(vec![mk("y"), mk("x")]);
        assert_eq!(
            forward, backward,
            "parallel edges differing only in acl must sort canonically regardless of insertion order"
        );
    }

    #[test]
    fn v2_temporal_and_acl_fields_round_trip_through_mmap_open() {
        let owned = inner_with_edge_fields(
            vec![("A", "m"), ("B", "m")],
            vec![("A", "B", "rel")],
            Some(10),
            Some(20),
            vec!["team-x", "team-y"],
        );
        let bytes = to_rkyv_v2(&owned);
        let base = open_csr(&bytes, Validate::Full).unwrap();
        let view = base.as_accessor();
        let nb = &view.outgoing_neighbors("A")[0];
        assert_eq!(nb.valid_from, Some(10));
        assert_eq!(nb.valid_to, Some(20));
        assert_eq!(nb.acl, vec!["team-x".to_string(), "team-y".to_string()]);

        let inc = &view.incoming_neighbors("B")[0];
        assert_eq!(inc.valid_from, Some(10));
        assert_eq!(inc.valid_to, Some(20));
        assert_eq!(inc.acl, vec!["team-x".to_string(), "team-y".to_string()]);
    }

    #[test]
    fn v2_owned_rebuild_preserves_temporal_and_acl_fields() {
        let owned = inner_with_edge_fields(
            vec![("A", "m"), ("B", "m")],
            vec![("A", "B", "rel")],
            Some(5),
            None,
            vec!["public-ish"],
        );
        let bytes = to_rkyv_v2(&owned);
        let rebuilt = from_rkyv_rebuild_v2(&bytes).unwrap();
        let nb = &rebuilt.outgoing_neighbors("A")[0];
        assert_eq!(nb.valid_from, Some(5));
        assert_eq!(nb.valid_to, None);
        assert_eq!(nb.acl, vec!["public-ish".to_string()]);
    }

    #[test]
    fn v2_metadata_round_trips_through_mmap_open() {
        // A metadata-bearing node must read its exact keys/values back after
        // serialize + mmap open — `CsrNode` stores metadata as a sorted Vec and
        // `get_node` rebuilds the HashMap. (Prior round-trip tests all used empty
        // metadata, so this path was untested.)
        let mut md = HashMap::new();
        md.insert("owner".to_string(), "alice".to_string());
        md.insert("zone".to_string(), "eu".to_string());
        md.insert("tier".to_string(), "gold".to_string());
        let nodes = vec![crate::types::NodeInput {
            name: "acct".into(),
            kind: "model".into(),
            metadata: md.clone(),
            base_weight: 1.0,
            noise_penalty: 0.0,
        }];
        let (g, m) = build_graph(nodes, vec![]);
        let bytes = to_rkyv_v2(&OrpheusGraphInner::new(g, m));
        let base = open_csr(&bytes, Validate::Full).unwrap();
        let node = base.as_accessor().get_node("acct").expect("node present");
        assert_eq!(node.metadata, md, "metadata must round-trip exactly");
    }

    // NOTE: ephemeral traversal-result determinism is covered TRANSITIVELY by the
    // byte-identity tests above (`v2_bytes_identical_across_source_insertion_order`,
    // which is mutation-verified to fail if the `build_graph` name-sort is removed):
    // identical snapshot bytes imply an identical graph, hence identical traversal.
    // A dedicated traversal test on a symmetric fixture was removed as vacuous — a
    // star graph sums N identical PageRank shares, which is order-invariant even
    // WITHOUT the fix, so it could never catch the regression it claimed to guard.

    // ---- owned open path -----------------------------------------------

    #[test]
    fn owned_open_matches_mmap_open() {
        let owned = sample();
        let bytes = to_rkyv_v2(&owned);
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
        let bytes = to_rkyv_v2(&owned);
        let crc = crc32fast::hash(&bytes);
        let (_dir, path) = write_temp(&bytes);
        std::mem::forget(_dir);
        let base = open_snapshot(&path, BaseMode::Mmap, Validate::Full, true, crc).unwrap();
        assert_eq!(base.as_accessor().node_count(), owned.node_count());
        assert_eq!(nbrs(base.as_accessor(), "A", true), nbrs(&owned, "A", true));
    }

    // ---- validate rejects hostile bytes (no panic) ---------------------

    fn hostile(mutate: impl FnOnce(&mut SerializableGraphV2)) -> Vec<u8> {
        // Start from a valid graph, then corrupt one invariant.
        let owned = inner(vec![("A", "m"), ("B", "m")], vec![("A", "B", "r")]);
        let good = to_rkyv_v2(&owned);
        let mut sg = rkyv::from_bytes::<SerializableGraphV2, rkyv::rancor::Error>(&good).unwrap();
        mutate(&mut sg);
        rkyv::to_bytes::<rkyv::rancor::Error>(&sg).unwrap().to_vec()
    }

    #[test]
    fn full_rejects_oob_edge_index() {
        let bytes = hostile(|sg| sg.edges[0].to_idx = 99);
        let crc = crc32fast::hash(&bytes);
        let err = validate_v2(&bytes, Validate::Full, crc).unwrap_err();
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
        let err = validate_v2(&bytes, Validate::Full, crc).unwrap_err();
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
        let err = validate_v2(&bytes, Validate::Full, crc).unwrap_err();
        assert!(matches!(err, PersistError::Corrupt(_)));
    }

    #[test]
    fn full_rejects_short_node_offsets() {
        let bytes = hostile(|sg| {
            sg.node_offsets.pop(); // now len n, not n+1
        });
        let crc = crc32fast::hash(&bytes);
        let err = validate_v2(&bytes, Validate::Full, crc).unwrap_err();
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
        let err = validate_v2(&bytes, Validate::Full, crc).unwrap_err();
        assert!(matches!(err, PersistError::Corrupt(_)));
    }

    #[test]
    fn full_rejects_nonzero_offsets_base() {
        // node_offsets[0] != 0 orphans edges [0, offsets[0]) (invisible to
        // outgoing traversal) while passing monotonic/sentinel — must be rejected.
        let bytes = hostile(|sg| sg.node_offsets[0] = 1);
        let crc = crc32fast::hash(&bytes);
        assert!(matches!(
            validate_v2(&bytes, Validate::Full, crc),
            Err(PersistError::Corrupt(_))
        ));
        // Symmetric for in_offsets (incoming traversal).
        let bytes2 = hostile(|sg| sg.in_offsets[0] = 1);
        let crc2 = crc32fast::hash(&bytes2);
        assert!(matches!(
            validate_v2(&bytes2, Validate::Full, crc2),
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
        let good = to_rkyv_v2(&owned);
        let mut sg = rkyv::from_bytes::<SerializableGraphV2, rkyv::rancor::Error>(&good).unwrap();
        // Force a duplicate (0,0) — no longer a permutation.
        sg.in_mirror = vec![0, 0];
        let bytes = rkyv::to_bytes::<rkyv::rancor::Error>(&sg).unwrap().to_vec();
        let crc = crc32fast::hash(&bytes);
        let err = validate_v2(&bytes, Validate::Full, crc).unwrap_err();
        assert!(matches!(err, PersistError::Corrupt(_)));
    }

    // ---- name_buckets (persisted open-addressing name index) -----------

    /// Build `total` distinct candidate names such that at least two of them
    /// land in the SAME `name_buckets` slot for a table sized for `total`
    /// entries — a deliberately engineered FNV-1a collision rather than a hope
    /// that enough nodes produce one by chance. Exercises the name-confirm-on-
    /// probe branch of `ArchivedCsrView::idx_of` (a hash hit that is NOT the
    /// right node, so probing must continue).
    fn names_with_forced_collision(total: usize) -> Vec<String> {
        let mask = (bucket_capacity(total) - 1) as u64;
        let pool: Vec<String> = (0..total * 50).map(|i| format!("node{i:05}")).collect();
        let mut by_bucket: Map<u64, Vec<usize>> = Map::new();
        for (i, name) in pool.iter().enumerate() {
            by_bucket
                .entry(fnv1a(name.as_bytes()) & mask)
                .or_default()
                .push(i);
        }
        let collide = by_bucket
            .values()
            .find(|v| v.len() >= 2)
            .expect("pool large enough to contain a same-bucket collision");
        let mut chosen: Vec<usize> = vec![collide[0], collide[1]];
        for i in 0..pool.len() {
            if chosen.len() == total {
                break;
            }
            if !chosen.contains(&i) {
                chosen.push(i);
            }
        }
        chosen.truncate(total);
        chosen.into_iter().map(|i| pool[i].clone()).collect()
    }

    #[test]
    fn v2_lookup_resolves_all_nodes_with_forced_hash_collision() {
        const N: usize = 40;
        let names = names_with_forced_collision(N);
        assert_eq!(names.len(), N);

        // Confirm the setup really forces a collision — this is the point of
        // the test, not an incidental fact about `idx_of`'s probe.
        let mask = (bucket_capacity(N) - 1) as u64;
        let mut by_bucket: Map<u64, usize> = Map::new();
        let mut collided = false;
        for name in &names {
            let b = fnv1a(name.as_bytes()) & mask;
            let count = by_bucket.entry(b).or_insert(0);
            *count += 1;
            if *count > 1 {
                collided = true;
            }
        }
        assert!(collided, "test setup must force a name_buckets collision");

        // Ring graph: node i -> node (i+1 mod N), "next". Exercises out+in.
        let node_pairs: Vec<(&str, &str)> = names.iter().map(|n| (n.as_str(), "model")).collect();
        let edge_triples: Vec<(&str, &str, &str)> = (0..N)
            .map(|i| (names[i].as_str(), names[(i + 1) % N].as_str(), "next"))
            .collect();
        let owned = inner(node_pairs, edge_triples);

        let bytes = to_rkyv_v2(&owned);
        let base = open_csr(&bytes, Validate::Full).unwrap();
        let view = base.as_accessor();

        assert_eq!(view.node_count(), N);
        for i in 0..N {
            let name = &names[i];
            let next = &names[(i + 1) % N];
            let got = view.get_node(name).unwrap();
            assert_eq!(&got.name, name);
            assert_eq!(got.kind, "model");

            let out = nbrs(view, name, true);
            assert_eq!(
                out,
                vec![(
                    next.clone(),
                    "next".to_string(),
                    format!("{name}_{next}"),
                    0.7f32.to_bits(),
                )],
                "outgoing for {name}"
            );

            let prev = &names[(i + N - 1) % N];
            let inc = nbrs(view, name, false);
            assert_eq!(
                inc,
                vec![(
                    prev.clone(),
                    "next".to_string(),
                    format!("{prev}_{name}"),
                    0.7f32.to_bits(),
                )],
                "incoming for {name}"
            );
        }
        assert!(view.get_node("definitely-not-present").is_none());
        assert!(view.outgoing_neighbors("definitely-not-present").is_empty());
        assert!(view.incoming_neighbors("definitely-not-present").is_empty());
    }

    #[test]
    fn full_rejects_name_buckets_non_power_of_two_length() {
        let bytes = hostile(|sg| sg.name_buckets.push(EMPTY_BUCKET));
        let crc = crc32fast::hash(&bytes);
        let err = validate_v2(&bytes, Validate::Full, crc).unwrap_err();
        assert!(matches!(err, PersistError::Corrupt(_)));
    }

    #[test]
    fn full_rejects_name_buckets_too_short_for_empty_slot() {
        // Power-of-two length equal to N leaves no EMPTY_BUCKET.
        let bytes = hostile(|sg| {
            let n = sg.nodes.len();
            sg.name_buckets.truncate(n);
        });
        let crc = crc32fast::hash(&bytes);
        let err = validate_v2(&bytes, Validate::Full, crc).unwrap_err();
        assert!(matches!(err, PersistError::Corrupt(_)));
    }

    #[test]
    fn full_rejects_name_bucket_index_out_of_range() {
        let bytes = hostile(|sg| {
            let slot = sg
                .name_buckets
                .iter()
                .position(|&b| b != EMPTY_BUCKET)
                .expect("at least one occupied slot");
            sg.name_buckets[slot] = sg.nodes.len() as u32; // == N, out of range
        });
        let crc = crc32fast::hash(&bytes);
        let err = validate_v2(&bytes, Validate::Full, crc).unwrap_err();
        assert!(matches!(err, PersistError::Corrupt(_)));
    }

    #[test]
    fn full_rejects_name_bucket_duplicate_index() {
        let bytes = hostile(|sg| {
            let occupied: Vec<usize> = (0..sg.name_buckets.len())
                .filter(|&i| sg.name_buckets[i] != EMPTY_BUCKET)
                .collect();
            assert!(occupied.len() >= 2, "test needs >= 2 occupied slots");
            let dup_value = sg.name_buckets[occupied[0]];
            sg.name_buckets[occupied[1]] = dup_value;
        });
        let crc = crc32fast::hash(&bytes);
        let err = validate_v2(&bytes, Validate::Full, crc).unwrap_err();
        assert!(matches!(err, PersistError::Corrupt(_)));
    }

    #[test]
    fn full_accepts_pathologically_clustered_names() {
        // Write/validate consistency guard: names mined to share ONE hash home
        // produce the worst-case linear-probe cluster `build_name_buckets` can
        // emit. 17 same-bucket names = 153 cumulative probes, which exceeded
        // the old `8N+16 = 152` reachability probe budget — a freshly-written
        // store failed its own default `Validate::Full` reopen as Corrupt. The
        // prefix-sum reachability check must accept ANY self-written table,
        // clustering included.
        let target_n = 17usize;
        let m = bucket_capacity(target_n);
        let mask = (m - 1) as u64;
        let mut names: Vec<String> = Vec::new();
        let mut i = 0u64;
        while names.len() < target_n {
            let cand = format!("collide{i}");
            if fnv1a(cand.as_bytes()) & mask == 0 {
                names.push(cand);
            }
            i += 1;
        }
        let nodes: Vec<(&str, &str)> = names.iter().map(|s| (s.as_str(), "m")).collect();
        let owned = inner(nodes, vec![]);
        let bytes = to_rkyv_v2(&owned);
        let crc = crc32fast::hash(&bytes);
        validate_v2(&bytes, Validate::Full, crc)
            .expect("a self-written clustered table must pass Full validation");
    }

    #[test]
    fn full_rejects_unreachable_node_behind_empty_gap() {
        // The critical (c) reachability case: a table that is a perfectly
        // valid bijection (every node index appears exactly once, in range)
        // can still strand a node behind an EMPTY_BUCKET slot planted between
        // its hash home and where it actually sits — `idx_of`'s linear probe
        // stops at the first EMPTY it sees, so it would silently MISS the
        // node. Constructed deliberately (not by chance): node 0 is placed two
        // slots past its own hash home, with its home's immediate successor
        // left EMPTY, so a probe starting at home dies at the gap.
        let bytes = hostile(|sg| {
            let m = sg.name_buckets.len();
            assert!(m.is_power_of_two() && m >= 4, "test assumes room for a gap");
            let mask = (m - 1) as u64;
            let home = (fnv1a(sg.nodes[0].name.as_bytes()) & mask) as usize;
            let gap = (home + 1) % m;
            let placed = (home + 2) % m;
            assert_ne!(gap, placed);

            let mut buckets = vec![EMPTY_BUCKET; m];
            buckets[placed] = 0; // node 0, two slots past its home
                                 // `gap` stays EMPTY_BUCKET: probing from `home` sees the gap and
                                 // stops before ever reaching `placed`.
            let mut free_slots = (0..m).filter(|&s| s != gap && s != placed);
            for idx in 1..sg.nodes.len() as u32 {
                let slot = free_slots
                    .next()
                    .expect("enough free slots for the remaining nodes");
                buckets[slot] = idx;
            }
            sg.name_buckets = buckets;
        });
        let crc = crc32fast::hash(&bytes);
        let err = validate_v2(&bytes, Validate::Full, crc).unwrap_err();
        assert!(matches!(err, PersistError::Corrupt(_)));
    }

    #[test]
    fn full_rejects_all_empty_name_buckets_when_n_positive() {
        let bytes = hostile(|sg| {
            let m = sg.name_buckets.len();
            sg.name_buckets = vec![EMPTY_BUCKET; m];
        });
        let crc = crc32fast::hash(&bytes);
        let err = validate_v2(&bytes, Validate::Full, crc).unwrap_err();
        assert!(matches!(err, PersistError::Corrupt(_)));
    }

    #[test]
    fn v2_round_trip_single_node_graph() {
        let owned = inner(vec![("Solo", "model")], vec![]);
        let bytes = to_rkyv_v2(&owned);
        let base = open_csr(&bytes, Validate::Full).unwrap();
        let view = base.as_accessor();

        assert_eq!(view.node_count(), 1);
        assert_eq!(view.edge_count(), 0);
        let node = view.get_node("Solo").unwrap();
        assert_eq!(node.name, "Solo");
        assert!(view.outgoing_neighbors("Solo").is_empty());
        assert!(view.incoming_neighbors("Solo").is_empty());
        assert!(view.get_node("Other").is_none());
    }

    #[test]
    fn v2_round_trip_empty_graph() {
        let owned = inner(vec![], vec![]);
        let bytes = to_rkyv_v2(&owned);
        let base = open_csr(&bytes, Validate::Full).unwrap();
        let view = base.as_accessor();

        assert_eq!(view.node_count(), 0);
        assert_eq!(view.edge_count(), 0);
        assert!(view.get_node("anything").is_none());
        assert!(view.outgoing_neighbors("anything").is_empty());
        assert!(view.incoming_neighbors("anything").is_empty());
    }

    // ---- crc vs none ----------------------------------------------------

    #[test]
    fn crc_rejects_bitflip_none_skips() {
        let owned = sample();
        let mut bytes = to_rkyv_v2(&owned);
        let good_crc = crc32fast::hash(&bytes);
        // Flip a byte: crc mode must reject.
        bytes[16] ^= 0xFF;
        assert!(matches!(
            validate_v2(&bytes, Validate::Crc, good_crc),
            Err(PersistError::Corrupt(_))
        ));
        // none mode skips all checks (documents the trust contract).
        assert!(validate_v2(&bytes, Validate::None, good_crc).is_ok());
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

    #[test]
    fn crc_and_none_modes_still_resolve_lookups_correctly() {
        // §5.1: `Crc`/`None` skip the semantic sweep (incl. name_buckets
        // reachability) but the table is trusted for a file this process
        // wrote — lookups must still resolve correctly through it.
        let owned = sample();
        let bytes = to_rkyv_v2(&owned);
        for mode in [Validate::Crc, Validate::None] {
            let base = open_csr(&bytes, mode).unwrap();
            let view = base.as_accessor();
            for name in ["A", "B", "C", "D"] {
                assert_eq!(
                    nbrs(view, name, true),
                    nbrs(&owned, name, true),
                    "{mode:?} out {name}"
                );
                assert_eq!(
                    nbrs(view, name, false),
                    nbrs(&owned, name, false),
                    "{mode:?} in {name}"
                );
            }
            assert!(view.get_node("nonexistent").is_none(), "{mode:?}");
        }
    }
}
