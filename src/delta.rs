//! # Mutable delta overlay (Phase 1)
//!
//! A `GraphDelta` is an append-only, tombstone-based overlay on top of an
//! immutable base graph. It records node upserts, node/edge removals and edge
//! additions, and exposes the merged view through a `DeltaAccessor` that
//! implements the *existing* [`GraphAccessor`] trait — so every traversal
//! (`beam_traverse`, `find_path`, `contextual_subgraph`, `neighbors_with_overlay`)
//! sees the overlaid graph transparently, with zero changes to those algorithms.
//!
//! Phase 1 scope: the delta + accessor + a `materialize` primitive only. No
//! `ArcSwap`/`GraphState` wrapper, no `seq`/CAS, no WAL/MANIFEST/mmap/compaction,
//! and no Python API — those all belong to Phase 2 and are deliberately absent.
//!
//! ## Failure-mode discipline
//! * `apply` is a **two-pass, all-or-nothing batch**: pass 1 validates every op
//!   against a sequentially-simulated intra-batch state and returns
//!   [`DeltaError`] on the first problem; pass 2 is infallible. A rejected batch
//!   leaves `self` bit-identical to its pre-batch state (atomicity).
//! * No `unwrap()`/`panic!` on a caller-error path — an unknown/tombstoned
//!   `AddEdge` endpoint is a hard error, never a silent skip.
//! * Four O(1)-at-query counters are maintained on mutation and cross-checked by
//!   a full recompute under `debug_assertions` after every `apply` (so a debug
//!   test run trips on any drift). Under `--release` that recompute is compiled
//!   out, so the release suite guards counters via explicit count assertions and
//!   a topology diff against `build_graph` in the randomized/property tests.

use std::collections::{HashMap, HashSet};

use indexmap::IndexMap;

use crate::accessor::{GraphAccessor, NeighborView, NodeView};
use crate::types::{EdgeData, EdgeInput, NodeData, NodeInput};

/// A single mutation. Also the future WAL op enum (§3.3/§4.2) — the `serde`
/// derive is free because `NodeData`/`EdgeData` already derive it and is inert
/// in Phase 1 (no WAL/postcard code yet).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum Op {
    /// Add a new node OR replace an existing one (last-write-wins).
    UpsertNode(NodeData),
    /// Tombstone a node and mask all its incident edges.
    RemoveNode { name: String },
    /// Add a new (parallel) edge; both endpoints must exist at this point.
    AddEdge {
        from: String,
        to: String,
        edge: EdgeData,
    },
    /// Tombstone every matching `(from, to, kind)` edge (base + delta). Idempotent.
    RemoveEdge {
        from: String,
        to: String,
        kind: String,
    },
}

/// Error returned by [`GraphDelta::apply`]. Only `AddEdge` can produce one.
#[derive(Debug, Clone, PartialEq)]
pub enum DeltaError {
    /// An `AddEdge` whose endpoint — in the sequentially-simulated intra-batch
    /// state — does not exist or is tombstoned (§3.3 rule 4: hard error, never a
    /// silent skip). `op_index` locates the offending op in the batch;
    /// `tombstoned` distinguishes "was removed" from "never existed".
    MissingEndpoint {
        op_index: usize,
        from: String,
        to: String,
        endpoint: String,
        tombstoned: bool,
    },
}

impl std::fmt::Display for DeltaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DeltaError::MissingEndpoint {
                op_index,
                from,
                to,
                endpoint,
                tombstoned,
            } => write!(
                f,
                "AddEdge {from:?}->{to:?} at op {op_index}: endpoint {endpoint:?} {}",
                if *tombstoned {
                    "is tombstoned"
                } else {
                    "does not exist"
                }
            ),
        }
    }
}

impl std::error::Error for DeltaError {}

/// Append-only delta edge. It is **never** `Vec::remove`d — removal sets `dead`
/// in place so the `u32` positions cached in `out_index`/`in_index` stay valid
/// (§3.3 rule 2). Physical reclamation is deferred to Phase-2 compaction.
#[derive(Debug, Clone)]
struct DeltaEdge {
    from: String,
    to: String,
    edge: EdgeData,
    dead: bool,
}

/// Mutable overlay owned by the graph. Insertion-ordered for deterministic
/// materialization / WAL replay. Derives `Clone` so the Phase-2 wrapper's
/// copy-on-write `apply` (§3.5) needs no delta-side change.
#[derive(Debug, Clone, Default)]
pub struct GraphDelta {
    /// Shadowed / added nodes, in insertion order (IndexMap: O(1) lookup +
    /// order-preserving `shift_remove`).
    added_nodes: IndexMap<String, NodeData>,
    /// Append-only, dead-flagged edges.
    added_edges: Vec<DeltaEdge>,
    /// Node tombstones (may include names absent from base).
    removed_nodes: HashSet<String>,
    /// Edge tombstones keyed by `(from, to, kind)`.
    removed_edges: HashSet<(String, String, String)>,
    /// `from` -> positions in `added_edges`.
    out_index: HashMap<String, Vec<u32>>,
    /// `to`   -> positions in `added_edges`.
    in_index: HashMap<String, Vec<u32>>,

    // ---- O(1)-at-query counters (maintained on apply, cross-checked in debug) ----
    /// `|removed_nodes ∩ base|`.
    tombstone_base_hits: usize,
    /// `|added_nodes.keys() \ base|`.
    added_non_base: usize,
    /// Base edges masked by a node OR edge tombstone, each counted exactly once.
    masked_base_edge_count: usize,
    /// `added_edges` where `!dead`.
    live_delta_edge_count: usize,
}

impl GraphDelta {
    pub fn new() -> Self {
        Self::default()
    }

    /// `true` when the overlay records nothing — enables the accessor fast path.
    pub fn is_empty(&self) -> bool {
        self.added_nodes.is_empty()
            && self.added_edges.is_empty()
            && self.removed_nodes.is_empty()
            && self.removed_edges.is_empty()
    }

    /// Apply an all-or-nothing batch of ops against `base` (§3.3 rule 4).
    ///
    /// `base` is passed in (not stored) so the delta stays base-agnostic in
    /// storage yet can validate endpoints and maintain the O(1)-query counters,
    /// matching the Phase-2 `GraphState` which owns base and delta separately.
    ///
    /// On `Err`, `self` is left exactly as it was before the call.
    pub fn apply(&mut self, base: &dyn GraphAccessor, ops: Vec<Op>) -> Result<(), DeltaError> {
        // ---- PASS 1: validate against sequentially-simulated intra-batch state ----
        // `scratch` holds simulated node-presence, lazily seeded from the current
        // delta+base view on first touch of a name.
        let mut scratch: HashMap<String, bool> = HashMap::new();
        for (i, op) in ops.iter().enumerate() {
            match op {
                Op::UpsertNode(nd) => {
                    scratch.insert(nd.name.clone(), true);
                }
                Op::RemoveNode { name } => {
                    scratch.insert(name.clone(), false);
                }
                Op::AddEdge { from, to, .. } => {
                    if !self.is_present(&scratch, base, from) {
                        return Err(DeltaError::MissingEndpoint {
                            op_index: i,
                            from: from.clone(),
                            to: to.clone(),
                            endpoint: from.clone(),
                            tombstoned: self.is_tombstoned(&scratch, from),
                        });
                    }
                    if !self.is_present(&scratch, base, to) {
                        return Err(DeltaError::MissingEndpoint {
                            op_index: i,
                            from: from.clone(),
                            to: to.clone(),
                            endpoint: to.clone(),
                            tombstoned: self.is_tombstoned(&scratch, to),
                        });
                    }
                }
                Op::RemoveEdge { .. } => {
                    // Idempotent, never errors — no validation.
                }
            }
        }

        // ---- PASS 2: infallible replay via the private mutators ----
        for op in ops {
            match op {
                Op::UpsertNode(nd) => self.upsert_node(base, nd),
                Op::RemoveNode { name } => self.remove_node(base, &name),
                Op::AddEdge { from, to, edge } => self.add_edge(from, to, edge),
                Op::RemoveEdge { from, to, kind } => self.remove_edge(base, from, to, kind),
            }
        }

        #[cfg(debug_assertions)]
        self.debug_check_invariants(base);

        Ok(())
    }

    /// Replay already-committed ops WITHOUT re-validation (spec §3.3 rule 4).
    ///
    /// Recovery replays WAL frames that were validated and made durable at their
    /// original `apply()`; re-running PASS-1 validation on replay would (a) after
    /// 2b compaction validate a frame against a different accumulated state than
    /// at apply time, and (b) let a newer version's tightened rule reject a batch
    /// a prior version legally committed — bricking a valid store. Validity is
    /// decided once, at apply; durability is permanent. This runs only the
    /// infallible PASS-2 mutators.
    pub fn replay(&mut self, base: &dyn GraphAccessor, ops: Vec<Op>) {
        for op in ops {
            match op {
                Op::UpsertNode(nd) => self.upsert_node(base, nd),
                Op::RemoveNode { name } => self.remove_node(base, &name),
                Op::AddEdge { from, to, edge } => self.add_edge(from, to, edge),
                Op::RemoveEdge { from, to, kind } => self.remove_edge(base, from, to, kind),
            }
        }
        #[cfg(debug_assertions)]
        self.debug_check_invariants(base);
    }

    /// Simulated presence of `x` given `scratch` + current delta + base.
    fn is_present(
        &self,
        scratch: &HashMap<String, bool>,
        base: &dyn GraphAccessor,
        x: &str,
    ) -> bool {
        if let Some(&p) = scratch.get(x) {
            return p;
        }
        self.added_nodes.contains_key(x)
            || (!self.removed_nodes.contains(x) && base.get_node(x).is_some())
    }

    /// For a *missing* endpoint, whether it is tombstoned (explicitly removed)
    /// rather than simply never having existed.
    fn is_tombstoned(&self, scratch: &HashMap<String, bool>, x: &str) -> bool {
        match scratch.get(x) {
            Some(&present) => !present, // false in scratch == removed-in-batch
            None => self.removed_nodes.contains(x),
        }
    }

    // ---------------------------------------------------------------------
    // Private, infallible mutators (only called after pass-1 validation).
    // ---------------------------------------------------------------------

    fn upsert_node(&mut self, base: &dyn GraphAccessor, nd: NodeData) {
        let name = nd.name.clone();
        let in_base = base.get_node(&name).is_some();

        // Clearing a tombstone restores the node and un-masks its incident base
        // edges. Order: remove from `removed_nodes` FIRST, so `unmask` sees the
        // node as present (mirrors `mask` which runs before the tombstone insert).
        if self.removed_nodes.remove(&name) && in_base {
            self.tombstone_base_hits -= 1;
            self.unmask_incident_base_edges(base, &name);
        }

        // §3.3 rule 5: NO normalization/clamp of base_weight/noise_penalty here
        // (unlike build_graph) — the caller supplies values already in [0,1].
        // We deliberately do NOT panic/assert on out-of-range input: the audit-#6
        // determinism guarantee requires that whatever value enters the delta is
        // stored VERBATIM, so the live view and a post-compaction rebuild
        // (`build_graph_prenormalized`) return the identical weight at the same
        // seq. Clamping/asserting here would diverge those two paths. The [0,1]
        // contract is enforced at the ingest boundary instead (the Python FFI
        // `require_unit` check); a Rust caller that violates it gets deterministic
        // (if unnormalized) scores, never a crash on a query path.
        let newly = self.added_nodes.insert(name, nd).is_none();
        if newly && !in_base {
            self.added_non_base += 1;
        }
    }

    fn remove_node(&mut self, base: &dyn GraphAccessor, name: &str) {
        let in_base = base.get_node(name).is_some();

        // Drop any shadow first (maintains disjointness: never both sets at once).
        if self.added_nodes.shift_remove(name).is_some() && !in_base {
            self.added_non_base -= 1;
        }

        // Mask incident base edges BEFORE inserting the tombstone, so that the
        // node still reads as "present" while we detect live->masked transitions
        // (this is what makes self-loops count exactly once — see mask helper).
        let newly_tombstoned = !self.removed_nodes.contains(name);
        if newly_tombstoned && in_base {
            self.mask_incident_base_edges(base, name);
            self.tombstone_base_hits += 1;
        }
        self.removed_nodes.insert(name.to_string());

        // Mark incident DELTA edges dead in place. A self-loop appears in both
        // out_index and in_index, so the `!dead` guard ensures a single decrement.
        let mut positions: Vec<u32> = Vec::new();
        if let Some(v) = self.out_index.get(name) {
            positions.extend_from_slice(v);
        }
        if let Some(v) = self.in_index.get(name) {
            positions.extend_from_slice(v);
        }
        for p in positions {
            let e = &mut self.added_edges[p as usize];
            if !e.dead {
                e.dead = true;
                self.live_delta_edge_count -= 1;
            }
        }
    }

    fn add_edge(&mut self, from: String, to: String, edge: EdgeData) {
        // Endpoints already validated. Always a NEW (parallel) edge.
        let p = self.added_edges.len() as u32;
        self.out_index.entry(from.clone()).or_default().push(p);
        self.in_index.entry(to.clone()).or_default().push(p);
        self.added_edges.push(DeltaEdge {
            from,
            to,
            edge,
            dead: false,
        });
        self.live_delta_edge_count += 1;
    }

    fn remove_edge(&mut self, base: &dyn GraphAccessor, from: String, to: String, kind: String) {
        // Kill ALL matching live delta edges (deterministic; avoids ambiguity
        // when base+delta both carry the triple).
        let positions: Vec<u32> = self.out_index.get(&from).cloned().unwrap_or_default();
        for p in positions {
            let e = &mut self.added_edges[p as usize];
            if !e.dead && e.to == to && e.edge.kind == kind {
                e.dead = true;
                self.live_delta_edge_count -= 1;
            }
        }

        // Tombstone matching base edge(s): +masked for each currently-live match
        // (0 if already masked by a node tombstone → no double-count).
        if self
            .removed_edges
            .insert((from.clone(), to.clone(), kind.clone()))
        {
            self.masked_base_edge_count +=
                self.count_live_base_edges_matching(base, &from, &to, &kind);
        }
    }

    /// Number of base edges `from -> to` with `kind` that are currently live
    /// (i.e. not already masked by a node tombstone). Used by `remove_edge` and
    /// by the debug recompute.
    fn count_live_base_edges_matching(
        &self,
        base: &dyn GraphAccessor,
        from: &str,
        to: &str,
        kind: &str,
    ) -> usize {
        if self.removed_nodes.contains(from) || self.removed_nodes.contains(to) {
            return 0; // already masked by a node tombstone
        }
        base.outgoing_neighbors(from)
            .iter()
            .filter(|nb| nb.target_name == to && nb.edge_kind == kind)
            .count()
    }

    /// +1 for each incident base edge of `name` that transitions live -> masked.
    /// Precondition: `name` is NOT yet in `removed_nodes`.
    fn mask_incident_base_edges(&mut self, base: &dyn GraphAccessor, name: &str) {
        for nb in base.outgoing_neighbors(name) {
            // A self-loop (target == name) is counted here exactly once; the
            // incoming pass skips it via the `== name` guard below.
            if self.removed_nodes.contains(&nb.target_name) {
                continue; // other endpoint already tombstoned -> already masked
            }
            if self.removed_edges.contains(&(
                name.to_string(),
                nb.target_name.clone(),
                nb.edge_kind.clone(),
            )) {
                continue; // already masked by an edge tombstone
            }
            self.masked_base_edge_count += 1;
        }
        for nb in base.incoming_neighbors(name) {
            let source = &nb.target_name; // incoming: the "other endpoint" is the source
            if source == name {
                continue; // self-loop, already counted in the outgoing pass
            }
            if self.removed_nodes.contains(source) {
                continue;
            }
            if self.removed_edges.contains(&(
                source.clone(),
                name.to_string(),
                nb.edge_kind.clone(),
            )) {
                continue;
            }
            self.masked_base_edge_count += 1;
        }
    }

    /// -1 for each incident base edge of `name` that transitions masked -> live.
    /// Precondition: `name` has ALREADY been removed from `removed_nodes`.
    fn unmask_incident_base_edges(&mut self, base: &dyn GraphAccessor, name: &str) {
        for nb in base.outgoing_neighbors(name) {
            if self.removed_nodes.contains(&nb.target_name) {
                continue; // still masked by the other endpoint's tombstone
            }
            if self.removed_edges.contains(&(
                name.to_string(),
                nb.target_name.clone(),
                nb.edge_kind.clone(),
            )) {
                continue; // still masked by an edge tombstone
            }
            self.masked_base_edge_count -= 1;
        }
        for nb in base.incoming_neighbors(name) {
            let source = &nb.target_name;
            if source == name {
                continue; // self-loop, already handled in the outgoing pass
            }
            if self.removed_nodes.contains(source) {
                continue;
            }
            if self.removed_edges.contains(&(
                source.clone(),
                name.to_string(),
                nb.edge_kind.clone(),
            )) {
                continue;
            }
            self.masked_base_edge_count -= 1;
        }
    }

    // ---------------------------------------------------------------------
    // Debug-only belt-and-suspenders: full recompute of every counter.
    // ---------------------------------------------------------------------

    #[cfg(debug_assertions)]
    fn debug_check_invariants(&self, base: &dyn GraphAccessor) {
        // Invariant 1: added_nodes.keys() ∩ removed_nodes == ∅.
        debug_assert!(
            self.added_nodes
                .keys()
                .all(|k| !self.removed_nodes.contains(k)),
            "disjointness violated: a name is both shadowed and tombstoned"
        );
        let (tbh, anb, masked, live) = self.recompute_counts(base);
        debug_assert_eq!(self.tombstone_base_hits, tbh, "tombstone_base_hits drift");
        debug_assert_eq!(self.added_non_base, anb, "added_non_base drift");
        debug_assert_eq!(
            self.masked_base_edge_count, masked,
            "masked_base_edge_count drift"
        );
        debug_assert_eq!(
            self.live_delta_edge_count, live,
            "live_delta_edge_count drift"
        );
    }

    /// Recompute all four counters from scratch (debug cross-check). Uses only
    /// delta-sized iteration — never a full base-edge scan — via
    /// inclusion-exclusion over the tombstone sets.
    #[cfg(debug_assertions)]
    fn recompute_counts(&self, base: &dyn GraphAccessor) -> (usize, usize, usize, usize) {
        let tbh = self
            .removed_nodes
            .iter()
            .filter(|n| base.get_node(n.as_str()).is_some())
            .count();
        let anb = self
            .added_nodes
            .keys()
            .filter(|n| base.get_node(n.as_str()).is_none())
            .count();
        let live = self.added_edges.iter().filter(|e| !e.dead).count();

        // Base edges masked by a NODE tombstone, each counted once. For a
        // tombstoned base node n: all outgoing edges, plus incoming edges whose
        // source is not itself tombstoned (else already counted via that
        // source's outgoing pass). Self-loops fall out naturally (the incoming
        // self-edge has source == n, which is tombstoned -> skipped).
        let mut node_masked = 0usize;
        for n in &self.removed_nodes {
            if base.get_node(n.as_str()).is_none() {
                continue; // tombstone of a non-base node masks nothing
            }
            node_masked += base.outgoing_neighbors(n).len();
            for nb in base.incoming_neighbors(n) {
                if !self.removed_nodes.contains(&nb.target_name) {
                    node_masked += 1;
                }
            }
        }

        // Base edges masked ONLY by an edge tombstone (both endpoints alive);
        // count_live returns 0 when an endpoint is tombstoned, so these are
        // disjoint from `node_masked`.
        let mut edge_only_masked = 0usize;
        for (f, t, k) in &self.removed_edges {
            edge_only_masked += self.count_live_base_edges_matching(base, f, t, k);
        }

        (tbh, anb, node_masked + edge_only_masked, live)
    }
}

// ---------------------------------------------------------------------------
// DeltaAccessor : GraphAccessor  (§3.2 merge rules)
// ---------------------------------------------------------------------------

/// Read-only merged view of `base` overlaid with `delta`. Implements the
/// existing [`GraphAccessor`] trait, so traversal code consumes it unchanged.
pub struct DeltaAccessor<'a> {
    base: &'a dyn GraphAccessor,
    delta: &'a GraphDelta,
}

impl<'a> DeltaAccessor<'a> {
    pub fn new(base: &'a dyn GraphAccessor, delta: &'a GraphDelta) -> Self {
        Self { base, delta }
    }
}

impl GraphAccessor for DeltaAccessor<'_> {
    fn node_count(&self) -> usize {
        // Add before subtract to keep the intermediate non-negative (usize).
        (self.base.node_count() + self.delta.added_non_base) - self.delta.tombstone_base_hits
    }

    fn edge_count(&self) -> usize {
        (self.base.edge_count() + self.delta.live_delta_edge_count)
            - self.delta.masked_base_edge_count
    }

    fn get_node(&self, name: &str) -> Option<NodeView> {
        if !self.delta.added_nodes.is_empty() {
            if let Some(nd) = self.delta.added_nodes.get(name) {
                return Some(NodeView::from(nd)); // shadow wins (§3.3 rule 1)
            }
        }
        if !self.delta.removed_nodes.is_empty() && self.delta.removed_nodes.contains(name) {
            return None; // tombstoned
        }
        self.base.get_node(name)
    }

    fn outgoing_neighbors(&self, name: &str) -> Vec<NeighborView> {
        if self.delta.is_empty() {
            return self.base.outgoing_neighbors(name); // §4.5 fast path
        }
        let removed_nodes = &self.delta.removed_nodes;
        let removed_edges = &self.delta.removed_edges;
        if !removed_nodes.is_empty() && removed_nodes.contains(name) {
            return vec![]; // tombstoned node exposes nothing
        }

        // Reuse the base's Vec and mask IN PLACE. The edge-mask check builds a
        // 3-String tuple key per base edge, so both mask checks run ONLY when
        // their tombstone set is actually non-empty — an additive delta (the
        // common case) pays zero allocations here, just the base call plus the
        // out_index probe below, instead of a `filter().collect()` that cloned
        // three strings per base edge on every visit regardless of removals.
        let mut out = self.base.outgoing_neighbors(name);
        if !removed_nodes.is_empty() || !removed_edges.is_empty() {
            out.retain(|nb| {
                if !removed_nodes.is_empty() && removed_nodes.contains(&nb.target_name) {
                    return false;
                }
                if !removed_edges.is_empty()
                    && removed_edges.contains(&(
                        name.to_string(),
                        nb.target_name.clone(),
                        nb.edge_kind.clone(),
                    ))
                {
                    return false;
                }
                true
            });
        }

        // Delta edges appended AFTER base edges, in insertion order.
        if let Some(positions) = self.delta.out_index.get(name) {
            for &p in positions {
                let e = &self.delta.added_edges[p as usize];
                if e.dead || (!removed_nodes.is_empty() && removed_nodes.contains(&e.to)) {
                    continue;
                }
                out.push(NeighborView {
                    target_name: e.to.clone(),
                    edge_kind: e.edge.kind.clone(),
                    field_name: e.edge.field_name.clone(),
                    edge_weight: e.edge.base_weight,
                });
            }
        }
        out
    }

    fn incoming_neighbors(&self, name: &str) -> Vec<NeighborView> {
        if self.delta.is_empty() {
            return self.base.incoming_neighbors(name); // §4.5 fast path
        }
        let removed_nodes = &self.delta.removed_nodes;
        let removed_edges = &self.delta.removed_edges;
        if !removed_nodes.is_empty() && removed_nodes.contains(name) {
            return vec![];
        }

        // For incoming edges, NeighborView.target_name carries the SOURCE. Same
        // in-place, allocation-free-on-additive-delta masking as outgoing above.
        let mut inc = self.base.incoming_neighbors(name);
        if !removed_nodes.is_empty() || !removed_edges.is_empty() {
            inc.retain(|nb| {
                if !removed_nodes.is_empty() && removed_nodes.contains(&nb.target_name) {
                    return false;
                }
                if !removed_edges.is_empty()
                    && removed_edges.contains(&(
                        nb.target_name.clone(),
                        name.to_string(),
                        nb.edge_kind.clone(),
                    ))
                {
                    return false;
                }
                true
            });
        }

        if let Some(positions) = self.delta.in_index.get(name) {
            for &p in positions {
                let e = &self.delta.added_edges[p as usize];
                if e.dead || (!removed_nodes.is_empty() && removed_nodes.contains(&e.from)) {
                    continue;
                }
                inc.push(NeighborView {
                    target_name: e.from.clone(), // incoming: report the source
                    edge_kind: e.edge.kind.clone(),
                    field_name: e.edge.field_name.clone(),
                    edge_weight: e.edge.base_weight,
                });
            }
        }
        inc
    }
}

// ---------------------------------------------------------------------------
// materialize — flatten (base ∘ delta) into build_graph inputs.
// ---------------------------------------------------------------------------

/// Flatten the merged view into `(nodes, edges)` suitable for `build_graph`.
///
/// Test-support helper, and the §4.4-step-1 compaction primitive for Phase 2.
///
/// `base_node_names` must list every node name present in `base`. It is an
/// explicit parameter because [`GraphAccessor`] exposes no node-name iterator
/// and Phase 1 must not change that trait (the Phase-2 caller owns the concrete
/// graph and passes `index_map().keys()`).
pub fn materialize(
    base: &dyn GraphAccessor,
    base_node_names: &[String],
    delta: &GraphDelta,
) -> (Vec<NodeInput>, Vec<EdgeInput>) {
    let acc = DeltaAccessor::new(base, delta);

    // Live node names: base names (not tombstoned, not shadowed) ∪ added_nodes.
    let mut names: Vec<String> = Vec::new();
    let mut seen: HashSet<String> = HashSet::new();
    for n in base_node_names {
        if delta.added_nodes.contains_key(n) || delta.removed_nodes.contains(n) {
            continue; // shadow emitted from added set; tombstone skipped
        }
        if seen.insert(n.clone()) {
            names.push(n.clone());
        }
    }
    for n in delta.added_nodes.keys() {
        if seen.insert(n.clone()) {
            names.push(n.clone());
        }
    }

    let mut nodes_out: Vec<NodeInput> = Vec::with_capacity(names.len());
    let mut edges_out: Vec<EdgeInput> = Vec::new();
    for name in &names {
        if let Some(nv) = acc.get_node(name) {
            nodes_out.push(NodeInput {
                name: nv.name,
                kind: nv.kind,
                metadata: nv.metadata,
                base_weight: nv.base_weight,
                noise_penalty: nv.noise_penalty,
            });
        }
        // Every live edge is emitted once, from its (live) source.
        for nb in acc.outgoing_neighbors(name) {
            edges_out.push(EdgeInput {
                from: name.clone(),
                to: nb.target_name,
                kind: nb.edge_kind,
                field_name: nb.field_name,
                base_weight: nb.edge_weight,
            });
        }
    }
    (nodes_out, edges_out)
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::build_graph;
    use crate::graph::OrpheusGraphInner;

    // ---- fixtures -------------------------------------------------------

    fn base_graph(nodes: Vec<(&str, &str)>, edges: Vec<(&str, &str, &str)>) -> OrpheusGraphInner {
        let n: Vec<NodeInput> = nodes
            .iter()
            .map(|(name, kind)| NodeInput {
                name: name.to_string(),
                kind: kind.to_string(),
                metadata: HashMap::new(),
                base_weight: 0.5,
                noise_penalty: 0.0,
            })
            .collect();
        let e: Vec<EdgeInput> = edges
            .iter()
            .map(|(f, t, k)| EdgeInput {
                from: f.to_string(),
                to: t.to_string(),
                kind: k.to_string(),
                field_name: None,
                base_weight: 1.0,
            })
            .collect();
        let (g, idx) = build_graph(n, e);
        OrpheusGraphInner::new(g, idx)
    }

    fn node(name: &str, kind: &str) -> NodeData {
        NodeData {
            name: name.into(),
            kind: kind.into(),
            metadata: HashMap::new(),
            base_weight: 0.5,
            noise_penalty: 0.0,
            pagerank_weight: 0.0,
        }
    }

    fn node_meta(name: &str, kind: &str, key: &str, val: &str) -> NodeData {
        let mut nd = node(name, kind);
        nd.metadata.insert(key.into(), val.into());
        nd
    }

    fn edata(kind: &str) -> EdgeData {
        EdgeData {
            kind: kind.into(),
            field_name: None,
            base_weight: 1.0,
        }
    }

    fn upsert(nd: NodeData) -> Op {
        Op::UpsertNode(nd)
    }
    fn rmnode(name: &str) -> Op {
        Op::RemoveNode { name: name.into() }
    }
    fn addedge(from: &str, to: &str, kind: &str) -> Op {
        Op::AddEdge {
            from: from.into(),
            to: to.into(),
            edge: edata(kind),
        }
    }
    fn rmedge(from: &str, to: &str, kind: &str) -> Op {
        Op::RemoveEdge {
            from: from.into(),
            to: to.into(),
            kind: kind.into(),
        }
    }

    /// (target, kind, field-or-empty) tuples, ORDER-preserving.
    fn triples(v: Vec<NeighborView>) -> Vec<(String, String, String)> {
        v.into_iter()
            .map(|nb| {
                (
                    nb.target_name,
                    nb.edge_kind,
                    nb.field_name.unwrap_or_default(),
                )
            })
            .collect()
    }

    fn sorted_triples(v: Vec<NeighborView>) -> Vec<(String, String, String)> {
        let mut t = triples(v);
        t.sort();
        t
    }

    // ---- get_node -------------------------------------------------------

    #[test]
    fn get_node_shadow_wins_over_base() {
        let base = base_graph(vec![("a", "model")], vec![]);
        let mut d = GraphDelta::new();
        d.apply(&base, vec![upsert(node_meta("a", "table", "src", "delta"))])
            .unwrap();
        let acc = DeltaAccessor::new(&base, &d);
        let nv = acc.get_node("a").unwrap();
        assert_eq!(nv.kind, "table");
        assert_eq!(nv.metadata.get("src").map(String::as_str), Some("delta"));
    }

    #[test]
    fn get_node_tombstoned_is_none() {
        let base = base_graph(vec![("a", "model")], vec![]);
        let mut d = GraphDelta::new();
        d.apply(&base, vec![rmnode("a")]).unwrap();
        let acc = DeltaAccessor::new(&base, &d);
        assert!(acc.get_node("a").is_none());
    }

    #[test]
    fn get_node_passthrough_to_base() {
        let base = base_graph(vec![("a", "model")], vec![]);
        let d = GraphDelta::new();
        let acc = DeltaAccessor::new(&base, &d);
        assert_eq!(acc.get_node("a").unwrap().kind, "model");
        assert!(acc.get_node("missing").is_none());
    }

    #[test]
    fn get_node_pure_delta_node() {
        let base = base_graph(vec![("a", "model")], vec![]);
        let mut d = GraphDelta::new();
        d.apply(&base, vec![upsert(node("x", "table"))]).unwrap();
        let acc = DeltaAccessor::new(&base, &d);
        assert_eq!(acc.get_node("x").unwrap().kind, "table");
    }

    // ---- outgoing_neighbors --------------------------------------------

    #[test]
    fn outgoing_no_delta_is_base_verbatim() {
        let base = base_graph(vec![("a", "m"), ("b", "m")], vec![("a", "b", "rel")]);
        let d = GraphDelta::new();
        let acc = DeltaAccessor::new(&base, &d);
        assert_eq!(
            triples(acc.outgoing_neighbors("a")),
            triples(base.outgoing_neighbors("a"))
        );
    }

    #[test]
    fn outgoing_drops_edge_when_target_tombstoned() {
        let base = base_graph(vec![("a", "m"), ("b", "m")], vec![("a", "b", "rel")]);
        let mut d = GraphDelta::new();
        d.apply(&base, vec![rmnode("b")]).unwrap();
        let acc = DeltaAccessor::new(&base, &d);
        assert!(acc.outgoing_neighbors("a").is_empty());
    }

    #[test]
    fn outgoing_drops_edge_when_triple_tombstoned() {
        let base = base_graph(vec![("a", "m"), ("b", "m")], vec![("a", "b", "rel")]);
        let mut d = GraphDelta::new();
        d.apply(&base, vec![rmedge("a", "b", "rel")]).unwrap();
        let acc = DeltaAccessor::new(&base, &d);
        assert!(acc.outgoing_neighbors("a").is_empty());
    }

    #[test]
    fn outgoing_delta_edges_after_base_in_insertion_order() {
        let base = base_graph(
            vec![("a", "m"), ("b", "m"), ("c", "m"), ("d", "m")],
            vec![("a", "b", "rel")],
        );
        let mut delta = GraphDelta::new();
        delta
            .apply(&base, vec![addedge("a", "c", "x"), addedge("a", "d", "y")])
            .unwrap();
        let acc = DeltaAccessor::new(&base, &delta);
        let got = triples(acc.outgoing_neighbors("a"));
        assert_eq!(
            got,
            vec![
                ("b".into(), "rel".into(), "".into()),
                ("c".into(), "x".into(), "".into()),
                ("d".into(), "y".into(), "".into()),
            ]
        );
    }

    #[test]
    fn outgoing_of_tombstoned_node_is_empty() {
        let base = base_graph(vec![("a", "m"), ("b", "m")], vec![("a", "b", "rel")]);
        let mut d = GraphDelta::new();
        d.apply(&base, vec![rmnode("a")]).unwrap();
        let acc = DeltaAccessor::new(&base, &d);
        assert!(
            acc.outgoing_neighbors("a").is_empty(),
            "tombstoned node exposes nothing"
        );
    }

    #[test]
    fn remove_middle_parallel_delta_edge_keeps_survivors_positionally_correct() {
        // Positional-integrity guard (spec §3.3 rule 2): removing ONE of several
        // parallel delta edges must not shift the u32 positions cached in
        // out_index/in_index for the survivors. If `dead` were ever replaced by
        // Vec::remove/swap_remove, the survivors would resolve to wrong
        // targets/kinds or wrong order and this test would fail.
        let base = base_graph(vec![("a", "m"), ("b", "m"), ("c", "m"), ("d", "m")], vec![]);
        let mut delta = GraphDelta::new();
        delta
            .apply(
                &base,
                vec![
                    addedge("a", "b", "x"),
                    addedge("a", "c", "y"),
                    addedge("a", "d", "z"),
                ],
            )
            .unwrap();
        // Kill the MIDDLE edge only.
        delta.apply(&base, vec![rmedge("a", "c", "y")]).unwrap();
        let acc = DeltaAccessor::new(&base, &delta);
        assert_eq!(
            triples(acc.outgoing_neighbors("a")),
            vec![
                ("b".into(), "x".into(), "".into()),
                ("d".into(), "z".into(), "".into()),
            ],
            "survivors must keep correct target/kind and insertion order after middle removal"
        );
        // Mirror index: c no longer has an incoming edge; b and d still do.
        assert!(acc.incoming_neighbors("c").is_empty());
        assert_eq!(acc.incoming_neighbors("b").len(), 1);
        assert_eq!(acc.incoming_neighbors("d").len(), 1);
        assert_eq!(acc.edge_count(), 2);
    }

    // ---- incoming_neighbors --------------------------------------------

    #[test]
    fn incoming_drops_edge_when_source_tombstoned() {
        let base = base_graph(vec![("a", "m"), ("b", "m")], vec![("a", "b", "rel")]);
        let mut d = GraphDelta::new();
        d.apply(&base, vec![rmnode("a")]).unwrap();
        let acc = DeltaAccessor::new(&base, &d);
        assert!(acc.incoming_neighbors("b").is_empty());
    }

    #[test]
    fn incoming_drops_edge_when_triple_tombstoned() {
        let base = base_graph(vec![("a", "m"), ("b", "m")], vec![("a", "b", "rel")]);
        let mut d = GraphDelta::new();
        d.apply(&base, vec![rmedge("a", "b", "rel")]).unwrap();
        let acc = DeltaAccessor::new(&base, &d);
        assert!(acc.incoming_neighbors("b").is_empty());
    }

    #[test]
    fn incoming_of_tombstoned_node_is_empty() {
        let base = base_graph(vec![("a", "m"), ("b", "m")], vec![("a", "b", "rel")]);
        let mut d = GraphDelta::new();
        d.apply(&base, vec![rmnode("b")]).unwrap();
        let acc = DeltaAccessor::new(&base, &d);
        assert!(acc.incoming_neighbors("b").is_empty());
    }

    #[test]
    fn incoming_delta_edges_insertion_order() {
        let base = base_graph(
            vec![("a", "m"), ("b", "m"), ("c", "m")],
            vec![("a", "c", "rel")],
        );
        let mut d = GraphDelta::new();
        d.apply(&base, vec![addedge("b", "c", "x")]).unwrap();
        let acc = DeltaAccessor::new(&base, &d);
        let got = triples(acc.incoming_neighbors("c"));
        assert_eq!(
            got,
            vec![
                ("a".into(), "rel".into(), "".into()),
                ("b".into(), "x".into(), "".into()),
            ]
        );
    }

    // ---- shadow non-destructive ----------------------------------------

    #[test]
    fn shadow_is_non_destructive() {
        // Hub "h" with base out (h->a) and in (b->h). A metadata-only upsert of
        // h keeps its full base topology and does not change node_count.
        let base = base_graph(
            vec![("h", "m"), ("a", "m"), ("b", "m")],
            vec![("h", "a", "rel"), ("b", "h", "rel")],
        );
        let mut d = GraphDelta::new();
        d.apply(&base, vec![upsert(node_meta("h", "m", "note", "hi"))])
            .unwrap();
        let acc = DeltaAccessor::new(&base, &d);
        assert_eq!(
            sorted_triples(acc.outgoing_neighbors("h")),
            sorted_triples(base.outgoing_neighbors("h"))
        );
        assert_eq!(
            sorted_triples(acc.incoming_neighbors("h")),
            sorted_triples(base.incoming_neighbors("h"))
        );
        assert_eq!(acc.node_count(), base.node_count());
        assert_eq!(
            acc.get_node("h")
                .unwrap()
                .metadata
                .get("note")
                .map(String::as_str),
            Some("hi")
        );
    }

    #[test]
    fn tombstone_masks_incident_base_edges_both_directions() {
        let base = base_graph(
            vec![("h", "m"), ("a", "m"), ("b", "m")],
            vec![("h", "a", "rel"), ("b", "h", "rel")],
        );
        let mut d = GraphDelta::new();
        d.apply(&base, vec![rmnode("h")]).unwrap();
        let acc = DeltaAccessor::new(&base, &d);
        // The tombstoned node itself exposes no edges in EITHER direction and
        // resolves to None (the audit-fixed rule §3.2).
        assert!(
            acc.get_node("h").is_none(),
            "tombstoned node must resolve to None"
        );
        assert!(
            acc.outgoing_neighbors("h").is_empty() && acc.incoming_neighbors("h").is_empty(),
            "tombstoned node must expose no edges in either direction"
        );
        // a's incoming (h->a) gone; b's outgoing (b->h) gone.
        assert!(acc.incoming_neighbors("a").is_empty());
        assert!(acc.outgoing_neighbors("b").is_empty());
        // 3 base nodes - 1 tombstone, 3 base edges? base has 2 edges, both masked.
        assert_eq!(acc.edge_count(), 0);
    }

    // ---- node_count -----------------------------------------------------

    #[test]
    fn node_count_transitions() {
        let base = base_graph(vec![("a", "m"), ("b", "m")], vec![]);
        let base_n = base.node_count();

        // shadow of base node -> unchanged
        let mut d = GraphDelta::new();
        d.apply(&base, vec![upsert(node("a", "table"))]).unwrap();
        assert_eq!(DeltaAccessor::new(&base, &d).node_count(), base_n);

        // pure delta add -> +1
        d.apply(&base, vec![upsert(node("x", "m"))]).unwrap();
        assert_eq!(DeltaAccessor::new(&base, &d).node_count(), base_n + 1);

        // tombstone base node -> back to base_n (had +1 from x, -1 from tombstone)
        d.apply(&base, vec![rmnode("b")]).unwrap();
        assert_eq!(DeltaAccessor::new(&base, &d).node_count(), base_n);

        // restore base node -> +1 again
        d.apply(&base, vec![upsert(node("b", "m"))]).unwrap();
        assert_eq!(DeltaAccessor::new(&base, &d).node_count(), base_n + 1);
    }

    // ---- edge_count -----------------------------------------------------

    #[test]
    fn edge_count_transitions() {
        let base = base_graph(
            vec![("a", "m"), ("b", "m"), ("c", "m")],
            vec![("a", "b", "rel")],
        );
        let base_e = base.edge_count();

        // add delta edge -> +1
        let mut d = GraphDelta::new();
        d.apply(&base, vec![addedge("a", "c", "x")]).unwrap();
        assert_eq!(DeltaAccessor::new(&base, &d).edge_count(), base_e + 1);

        // remove base edge -> back to base_e (delta +1, base -1)
        d.apply(&base, vec![rmedge("a", "b", "rel")]).unwrap();
        assert_eq!(DeltaAccessor::new(&base, &d).edge_count(), base_e);

        // remove the delta edge -> base_e - 1
        d.apply(&base, vec![rmedge("a", "c", "x")]).unwrap();
        assert_eq!(DeltaAccessor::new(&base, &d).edge_count(), base_e - 1);
    }

    #[test]
    fn edge_count_tombstone_masks_by_degree_no_double_count() {
        // a->b and b->a (b has degree 2). Tombstone a then b: masks exactly
        // the 2 base edges, no double count for the a<->b pair.
        let base = base_graph(
            vec![("a", "m"), ("b", "m")],
            vec![("a", "b", "rel"), ("b", "a", "rel")],
        );
        assert_eq!(base.edge_count(), 2);
        let mut d = GraphDelta::new();
        d.apply(&base, vec![rmnode("a")]).unwrap();
        // both edges incident to a -> both masked
        assert_eq!(DeltaAccessor::new(&base, &d).edge_count(), 0);
        d.apply(&base, vec![rmnode("b")]).unwrap();
        assert_eq!(DeltaAccessor::new(&base, &d).edge_count(), 0);
    }

    #[test]
    fn edge_count_remove_edge_on_node_masked_edge_no_double_decrement() {
        let base = base_graph(vec![("a", "m"), ("b", "m")], vec![("a", "b", "rel")]);
        let mut d = GraphDelta::new();
        d.apply(&base, vec![rmnode("a")]).unwrap();
        assert_eq!(DeltaAccessor::new(&base, &d).edge_count(), 0);
        // The base edge is already node-masked; tombstoning the triple must not
        // decrement again.
        d.apply(&base, vec![rmedge("a", "b", "rel")]).unwrap();
        assert_eq!(DeltaAccessor::new(&base, &d).edge_count(), 0);
    }

    // ---- symmetry -------------------------------------------------------

    #[test]
    fn upsert_after_remove_restores_baseline() {
        let base = base_graph(
            vec![("h", "m"), ("a", "m"), ("b", "m")],
            vec![("h", "a", "rel"), ("b", "h", "rel")],
        );
        let (bn, be) = (base.node_count(), base.edge_count());
        let mut d = GraphDelta::new();
        d.apply(&base, vec![rmnode("h")]).unwrap();
        d.apply(&base, vec![upsert(node("h", "m"))]).unwrap();
        let acc = DeltaAccessor::new(&base, &d);
        assert_eq!(acc.node_count(), bn);
        assert_eq!(acc.edge_count(), be, "incident base edges restored");
        assert!(acc.get_node("h").is_some());
        assert_eq!(
            sorted_triples(acc.outgoing_neighbors("h")),
            sorted_triples(base.outgoing_neighbors("h"))
        );
    }

    #[test]
    fn remove_after_upsert_symmetry() {
        let base = base_graph(vec![("a", "m"), ("b", "m")], vec![("a", "b", "rel")]);
        let mut d = GraphDelta::new();
        d.apply(&base, vec![upsert(node("a", "table")), rmnode("a")])
            .unwrap();
        let acc = DeltaAccessor::new(&base, &d);
        assert!(acc.get_node("a").is_none(), "final tombstoned");
        assert!(acc.outgoing_neighbors("a").is_empty());
        assert_eq!(acc.edge_count(), 0, "incident base edge masked");
    }

    #[test]
    fn edge_remove_then_readd_reappears() {
        let base = base_graph(vec![("a", "m"), ("b", "m")], vec![("a", "b", "rel")]);
        let mut d = GraphDelta::new();
        d.apply(&base, vec![rmedge("a", "b", "rel")]).unwrap();
        assert!(DeltaAccessor::new(&base, &d)
            .outgoing_neighbors("a")
            .is_empty());
        d.apply(&base, vec![addedge("a", "b", "rel")]).unwrap();
        let acc = DeltaAccessor::new(&base, &d);
        assert_eq!(
            triples(acc.outgoing_neighbors("a")),
            vec![("b".into(), "rel".into(), "".into())],
            "re-added delta edge reappears"
        );
    }

    #[test]
    fn self_loop_delta_edge_dead_once() {
        let base = base_graph(vec![("a", "m")], vec![]);
        let mut d = GraphDelta::new();
        d.apply(&base, vec![addedge("a", "a", "self")]).unwrap();
        assert_eq!(DeltaAccessor::new(&base, &d).edge_count(), 1);
        // Removing the node marks the self-loop dead exactly once (it is in both
        // out_index and in_index) — a double decrement would trip the debug
        // recompute inside apply.
        d.apply(&base, vec![rmnode("a")]).unwrap();
        assert_eq!(DeltaAccessor::new(&base, &d).edge_count(), 0);
    }

    #[test]
    fn self_loop_base_edge_masked_once_on_node_tombstone() {
        let base = base_graph(vec![("a", "m")], vec![("a", "a", "self")]);
        assert_eq!(base.edge_count(), 1);
        let mut d = GraphDelta::new();
        // If the base self-loop were masked twice (out + in pass) the debug
        // recompute would catch it; edge_count must land at 0.
        d.apply(&base, vec![rmnode("a")]).unwrap();
        assert_eq!(DeltaAccessor::new(&base, &d).edge_count(), 0);
    }

    // ---- batch validation ----------------------------------------------

    #[test]
    fn batch_accept_upsert_then_edge_to_base() {
        let base = base_graph(vec![("base_node", "m")], vec![]);
        let mut d = GraphDelta::new();
        d.apply(
            &base,
            vec![upsert(node("x", "m")), addedge("x", "base_node", "rel")],
        )
        .unwrap();
        let acc = DeltaAccessor::new(&base, &d);
        assert!(acc.get_node("x").is_some());
        assert_eq!(
            triples(acc.outgoing_neighbors("x")),
            vec![("base_node".into(), "rel".into(), "".into())]
        );
    }

    #[test]
    fn batch_accept_intra_batch_endpoints() {
        let base = base_graph(vec![], vec![]);
        let mut d = GraphDelta::new();
        d.apply(
            &base,
            vec![
                upsert(node("a", "m")),
                upsert(node("b", "m")),
                addedge("a", "b", "rel"),
            ],
        )
        .unwrap();
        let acc = DeltaAccessor::new(&base, &d);
        assert_eq!(acc.node_count(), 2);
        assert_eq!(acc.edge_count(), 1);
    }

    #[test]
    fn batch_reject_edge_to_tombstoned() {
        let base = base_graph(vec![("a", "m"), ("b", "m")], vec![]);
        let mut d = GraphDelta::new();
        let err = d
            .apply(&base, vec![rmnode("a"), addedge("b", "a", "rel")])
            .unwrap_err();
        match err {
            DeltaError::MissingEndpoint {
                endpoint,
                tombstoned,
                ..
            } => {
                assert_eq!(endpoint, "a");
                assert!(tombstoned);
            }
        }
        assert!(d.is_empty(), "rejected batch leaves delta untouched");
    }

    #[test]
    fn batch_reject_edge_to_unknown() {
        let base = base_graph(vec![("a", "m")], vec![]);
        let mut d = GraphDelta::new();
        let err = d
            .apply(&base, vec![addedge("a", "ghost", "rel")])
            .unwrap_err();
        match err {
            DeltaError::MissingEndpoint {
                endpoint,
                tombstoned,
                ..
            } => {
                assert_eq!(endpoint, "ghost");
                assert!(!tombstoned, "never existed, not tombstoned");
            }
        }
        assert!(d.is_empty());
    }

    #[test]
    fn batch_all_or_nothing_no_partial_mutation() {
        let base = base_graph(vec![("a", "m"), ("b", "m")], vec![("a", "b", "rel")]);
        // Seed a non-empty delta first.
        let mut d = GraphDelta::new();
        d.apply(
            &base,
            vec![upsert(node("x", "m")), addedge("x", "a", "rel")],
        )
        .unwrap();
        let (n0, e0) = {
            let acc = DeltaAccessor::new(&base, &d);
            (acc.node_count(), acc.edge_count())
        };
        let x_out = triples(DeltaAccessor::new(&base, &d).outgoing_neighbors("x"));

        // A batch of valid ops followed by one bad AddEdge must mutate nothing.
        let err = d
            .apply(
                &base,
                vec![
                    upsert(node("y", "m")),
                    addedge("y", "a", "rel"),
                    addedge("y", "ghost", "rel"), // bad
                ],
            )
            .unwrap_err();
        assert!(matches!(err, DeltaError::MissingEndpoint { .. }));

        let acc = DeltaAccessor::new(&base, &d);
        assert_eq!(acc.node_count(), n0);
        assert_eq!(acc.edge_count(), e0);
        assert!(acc.get_node("y").is_none());
        assert_eq!(triples(acc.outgoing_neighbors("x")), x_out);
    }

    #[test]
    fn batch_conflicting_upsert_then_remove_and_reverse() {
        let base = base_graph(vec![("a", "m")], vec![]);

        let mut d1 = GraphDelta::new();
        d1.apply(&base, vec![upsert(node("a", "table")), rmnode("a")])
            .unwrap();
        assert!(
            DeltaAccessor::new(&base, &d1).get_node("a").is_none(),
            "final tombstoned"
        );

        let mut d2 = GraphDelta::new();
        d2.apply(&base, vec![rmnode("a"), upsert(node("a", "table"))])
            .unwrap();
        let acc = DeltaAccessor::new(&base, &d2);
        assert_eq!(
            acc.get_node("a").unwrap().kind,
            "table",
            "final shadowed present"
        );

        // disjointness holds for both
        for d in [&d1, &d2] {
            assert!(d.added_nodes.keys().all(|k| !d.removed_nodes.contains(k)));
        }
    }

    // ---- empty-delta fast path -----------------------------------------

    #[test]
    fn empty_delta_matches_base_exactly() {
        let base = base_graph(
            vec![("a", "m"), ("b", "m")],
            vec![("a", "b", "rel"), ("b", "a", "back")],
        );
        let d = GraphDelta::new();
        let acc = DeltaAccessor::new(&base, &d);
        assert_eq!(acc.node_count(), base.node_count());
        assert_eq!(acc.edge_count(), base.edge_count());
        for n in ["a", "b"] {
            assert_eq!(
                acc.get_node(n).map(|v| v.kind.clone()),
                base.get_node(n).map(|v| v.kind.clone())
            );
            assert_eq!(
                triples(acc.outgoing_neighbors(n)),
                triples(base.outgoing_neighbors(n))
            );
            assert_eq!(
                triples(acc.incoming_neighbors(n)),
                triples(base.incoming_neighbors(n))
            );
        }
    }

    // ---- randomized counter / disjointness cross-check -----------------

    #[test]
    fn randomized_op_mix_keeps_counters_and_disjointness() {
        // Deterministic pseudo-random mix (no proptest dep needed here).
        let base = base_graph(
            vec![("n0", "m"), ("n1", "m"), ("n2", "f"), ("n3", "f")],
            vec![
                ("n0", "n1", "rel"),
                ("n1", "n2", "rel"),
                ("n0", "n3", "rel"),
                ("n2", "n0", "back"),
            ],
        );
        let base_names: Vec<String> = vec!["n0", "n1", "n2", "n3"]
            .into_iter()
            .map(String::from)
            .collect();
        let names = ["n0", "n1", "n2", "n3", "n4", "n5"];
        let kinds = ["rel", "back", "contains"];

        let mut d = GraphDelta::new();
        let mut seed: u64 = 0x1234_5678;
        let next = |seed: &mut u64, m: u64| {
            *seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (*seed >> 33) % m
        };
        for _ in 0..400 {
            let op = match next(&mut seed, 4) {
                0 => {
                    let n = names[next(&mut seed, 6) as usize];
                    let k = kinds[next(&mut seed, 3) as usize];
                    upsert(node(n, k))
                }
                1 => rmnode(names[next(&mut seed, 6) as usize]),
                2 => addedge(
                    names[next(&mut seed, 6) as usize],
                    names[next(&mut seed, 6) as usize],
                    kinds[next(&mut seed, 3) as usize],
                ),
                _ => rmedge(
                    names[next(&mut seed, 6) as usize],
                    names[next(&mut seed, 6) as usize],
                    kinds[next(&mut seed, 3) as usize],
                ),
            };
            // apply single-op batches; ignore validation errors. The debug
            // recompute inside apply asserts every counter on each accepted op.
            let _ = d.apply(&base, vec![op]);
            assert!(d.added_nodes.keys().all(|k| !d.removed_nodes.contains(k)));
        }

        // Independent from-scratch recompute via materialize + build_graph.
        let (mn, me) = materialize(&base, &base_names, &d);
        let (g, idx) = build_graph(mn, me);
        let rebuilt = OrpheusGraphInner::new(g, idx);
        let acc = DeltaAccessor::new(&base, &d);
        assert_eq!(acc.node_count(), rebuilt.node_count());
        assert_eq!(acc.edge_count(), rebuilt.edge_count());
        // Compare full per-node TOPOLOGY, not just counts — a bug that keeps
        // counts right but corrupts adjacency (wrong target/order) must fail
        // here even under --release (where the debug recompute is compiled out).
        for n in &names {
            let mut a_out = triples(acc.outgoing_neighbors(n));
            let mut r_out = triples(rebuilt.outgoing_neighbors(n));
            a_out.sort();
            r_out.sort();
            assert_eq!(a_out, r_out, "outgoing topology mismatch at {n}");
            let mut a_in = triples(acc.incoming_neighbors(n));
            let mut r_in = triples(rebuilt.incoming_neighbors(n));
            a_in.sort();
            r_in.sort();
            assert_eq!(a_in, r_in, "incoming topology mismatch at {n}");
        }
    }

    // ---- PROPTEST: semantic invisibility -------------------------------

    use proptest::prelude::*;

    fn name_strat() -> impl Strategy<Value = String> {
        prop::sample::select(vec!["n0", "n1", "n2", "n3", "n4", "n5"]).prop_map(String::from)
    }
    fn kind_strat() -> impl Strategy<Value = String> {
        prop::sample::select(vec!["rel", "back", "contains"]).prop_map(String::from)
    }
    fn op_strat() -> impl Strategy<Value = Op> {
        prop_oneof![
            (name_strat(), kind_strat()).prop_map(|(n, k)| Op::UpsertNode(node(&n, &k))),
            name_strat().prop_map(|n| Op::RemoveNode { name: n }),
            (name_strat(), name_strat(), kind_strat()).prop_map(|(f, t, k)| Op::AddEdge {
                from: f,
                to: t,
                edge: edata(&k),
            }),
            (name_strat(), name_strat(), kind_strat()).prop_map(|(f, t, k)| Op::RemoveEdge {
                from: f,
                to: t,
                kind: k
            }),
        ]
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(256))]
        #[test]
        fn prop_delta_topology_equals_rebuild(ops in prop::collection::vec(op_strat(), 0..40)) {
            let base = base_graph(
                vec![("n0", "m"), ("n1", "m"), ("n2", "f"), ("n3", "f")],
                vec![("n0", "n1", "rel"), ("n1", "n2", "rel"), ("n0", "n3", "rel"), ("n2", "n0", "back")],
            );
            let base_names: Vec<String> =
                vec!["n0", "n1", "n2", "n3"].into_iter().map(String::from).collect();

            let mut delta = GraphDelta::new();
            for op in ops {
                // Single-op batches; validation errors leave the delta valid.
                let _ = delta.apply(&base, vec![op]);
            }

            let (mn, me) = materialize(&base, &base_names, &delta);
            let (g, idx) = build_graph(mn, me);
            let rebuilt = OrpheusGraphInner::new(g, idx);
            let acc = DeltaAccessor::new(&base, &delta);

            // Counts (independently computed: counters vs petgraph).
            prop_assert_eq!(acc.node_count(), rebuilt.node_count());
            prop_assert_eq!(acc.edge_count(), rebuilt.edge_count());

            // Per-node topology: node presence + outgoing/incoming multisets
            // (target+kind+field). Weights/pagerank excluded by design (§3.4).
            for name in ["n0", "n1", "n2", "n3", "n4", "n5"] {
                prop_assert_eq!(acc.get_node(name).is_some(), rebuilt.get_node(name).is_some());
                prop_assert_eq!(
                    sorted_triples(acc.outgoing_neighbors(name)),
                    sorted_triples(rebuilt.outgoing_neighbors(name))
                );
                prop_assert_eq!(
                    sorted_triples(acc.incoming_neighbors(name)),
                    sorted_triples(rebuilt.incoming_neighbors(name))
                );
            }
        }
    }
}
