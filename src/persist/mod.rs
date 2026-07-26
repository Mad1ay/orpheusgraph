//! # Persistent graph store (Phase 2a)
//!
//! Wraps an immutable base graph + a mutable [`GraphDelta`] behind a
//! crash-safe, single-writer / lock-free-multi-reader store:
//!
//! * **Readers** call [`PersistentGraph::snapshot`] — one atomic `ArcSwap`
//!   load returning a consistent, immutable `(base, delta, seq, epoch)`
//!   ([`GraphState`]). The whole traversal runs against that pair with zero
//!   locking; a concurrent `apply` publishing a new state never mutates the one
//!   a reader holds (§3.5).
//! * **Writers** serialize through a single `Mutex<Writer>`. `apply()` validates
//!   + CAS-checks BEFORE any WAL write, appends (the commit point), fsyncs per
//!   policy, then advances in-memory seq/delta and publishes — so a crash
//!   between append and publish loses nothing (§4.2).
//! * **Durability**: a write-ahead log (crc-framed, poison-guarded) plus a
//!   MANIFEST + snapshot written with tmp+fsync+rename+fsync(dir). Recovery
//!   replays the WAL, truncates a torn tail, and re-mints the epoch on any
//!   timeline fork (§4.3). No recovery path panics on disk bytes.
//!
//! ## Scope (2a)
//! Base is always [`BaseGraph::Owned`] loaded via `from_rkyv_rebuild` (flat
//! `format_version` 0). CSR V1, mmap bases, compaction, open-time GC and
//! validate-modes are Phase 2b and are deliberately absent. There is no Python
//! API here — Rust-native surface + Rust tests only.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use arc_swap::ArcSwap;

use crate::accessor::GraphAccessor;
use crate::builder::build_graph;
use crate::delta::{GraphDelta, Op};
use crate::graph::OrpheusGraphInner;
use crate::serialization::{from_rkyv_rebuild, to_rkyv};

mod error;
mod lock;
mod manifest;
mod wal;

pub use error::PersistError;

use lock::DirLock;
use manifest::{
    fsync_dir, mint_epoch, read_manifest, write_file_atomic, write_manifest_atomic, Manifest,
    CREATED_BY, FORMAT_VERSION,
};
use wal::{read_and_scan, WalRecord, WalWriter};

/// When the WAL is forced to durable storage.
#[derive(Clone, Copy, Debug)]
pub enum FsyncPolicy {
    /// fsync only on explicit `flush()`/`close()` (default). Write-through means
    /// process death still loses nothing; only power loss needs the flush.
    OnFlush,
    /// fsync after every committed batch (strongest per-batch durability).
    EveryBatch,
    /// fsync once every N committed batches.
    EveryN(u32),
}

impl Default for FsyncPolicy {
    fn default() -> Self {
        FsyncPolicy::OnFlush
    }
}

/// The base half of a [`GraphState`]. An enum for 2b headroom
/// (`Archived(Mmap)`), but only `Owned` is constructed in 2a.
pub enum BaseGraph {
    Owned(OrpheusGraphInner),
    // Archived(Mmap) — Phase 2b.
}

impl BaseGraph {
    /// Borrow as the trait object every traversal / delta op consumes.
    pub fn as_accessor(&self) -> &dyn GraphAccessor {
        match self {
            BaseGraph::Owned(g) => g,
        }
    }
}

/// The immutable snapshot readers observe. Publishing a new one is a single
/// `ArcSwap` pointer swap; a reader holding an old `Arc` is never mutated.
pub struct GraphState {
    pub base: Arc<BaseGraph>,
    pub delta: Arc<GraphDelta>,
    pub seq: u64,
    pub epoch: u128,
}

/// Everything mutated only under the writer `Mutex`.
struct Writer {
    wal: WalWriter,
    /// Durable commit counter — mirror of the last WAL frame's seq (§4.2).
    seq: u64,
    /// Current incarnation id (§4.3).
    epoch: u128,
    /// 2a: owned base, shared with the published `GraphState` via the same `Arc`.
    base: Arc<BaseGraph>,
    /// Current live delta (also in the published `GraphState`).
    delta: Arc<GraphDelta>,
    /// Seq folded into `base` on disk (0 in 2a; no compaction).
    snapshot_seq: u64,
    /// Snapshot filename/crc, retained so MANIFEST rewrites are faithful.
    snapshot_file: String,
    snapshot_crc32: u32,
}

impl Writer {
    fn build_manifest(&self, clean_shutdown: bool) -> Manifest {
        Manifest {
            format_version: FORMAT_VERSION,
            snapshot_file: self.snapshot_file.clone(),
            snapshot_seq: self.snapshot_seq,
            snapshot_crc32: self.snapshot_crc32,
            epoch: self.epoch,
            clean_shutdown,
            created_by: CREATED_BY.to_string(),
        }
    }
}

/// A crash-safe, single-writer / multi-reader persistent graph.
pub struct PersistentGraph {
    dir: PathBuf,
    /// Reader publisher (§3.5).
    state: ArcSwap<GraphState>,
    /// Single-writer serialization (§3.5). Everything durable happens here.
    writer: Mutex<Writer>,
    /// flock held for the process lifetime (§4.1). Released on drop.
    _lock: DirLock,
}

// ArcSwap/Mutex/DirLock don't derive Debug; a minimal manual impl is enough for
// `.unwrap()`/`assert!` on Result<PersistentGraph, _> in tests and callers.
impl std::fmt::Debug for PersistentGraph {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = self.state.load();
        f.debug_struct("PersistentGraph")
            .field("dir", &self.dir)
            .field("seq", &s.seq)
            .field("epoch", &s.epoch)
            .finish_non_exhaustive()
    }
}

fn empty_inner() -> OrpheusGraphInner {
    let (g, idx) = build_graph(Vec::new(), Vec::new());
    OrpheusGraphInner::new(g, idx)
}

fn snapshot_name(seq: u64) -> String {
    format!("snapshot-{seq:020}.og")
}

impl PersistentGraph {
    /// Create a fresh store at `dir` from `base_graph`. Refuses to clobber an
    /// existing MANIFEST. Writes snapshot + empty WAL + MANIFEST with the
    /// tmp+fsync+rename+fsync(dir) discipline; `clean_shutdown` starts `false`.
    pub fn create(
        dir: impl AsRef<Path>,
        base_graph: OrpheusGraphInner,
    ) -> Result<Self, PersistError> {
        let dir = dir.as_ref().to_path_buf();
        std::fs::create_dir_all(&dir)?;
        let _lock = DirLock::acquire(&dir)?;

        let manifest_path = dir.join("MANIFEST.json");
        if manifest_path.exists() {
            return Err(PersistError::Corrupt(format!(
                "refusing to clobber existing store at {}",
                dir.display()
            )));
        }

        // Snapshot (flat rkyv V0), crc-guarded, atomically written.
        let bytes = to_rkyv(&base_graph);
        let snapshot_crc32 = crc32fast::hash(&bytes);
        let snapshot_file = snapshot_name(0);
        write_file_atomic(&dir, &snapshot_file, &bytes)?;

        // Empty WAL, durably created.
        let wal_path = dir.join("wal.log");
        {
            let f = std::fs::File::create(&wal_path)?;
            f.sync_all()?;
        }
        fsync_dir(&dir)?;

        let epoch = mint_epoch()?;
        let manifest = Manifest {
            format_version: FORMAT_VERSION,
            snapshot_file: snapshot_file.clone(),
            snapshot_seq: 0,
            snapshot_crc32,
            epoch,
            clean_shutdown: false,
            created_by: CREATED_BY.to_string(),
        };
        write_manifest_atomic(&dir, &manifest)?;

        let wal_file = std::fs::OpenOptions::new()
            .read(true)
            .append(true)
            .open(&wal_path)?;
        let wal = WalWriter::new(wal_file, FsyncPolicy::default(), 0);

        let base = Arc::new(BaseGraph::Owned(base_graph));
        let delta = Arc::new(GraphDelta::new());
        let state = ArcSwap::new(Arc::new(GraphState {
            base: base.clone(),
            delta: delta.clone(),
            seq: 0,
            epoch,
        }));
        let writer = Mutex::new(Writer {
            wal,
            seq: 0,
            epoch,
            base,
            delta,
            snapshot_seq: 0,
            snapshot_file,
            snapshot_crc32,
        });

        Ok(Self {
            dir,
            state,
            writer,
            _lock,
        })
    }

    /// Open the store at `dir`, running full §4.3 recovery. If there is no
    /// MANIFEST: `create=true` initializes an empty store at seq 0 with a fresh
    /// epoch; `create=false` returns [`PersistError::NotFound`].
    pub fn open(dir: impl AsRef<Path>, create: bool) -> Result<Self, PersistError> {
        let dir = dir.as_ref().to_path_buf();
        let manifest_path = dir.join("MANIFEST.json");

        if !manifest_path.exists() {
            if create {
                return Self::create(&dir, empty_inner());
            }
            return Err(PersistError::NotFound(dir));
        }

        // 0. Single-writer: take the flock before recovery.
        let _lock = DirLock::acquire(&dir)?;

        // 1. MANIFEST + version gate.
        let manifest = read_manifest(&manifest_path)?;
        if manifest.format_version > FORMAT_VERSION {
            return Err(PersistError::UnsupportedVersion {
                found: manifest.format_version,
                max: FORMAT_VERSION,
            });
        }
        let clean_shutdown = manifest.clean_shutdown;

        // 2. Load + integrity-check the snapshot.
        let snap_path = dir.join(&manifest.snapshot_file);
        let snap_bytes = std::fs::read(&snap_path)?;
        let computed = crc32fast::hash(&snap_bytes);
        if computed != manifest.snapshot_crc32 {
            return Err(PersistError::Corrupt(format!(
                "snapshot crc mismatch: computed {computed}, manifest {}",
                manifest.snapshot_crc32
            )));
        }
        let inner = from_rkyv_rebuild(&snap_bytes).map_err(PersistError::Corrupt)?;
        let base = Arc::new(BaseGraph::Owned(inner));
        let snapshot_seq = manifest.snapshot_seq;

        // 3. Scan + fold the WAL. A MANIFEST is present, so wal.log MUST exist —
        // do NOT create(true) here (that would silently treat a missing WAL as
        // an empty one and discard every committed post-snapshot batch, spec §7
        // "corruption -> raise, never auto-heal by discarding data"). Only
        // create() mints a fresh WAL.
        let wal_path = dir.join("wal.log");
        let mut wal_file = std::fs::OpenOptions::new()
            .read(true)
            .append(true)
            .open(&wal_path)
            .map_err(|e| {
                if e.kind() == std::io::ErrorKind::NotFound {
                    PersistError::Corrupt(format!(
                        "MANIFEST present but wal.log missing at {}",
                        wal_path.display()
                    ))
                } else {
                    PersistError::Io(e)
                }
            })?;
        let scan = read_and_scan(&mut wal_file)?;

        let mut delta = GraphDelta::new();
        let mut expected_next = snapshot_seq + 1;
        let mut last_applied = snapshot_seq;
        for rec in &scan.records {
            if rec.seq <= snapshot_seq {
                // Already folded into the snapshot (compaction crash window). Skip.
                continue;
            }
            if rec.seq != expected_next {
                return Err(PersistError::Corrupt(format!(
                    "WAL seq gap: expected {expected_next}, found {}",
                    rec.seq
                )));
            }
            // Durably-committed frames replay WITHOUT re-validation (spec §3.3
            // rule 4): validity was decided once at apply; re-validating here
            // would brick a valid store after 2b compaction or a rule tightening.
            delta.replay(base.as_accessor(), rec.ops.clone());
            last_applied = rec.seq;
            expected_next += 1;
        }

        // 4. Truncate a torn tail (guaranteed the last frame by poisoning).
        let tail_truncated = scan.tail_truncated;
        if tail_truncated {
            wal_file.set_len(scan.valid_end)?;
            wal_file.sync_all()?;
            fsync_dir(&dir)?;
            eprintln!(
                "orpheusgraph: WAL torn tail at offset {}, dropped {} frame(s) during recovery",
                scan.valid_end, scan.dropped
            );
        }

        // 5. Timeline-fork detection & epoch re-mint.
        let epoch = if tail_truncated || !clean_shutdown {
            mint_epoch()?
        } else {
            manifest.epoch
        };

        // 6. Persist clean_shutdown=false for this (now open) incarnation. This
        //    also lands the possibly re-minted epoch from step 5.
        let out_manifest = Manifest {
            format_version: FORMAT_VERSION,
            snapshot_file: manifest.snapshot_file.clone(),
            snapshot_seq,
            snapshot_crc32: manifest.snapshot_crc32,
            epoch,
            clean_shutdown: false,
            created_by: manifest.created_by.clone(),
        };
        write_manifest_atomic(&dir, &out_manifest)?;

        // 7. Prime the writer at end-of-valid-WAL.
        let wal = WalWriter::new(wal_file, FsyncPolicy::default(), scan.valid_end);
        let delta = Arc::new(delta);
        let writer = Mutex::new(Writer {
            wal,
            seq: last_applied,
            epoch,
            base: base.clone(),
            delta: delta.clone(),
            snapshot_seq,
            snapshot_file: manifest.snapshot_file.clone(),
            snapshot_crc32: manifest.snapshot_crc32,
        });

        // 8. Publish.
        let state = ArcSwap::new(Arc::new(GraphState {
            base,
            delta,
            seq: last_applied,
            epoch,
        }));

        Ok(Self {
            dir,
            state,
            writer,
            _lock,
        })
    }

    /// Apply an all-or-nothing batch. Optional `expected_seq` is an in-process
    /// CAS token checked atomically against the durable counter (§3.3.6).
    /// Returns the new durable seq on success.
    pub fn apply(&self, ops: Vec<Op>, expected_seq: Option<u64>) -> Result<u64, PersistError> {
        let mut w = self.writer.lock().unwrap_or_else(|e| e.into_inner());

        // 1. Poison short-circuits before any work.
        if w.wal.poisoned {
            return Err(PersistError::Poisoned);
        }

        // 2. CAS, atomic under this same lock. No WAL touched on mismatch.
        if let Some(e) = expected_seq {
            if e != w.seq {
                return Err(PersistError::Conflict {
                    expected: e,
                    actual: w.seq,
                });
            }
        }

        // 3. Copy-on-write: validate on a clone. A DeltaError never becomes a
        //    WAL frame; the clone is dropped and published state is untouched.
        let mut d = (*w.delta).clone();
        d.apply(w.base.as_accessor(), ops.clone())
            .map_err(PersistError::Delta)?;

        // 4/5. Encode + append (the commit point). Append poisons on failure.
        let new_seq = w.seq + 1;
        let rec = WalRecord {
            seq: new_seq,
            ops,
        };
        let frame = wal::encode_frame(&rec)?;
        w.wal.append(&frame)?; // fsync per policy happens inside append (step 6)

        // 7. Advance in-memory ONLY after a durable append.
        w.seq = new_seq;
        let new_delta = Arc::new(d);
        w.delta = new_delta.clone();

        // 8. Publish the new consistent state.
        let base = w.base.clone();
        let epoch = w.epoch;
        self.state.store(Arc::new(GraphState {
            base,
            delta: new_delta,
            seq: new_seq,
            epoch,
        }));

        Ok(new_seq)
    }

    /// Lock-free reader load: a consistent, immutable `(base, delta, seq, epoch)`.
    pub fn snapshot(&self) -> Arc<GraphState> {
        self.state.load_full()
    }

    /// Like [`snapshot`](Self::snapshot) but returns the `ArcSwap` guard
    /// (cheaper when the state is used only transiently).
    pub fn state(&self) -> arc_swap::Guard<Arc<GraphState>> {
        self.state.load()
    }

    /// The current durable commit seq (a valid CAS token within this incarnation).
    pub fn seq(&self) -> u64 {
        self.state.load().seq
    }

    /// The current incarnation epoch.
    pub fn epoch(&self) -> u128 {
        self.state.load().epoch
    }

    /// Set the WAL fsync policy (runtime knob; not persisted).
    pub fn set_fsync_policy(&self, policy: FsyncPolicy) {
        let mut w = self.writer.lock().unwrap_or_else(|e| e.into_inner());
        w.wal.set_policy(policy);
    }

    /// Force the WAL to durable storage. Under `OnFlush` this is the durability
    /// point. Idempotent; poison-checked.
    pub fn flush(&self) -> Result<(), PersistError> {
        let mut w = self.writer.lock().unwrap_or_else(|e| e.into_inner());
        if w.wal.poisoned {
            return Err(PersistError::Poisoned);
        }
        w.wal.flush()
    }

    /// Cleanly close: fsync the WAL, then mark `clean_shutdown=true` in the
    /// MANIFEST (same snapshot_seq/epoch). Consumes `self`, dropping the flock.
    ///
    /// If the writer is poisoned (a prior append/fsync failed), close MUST NOT
    /// claim a clean shutdown: a durability-affecting write failed, so the next
    /// open must re-mint the epoch (timeline-fork safety, §4.3). It writes
    /// `clean_shutdown=false` and returns `Poisoned` — never `Ok` with a clean
    /// flag on a poisoned store.
    pub fn close(self) -> Result<(), PersistError> {
        let (manifest, poisoned) = {
            let mut w = self.writer.lock().unwrap_or_else(|e| e.into_inner());
            if w.wal.poisoned {
                // Do NOT flush (would re-error); record an UNCLEAN shutdown so
                // recovery re-mints the epoch.
                (w.build_manifest(false), true)
            } else {
                // Best-effort durability of everything acknowledged so far. A
                // flush failure here poisons and is surfaced below.
                match w.wal.flush() {
                    Ok(()) => (w.build_manifest(true), false),
                    Err(_) => (w.build_manifest(false), true),
                }
            }
        };
        write_manifest_atomic(&self.dir, &manifest)?;
        // self (and thus `_lock`) drops here, releasing the flock.
        if poisoned {
            Err(PersistError::Poisoned)
        } else {
            Ok(())
        }
    }

    // ---- test-only seams ------------------------------------------------

    /// Arm the WAL so the next `apply` append fails (poison-propagation tests).
    #[cfg(test)]
    fn arm_append_failure(&self) {
        let mut w = self.writer.lock().unwrap_or_else(|e| e.into_inner());
        w.wal.arm_fail();
    }
}

// ===========================================================================
// Integration tests (on-disk dirs via tempfile; Rust-native only — SCOPE 2a).
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::accessor::GraphAccessor;
    use crate::delta::{DeltaAccessor, DeltaError};
    use crate::types::{EdgeData, EdgeInput, NodeData, NodeInput};
    use std::collections::HashMap;
    use std::sync::Arc;

    // ---- fixtures -------------------------------------------------------

    fn base_inner(
        nodes: Vec<(&str, &str)>,
        edges: Vec<(&str, &str, &str)>,
    ) -> OrpheusGraphInner {
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

    fn upsert(name: &str, kind: &str) -> Op {
        Op::UpsertNode(node(name, kind))
    }
    fn addedge(from: &str, to: &str, kind: &str) -> Op {
        Op::AddEdge {
            from: from.into(),
            to: to.into(),
            edge: EdgeData {
                kind: kind.into(),
                field_name: None,
                base_weight: 1.0,
            },
        }
    }

    fn wal_len(dir: &Path) -> u64 {
        std::fs::metadata(dir.join("wal.log")).unwrap().len()
    }

    /// (target, kind) neighbor pairs, sorted for order-insensitive comparison.
    fn out_pairs(acc: &dyn GraphAccessor, name: &str) -> Vec<(String, String)> {
        let mut v: Vec<(String, String)> = acc
            .outgoing_neighbors(name)
            .into_iter()
            .map(|nb| (nb.target_name, nb.edge_kind))
            .collect();
        v.sort();
        v
    }

    // ---- round-trip -----------------------------------------------------

    #[test]
    fn round_trip_reopen_replays_all_batches() {
        let dir = tempfile::tempdir().unwrap();
        {
            let pg = PersistentGraph::create(
                dir.path(),
                base_inner(vec![("root", "m"), ("leaf", "m")], vec![("root", "leaf", "rel")]),
            )
            .unwrap();
            assert_eq!(pg.apply(vec![upsert("mid", "m"), addedge("root", "mid", "rel")], None).unwrap(), 1);
            assert_eq!(pg.apply(vec![upsert("tip", "m"), addedge("mid", "tip", "rel")], None).unwrap(), 2);
            // drop without close -> crash-like; batches were write-through.
        }

        let pg = PersistentGraph::open(dir.path(), false).unwrap();
        let s = pg.snapshot();
        assert_eq!(s.seq, 2);
        let acc = DeltaAccessor::new(s.base.as_accessor(), s.delta.as_ref());
        assert!(acc.get_node("mid").is_some());
        assert!(acc.get_node("tip").is_some());
        assert_eq!(
            out_pairs(&acc, "root"),
            vec![("leaf".into(), "rel".into()), ("mid".into(), "rel".into())]
        );
        assert_eq!(out_pairs(&acc, "mid"), vec![("tip".into(), "rel".into())]);
        pg.close().unwrap();
    }

    // ---- CAS ------------------------------------------------------------

    #[test]
    fn cas_success_commits() {
        let dir = tempfile::tempdir().unwrap();
        let pg = PersistentGraph::create(dir.path(), base_inner(vec![], vec![])).unwrap();
        assert_eq!(pg.seq(), 0);
        let n = pg.apply(vec![upsert("a", "m")], Some(0)).unwrap();
        assert_eq!(n, 1);
        assert_eq!(pg.seq(), 1);
    }

    #[test]
    fn cas_conflict_no_phantom_frame() {
        let dir = tempfile::tempdir().unwrap();
        let pg = PersistentGraph::create(dir.path(), base_inner(vec![], vec![])).unwrap();
        pg.apply(vec![upsert("a", "m")], None).unwrap(); // seq 1
        let n = pg.seq();
        assert_eq!(n, 1);
        pg.apply(vec![upsert("b", "m")], None).unwrap(); // seq 2, advanced by "another writer"
        assert_eq!(pg.seq(), 2);

        let len_before = wal_len(dir.path());
        let err = pg.apply(vec![upsert("c", "m")], Some(n)).unwrap_err();
        match err {
            PersistError::Conflict { expected, actual } => {
                assert_eq!(expected, 1);
                assert_eq!(actual, 2);
            }
            other => panic!("expected Conflict, got {other:?}"),
        }
        // No phantom frame, state untouched.
        assert_eq!(wal_len(dir.path()), len_before);
        assert_eq!(pg.seq(), 2);
    }

    // ---- DeltaError never durable --------------------------------------

    #[test]
    fn delta_error_never_reaches_wal() {
        let dir = tempfile::tempdir().unwrap();
        let pg = PersistentGraph::create(dir.path(), base_inner(vec![("a", "m")], vec![])).unwrap();
        pg.apply(vec![upsert("b", "m")], None).unwrap(); // seq 1
        let len_before = wal_len(dir.path());
        let seq_before = pg.seq();

        // AddEdge to a missing endpoint -> validation error on the clone.
        let err = pg.apply(vec![addedge("a", "ghost", "rel")], None).unwrap_err();
        assert!(matches!(err, PersistError::Delta(DeltaError::MissingEndpoint { .. })));

        assert_eq!(wal_len(dir.path()), len_before, "no frame written");
        assert_eq!(pg.seq(), seq_before, "seq unchanged");

        // Reopen replays only the one good batch.
        drop(pg);
        let pg2 = PersistentGraph::open(dir.path(), false).unwrap();
        assert_eq!(pg2.seq(), 1);
    }

    // ---- writer poisoning ----------------------------------------------

    #[test]
    fn poisoning_blocks_further_applies_but_reads_survive() {
        let dir = tempfile::tempdir().unwrap();
        let pg = PersistentGraph::create(dir.path(), base_inner(vec![("a", "m")], vec![])).unwrap();
        pg.apply(vec![upsert("b", "m")], None).unwrap(); // seq 1 (good)
        let len_after_good = wal_len(dir.path());

        // Arm an injected append failure; the next apply fails and poisons.
        pg.arm_append_failure();
        let err = pg.apply(vec![upsert("c", "m")], None).unwrap_err();
        assert!(matches!(err, PersistError::Io(_)));
        assert_eq!(wal_len(dir.path()), len_after_good, "failed append wrote nothing");

        // Subsequent apply short-circuits with Poisoned, still no write.
        let err2 = pg.apply(vec![upsert("d", "m")], None).unwrap_err();
        assert!(matches!(err2, PersistError::Poisoned));
        assert_eq!(wal_len(dir.path()), len_after_good);

        // Reads still serve the last good state.
        let s = pg.snapshot();
        assert_eq!(s.seq, 1);
        let acc = DeltaAccessor::new(s.base.as_accessor(), s.delta.as_ref());
        assert!(acc.get_node("b").is_some());
        assert!(acc.get_node("c").is_none());

        // Reopen recovers the pre-poison prefix exactly.
        drop(pg);
        let pg2 = PersistentGraph::open(dir.path(), false).unwrap();
        assert_eq!(pg2.seq(), 1);
        let s2 = pg2.snapshot();
        let acc2 = DeltaAccessor::new(s2.base.as_accessor(), s2.delta.as_ref());
        assert!(acc2.get_node("b").is_some());
    }

    // ---- recovery: skip <= snapshot_seq --------------------------------

    #[test]
    fn recovery_skips_frames_at_or_below_snapshot_seq() {
        let dir = tempfile::tempdir().unwrap();
        // Fresh empty store: snapshot-0 (empty graph), empty WAL, snapshot_seq 0.
        let pg = PersistentGraph::create(dir.path(), base_inner(vec![], vec![])).unwrap();
        drop(pg); // release lock

        // Hand-craft a WAL: seq 1 ("a"), seq 2 ("b") to be skipped, seq 3 ("z")
        // to be replayed — with the MANIFEST claiming snapshot_seq = 2.
        let mut buf = Vec::new();
        for (s, name) in [(1u64, "a"), (2, "b"), (3, "z")] {
            buf.extend_from_slice(
                &wal::encode_frame(&WalRecord { seq: s, ops: vec![upsert(name, "m")] }).unwrap(),
            );
        }
        std::fs::write(dir.path().join("wal.log"), &buf).unwrap();

        // Bump snapshot_seq to 2 in the MANIFEST (snapshot bytes/crc unchanged).
        let mut m = read_manifest(&dir.path().join("MANIFEST.json")).unwrap();
        m.snapshot_seq = 2;
        write_manifest_atomic(dir.path(), &m).unwrap();

        let pg = PersistentGraph::open(dir.path(), false).unwrap();
        assert_eq!(pg.seq(), 3, "only the > snapshot_seq frame replayed");
        let s = pg.snapshot();
        let acc = DeltaAccessor::new(s.base.as_accessor(), s.delta.as_ref());
        assert!(acc.get_node("z").is_some(), "seq 3 replayed");
        assert!(acc.get_node("a").is_none(), "seq 1 skipped");
        assert!(acc.get_node("b").is_none(), "seq 2 skipped");
    }

    // ---- recovery: torn tail drops only the last frame -----------------

    #[test]
    fn torn_tail_truncates_only_the_last_frame() {
        let dir = tempfile::tempdir().unwrap();
        let pg = PersistentGraph::create(dir.path(), base_inner(vec![], vec![])).unwrap();
        for i in 0..4 {
            pg.apply(vec![upsert(&format!("n{i}"), "m")], None).unwrap();
        }
        pg.close().unwrap();
        let good_len = wal_len(dir.path());

        // Append a truncated/garbage extra frame after the 4 intact ones.
        {
            use std::io::Write;
            let mut f = std::fs::OpenOptions::new().append(true).open(dir.path().join("wal.log")).unwrap();
            // A plausible-looking header claiming 100 payload bytes, but only 5 follow.
            f.write_all(&100u32.to_le_bytes()).unwrap();
            f.write_all(&0u32.to_le_bytes()).unwrap();
            f.write_all(&[1, 2, 3, 4, 5]).unwrap();
        }
        assert!(wal_len(dir.path()) > good_len);

        let pg = PersistentGraph::open(dir.path(), false).unwrap();
        assert_eq!(pg.seq(), 4, "4 intact frames replayed");
        assert_eq!(wal_len(dir.path()), good_len, "torn tail truncated to boundary");
    }

    // ---- recovery: seq gap is a hard error -----------------------------

    #[test]
    fn seq_gap_is_corrupt_hard_error() {
        let dir = tempfile::tempdir().unwrap();
        let pg = PersistentGraph::create(dir.path(), base_inner(vec![], vec![])).unwrap();
        drop(pg);

        // Frames seq {1, 2, 4} — 3 is missing.
        let mut buf = Vec::new();
        for s in [1u64, 2, 4] {
            buf.extend_from_slice(
                &wal::encode_frame(&WalRecord { seq: s, ops: vec![upsert(&format!("n{s}"), "m")] }).unwrap(),
            );
        }
        std::fs::write(dir.path().join("wal.log"), &buf).unwrap();

        let err = PersistentGraph::open(dir.path(), false).unwrap_err();
        match err {
            PersistError::Corrupt(msg) => assert!(msg.contains("gap"), "message should mention gap: {msg}"),
            other => panic!("expected Corrupt gap, got {other:?}"),
        }
    }

    // ---- epoch minting / re-mint ---------------------------------------

    #[test]
    fn epoch_reminted_on_unclean_reopen_kept_on_clean() {
        let dir = tempfile::tempdir().unwrap();
        let pg1 = PersistentGraph::create(dir.path(), base_inner(vec![], vec![])).unwrap();
        pg1.apply(vec![upsert("a", "m")], None).unwrap();
        let e1 = pg1.epoch();
        drop(pg1); // NO close -> clean_shutdown stays false (crash-like)

        // Unclean reopen re-mints the epoch.
        let pg2 = PersistentGraph::open(dir.path(), false).unwrap();
        let e2 = pg2.epoch();
        assert_ne!(e2, e1, "unclean reopen must re-mint epoch");
        pg2.close().unwrap(); // clean_shutdown = true

        // Clean reopen keeps the epoch.
        let pg3 = PersistentGraph::open(dir.path(), false).unwrap();
        let e3 = pg3.epoch();
        assert_eq!(e3, e2, "clean close + reopen keeps the same epoch");
    }

    // ---- flock single-writer -------------------------------------------

    #[test]
    fn second_open_while_held_is_lock_held() {
        let dir = tempfile::tempdir().unwrap();
        let pg = PersistentGraph::create(dir.path(), base_inner(vec![], vec![])).unwrap();
        let err = PersistentGraph::open(dir.path(), false).unwrap_err();
        assert!(matches!(err, PersistError::LockHeld(_)));
        drop(pg);
        // After releasing, a fresh open succeeds.
        let pg2 = PersistentGraph::open(dir.path(), false).unwrap();
        assert_eq!(pg2.seq(), 0);
    }

    // ---- open create flag ----------------------------------------------

    #[test]
    fn open_missing_dir_respects_create_flag() {
        let root = tempfile::tempdir().unwrap();
        let missing = root.path().join("does-not-exist");
        let err = PersistentGraph::open(&missing, false).unwrap_err();
        assert!(matches!(err, PersistError::NotFound(_)));

        let fresh = root.path().join("fresh");
        let pg = PersistentGraph::open(&fresh, true).unwrap();
        assert_eq!(pg.seq(), 0);
        let _fresh_epoch = pg.epoch(); // a fresh epoch is minted on create
        let s = pg.snapshot();
        let acc = DeltaAccessor::new(s.base.as_accessor(), s.delta.as_ref());
        assert_eq!(acc.node_count(), 0);
    }

    // ---- version gate --------------------------------------------------

    #[test]
    fn future_format_version_is_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let pg = PersistentGraph::create(dir.path(), base_inner(vec![], vec![])).unwrap();
        drop(pg);
        let mut m = read_manifest(&dir.path().join("MANIFEST.json")).unwrap();
        m.format_version = FORMAT_VERSION + 1;
        write_manifest_atomic(dir.path(), &m).unwrap();

        let err = PersistentGraph::open(dir.path(), false).unwrap_err();
        match err {
            PersistError::UnsupportedVersion { found, max } => {
                assert_eq!(found, FORMAT_VERSION + 1);
                assert_eq!(max, FORMAT_VERSION);
            }
            other => panic!("expected UnsupportedVersion, got {other:?}"),
        }
    }

    // ---- snapshot crc corruption ---------------------------------------

    #[test]
    fn snapshot_crc_mismatch_is_corrupt() {
        let dir = tempfile::tempdir().unwrap();
        let pg = PersistentGraph::create(dir.path(), base_inner(vec![("a", "m")], vec![])).unwrap();
        let snap_file = {
            let w = pg.writer.lock().unwrap();
            w.snapshot_file.clone()
        };
        drop(pg);
        // Corrupt one snapshot byte so the crc no longer matches the MANIFEST.
        let path = dir.path().join(&snap_file);
        let mut bytes = std::fs::read(&path).unwrap();
        bytes[0] ^= 0xFF;
        std::fs::write(&path, &bytes).unwrap();

        let err = PersistentGraph::open(dir.path(), false).unwrap_err();
        assert!(matches!(err, PersistError::Corrupt(_)));
    }

    // ---- fsync policy knob ---------------------------------------------

    #[test]
    fn every_batch_policy_commits_and_recovers() {
        let dir = tempfile::tempdir().unwrap();
        let pg = PersistentGraph::create(dir.path(), base_inner(vec![], vec![])).unwrap();
        pg.set_fsync_policy(FsyncPolicy::EveryBatch);
        for i in 0..3 {
            pg.apply(vec![upsert(&format!("n{i}"), "m")], None).unwrap();
        }
        assert_eq!(pg.seq(), 3);
        drop(pg);
        let pg2 = PersistentGraph::open(dir.path(), false).unwrap();
        assert_eq!(pg2.seq(), 3);
    }

    // ---- OnFlush process-death durability (in-process crash simulation) --

    #[test]
    fn onflush_drop_without_close_loses_nothing() {
        // Write-through (no fsync under OnFlush) still survives process death:
        // simulate by dropping the handle WITHOUT close(), then reopening.
        let dir = tempfile::tempdir().unwrap();
        {
            let pg = PersistentGraph::create(dir.path(), base_inner(vec![], vec![])).unwrap();
            // default policy is OnFlush; no flush() calls.
            for i in 0..5 {
                pg.apply(vec![upsert(&format!("n{i}"), "m")], None).unwrap();
            }
            // no close/flush
        }
        let pg2 = PersistentGraph::open(dir.path(), false).unwrap();
        assert_eq!(pg2.seq(), 5, "all acknowledged batches present after crash-like drop");
        let s = pg2.snapshot();
        let acc = DeltaAccessor::new(s.base.as_accessor(), s.delta.as_ref());
        for i in 0..5 {
            assert!(acc.get_node(&format!("n{i}")).is_some());
        }
    }

    // ---- concurrent readers during apply -------------------------------

    #[test]
    fn concurrent_readers_see_consistent_states() {
        let dir = tempfile::tempdir().unwrap();
        let pg = Arc::new(
            PersistentGraph::create(
                dir.path(),
                base_inner(vec![("root", "m")], vec![]),
            )
            .unwrap(),
        );

        const READERS: usize = 4;
        const BATCHES: u64 = 200;

        let mut handles = Vec::new();
        for _ in 0..READERS {
            let pg = Arc::clone(&pg);
            handles.push(std::thread::spawn(move || {
                for _ in 0..2000 {
                    let s = pg.snapshot();
                    let acc = DeltaAccessor::new(s.base.as_accessor(), s.delta.as_ref());
                    // A self-consistent (base, delta, seq): counts never
                    // underflow (would panic in debug), neighbors never panic,
                    // and seq is within the produced range.
                    let _ = acc.node_count();
                    let _ = acc.edge_count();
                    let _ = acc.outgoing_neighbors("root");
                    assert!(s.seq <= BATCHES);
                }
            }));
        }

        for i in 0..BATCHES {
            // root + n{i}; edge root->n{i}. Every published state is consistent.
            pg.apply(
                vec![upsert(&format!("n{i}"), "m"), addedge("root", &format!("n{i}"), "rel")],
                None,
            )
            .unwrap();
        }

        for h in handles {
            h.join().unwrap();
        }
        assert_eq!(pg.seq(), BATCHES);
    }
}
