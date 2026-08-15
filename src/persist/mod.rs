//! # Persistent graph store (Phase 2a + 2b)
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
//!   and CAS-checks BEFORE any WAL write, appends (the commit point), fsyncs per
//!   policy, then advances in-memory seq/delta and publishes — so a crash
//!   between append and publish loses nothing (§4.2).
//! * **Durability**: a write-ahead log (crc-framed, poison-guarded) plus a
//!   MANIFEST + snapshot written with tmp+fsync+rename+fsync(dir). Recovery
//!   replays the WAL, truncates a torn tail, and re-mints the epoch on any
//!   timeline fork (§4.3). No recovery path panics on disk bytes.
//!
//! ## Phase 2b (this module + [`snapshot`])
//! `create()` writes the **V2 CSR** snapshot (`format_version` 2); `open()`
//! reads ONLY V2 (mmap-traversed [`BaseGraph::Archived`], honoring
//! [`BaseMode`]/[`Validate`]/`prefault`) and REJECTS legacy formats 0 and 1 as
//! [`PersistError::Corrupt`] — pre-release, no in-place migration; recreate
//! the store instead. Added: crash-safe [`compact`] +
//! auto-compaction (fold the delta into a fresh base), a two-tier snapshot GC
//! (inline post-compaction delete + open-time orphan sweep), and the §5.1
//! trust-boundary validation of untrusted snapshots. The CSR format, the
//! mmap-backed [`ArchivedCsrView`] and the validate modes live in [`snapshot`].
//! There is no Python API here — Rust-native surface + Rust tests only.
//!
//! [`compact`]: PersistentGraph::compact

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use arc_swap::ArcSwap;

use crate::accessor::GraphAccessor;
use crate::builder::{build_graph, build_graph_prenormalized};
use crate::delta::{materialize, GraphDelta, Op};
use crate::graph::OrpheusGraphInner;

mod error;
mod lock;
mod manifest;
mod snapshot;
mod wal;

pub use error::PersistError;
pub use snapshot::{ArchivedCsrView, Validate};

use lock::DirLock;
use manifest::{
    fsync_dir, mint_epoch, read_manifest, write_file_atomic, write_manifest_atomic, Manifest,
    CREATED_BY, FORMAT_VERSION,
};
use snapshot::{open_snapshot, to_rkyv_v2};
use wal::{read_and_scan, WalRecord, WalWriter};

/// When the WAL is forced to durable storage.
#[derive(Clone, Copy, Debug, Default)]
pub enum FsyncPolicy {
    /// fsync only on explicit `flush()`/`close()` (default). Write-through means
    /// process death still loses nothing; only power loss needs the flush.
    #[default]
    OnFlush,
    /// fsync after every committed batch (strongest per-batch durability).
    EveryBatch,
    /// fsync once every N committed batches.
    EveryN(u32),
}

/// Which representation the snapshot base is loaded as.
///
/// `Mmap` is the default and the only option for larger-than-RAM bases (the OS
/// pages the CSR file lazily). `Owned` materializes a petgraph at open —
/// exactly the 2a hot path, and the mandatory choice for network /
/// larger-than-RAM-under-pressure callers who must read + validate up front
/// rather than risk a lazy-fault SIGBUS (§4.5).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum BaseMode {
    Owned,
    #[default]
    Mmap,
}

/// The base half of a [`GraphState`]. `Owned` is a materialized petgraph;
/// `Archived` is a zero-copy CSR view over a memory-mapped V2 snapshot (§4.4b).
pub enum BaseGraph {
    Owned(OrpheusGraphInner),
    Archived(ArchivedCsrView),
}

impl BaseGraph {
    /// Borrow as the trait object every traversal / delta op consumes.
    pub fn as_accessor(&self) -> &dyn GraphAccessor {
        match self {
            BaseGraph::Owned(g) => g,
            BaseGraph::Archived(v) => v,
        }
    }

    /// Enumerate every node name in the base — the input `materialize` (and thus
    /// compaction) needs since [`GraphAccessor`] exposes no name iterator.
    fn node_names(&self) -> Vec<String> {
        match self {
            BaseGraph::Owned(g) => g.index_map().keys().cloned().collect(),
            BaseGraph::Archived(v) => v.node_names(),
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
    /// Commit counter — mirror of the last appended WAL frame's seq. Advanced
    /// after every accepted append, even under `OnFlush` where the frame is only
    /// write-through (not yet fsync'd) (§4.2).
    seq: u64,
    /// Highest seq that is actually on STABLE STORAGE — advanced only by a
    /// successful fsync (EveryBatch/EveryN append, flush, or a compaction that
    /// folds it into the durable snapshot). `high_seq` in the MANIFEST is set
    /// from THIS, never from `seq`, so it never over-reports durability and a
    /// legitimate loss of the un-fsync'd tail cannot false-brick the store.
    durable_seq: u64,
    /// Current incarnation id (§4.3).
    epoch: u128,
    /// Base graph, shared with the published `GraphState` via the same `Arc`.
    base: Arc<BaseGraph>,
    /// Current live delta (also in the published `GraphState`).
    delta: Arc<GraphDelta>,
    /// Seq folded into `base` on disk. Advanced by compaction (§4.4).
    snapshot_seq: u64,
    /// Snapshot filename/crc, retained so MANIFEST rewrites are faithful.
    snapshot_file: String,
    snapshot_crc32: u32,
    /// On-disk snapshot encoding version of the CURRENT `snapshot_file`. Tracked
    /// per-writer rather than just reading the FORMAT_VERSION constant so this
    /// field stays faithful to what's actually on disk (relevant if a future
    /// format bump adds a live in-place upgrade path). Today `open()` rejects
    /// any format below FORMAT_VERSION outright (legacy 0/1 -> `Corrupt`), so
    /// this is always FORMAT_VERSION for any store that opened successfully.
    format_version: u32,
    /// Monotonic compaction id; the current snapshot's cid (§4.4b).
    compaction_id: u64,
    /// Ops applied since the last compaction — the auto-compaction trigger.
    delta_ops: usize,
    /// Overrides the computed auto-compaction threshold when set (runtime knob,
    /// not persisted). Lets a test force frequent compaction so a kill -9 harness
    /// can land inside `compact_locked`'s non-atomic on-disk sequence.
    auto_compact_override: Option<usize>,
    /// How the base is (re)loaded — honored by post-compaction republish (§4.4).
    mode: BaseMode,
    prefault: bool,
}

impl Writer {
    fn build_manifest(&self, clean_shutdown: bool) -> Manifest {
        Manifest {
            format_version: self.format_version,
            snapshot_file: self.snapshot_file.clone(),
            snapshot_seq: self.snapshot_seq,
            snapshot_crc32: self.snapshot_crc32,
            compaction_id: self.compaction_id,
            epoch: self.epoch,
            // Record the DURABLE high-water (fsync'd), never the in-memory seq:
            // under OnFlush a poisoned/flush-failed close has un-fsync'd frames
            // whose loss on power-off is contractual, so high_seq must not cover
            // them (else reopen would false-brick). The clean close path flushes
            // first (durable_seq == seq), so it still records the full seq.
            high_seq: self.durable_seq,
            clean_shutdown,
            created_by: CREATED_BY.to_string(),
            checksum: 0,
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

/// Auto-compaction trigger: fold once the delta exceeds `max(1000, N/10)` ops,
/// where `N` is the base node count. The floor keeps small graphs from
/// compacting on every handful of ops; the proportional term keeps a large
/// graph's delta from growing unbounded relative to its base (§4.4b).
fn auto_compact_threshold(base_node_count: usize) -> usize {
    1000.max(base_node_count / 10)
}

/// V0 flat-snapshot filename (legacy; still matched by the GC prefix/suffix
/// rule). Only the V0 back-compat test writes one now; V2 uses
/// [`snapshot_v2_name`].
#[cfg(test)]
fn snapshot_name(seq: u64) -> String {
    format!("snapshot-{seq:020}.og")
}

/// V2 CSR snapshot filename. The monotonic `cid` makes successive filenames
/// distinct even when `seq` is unchanged (compaction folds AT the current seq),
/// so the post-compaction GC can always name a file distinct from the live one.
fn snapshot_v2_name(seq: u64, cid: u64) -> String {
    format!("snapshot-{seq:020}-{cid:010}.og")
}

/// Delete a superseded snapshot file, tolerating the Windows "mapped file is
/// busy" sharing violation. On Linux, `unlink` of a still-mmap'd file is safe —
/// the inode/pages stay alive for existing readers until they drop the mapping
/// (same argument as the rename discipline). On Windows a delete of a mapped
/// file fails; we swallow that error and DEFER the reclaim to the next
/// open()-time GC sweep (which runs before the file is re-mapped). Every removal
/// (and every deferral) is logged.
fn remove_snapshot_file(path: &Path) {
    match std::fs::remove_file(path) {
        Ok(()) => eprintln!("orpheusgraph: GC removed snapshot {}", path.display()),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
        Err(e) => {
            // On Windows a mapped-file delete is a sharing violation; defer.
            eprintln!(
                "orpheusgraph: GC could not remove {} ({e}); deferring to next open sweep",
                path.display()
            );
        }
    }
}

/// Open-time GC backstop (§4.3 step 6 / §4.4b): remove every `snapshot-*.og`
/// whose basename is not the live `keep` file. Catches crash orphans (e.g. a
/// snapshot written but never referenced because the process died before the
/// MANIFEST rename) and any Windows-deferred deletes. NEVER removes `keep`, and
/// never touches `.tmp` files (there is no concurrent writer — this runs under
/// the writer path, single-writer). Best-effort: a `read_dir` error is logged,
/// not fatal.
fn gc_orphan_snapshots(dir: &Path, keep: &str) {
    let rd = match std::fs::read_dir(dir) {
        Ok(rd) => rd,
        Err(e) => {
            eprintln!("orpheusgraph: GC read_dir failed on {}: {e}", dir.display());
            return;
        }
    };
    for entry in rd.flatten() {
        let name = entry.file_name();
        let Some(name) = name.to_str() else { continue };
        if name == keep {
            continue;
        }
        // Orphan snapshots from an interrupted compaction, AND stray
        // tmp files a real kill -9 can leave mid tmp+fsync+rename (a
        // `.og.tmp` that never got renamed, or a `MANIFEST.json.tmp`).
        // None are ever the live file (that's `keep` / MANIFEST.json).
        let is_orphan_snapshot =
            name.starts_with("snapshot-") && (name.ends_with(".og") || name.ends_with(".og.tmp"));
        let is_stray_tmp = name == "MANIFEST.json.tmp";
        if is_orphan_snapshot || is_stray_tmp {
            remove_snapshot_file(&entry.path());
        }
    }
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

        // Snapshot (V2 CSR going forward), crc-guarded, atomically written.
        let bytes = to_rkyv_v2(&base_graph);
        let snapshot_crc32 = crc32fast::hash(&bytes);
        let compaction_id = 0u64;
        let snapshot_file = snapshot_v2_name(0, compaction_id);
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
            compaction_id,
            epoch,
            high_seq: 0, // fresh store: nothing acked yet
            clean_shutdown: false,
            created_by: CREATED_BY.to_string(),
            checksum: 0,
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
            durable_seq: 0, // fresh store: empty WAL fsync'd, nothing acked yet
            epoch,
            base,
            delta,
            snapshot_seq: 0,
            snapshot_file,
            snapshot_crc32,
            format_version: FORMAT_VERSION,
            compaction_id,
            delta_ops: 0,
            auto_compact_override: None,
            mode: BaseMode::default(),
            prefault: false,
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
    ///
    /// Defaults to `mode=Mmap`, `validate=Full`, `prefault=false`. Use
    /// [`open_with`](Self::open_with) to override.
    pub fn open(dir: impl AsRef<Path>, create: bool) -> Result<Self, PersistError> {
        Self::open_with(dir, create, BaseMode::default(), Validate::default(), false)
    }

    /// [`open`](Self::open) with explicit base [`BaseMode`], [`Validate`] mode
    /// and `prefault` (madvise WILLNEED), applied to the V2 CSR path — the only
    /// format `open()` accepts. Legacy formats 0 and 1 are rejected before any
    /// of these options are consulted (see the `format_version` match below).
    pub fn open_with(
        dir: impl AsRef<Path>,
        create: bool,
        mode: BaseMode,
        validate: Validate,
        prefault: bool,
    ) -> Result<Self, PersistError> {
        // SOUNDNESS: `Validate::None` reaches `rkyv::access_unchecked`, which is
        // UB on structurally-invalid bytes. A caller-provided path may hold
        // untrusted/bit-rotted bytes, so a SAFE public open must never use it —
        // upgrade None to Crc (still O(1)-ish: crc + a *checked* structural
        // access, no UB). `None` remains valid ONLY on the internal
        // post-compaction re-mmap of a file this process just wrote, which calls
        // `open_snapshot` directly, not through this public entry.
        let validate = match validate {
            Validate::None => Validate::Crc,
            v => v,
        };
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
        let snapshot_seq = manifest.snapshot_seq;

        // 2. Load + integrity-check the snapshot. Only format_version 2 (V2 CSR)
        //    is accepted below (legacy 0/1 are rejected, see the match arm's own
        //    comment); the accepted path honors mode/validate/prefault via
        //    open_snapshot, whose full/crc modes do the crc check internally.
        let snap_path = dir.join(&manifest.snapshot_file);
        let base = match manifest.format_version {
            // V0 (flat) and V1 (CSR without the persisted name index) are legacy
            // on-disk formats this build no longer reads. They are REJECTED, not
            // silently upgraded, so a stale store fails loudly instead of feeding
            // a mislabelled snapshot to the V2 reader. Recreate the store to
            // migrate (pre-release: no in-place migration path).
            0 | 1 => {
                return Err(PersistError::Corrupt(format!(
                    "legacy snapshot format_version {} is no longer supported \
                     (recreate the store; current format is {FORMAT_VERSION})",
                    manifest.format_version
                )))
            }
            2 => Arc::new(open_snapshot(
                &snap_path,
                mode,
                validate,
                prefault,
                manifest.snapshot_crc32,
            )?),
            v => {
                return Err(PersistError::UnsupportedVersion {
                    found: v,
                    max: FORMAT_VERSION,
                })
            }
        };

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

        // 3b. Durable high-water check (§7). `high_seq` is the seq durably
        //     recorded at the last close/compaction (MANIFEST rewrite points).
        //     If replay yielded LESS, the WAL lost acked data (external truncation
        //     / whole-frame suffix loss that lands on a boundary, so it isn't seen
        //     as a torn tail) — raise rather than silently open a regressed store.
        //     Closes the asymmetry where `rm wal.log` was Corrupt but
        //     `truncate -s0 wal.log` was silently accepted.
        //
        //     GRANULARITY (documented limitation, audit #6): this is
        //     defense-in-depth against EXTERNAL FS faults, and its floor is the
        //     last MANIFEST high-water, NOT the last fsync. Frames fsync'd under
        //     EveryBatch AFTER the last close/compaction are durable but not yet
        //     reflected in `high_seq`, so an external truncation of ONLY those
        //     frames would not be detected here (it opens at the earlier seq).
        //     The core durability guarantees are unaffected — process death and
        //     power-loss of the un-fsync'd tail are handled exactly; this check
        //     only widens detection of a rarer external-corruption class.
        //     Persisting high_seq per-fsync would need a MANIFEST rewrite per
        //     batch (rejected: it burdens the durability hot path for a
        //     defense-in-depth check).
        if last_applied < manifest.high_seq {
            return Err(PersistError::Corrupt(format!(
                "WAL lost acked data: recovered seq {last_applied} < durable high-water {} \
                 (external WAL truncation/suffix loss)",
                manifest.high_seq
            )));
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
        //    also lands the possibly re-minted epoch from step 5. format_version
        //    and compaction_id are PRESERVED from the on-disk manifest rather
        //    than overwritten with the FORMAT_VERSION constant, so this stays
        //    correct if a future format bump adds an in-place upgrade path
        //    (today it is always FORMAT_VERSION, since anything older is
        //    rejected in step 2).
        let out_manifest = Manifest {
            format_version: manifest.format_version,
            snapshot_file: manifest.snapshot_file.clone(),
            snapshot_seq,
            snapshot_crc32: manifest.snapshot_crc32,
            compaction_id: manifest.compaction_id,
            epoch,
            // Do NOT promote high_seq to last_applied: under OnFlush the replayed
            // frames survived the crash in the page cache but were never fsync'd,
            // so they are NOT durable — a later power-loss legitimately drops
            // them. Promoting here would then false-brick the store on reopen
            // (audit P1). Keep the last genuinely-durable high-water; new durable
            // data re-advances it via the next flush/close/compaction.
            high_seq: manifest.high_seq,
            clean_shutdown: false,
            created_by: manifest.created_by.clone(),
            checksum: 0,
        };
        write_manifest_atomic(&dir, &out_manifest)?;

        // 6b. Open-time GC backstop (§4.3 step 6 / §4.4b): reclaim crash-orphan
        //     snapshots (and any Windows-deferred deletes) left by a compaction
        //     that died between writing a new snapshot and renaming the MANIFEST.
        //     Runs under the writer lock, before priming; never touches the live
        //     file. For V2 mmap the live file is already mapped in `base` — on
        //     Linux unlinking OTHER snapshots is safe.
        gc_orphan_snapshots(&dir, &manifest.snapshot_file);

        // 7. Prime the writer at end-of-valid-WAL.
        let wal = WalWriter::new(wal_file, FsyncPolicy::default(), scan.valid_end);
        let delta = Arc::new(delta);
        let writer = Mutex::new(Writer {
            wal,
            seq: last_applied,
            // The genuinely-durable high-water is the on-disk MANIFEST's high_seq
            // (recovery under OnFlush does not newly fsync the replayed frames).
            // Never set this to last_applied — that would over-report durability.
            durable_seq: manifest.high_seq,
            epoch,
            base: base.clone(),
            delta: delta.clone(),
            snapshot_seq,
            snapshot_file: manifest.snapshot_file.clone(),
            snapshot_crc32: manifest.snapshot_crc32,
            format_version: manifest.format_version,
            compaction_id: manifest.compaction_id,
            delta_ops: 0,
            auto_compact_override: None,
            mode,
            prefault,
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
        let rec = WalRecord { seq: new_seq, ops };
        let frame = wal::encode_frame(&rec)?;
        let fsynced = w.wal.append(&frame)?; // fsync per policy happens inside append

        // 7. Advance in-memory ONLY after a durable append.
        w.seq = new_seq;
        // Advance the durable high-water ONLY when the append actually fsync'd
        // (EveryBatch always; EveryN on a boundary). Under OnFlush the frame is
        // only write-through — durable_seq stays until flush()/close/compaction.
        // high_seq is derived from durable_seq, so it never over-reports what is
        // on stable storage (else a legitimate power-loss of the un-fsync'd tail
        // would falsely brick the store).
        if fsynced {
            w.durable_seq = new_seq;
        }
        let op_count = rec.ops.len();
        w.delta_ops += op_count;
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

        // 9. Auto-compaction (§4.4b): once the delta has accumulated more than
        //    max(1000, base_node_count/10) ops, fold it into a fresh base under
        //    the held writer lock. compact_locked is a no-op if the delta is
        //    already empty, and resets delta_ops. This keeps a run-for-months
        //    process's delta bounded and its neighbor lookups on CSR asymptotics.
        //
        //    CALLER CONTRACT: this batch is ALREADY durably committed (the WAL
        //    append above was the commit point) and published. A compaction that
        //    fails now must NOT turn this successful apply into an `Err` — a
        //    caller must never see Err for a committed batch and retry it into a
        //    duplicate. So auto-compaction is best-effort: on error we log and
        //    still return Ok(new_seq). If the failure poisoned the writer, the
        //    NEXT apply surfaces `Poisoned`; the on-disk state stays recoverable
        //    (frames <= snapshot_seq are skipped, or the un-renamed MANIFEST
        //    keeps the pre-compaction snapshot live). Explicit `compact()` DOES
        //    propagate errors — only this auto path swallows them.
        let threshold = w
            .auto_compact_override
            .unwrap_or_else(|| auto_compact_threshold(w.base.as_accessor().node_count()));
        if w.delta_ops > threshold {
            if let Err(e) = self.compact_locked(&mut w) {
                eprintln!(
                    "orpheusgraph: auto-compaction failed (batch {new_seq} still committed): {e}"
                );
                // Reset the counter even on failure: leaving delta_ops above the
                // threshold would retry a full-graph rebuild on EVERY subsequent
                // apply (a compaction storm after any transient error). The delta
                // simply keeps growing until it crosses the threshold again; if
                // the failure poisoned the writer, the next apply returns Poisoned
                // regardless.
                w.delta_ops = 0;
            }
        }

        Ok(new_seq)
    }

    /// Fold the live delta into a fresh base snapshot, then empty the delta.
    /// Takes the writer lock; a no-op if the delta is already empty (§4.4b).
    pub fn compact(&self) -> Result<(), PersistError> {
        let mut w = self.writer.lock().unwrap_or_else(|e| e.into_inner());
        if w.wal.poisoned {
            return Err(PersistError::Poisoned);
        }
        self.compact_locked(&mut w)
    }

    /// The real compaction steps, reusing an already-held writer lock (so the
    /// end-of-`apply` auto-trigger has no re-entrancy). Crash-safe: the new
    /// snapshot is written tmp+fsync+rename+fsync(dir), the MANIFEST rename is
    /// the commit point, and only then is the WAL truncated. Folds AT the current
    /// seq (does not mint a new one); durable identity (epoch, seq) is preserved.
    fn compact_locked(&self, w: &mut Writer) -> Result<(), PersistError> {
        // Empty-delta NO-OP guard: nothing to fold => no new snapshot, no delete,
        // seq/snapshot_seq untouched. Required so `compact()` on a quiescent store
        // is free and does not churn the on-disk file (§4.4b test).
        if w.delta.is_empty() {
            w.delta_ops = 0;
            return Ok(());
        }

        // 1. Materialize (base ∘ delta) and rebuild a fresh owned graph. Uses
        //    build_graph_prenormalized: base weights are already normalized and
        //    delta weights are in [0,1], so re-normalizing would rescale
        //    survivors (changing scores at an unchanged seq — audit P2). PageRank
        //    IS recomputed (documented drift, §3.4); the folded graph is the
        //    logical (base+delta) view.
        let base_node_names = w.base.node_names();
        let (nodes, edges) = materialize(w.base.as_accessor(), &base_node_names, &w.delta);
        let (g, idx) = build_graph_prenormalized(nodes, edges);
        let folded = OrpheusGraphInner::new(g, idx);

        // 2. Serialize V2 CSR + crc.
        let bytes = to_rkyv_v2(&folded);
        let new_crc = crc32fast::hash(&bytes);

        // 3. New filename: fold AT the current seq, monotonic cid guarantees a
        //    filename distinct from the live one (empty-delta guard already made
        //    the new_seq==old_seq collision impossible for a real fold).
        let fold_seq = w.seq;
        let new_cid = w.compaction_id + 1;
        let old_snapshot_file = w.snapshot_file.clone();
        let new_snapshot_file = snapshot_v2_name(fold_seq, new_cid);

        // 4. Write the new snapshot durably (tmp+fsync+rename+fsync(dir)).
        write_file_atomic(&self.dir, &new_snapshot_file, &bytes)?;

        // 5. Commit point: rename the MANIFEST to point at the new V2 snapshot.
        let new_manifest = Manifest {
            format_version: FORMAT_VERSION,
            snapshot_file: new_snapshot_file.clone(),
            snapshot_seq: fold_seq,
            snapshot_crc32: new_crc,
            compaction_id: new_cid,
            epoch: w.epoch,
            // Everything up to fold_seq is now folded into the snapshot and
            // durable — record it as the high-water.
            high_seq: fold_seq,
            clean_shutdown: false,
            created_by: CREATED_BY.to_string(),
            checksum: 0,
        };
        write_manifest_atomic(&self.dir, &new_manifest)?;

        // 5b. The MANIFEST above is the durable commit. Advance the in-memory
        //     snapshot pointer NOW, before the WAL-truncate/re-mmap steps that
        //     can still fail: any later failure (poison) must leave the Writer's
        //     snapshot_* fields consistent with the just-committed MANIFEST, so
        //     a subsequent poisoned close() rebuilds the CORRECT pointer instead
        //     of rolling back to the pre-compaction snapshot and losing the
        //     folded WAL (audit P1).
        w.snapshot_seq = fold_seq;
        w.snapshot_file = new_snapshot_file.clone();
        w.snapshot_crc32 = new_crc;
        w.format_version = FORMAT_VERSION;
        w.compaction_id = new_cid;
        // fold_seq is now durable IN THE SNAPSHOT (fsync'd + renamed above), so
        // the durable high-water advances to it even under OnFlush.
        w.durable_seq = w.durable_seq.max(fold_seq);

        // 6. Rotate the WAL: all frames <= fold_seq are now folded. A crash
        //    between step 5 and here leaves stale frames <= snapshot_seq that the
        //    next open() skips (already folded); the next compaction re-truncates.
        w.wal.truncate()?;

        // 7. Republish the folded base honoring the OPEN mode. mode=Owned keeps
        //    the just-built inner (no mmap round-trip); mode=Mmap drops it and
        //    re-mmaps the freshly-written file with validate=None (sound — this
        //    process wrote it this run) so a months-long larger-than-RAM process
        //    actually stays on mmap.
        let new_base = match w.mode {
            BaseMode::Owned => Arc::new(BaseGraph::Owned(folded)),
            BaseMode::Mmap => {
                drop(folded);
                let bg = open_snapshot(
                    &self.dir.join(&new_snapshot_file),
                    BaseMode::Mmap,
                    Validate::None,
                    w.prefault,
                    new_crc,
                )?;
                Arc::new(bg)
            }
        };
        let new_delta = Arc::new(GraphDelta::new());

        w.base = new_base.clone();
        w.delta = new_delta.clone();
        w.delta_ops = 0;
        // snapshot_seq/file/crc/format_version/compaction_id were advanced at
        // step 5b (right after the durable MANIFEST commit).

        // Publish the compacted state (same seq/epoch).
        self.state.store(Arc::new(GraphState {
            base: new_base,
            delta: new_delta,
            seq: w.seq,
            epoch: w.epoch,
        }));

        // 8. INLINE GC (§4.4b): reclaim the immediately-superseded snapshot,
        //    guarded on distinct paths (the cid always differs; defensive). On
        //    Linux unlinking a still-mmap'd old file is safe (existing readers
        //    keep their mapping); on Windows remove_snapshot_file defers.
        if old_snapshot_file != new_snapshot_file {
            remove_snapshot_file(&self.dir.join(&old_snapshot_file));
        }

        Ok(())
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

    /// Override the auto-compaction threshold (runtime knob; not persisted).
    /// Primarily for tests/harnesses that need compaction to fire frequently so
    /// a crash can land inside the compaction sequence. `None` restores the
    /// computed `max(1000, base_nodes/10)`.
    pub fn set_auto_compact_threshold(&self, threshold: Option<usize>) {
        let mut w = self.writer.lock().unwrap_or_else(|e| e.into_inner());
        w.auto_compact_override = threshold;
    }

    /// Force the WAL to durable storage. Under `OnFlush` this is the durability
    /// point. Idempotent; poison-checked.
    pub fn flush(&self) -> Result<(), PersistError> {
        let mut w = self.writer.lock().unwrap_or_else(|e| e.into_inner());
        if w.wal.poisoned {
            return Err(PersistError::Poisoned);
        }
        w.wal.flush()?;
        // Everything appended so far is now on stable storage.
        w.durable_seq = w.seq;
        Ok(())
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
                    Ok(()) => {
                        // All appended frames are now durable.
                        w.durable_seq = w.seq;
                        (w.build_manifest(true), false)
                    }
                    // Flush failed: durable_seq stays at the last successful
                    // fsync, so build_manifest(false) records only durable data.
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

    /// Fold the current (base ∘ delta) exactly as `compact_locked` steps 1-3
    /// would, returning `(bytes, crc, new_file, fold_seq, new_cid)` WITHOUT
    /// writing anything. Lets a test replay the on-disk half of a compaction up
    /// to an arbitrary crash point (before/after the MANIFEST rename) and verify
    /// recovery. Test-only seam.
    #[cfg(test)]
    fn test_fold_bytes(&self) -> (Vec<u8>, u32, String, u64, u64) {
        let w = self.writer.lock().unwrap_or_else(|e| e.into_inner());
        let base_node_names = w.base.node_names();
        let (nodes, edges) = materialize(w.base.as_accessor(), &base_node_names, &w.delta);
        let (g, idx) = build_graph(nodes, edges);
        let folded = OrpheusGraphInner::new(g, idx);
        let bytes = to_rkyv_v2(&folded);
        let crc = crc32fast::hash(&bytes);
        let new_cid = w.compaction_id + 1;
        let file = snapshot_v2_name(w.seq, new_cid);
        (bytes, crc, file, w.seq, new_cid)
    }

    /// Current snapshot filename (test assertions on GC / on-disk layout).
    #[cfg(test)]
    fn snapshot_file_name(&self) -> String {
        self.writer
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .snapshot_file
            .clone()
    }

    /// Current delta_ops counter (auto-compaction assertions).
    #[cfg(test)]
    fn delta_ops(&self) -> usize {
        self.writer
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .delta_ops
    }

    /// Seq folded into the on-disk base (compaction assertions).
    #[cfg(test)]
    fn snapshot_seq(&self) -> u64 {
        self.writer
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .snapshot_seq
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

    fn base_inner(nodes: Vec<(&str, &str)>, edges: Vec<(&str, &str, &str)>) -> OrpheusGraphInner {
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
                base_inner(
                    vec![("root", "m"), ("leaf", "m")],
                    vec![("root", "leaf", "rel")],
                ),
            )
            .unwrap();
            assert_eq!(
                pg.apply(
                    vec![upsert("mid", "m"), addedge("root", "mid", "rel")],
                    None
                )
                .unwrap(),
                1
            );
            assert_eq!(
                pg.apply(vec![upsert("tip", "m"), addedge("mid", "tip", "rel")], None)
                    .unwrap(),
                2
            );
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
        let err = pg
            .apply(vec![addedge("a", "ghost", "rel")], None)
            .unwrap_err();
        assert!(matches!(
            err,
            PersistError::Delta(DeltaError::MissingEndpoint { .. })
        ));

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
        assert_eq!(
            wal_len(dir.path()),
            len_after_good,
            "failed append wrote nothing"
        );

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
                &wal::encode_frame(&WalRecord {
                    seq: s,
                    ops: vec![upsert(name, "m")],
                })
                .unwrap(),
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
            let mut f = std::fs::OpenOptions::new()
                .append(true)
                .open(dir.path().join("wal.log"))
                .unwrap();
            // A plausible-looking header claiming 100 payload bytes, but only 5 follow.
            f.write_all(&100u32.to_le_bytes()).unwrap();
            f.write_all(&0u32.to_le_bytes()).unwrap();
            f.write_all(&[1, 2, 3, 4, 5]).unwrap();
        }
        assert!(wal_len(dir.path()) > good_len);

        let pg = PersistentGraph::open(dir.path(), false).unwrap();
        assert_eq!(pg.seq(), 4, "4 intact frames replayed");
        assert_eq!(
            wal_len(dir.path()),
            good_len,
            "torn tail truncated to boundary"
        );
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
                &wal::encode_frame(&WalRecord {
                    seq: s,
                    ops: vec![upsert(&format!("n{s}"), "m")],
                })
                .unwrap(),
            );
        }
        std::fs::write(dir.path().join("wal.log"), &buf).unwrap();

        let err = PersistentGraph::open(dir.path(), false).unwrap_err();
        match err {
            PersistError::Corrupt(msg) => {
                assert!(msg.contains("gap"), "message should mention gap: {msg}")
            }
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
        assert_eq!(
            pg2.seq(),
            5,
            "all acknowledged batches present after crash-like drop"
        );
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
            PersistentGraph::create(dir.path(), base_inner(vec![("root", "m")], vec![])).unwrap(),
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
                vec![
                    upsert(&format!("n{i}"), "m"),
                    addedge("root", &format!("n{i}"), "rel"),
                ],
                None,
            )
            .unwrap();
        }

        for h in handles {
            h.join().unwrap();
        }
        assert_eq!(pg.seq(), BATCHES);
    }

    // =======================================================================
    // Phase 2b: CSR snapshot, compaction, mmap, GC, auto-compaction, back-compat
    // =======================================================================

    use crate::serialization::to_rkyv;

    /// Per-node (name, is_outgoing, sorted (target, kind) pairs) — the oracle
    /// shape for "logical graph unchanged" comparisons.
    type Topology = Vec<(String, bool, Vec<(String, String)>)>;

    /// All-node outgoing+incoming neighbor sets (sorted) for a whole accessor.
    /// The oracle for "logical graph unchanged" across compaction / mmap.
    fn full_topology(acc: &dyn GraphAccessor, names: &[String]) -> Topology {
        let mut out = Vec::new();
        for n in names {
            out.push((n.clone(), true, out_pairs(acc, n)));
            let mut inc: Vec<(String, String)> = acc
                .incoming_neighbors(n)
                .into_iter()
                .map(|nb| (nb.target_name, nb.edge_kind))
                .collect();
            inc.sort();
            out.push((n.clone(), false, inc));
        }
        out.sort();
        out
    }

    /// Recompute the logical (base+delta) graph independently via
    /// materialize->build_graph and return its topology over `names`.
    fn recompute_topology(pg: &PersistentGraph, names: &[String]) -> Topology {
        let s = pg.snapshot();
        let base_names = s.base.node_names();
        let (nodes, edges) = materialize(s.base.as_accessor(), &base_names, s.delta.as_ref());
        let (g, idx) = build_graph(nodes, edges);
        let recomputed = OrpheusGraphInner::new(g, idx);
        full_topology(&recomputed, names)
    }

    fn live_node_names(pg: &PersistentGraph) -> Vec<String> {
        let s = pg.snapshot();
        let acc = DeltaAccessor::new(s.base.as_accessor(), s.delta.as_ref());
        // Enumerate base names + any delta-added names via a materialize pass.
        let base_names = s.base.node_names();
        let (nodes, _edges) = materialize(s.base.as_accessor(), &base_names, s.delta.as_ref());
        let mut names: Vec<String> = nodes.into_iter().map(|n| n.name).collect();
        names.sort();
        names.dedup();
        // Guard: every listed name is actually live in the merged view.
        names.retain(|n| acc.get_node(n).is_some());
        names
    }

    // ---- compact folds delta and empties it ----------------------------

    #[test]
    fn compact_folds_delta_and_empties_it() {
        let dir = tempfile::tempdir().unwrap();
        let pg =
            PersistentGraph::create(dir.path(), base_inner(vec![("root", "m")], vec![])).unwrap();

        for i in 0..20 {
            pg.apply(
                vec![
                    upsert(&format!("n{i}"), "m"),
                    addedge("root", &format!("n{i}"), "rel"),
                ],
                None,
            )
            .unwrap();
        }
        let seq_before = pg.seq();
        let names = live_node_names(&pg);
        let want = recompute_topology(&pg, &names);
        let wal_before = wal_len(dir.path());

        pg.compact().unwrap();

        // Delta empty, snapshot advanced to seq, seq unchanged.
        {
            let s = pg.snapshot();
            assert!(s.delta.is_empty(), "delta not empty after compaction");
        }
        assert_eq!(pg.snapshot_seq(), seq_before);
        // WAL was actually rotated: it shrank to (near) empty. Without this,
        // removing the compact-time `wal.truncate()` would leave every test
        // green (recovery unconditionally skips frames <= snapshot_seq) while
        // the WAL grew unbounded across compactions — a mutation-adequacy hole.
        let wal_after = wal_len(dir.path());
        assert!(
            wal_after < wal_before && wal_after == 0,
            "compaction must truncate the WAL: {wal_before} -> {wal_after}"
        );
        assert_eq!(pg.seq(), seq_before, "compaction must not mint a new seq");

        // Traversal identical to the independent recompute.
        let s = pg.snapshot();
        let acc = DeltaAccessor::new(s.base.as_accessor(), s.delta.as_ref());
        assert_eq!(full_topology(&acc, &names), want);

        // Reopen: state survives, still folded.
        drop(pg);
        let pg2 = PersistentGraph::open(dir.path(), false).unwrap();
        assert_eq!(pg2.seq(), seq_before);
        let s2 = pg2.snapshot();
        let acc2 = DeltaAccessor::new(s2.base.as_accessor(), s2.delta.as_ref());
        assert_eq!(full_topology(&acc2, &names), want);
    }

    // ---- compaction determinism: no base_weight re-normalization (audit P2) ----

    #[test]
    fn compaction_does_not_renormalize_surviving_base_weight() {
        let dir = tempfile::tempdir().unwrap();
        // A=1.0 is the max, so build_graph normalizes A->1.0, B->0.5.
        let mk = |name: &str, w: f32| NodeInput {
            name: name.into(),
            kind: "m".into(),
            metadata: HashMap::new(),
            base_weight: w,
            noise_penalty: 0.0,
        };
        let (g, idx) = build_graph(vec![mk("A", 1.0), mk("B", 0.5)], vec![]);
        let pg = PersistentGraph::create(dir.path(), OrpheusGraphInner::new(g, idx)).unwrap();

        pg.apply(vec![Op::RemoveNode { name: "A".into() }], None)
            .unwrap();
        let seq = pg.seq();
        let read_b = |pg: &PersistentGraph| {
            let s = pg.snapshot();
            let acc = DeltaAccessor::new(s.base.as_accessor(), s.delta.as_ref());
            acc.get_node("B").unwrap().base_weight
        };
        let before = read_b(&pg);

        pg.compact().unwrap();

        assert_eq!(pg.seq(), seq, "compaction must not mint a new seq");
        // B must NOT be rescaled to 1.0 by re-normalization over the survivors —
        // that would change the scoring base_component at an identical seq.
        assert_eq!(
            read_b(&pg),
            before,
            "surviving base_weight rescaled by compaction"
        );
        assert!(
            (read_b(&pg) - 0.5).abs() < 1e-6,
            "B should keep its normalized 0.5"
        );
    }

    #[test]
    fn compaction_preserves_out_of_range_delta_weight() {
        // An out-of-[0,1] delta node weight (contract violation) must read
        // IDENTICALLY before and after compaction — prenormalized must NOT
        // sanitize it to 0.0 while the live delta view returns it raw (audit #6
        // same-seq scoring divergence).
        let dir = tempfile::tempdir().unwrap();
        let pg =
            PersistentGraph::create(dir.path(), base_inner(vec![("root", "m")], vec![])).unwrap();
        let bad = NodeData {
            name: "x".into(),
            kind: "m".into(),
            metadata: HashMap::new(),
            base_weight: -0.5, // out of contract
            noise_penalty: 0.0,
            pagerank_weight: 0.0,
        };
        pg.apply(vec![Op::UpsertNode(bad)], None).unwrap();
        let seq = pg.seq();
        let read_x = |pg: &PersistentGraph| {
            let s = pg.snapshot();
            let acc = DeltaAccessor::new(s.base.as_accessor(), s.delta.as_ref());
            acc.get_node("x").unwrap().base_weight
        };
        let before = read_x(&pg);
        assert_eq!(before, -0.5, "delta view returns raw weight");
        pg.compact().unwrap();
        assert_eq!(pg.seq(), seq);
        assert_eq!(
            read_x(&pg),
            before,
            "compaction changed an out-of-range weight (same-seq divergence)"
        );
    }

    #[test]
    fn open_with_none_on_corrupt_snapshot_is_error_not_ub() {
        // A SAFE public open must never reach access_unchecked on untrusted bytes:
        // Validate::None is upgraded to Crc, so a bit-rotted (crc-mismatch)
        // snapshot is a typed Corrupt, not UB.
        let dir = tempfile::tempdir().unwrap();
        {
            let pg = PersistentGraph::create(dir.path(), base_inner(vec![("root", "m")], vec![]))
                .unwrap();
            pg.apply(vec![upsert("a", "m")], None).unwrap();
            pg.compact().unwrap(); // ensure a V2 snapshot exists
            pg.close().unwrap();
        }
        // Bit-rot the snapshot file (crc will mismatch).
        let snap = std::fs::read_dir(dir.path())
            .unwrap()
            .flatten()
            .map(|e| e.file_name().into_string().unwrap())
            .find(|n| n.starts_with("snapshot-") && n.ends_with(".og"))
            .unwrap();
        let p = dir.path().join(&snap);
        let mut bytes = std::fs::read(&p).unwrap();
        let mid = bytes.len() / 2;
        bytes[mid] ^= 0xFF;
        std::fs::write(&p, &bytes).unwrap();
        // open_with(None) must return a typed error, never panic/UB.
        match PersistentGraph::open_with(dir.path(), false, BaseMode::Mmap, Validate::None, false) {
            Err(PersistError::Corrupt(_)) => {}
            other => panic!("expected Corrupt (None upgraded to Crc), got {other:?}"),
        }
    }

    // ---- WAL suffix-loss below the durable high-water is Corrupt (audit P2) ----

    #[test]
    fn wal_truncate_to_zero_after_acked_applies_is_corrupt() {
        let dir = tempfile::tempdir().unwrap();
        let pg =
            PersistentGraph::create(dir.path(), base_inner(vec![("root", "m")], vec![])).unwrap();
        pg.apply(vec![upsert("a", "m")], None).unwrap();
        pg.apply(vec![upsert("b", "m")], None).unwrap();
        assert_eq!(pg.seq(), 2);
        pg.close().unwrap(); // persists high_seq=2, clean_shutdown=true

        // External fault: WAL truncated to empty on a frame boundary (NOT a torn
        // tail). Without the high-water guard this reopens silently at seq 0.
        let wal = std::fs::OpenOptions::new()
            .write(true)
            .open(dir.path().join("wal.log"))
            .unwrap();
        wal.set_len(0).unwrap();
        wal.sync_all().unwrap();

        match PersistentGraph::open(dir.path(), false) {
            Err(PersistError::Corrupt(_)) => {}
            other => panic!("expected Corrupt on WAL suffix-loss below high-water, got {other:?}"),
        }
    }

    fn truncate_wal(dir: &Path) {
        let f = std::fs::OpenOptions::new()
            .write(true)
            .open(dir.join("wal.log"))
            .unwrap();
        f.set_len(0).unwrap();
        f.sync_all().unwrap();
    }

    #[test]
    fn manifest_field_bitrot_is_corrupt() {
        let dir = tempfile::tempdir().unwrap();
        let pg =
            PersistentGraph::create(dir.path(), base_inner(vec![("root", "m")], vec![])).unwrap();
        pg.apply(vec![upsert("a", "m")], None).unwrap();
        pg.close().unwrap();
        // Bit-rot snapshot_seq in the MANIFEST to a valid-but-wrong value (would
        // otherwise silently skip WAL frames). The checksum must catch it.
        let mpath = dir.path().join("MANIFEST.json");
        let text = std::fs::read_to_string(&mpath).unwrap();
        let tampered = text.replace("\"snapshot_seq\": 0", "\"snapshot_seq\": 999");
        assert_ne!(tampered, text, "expected to find snapshot_seq to tamper");
        std::fs::write(&mpath, tampered).unwrap();
        match PersistentGraph::open(dir.path(), false) {
            Err(PersistError::Corrupt(m)) if m.contains("checksum") => {}
            other => panic!("expected MANIFEST checksum Corrupt, got {other:?}"),
        }
    }

    // The mirror of the test above (audit P1 regressions): under OnFlush, frames
    // that were only WRITE-THROUGH (never fsync'd) are not durable, so losing
    // them on power-off is contractual — reopen must NOT hard-error. high_seq
    // must never cover un-fsync'd frames.

    #[test]
    fn onflush_crash_recover_then_powerloss_reopens_gracefully() {
        let dir = tempfile::tempdir().unwrap();
        {
            let pg = PersistentGraph::create(dir.path(), base_inner(vec![("root", "m")], vec![]))
                .unwrap();
            // OnFlush (default): applies are write-through only, never fsync'd.
            pg.apply(vec![upsert("a", "m")], None).unwrap();
            pg.apply(vec![upsert("b", "m")], None).unwrap();
            pg.apply(vec![upsert("c", "m")], None).unwrap();
            assert_eq!(pg.seq(), 3);
            drop(pg); // kill -9 model: no close; page-cache WAL survives
        }
        {
            // Crash-recovery reopen. Must NOT durably promote high_seq to 3.
            let pg = PersistentGraph::open(dir.path(), false).unwrap();
            assert_eq!(pg.seq(), 3);
            drop(pg); // second crash, still no flush/close
        }
        // Power loss drops the never-fsync'd WAL tail (reverts to durable = empty).
        truncate_wal(dir.path());
        // Must reopen at the last DURABLE seq (0), not Corrupt-brick.
        let pg = PersistentGraph::open(dir.path(), false)
            .expect("OnFlush power-loss after a crash-recovery reopen must open, not brick");
        assert_eq!(pg.seq(), 0);
    }

    #[test]
    fn onflush_poisoned_close_then_powerloss_reopens_gracefully() {
        let dir = tempfile::tempdir().unwrap();
        let pg =
            PersistentGraph::create(dir.path(), base_inner(vec![("root", "m")], vec![])).unwrap();
        pg.apply(vec![upsert("a", "m")], None).unwrap();
        pg.apply(vec![upsert("b", "m")], None).unwrap();
        assert_eq!(pg.seq(), 2);
        // Poison the writer on the next append; seq stays 2, nothing was flushed.
        pg.arm_append_failure();
        assert!(pg.apply(vec![upsert("c", "m")], None).is_err());
        // Poisoned close must record high_seq = durable_seq (0, nothing fsync'd),
        // NOT seq=2 — else the following power loss would false-brick.
        assert!(matches!(pg.close(), Err(PersistError::Poisoned)));
        truncate_wal(dir.path());
        let pg2 = PersistentGraph::open(dir.path(), false)
            .expect("power loss after a poisoned OnFlush close must open, not brick");
        assert_eq!(pg2.seq(), 0);
    }

    // ---- empty-delta compaction is a no-op -----------------------------

    #[test]
    fn empty_delta_compaction_is_noop() {
        let dir = tempfile::tempdir().unwrap();
        let pg = PersistentGraph::create(dir.path(), base_inner(vec![("a", "m")], vec![])).unwrap();
        // Quiescent store: delta already empty.
        let file_before = pg.snapshot_file_name();
        let listing_before: Vec<String> = std::fs::read_dir(dir.path())
            .unwrap()
            .flatten()
            .map(|e| e.file_name().to_string_lossy().to_string())
            .collect();

        pg.compact().unwrap();
        pg.compact().unwrap(); // twice in a row

        assert_eq!(pg.snapshot_file_name(), file_before, "no new snapshot file");
        assert_eq!(pg.snapshot_seq(), 0);
        assert_eq!(pg.seq(), 0);
        let listing_after: Vec<String> = std::fs::read_dir(dir.path())
            .unwrap()
            .flatten()
            .map(|e| e.file_name().to_string_lossy().to_string())
            .collect();
        let mut a = listing_before.clone();
        let mut b = listing_after.clone();
        a.sort();
        b.sort();
        assert_eq!(a, b, "no files created/removed by no-op compaction");
        // Live snapshot file still present.
        assert!(dir.path().join(&file_before).exists());
    }

    // ---- crash between MANIFEST rename and WAL truncate -----------------

    #[test]
    fn crash_after_manifest_before_wal_truncate_recovers_folded() {
        let dir = tempfile::tempdir().unwrap();
        let pg =
            PersistentGraph::create(dir.path(), base_inner(vec![("root", "m")], vec![])).unwrap();
        for i in 0..8 {
            pg.apply(
                vec![
                    upsert(&format!("n{i}"), "m"),
                    addedge("root", &format!("n{i}"), "rel"),
                ],
                None,
            )
            .unwrap();
        }
        let seq = pg.seq();
        let names = live_node_names(&pg);
        let want = recompute_topology(&pg, &names);

        // Replay the on-disk half of a compaction up to (and including) the
        // MANIFEST rename, but DO NOT truncate the WAL — simulating a crash in
        // that window. The WAL keeps frames 1..=seq (all <= new snapshot_seq).
        let (bytes, crc, file, fold_seq, cid) = pg.test_fold_bytes();
        write_file_atomic(dir.path(), &file, &bytes).unwrap();
        let mut m = read_manifest(&dir.path().join("MANIFEST.json")).unwrap();
        m.format_version = FORMAT_VERSION;
        m.snapshot_file = file.clone();
        m.snapshot_seq = fold_seq;
        m.snapshot_crc32 = crc;
        m.compaction_id = cid;
        write_manifest_atomic(dir.path(), &m).unwrap();
        drop(pg); // "crash": no truncate, no clean close

        // Reopen: frames <= snapshot_seq are skipped (already folded), state is
        // the folded graph, no data loss, no error.
        let pg2 = PersistentGraph::open(dir.path(), false).unwrap();
        assert_eq!(pg2.seq(), seq);
        let s = pg2.snapshot();
        let acc = DeltaAccessor::new(s.base.as_accessor(), s.delta.as_ref());
        assert_eq!(full_topology(&acc, &names), want);
    }

    // ---- crash between snapshot rename and MANIFEST rename (orphan) -----

    #[test]
    fn crash_after_snapshot_before_manifest_gc_removes_orphan() {
        let dir = tempfile::tempdir().unwrap();
        let pg =
            PersistentGraph::create(dir.path(), base_inner(vec![("root", "m")], vec![])).unwrap();
        for i in 0..6 {
            pg.apply(vec![upsert(&format!("n{i}"), "m")], None).unwrap();
        }
        let seq = pg.seq();
        let names = live_node_names(&pg);
        let want = recompute_topology(&pg, &names);
        let live_file = pg.snapshot_file_name();

        // Write ONLY the new snapshot; DO NOT update the MANIFEST — a crash
        // before the commit point. The new file is an orphan.
        let (bytes, _crc, orphan_file, _fs, _cid) = pg.test_fold_bytes();
        write_file_atomic(dir.path(), &orphan_file, &bytes).unwrap();
        drop(pg);
        assert!(dir.path().join(&orphan_file).exists());
        assert_ne!(orphan_file, live_file);

        // Reopen: old MANIFEST still live -> WAL replays -> state == pre-compaction;
        // GC removes the orphan.
        let pg2 = PersistentGraph::open(dir.path(), false).unwrap();
        assert_eq!(pg2.seq(), seq);
        let s = pg2.snapshot();
        let acc = DeltaAccessor::new(s.base.as_accessor(), s.delta.as_ref());
        assert_eq!(full_topology(&acc, &names), want);
        assert!(!dir.path().join(&orphan_file).exists(), "orphan not GC'd");
        assert!(
            dir.path().join(&live_file).exists(),
            "live file wrongly removed"
        );
    }

    // ---- mmap open == owned open ---------------------------------------

    #[test]
    fn mmap_open_equals_owned_open() {
        let dir = tempfile::tempdir().unwrap();
        {
            let pg = PersistentGraph::create(dir.path(), base_inner(vec![("root", "m")], vec![]))
                .unwrap();
            for i in 0..10 {
                pg.apply(
                    vec![
                        upsert(&format!("n{i}"), "m"),
                        addedge("root", &format!("n{i}"), "rel"),
                    ],
                    None,
                )
                .unwrap();
            }
            // Compact so the base carries all the data (delta empty) -> the CSR
            // is what both opens traverse.
            pg.compact().unwrap();
            pg.close().unwrap();
        }

        let mmap =
            PersistentGraph::open_with(dir.path(), false, BaseMode::Mmap, Validate::Full, false)
                .unwrap();
        let names = live_node_names(&mmap);
        let t_mmap = {
            let s = mmap.snapshot();
            let acc = DeltaAccessor::new(s.base.as_accessor(), s.delta.as_ref());
            let t = full_topology(&acc, &names);
            let nc = acc.node_count();
            (t, nc)
        };
        drop(mmap);

        let owned =
            PersistentGraph::open_with(dir.path(), false, BaseMode::Owned, Validate::Full, false)
                .unwrap();
        let t_owned = {
            let s = owned.snapshot();
            let acc = DeltaAccessor::new(s.base.as_accessor(), s.delta.as_ref());
            (full_topology(&acc, &names), acc.node_count())
        };
        assert_eq!(t_mmap, t_owned);
    }

    // ---- prefault open --------------------------------------------------

    #[test]
    fn prefault_open_is_transparent() {
        let dir = tempfile::tempdir().unwrap();
        {
            let pg = PersistentGraph::create(dir.path(), base_inner(vec![("root", "m")], vec![]))
                .unwrap();
            for i in 0..5 {
                pg.apply(
                    vec![
                        upsert(&format!("n{i}"), "m"),
                        addedge("root", &format!("n{i}"), "rel"),
                    ],
                    None,
                )
                .unwrap();
            }
            pg.compact().unwrap();
            pg.close().unwrap();
        }
        let plain =
            PersistentGraph::open_with(dir.path(), false, BaseMode::Mmap, Validate::Full, false)
                .unwrap();
        let names = live_node_names(&plain);
        let want = {
            let s = plain.snapshot();
            let acc = DeltaAccessor::new(s.base.as_accessor(), s.delta.as_ref());
            full_topology(&acc, &names)
        };
        drop(plain);

        let pf =
            PersistentGraph::open_with(dir.path(), false, BaseMode::Mmap, Validate::Full, true)
                .unwrap();
        let s = pf.snapshot();
        let acc = DeltaAccessor::new(s.base.as_accessor(), s.delta.as_ref());
        assert_eq!(full_topology(&acc, &names), want);
    }

    // ---- GC removes orphan snapshots at open ---------------------------

    #[test]
    fn open_time_gc_removes_orphan_snapshots() {
        let dir = tempfile::tempdir().unwrap();
        let pg = PersistentGraph::create(dir.path(), base_inner(vec![("a", "m")], vec![])).unwrap();
        let live = pg.snapshot_file_name();
        pg.close().unwrap();

        // Drop a stray snapshot-*.og not named by the MANIFEST.
        let orphan = dir
            .path()
            .join("snapshot-00000000000000000009-0000000007.og");
        std::fs::write(&orphan, b"garbage-orphan").unwrap();
        assert!(orphan.exists());

        let pg2 = PersistentGraph::open(dir.path(), false).unwrap();
        assert!(!orphan.exists(), "open-time GC did not remove the orphan");
        assert!(
            dir.path().join(&live).exists(),
            "live snapshot wrongly removed"
        );
        drop(pg2);
    }

    // ---- auto-compaction fires at threshold ----------------------------

    #[test]
    fn auto_compaction_fires_and_bounds_delta() {
        let dir = tempfile::tempdir().unwrap();
        // Empty base => threshold = max(1000, 0) = 1000 ops.
        let pg = PersistentGraph::create(dir.path(), base_inner(vec![], vec![])).unwrap();
        assert_eq!(pg.snapshot_seq(), 0);

        // Apply > 1000 ops WITHOUT any manual compact(). Each batch is 1 op.
        let n_batches = 1100u64;
        for i in 0..n_batches {
            pg.apply(vec![upsert(&format!("n{i}"), "m")], None).unwrap();
        }

        // A compaction must have fired: snapshot_seq advanced past 0, delta_ops
        // reset below threshold, delta empty at the fold point.
        assert!(
            pg.snapshot_seq() > 0,
            "auto-compaction never advanced snapshot_seq"
        );
        assert!(
            pg.delta_ops() <= auto_compact_threshold(pg.snapshot().base.as_accessor().node_count()),
            "delta_ops not bounded by threshold after auto-compaction"
        );
        assert_eq!(pg.seq(), n_batches, "seq preserved across auto-compaction");

        // Everything still present.
        let s = pg.snapshot();
        let acc = DeltaAccessor::new(s.base.as_accessor(), s.delta.as_ref());
        assert!(acc.get_node("n0").is_some());
        assert!(acc.get_node(&format!("n{}", n_batches - 1)).is_some());
    }

    // ---- post-compaction inline delete (Linux) -------------------------

    #[test]
    fn post_compaction_old_snapshot_deleted_inline() {
        let dir = tempfile::tempdir().unwrap();
        let pg =
            PersistentGraph::create(dir.path(), base_inner(vec![("root", "m")], vec![])).unwrap();
        let first_file = pg.snapshot_file_name();
        for i in 0..12 {
            pg.apply(vec![upsert(&format!("n{i}"), "m")], None).unwrap();
        }
        pg.compact().unwrap();
        let new_file = pg.snapshot_file_name();
        assert_ne!(new_file, first_file);
        assert!(
            !dir.path().join(&first_file).exists(),
            "old snapshot not deleted inline"
        );
        assert!(dir.path().join(&new_file).exists(), "new snapshot missing");

        // New base is traversable.
        let s = pg.snapshot();
        let acc = DeltaAccessor::new(s.base.as_accessor(), s.delta.as_ref());
        assert!(acc.get_node("n0").is_some());
        assert!(acc.get_node("n11").is_some());
    }

    // ---- legacy format_version 0 / 1 are rejected ----------------------

    #[test]
    fn legacy_format_versions_are_rejected() {
        // A legacy store (flat V0 rkyv snapshot + a MANIFEST tagging it) must be
        // rejected as Corrupt, not silently opened by the V2 reader. Same for a
        // MANIFEST claiming V1. Both are `< FORMAT_VERSION`, so they slip past the
        // `> FORMAT_VERSION` gate and are caught by the explicit legacy arm.
        for legacy in [0u32, 1u32] {
            let dir = tempfile::tempdir().unwrap();
            let base = base_inner(vec![("a", "m"), ("b", "m")], vec![("a", "b", "rel")]);
            let bytes = to_rkyv(&base);
            let crc = crc32fast::hash(&bytes);
            let file = snapshot_name(0);
            write_file_atomic(dir.path(), &file, &bytes).unwrap();
            {
                let f = std::fs::File::create(dir.path().join("wal.log")).unwrap();
                f.sync_all().unwrap();
            }
            let m = Manifest {
                format_version: legacy,
                snapshot_file: file.clone(),
                snapshot_seq: 0,
                snapshot_crc32: crc,
                compaction_id: 0,
                epoch: 123456789,
                high_seq: 0,
                clean_shutdown: true,
                created_by: "test-legacy".into(),
                checksum: 0,
            };
            write_manifest_atomic(dir.path(), &m).unwrap();

            let err = PersistentGraph::open(dir.path(), false).unwrap_err();
            assert!(
                matches!(err, PersistError::Corrupt(_)),
                "legacy format_version {legacy} must be rejected as Corrupt, got {err:?}"
            );
        }
    }

    // ---- property: compaction preserves the logical graph --------------

    use proptest::prelude::*;

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(48))]
        #[test]
        fn prop_compaction_preserves_logical_graph(
            cmds in prop::collection::vec(0u8..4, 1..40usize),
            compact_at in 0usize..40usize,
        ) {
            let dir = tempfile::tempdir().unwrap();
            let pg = PersistentGraph::create(dir.path(), base_inner(vec![("root", "m")], vec![])).unwrap();
            // Track live node names to only ever AddEdge between existing nodes.
            let mut live: Vec<String> = vec!["root".into()];
            let mut counter = 0usize;

            for (i, c) in cmds.iter().enumerate() {
                match c {
                    0 | 1 => {
                        // UpsertNode a fresh node + connect it (keeps the graph linked).
                        let name = format!("p{counter}");
                        counter += 1;
                        let anchor = live[counter % live.len().max(1)].clone();
                        pg.apply(vec![upsert(&name, "m"), addedge(&anchor, &name, "rel")], None).unwrap();
                        live.push(name);
                    }
                    2 => {
                        // AddEdge between two existing live nodes.
                        if live.len() >= 2 {
                            let a = live[i % live.len()].clone();
                            let b = live[(i * 7 + 1) % live.len()].clone();
                            pg.apply(vec![addedge(&a, &b, "x")], None).unwrap();
                        }
                    }
                    _ => {
                        // RemoveNode a non-root live node (idempotent-safe).
                        if live.len() > 1 {
                            let idx = 1 + (i % (live.len() - 1));
                            let victim = live.remove(idx);
                            pg.apply(vec![Op::RemoveNode { name: victim }], None).unwrap();
                        }
                    }
                }
                if i == compact_at {
                    let names = live_node_names(&pg);
                    let before = recompute_topology(&pg, &names);
                    pg.compact().unwrap();
                    let after = {
                        let s = pg.snapshot();
                        let acc = DeltaAccessor::new(s.base.as_accessor(), s.delta.as_ref());
                        full_topology(&acc, &names)
                    };
                    prop_assert_eq!(before, after);
                }
            }

            // Final: traverse(base+delta) == recompute(materialize) after everything.
            let names = live_node_names(&pg);
            let want = recompute_topology(&pg, &names);
            let s = pg.snapshot();
            let acc = DeltaAccessor::new(s.base.as_accessor(), s.delta.as_ref());
            prop_assert_eq!(full_topology(&acc, &names), want);
        }
    }
}
