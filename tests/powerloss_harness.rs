//! Power-loss fault-injection harness (the deterministic sibling of the kill -9
//! `crash_harness`). Spec §6 asks for `dm-flakey`/a VM to validate the loss of
//! the OS page cache on power loss; instead this builds a userspace
//! fault-injection [`Vfs`] (the RocksDB `FaultInjectionTestFS` pattern) — no
//! root, fully deterministic (seeded), CI-able.
//!
//! ## The fault model (why it is faithful)
//! [`FaultVfs`] passes EVERY operation through to the real filesystem (so the
//! snapshot mmap, external readers and all real I/O work naturally), while
//! shadowing, per file, the LAST-SYNCED image plus a journal of un-synced
//! mutations. `power_cut()` then rewrites the REAL directory to the crash image:
//!
//! * every file reverts to its last successful `fsync` content (the durable
//!   image), because un-fsync'd page-cache bytes are exactly what power loss
//!   drops;
//! * a configurable prefix of the last un-synced write MAY survive (torn
//!   sector write, byte-granular, driven by the harness RNG);
//! * un-`fsync_dir`'d namespace ops are undone — a never-dir-synced create is
//!   removed, a never-dir-synced rename is reverted, a never-dir-synced remove
//!   is resurrected.
//!
//! `fail_next_fsync(persist=true)` returns `Err` yet KEEPS the data (as if the
//! kernel wrote it back anyway) — the exact fsync-failure ambiguity the COMMIT
//! marker exists for.
//!
//! ## Ground truth
//! The harness tracks, per applied batch, the ops it applied (`reference`) and
//! the store's `committed_seq()` after each ack (`max_committed`). After a pure
//! power cut + reopen it asserts the recovered state is EXACTLY the committed
//! prefix: `committed_seq()` never regresses below the last observed durable
//! value, every committed batch is present, and NOTHING beyond the marker is
//! (node_count/edge_count are exact — a phantom or a hole fails).
//!
//! Iterations scale via `OG_POWERLOSS_ITERS` (like `OG_CRASH_ITERS`); every
//! failure message prints the seed for replay.

use orpheusgraph::accessor::GraphAccessor;
use orpheusgraph::persist::vfs::{Vfs, VfsFile};
use orpheusgraph::types::{EdgeData, NodeData, NodeInput};
use orpheusgraph::{
    build_graph, BaseMode, DeltaAccessor, FsyncPolicy, Op, OrpheusGraphInner, PersistentGraph,
    Validate,
};
use std::collections::HashMap;
use std::fs::OpenOptions;
use std::io::{self, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

// ===========================================================================
// Deterministic PRNG (SplitMix64) — no `rand` dependency, fully reproducible.
// ===========================================================================

struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Rng(seed)
    }
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    /// Uniform in `[0, n)`; `0` for `n == 0`.
    fn below(&mut self, n: u64) -> u64 {
        if n == 0 {
            0
        } else {
            self.next_u64() % n
        }
    }
    /// Uniform in `[lo, hi]`.
    fn range(&mut self, lo: u64, hi: u64) -> u64 {
        lo + self.below(hi - lo + 1)
    }
    fn boolean(&mut self) -> bool {
        self.next_u64() & 1 == 1
    }
}

// ===========================================================================
// FaultVfs — pass-through + per-file shadow + power_cut restore.
// ===========================================================================

/// One un-synced content mutation, replayed (last one possibly torn) on a cut.
#[derive(Clone)]
enum FileMut {
    Write { offset: u64, bytes: Vec<u8> },
    SetLen(u64),
}

/// Per-file shadow. `durable` is the content at the last SUCCESSFUL `sync_all`
/// (never torn, never below committed data); `pending` is everything since;
/// `live` mirrors the real file (== durable folded with all pending).
#[derive(Clone, Default)]
struct Tracked {
    durable: Vec<u8>,
    pending: Vec<FileMut>,
    live: Vec<u8>,
}

/// A namespace mutation, undone on a cut if its containing dir was never fsync'd.
enum EntryOp {
    Create(PathBuf),
    Remove {
        path: PathBuf,
        durable: Vec<u8>,
    },
    Rename {
        from: PathBuf,
        to: PathBuf,
        /// `to`'s durable content before this (overwriting) rename, to restore
        /// on undo. `None` if `to` did not previously exist.
        to_prior: Option<Vec<u8>>,
    },
}

/// A scripted one-shot fsync failure matched by a path substring.
struct FsyncFail {
    needle: String,
    /// `true` => the data still persists (folds into `durable`) despite the
    /// returned `Err` — the fsync-failure ambiguity the COMMIT marker resolves.
    persist: bool,
}

#[derive(Default)]
struct Inner {
    tracked: HashMap<PathBuf, Tracked>,
    /// Un-`fsync_dir`'d namespace ops (the whole store lives in one directory).
    journal: Vec<EntryOp>,
    /// FIFO one-shot fsync failures.
    fsync_fail: Vec<FsyncFail>,
    /// Count of upcoming `fsync_dir` calls that return `Err` WITHOUT making the
    /// pending namespace ops durable (models a rename not reaching the disk).
    dir_fsync_fail: usize,
}

/// How the un-synced tail is treated on a cut.
enum Tear<'a> {
    /// Drop every un-synced mutation (the clean, deterministic crash image).
    DropAll,
    /// Random torn prefixes (models real sector-granular loss).
    Random(&'a mut Rng),
    /// Apply every pending mutation but the last fully, then keep exactly `keep`
    /// bytes of the last write (a forced torn-write, for targeted tests).
    LastKeep(usize),
}

pub struct FaultVfs {
    inner: Arc<Mutex<Inner>>,
}

/// Apply one content mutation to a byte buffer (holes are zero-filled).
fn apply_mut(buf: &mut Vec<u8>, m: &FileMut) {
    match m {
        FileMut::Write { offset, bytes } => {
            let end = *offset as usize + bytes.len();
            if buf.len() < end {
                buf.resize(end, 0);
            }
            buf[*offset as usize..end].copy_from_slice(bytes);
        }
        FileMut::SetLen(n) => buf.resize(*n as usize, 0),
    }
}

/// Reconstruct the crash-image content from the durable base + the un-synced
/// pending mutations, per the tear policy. NEVER removes durable (committed)
/// bytes — only the un-synced tail is at risk.
fn reconstruct(durable: &[u8], pending: &[FileMut], tear: &mut Tear) -> Vec<u8> {
    let mut r = durable.to_vec();
    let n = pending.len();
    if n == 0 {
        return r;
    }
    // Number of leading pending mutations applied in full.
    let k = match tear {
        Tear::DropAll => 0,
        Tear::LastKeep(_) => n - 1,
        Tear::Random(rng) => rng.below(n as u64 + 1) as usize,
    };
    for m in &pending[..k] {
        apply_mut(&mut r, m);
    }
    if k < n {
        match &pending[k] {
            FileMut::Write { offset, bytes } => {
                let keep = match tear {
                    Tear::DropAll => 0,
                    Tear::LastKeep(keep) => (*keep).min(bytes.len()),
                    Tear::Random(rng) => rng.below(bytes.len() as u64 + 1) as usize,
                };
                apply_mut(
                    &mut r,
                    &FileMut::Write {
                        offset: *offset,
                        bytes: bytes[..keep].to_vec(),
                    },
                );
            }
            FileMut::SetLen(nn) => {
                // A metadata op is atomic: applied or not (never "half").
                let apply = match tear {
                    Tear::DropAll => false,
                    Tear::LastKeep(_) => true,
                    Tear::Random(rng) => rng.boolean(),
                };
                if apply {
                    apply_mut(&mut r, &FileMut::SetLen(*nn));
                }
            }
        }
    }
    r
}

/// Overwrite a real file with exact bytes (create+truncate+write+fsync).
fn write_real(path: &Path, bytes: &[u8]) {
    if let Ok(mut f) = OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(path)
    {
        let _ = f.write_all(bytes);
        let _ = f.sync_all();
    }
}

impl FaultVfs {
    fn new() -> Self {
        FaultVfs {
            inner: Arc::new(Mutex::new(Inner::default())),
        }
    }

    /// Arm a one-shot fsync failure on the next file whose path contains
    /// `needle`. `persist` = the bytes survive the cut anyway (kernel writeback).
    fn fail_next_fsync(&self, needle: &str, persist: bool) {
        self.inner.lock().unwrap().fsync_fail.push(FsyncFail {
            needle: needle.to_string(),
            persist,
        });
    }

    /// Arm the next `fsync_dir` to fail WITHOUT making its pending namespace ops
    /// durable (so a preceding rename is undone on the cut).
    fn fail_next_dir_fsync(&self) {
        self.inner.lock().unwrap().dir_fsync_fail += 1;
    }

    /// Lazily shadow an existing on-disk file's content as its durable image.
    fn ensure_tracked_from_disk(&self, path: &Path) {
        let mut inner = self.inner.lock().unwrap();
        if !inner.tracked.contains_key(path) {
            let bytes = std::fs::read(path).unwrap_or_default();
            inner.tracked.insert(
                path.to_path_buf(),
                Tracked {
                    durable: bytes.clone(),
                    pending: Vec::new(),
                    live: bytes,
                },
            );
        }
    }

    /// Power loss with random torn tails.
    fn power_cut(&self, rng: &mut Rng) {
        self.restore(Tear::Random(rng));
    }
    /// Power loss dropping the entire un-synced tail (deterministic crash image).
    fn power_cut_dropping_unsynced(&self) {
        self.restore(Tear::DropAll);
    }
    /// Power loss forcing a torn last write to keep exactly `keep` bytes.
    fn power_cut_torn_keep(&self, keep: usize) {
        self.restore(Tear::LastKeep(keep));
    }

    fn restore(&self, mut tear: Tear) {
        let mut inner = self.inner.lock().unwrap();
        // Phase A: rewrite each tracked file to its crash-image content.
        let paths: Vec<PathBuf> = inner.tracked.keys().cloned().collect();
        for p in &paths {
            let crash = {
                let t = &inner.tracked[p];
                reconstruct(&t.durable, &t.pending, &mut tear)
            };
            write_real(p, &crash);
        }
        // Phase B: undo un-dir-fsync'd namespace ops, newest first.
        let journal = std::mem::take(&mut inner.journal);
        for op in journal.iter().rev() {
            match op {
                EntryOp::Create(p) => {
                    let _ = std::fs::remove_file(p);
                }
                EntryOp::Remove { path, durable } => write_real(path, durable),
                EntryOp::Rename { from, to, to_prior } => {
                    let _ = std::fs::rename(to, from);
                    match to_prior {
                        Some(b) => write_real(to, b),
                        None => {
                            let _ = std::fs::remove_file(to);
                        }
                    }
                }
            }
        }
        // Phase C: the real directory IS the crash image now and everything on it
        // is durable — reset the shadow so a reopen re-inits from disk.
        inner.tracked.clear();
        inner.journal.clear();
        inner.fsync_fail.clear();
        inner.dir_fsync_fail = 0;
    }
}

impl Vfs for FaultVfs {
    fn create(&self, path: &Path) -> io::Result<Box<dyn VfsFile>> {
        let real = OpenOptions::new()
            .create(true)
            .read(true)
            .write(true)
            .truncate(true)
            .open(path)?;
        {
            let mut inner = self.inner.lock().unwrap();
            // Keep any prior durable content (a create/truncate is not durable
            // until fsync); model the truncate as a pending SetLen(0). For a
            // brand-new file the prior durable is empty (a harmless no-op).
            let prior = inner
                .tracked
                .get(path)
                .map(|t| t.durable.clone())
                .unwrap_or_default();
            inner.tracked.insert(
                path.to_path_buf(),
                Tracked {
                    durable: prior,
                    pending: vec![FileMut::SetLen(0)],
                    live: Vec::new(),
                },
            );
            // Creation of the directory entry is not durable until fsync_dir.
            inner.journal.push(EntryOp::Create(path.to_path_buf()));
        }
        Ok(Box::new(FaultFile {
            inner: self.inner.clone(),
            real,
            path: path.to_path_buf(),
            append: false,
            pos: 0,
        }))
    }

    fn open_rw(&self, path: &Path) -> io::Result<Box<dyn VfsFile>> {
        let real = OpenOptions::new().read(true).write(true).open(path)?;
        self.ensure_tracked_from_disk(path);
        Ok(Box::new(FaultFile {
            inner: self.inner.clone(),
            real,
            path: path.to_path_buf(),
            append: false,
            pos: 0,
        }))
    }

    fn open_append(&self, path: &Path) -> io::Result<Box<dyn VfsFile>> {
        let real = OpenOptions::new().read(true).append(true).open(path)?;
        self.ensure_tracked_from_disk(path);
        Ok(Box::new(FaultFile {
            inner: self.inner.clone(),
            real,
            path: path.to_path_buf(),
            append: true,
            pos: 0,
        }))
    }

    fn rename(&self, from: &Path, to: &Path) -> io::Result<()> {
        std::fs::rename(from, to)?;
        let mut inner = self.inner.lock().unwrap();
        let to_prior = inner.tracked.get(to).map(|t| t.durable.clone());
        // The inode content follows the name.
        if let Some(src) = inner.tracked.remove(from) {
            inner.tracked.insert(to.to_path_buf(), src);
        }
        inner.journal.push(EntryOp::Rename {
            from: from.to_path_buf(),
            to: to.to_path_buf(),
            to_prior,
        });
        Ok(())
    }

    fn remove_file(&self, path: &Path) -> io::Result<()> {
        let durable = {
            let inner = self.inner.lock().unwrap();
            inner.tracked.get(path).map(|t| t.durable.clone())
        };
        // If not shadowed (e.g. an old snapshot from a prior generation), the
        // durable image is whatever is on disk right now.
        let durable = match durable {
            Some(d) => d,
            None => std::fs::read(path).unwrap_or_default(),
        };
        std::fs::remove_file(path)?;
        let mut inner = self.inner.lock().unwrap();
        inner.tracked.remove(path);
        inner.journal.push(EntryOp::Remove {
            path: path.to_path_buf(),
            durable,
        });
        Ok(())
    }

    fn fsync_dir(&self, dir: &Path) -> io::Result<()> {
        let mut inner = self.inner.lock().unwrap();
        if inner.dir_fsync_fail > 0 {
            inner.dir_fsync_fail -= 1;
            // Return Err WITHOUT clearing the journal: the pending namespace ops
            // stay un-durable and are undone on the next power_cut.
            return Err(io::Error::other("injected fsync_dir failure"));
        }
        std::fs::File::open(dir)?.sync_all()?;
        // Every pending namespace op is now durable.
        inner.journal.clear();
        Ok(())
    }
}

struct FaultFile {
    inner: Arc<Mutex<Inner>>,
    real: std::fs::File,
    path: PathBuf,
    append: bool,
    pos: u64,
}

impl VfsFile for FaultFile {
    fn read_to_end(&mut self, buf: &mut Vec<u8>) -> io::Result<usize> {
        // Read from the shadow `live` (== the real file, passthrough) so reads
        // are consistent with the tracked content after a reset+reopen.
        let inner = self.inner.lock().unwrap();
        let t = inner.tracked.get(&self.path).expect("tracked file");
        let len = t.live.len();
        let start = (self.pos as usize).min(len);
        buf.extend_from_slice(&t.live[start..]);
        self.pos = len as u64;
        Ok(len - start)
    }

    fn write_all(&mut self, buf: &[u8]) -> io::Result<()> {
        self.real.write_all(buf)?; // passthrough (append mode -> physical end)
        let mut inner = self.inner.lock().unwrap();
        let t = inner.tracked.get_mut(&self.path).expect("tracked file");
        let offset = if self.append {
            t.live.len() as u64
        } else {
            self.pos
        };
        let m = FileMut::Write {
            offset,
            bytes: buf.to_vec(),
        };
        apply_mut(&mut t.live, &m);
        t.pending.push(m);
        if !self.append {
            self.pos += buf.len() as u64;
        }
        Ok(())
    }

    fn seek(&mut self, pos: SeekFrom) -> io::Result<u64> {
        self.real.seek(pos)?; // keep the real cursor coherent
        let len = {
            let inner = self.inner.lock().unwrap();
            inner
                .tracked
                .get(&self.path)
                .map(|t| t.live.len() as u64)
                .unwrap_or(0)
        };
        let np = match pos {
            SeekFrom::Start(n) => n,
            SeekFrom::End(n) => (len as i64 + n).max(0) as u64,
            SeekFrom::Current(n) => (self.pos as i64 + n).max(0) as u64,
        };
        self.pos = np;
        Ok(np)
    }

    fn sync_all(&mut self) -> io::Result<()> {
        let mut inner = self.inner.lock().unwrap();
        if let Some(idx) = inner
            .fsync_fail
            .iter()
            .position(|r| self.path.to_string_lossy().contains(&r.needle))
        {
            let rule = inner.fsync_fail.remove(idx);
            if rule.persist {
                // Data persists despite the Err: fold pending into durable so the
                // cut keeps it (crc-valid frame beyond the un-advanced marker).
                let t = inner.tracked.get_mut(&self.path).expect("tracked file");
                t.durable = t.live.clone();
                t.pending.clear();
            }
            return Err(io::Error::other("injected fsync failure"));
        }
        self.real.sync_all()?;
        let t = inner.tracked.get_mut(&self.path).expect("tracked file");
        t.durable = t.live.clone();
        t.pending.clear();
        Ok(())
    }

    fn set_len(&mut self, size: u64) -> io::Result<()> {
        self.real.set_len(size)?;
        let mut inner = self.inner.lock().unwrap();
        let t = inner.tracked.get_mut(&self.path).expect("tracked file");
        let m = FileMut::SetLen(size);
        apply_mut(&mut t.live, &m);
        t.pending.push(m);
        Ok(())
    }

    fn len(&mut self) -> io::Result<u64> {
        let inner = self.inner.lock().unwrap();
        Ok(inner
            .tracked
            .get(&self.path)
            .map(|t| t.live.len() as u64)
            .unwrap_or(0))
    }
}

// ===========================================================================
// Harness scaffolding: deterministic additive batches + reference state.
// ===========================================================================

fn root_base() -> OrpheusGraphInner {
    let (g, idx) = build_graph(
        vec![NodeInput {
            name: "root".into(),
            kind: "m".into(),
            metadata: HashMap::new(),
            base_weight: 0.5,
            noise_penalty: 0.0,
        }],
        vec![],
    );
    OrpheusGraphInner::new(g, idx)
}

fn mk_node(name: &str) -> Op {
    Op::UpsertNode(NodeData {
        name: name.into(),
        kind: "m".into(),
        metadata: HashMap::new(),
        base_weight: 0.5,
        noise_penalty: 0.0,
        pagerank_weight: 0.0,
    })
}

fn mk_edge(from: &str, to: &str) -> Op {
    Op::AddEdge {
        from: from.into(),
        to: to.into(),
        edge: EdgeData {
            kind: "rel".into(),
            field_name: None,
            base_weight: 1.0,
            valid_from: None,
            valid_to: None,
            acl: Vec::new(),
        },
    }
}

/// An all-or-nothing batch at `seq`: `k` fresh nodes `mark-{seq}-{j}` plus one
/// edge `mark-{seq}-0 -> root`. Returns the ops and the node names.
fn batch_ops(seq: u64, k: usize) -> (Vec<Op>, Vec<String>) {
    let mut ops = Vec::new();
    let mut names = Vec::new();
    for j in 0..k {
        let n = format!("mark-{seq}-{j}");
        ops.push(mk_node(&n));
        names.push(n);
    }
    ops.push(mk_edge(&format!("mark-{seq}-0"), "root"));
    (ops, names)
}

fn open(vfs: Arc<dyn Vfs>, dir: &Path, mode: BaseMode) -> Result<PersistentGraph, String> {
    PersistentGraph::open_with_vfs(vfs, dir, false, mode, Validate::Full, false)
        .map_err(|e| format!("{e}"))
}

/// Assert the reopened store is EXACTLY the committed prefix of `reference`.
///
/// The ACHIEVABLE recovery contract (see `commit.rs` / the "Honest apply()
/// durability contract" impl-log entry): whatever `committed_seq` the store
/// reopens at, the recovered CONTENT must equal the materialization of EXACTLY
/// that prefix (consistent prefix — no torn/half-applied batch), and no
/// durably-acked batch (`seq <= max_committed`) may be lost. This does NOT assume
/// per-Err invisibility: under the COMMIT-fsync-persist injection an Err'd batch
/// may legally reappear, but ONLY as a fully-materialized prefix extension.
fn verify(
    pg: &PersistentGraph,
    reference: &HashMap<u64, Vec<String>>,
    max_committed: u64,
    seed: u64,
    gen: usize,
) {
    let r = pg.seq();
    // (a) is checked at the open call site (never Corrupt). (b): recovery opens at
    // the marker, so seq == committed_seq, and it never regresses below the last
    // observed durable high-water — no acked batch is silently dropped.
    assert_eq!(
        pg.committed_seq(),
        r,
        "[seed {seed:#x} gen {gen}] recovered committed_seq {} != seq {r}",
        pg.committed_seq()
    );
    assert!(
        r >= max_committed,
        "[seed {seed:#x} gen {gen}] recovered seq {r} < last observed durable committed {max_committed} (acked-batch loss)"
    );

    let s = pg.snapshot();
    let acc = DeltaAccessor::new(s.base.as_accessor(), s.delta.as_ref());

    // (c) CONSISTENT-PREFIX oracle. Materialize, from the harness-side `reference`,
    // the EXACT graph implied by the reopened prefix `r`, and compare content:
    // every committed batch's nodes AND its edge must be present. Combined with the
    // exact node/edge counts below (which forbid any extra), this pins the recovered
    // graph to the prefix-`r` materialization set-for-set (no hole, no phantom, no
    // torn batch) — for these additive batches presence-of-all + exact-count is set
    // equality by pigeonhole.
    let mut expected_nodes = 1usize; // root
    for seq in 1..=r {
        let names = reference.get(&seq).unwrap_or_else(|| {
            panic!("[seed {seed:#x} gen {gen}] reference missing committed seq {seq} (<= recovered {r})")
        });
        for n in names {
            assert!(
                acc.get_node(n).is_some(),
                "[seed {seed:#x} gen {gen}] committed node {n} (seq {seq}) missing after reopen"
            );
        }
        expected_nodes += names.len();
        assert!(
            acc.outgoing_neighbors(&format!("mark-{seq}-0"))
                .iter()
                .any(|nb| nb.target_name == "root"),
            "[seed {seed:#x} gen {gen}] committed edge for seq {seq} missing after reopen"
        );
    }

    // NO-ACKED-LOSS oracle (explicit, independent of the prefix loop above so a
    // future loosening of that loop cannot silently drop this guarantee): every
    // DURABLY-acked batch (seq <= the highest committed_seq the harness ever
    // observed) MUST be present and fully materialized after any crash+reopen. An
    // `apply`-Ok under OnFlush is NOT yet durable, so `max_committed` — the running
    // max of committed_seq() — is the correct "acked & durable" high-water here.
    for seq in 1..=max_committed {
        let names = reference.get(&seq).unwrap_or_else(|| {
            panic!("[seed {seed:#x} gen {gen}] reference missing durably-acked seq {seq}")
        });
        for n in names {
            assert!(
                acc.get_node(n).is_some(),
                "[seed {seed:#x} gen {gen}] durably-acked node {n} (seq {seq}) lost after reopen"
            );
        }
        assert!(
            acc.outgoing_neighbors(&format!("mark-{seq}-0"))
                .iter()
                .any(|nb| nb.target_name == "root"),
            "[seed {seed:#x} gen {gen}] durably-acked edge for seq {seq} lost after reopen"
        );
    }

    // (d) nothing beyond the marker: the next batch, if it was ever applied, is gone.
    if let Some(names) = reference.get(&(r + 1)) {
        for n in names {
            assert!(
                acc.get_node(n).is_none(),
                "[seed {seed:#x} gen {gen}] phantom node {n} beyond recovered seq {r}"
            );
        }
    }
    // Exact counts catch BOTH a hole (missing committed) and a phantom (leaked),
    // closing the consistent-prefix check into a set equality.
    assert_eq!(
        acc.node_count(),
        expected_nodes,
        "[seed {seed:#x} gen {gen}] node_count {} != expected {expected_nodes} (prefix len {r})",
        acc.node_count()
    );
    assert_eq!(
        acc.edge_count(),
        r as usize,
        "[seed {seed:#x} gen {gen}] edge_count {} != expected {r}",
        acc.edge_count()
    );
}

// ===========================================================================
// The main randomized, seeded, multi-generation harness.
// ===========================================================================

#[test]
fn powerloss_multi_generation_harness() {
    let iters: u64 = std::env::var("OG_POWERLOSS_ITERS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(16);
    const GENERATIONS: usize = 4; // >= 3: recovery-of-a-recovery-of-a-recovery
    const SEED_BASE: u64 = 0x0DD0_5EED_0000_0000;

    let mut global_progress = 0u64;

    for i in 0..iters {
        let seed = SEED_BASE ^ i.wrapping_mul(0x9E37_79B9_7F4A_7C15);
        let mut rng = Rng::new(seed);
        let tmp = tempfile::tempdir().expect("tempdir");
        let dir = tmp.path().to_path_buf();
        let vfs = Arc::new(FaultVfs::new());

        // Durable baseline: a clean store with just `root`.
        PersistentGraph::create_with_vfs(vfs.clone(), &dir, root_base())
            .expect("create")
            .close()
            .expect("close");

        let mut reference: HashMap<u64, Vec<String>> = HashMap::new();
        let mut max_committed: u64 = 0;
        let mut mode = BaseMode::Mmap;
        let mut pg = open(vfs.clone() as Arc<dyn Vfs>, &dir, mode)
            .unwrap_or_else(|e| panic!("[seed {seed:#x}] initial open failed: {e}"));

        for gen in 0..GENERATIONS {
            // Gen 0 always EveryBatch (guaranteed progress); later gens random.
            let policy = if gen == 0 {
                FsyncPolicy::EveryBatch
            } else {
                match rng.below(3) {
                    0 => FsyncPolicy::EveryBatch,
                    1 => FsyncPolicy::EveryN(rng.range(2, 4) as u32),
                    _ => FsyncPolicy::OnFlush,
                }
            };
            pg.set_fsync_policy(policy);
            // Force frequent compaction so cuts land around the non-atomic
            // on-disk compaction sequence.
            if rng.boolean() {
                pg.set_auto_compact_threshold(Some(rng.range(2, 8) as usize));
            }

            let nbatches = rng.range(3, 20);
            let mut poisoned = false;
            for _ in 0..nbatches {
                let seq = pg.seq() + 1;
                let k = rng.range(1, 3) as usize;
                let (ops, names) = batch_ops(seq, k);
                reference.insert(seq, names);
                if rng.below(7) == 0 {
                    let _ = pg.compact(); // occasional explicit compaction
                }
                match pg.apply(ops, None) {
                    Ok(_) => {
                        max_committed = max_committed.max(pg.committed_seq());
                        if rng.below(4) == 0 {
                            let _ = pg.flush();
                            max_committed = max_committed.max(pg.committed_seq());
                        }
                    }
                    Err(_) => {
                        poisoned = true;
                        break;
                    }
                }
            }

            // Occasionally mix in a durability injection just before the cut.
            let mut injected = false;
            if !poisoned && gen > 0 {
                match rng.below(4) {
                    // Crown jewel: a WAL fsync that FAILS yet the frame persists.
                    // apply returns Err, the marker does NOT advance, and the
                    // persisted crc-valid frame is discarded (counted) on reopen.
                    0 => {
                        pg.set_fsync_policy(FsyncPolicy::EveryBatch);
                        let seq = pg.seq() + 1;
                        let (ops, names) = batch_ops(seq, 1);
                        reference.insert(seq, names);
                        vfs.fail_next_fsync("wal.log", true);
                        assert!(
                            pg.apply(ops, None).is_err(),
                            "[seed {seed:#x} gen {gen}] injected WAL fsync failure must Err"
                        );
                        injected = true;
                    }
                    // Mid-compaction: snapshot + MANIFEST durable, COMMIT advance
                    // fsync fails. Recovery must use the snapshot_seq floor.
                    1 => {
                        vfs.fail_next_fsync("COMMIT", false);
                        let _ = pg.compact(); // Err (advance_commit fails) is fine
                        injected = true;
                    }
                    // The irreducible ambiguity direction (previously unexercised —
                    // the gap that hid the bug): a COMMIT-slot fsync FAILS yet its
                    // fully-written slot PERSISTS. apply Errs, but the marker on disk
                    // now advances, so the Err'd batch's durable WAL frame becomes
                    // VISIBLE on reopen. Legal under the honest contract — recovery
                    // must still yield a CONSISTENT prefix (the strengthened oracle
                    // enforces it), never a torn/half-applied batch.
                    2 => {
                        pg.set_fsync_policy(FsyncPolicy::EveryBatch);
                        let seq = pg.seq() + 1;
                        let (ops, names) = batch_ops(seq, 1);
                        reference.insert(seq, names);
                        vfs.fail_next_fsync("COMMIT", true);
                        assert!(
                            pg.apply(ops, None).is_err(),
                            "[seed {seed:#x} gen {gen}] injected COMMIT fsync-persist must Err"
                        );
                        injected = true;
                    }
                    _ => {}
                }
            }

            // The cut. After an injection use a deterministic drop-all so the
            // recovered state is unambiguous; otherwise random torn tails.
            if injected || poisoned {
                vfs.power_cut_dropping_unsynced();
            } else if rng.boolean() {
                vfs.power_cut(&mut rng);
            } else {
                vfs.power_cut_dropping_unsynced();
            }
            drop(pg); // abandon the handle WITHOUT close (a crash)

            // Reopen from the real (crash-image) directory; alternate the base mode.
            mode = if rng.boolean() {
                BaseMode::Mmap
            } else {
                BaseMode::Owned
            };
            pg = open(vfs.clone() as Arc<dyn Vfs>, &dir, mode).unwrap_or_else(|e| {
                panic!("[seed {seed:#x} gen {gen}] reopen after power cut is Corrupt/failed: {e}")
            });
            verify(&pg, &reference, max_committed, seed, gen);
            global_progress = global_progress.max(pg.seq());
        }
        drop(pg);
    }

    assert!(
        global_progress > 0,
        "harness never committed a single batch across all iterations"
    );
}

// ===========================================================================
// Targeted, deterministic scenarios (the explicit bullets in the mandate).
// ===========================================================================

/// (d) THE crown jewel: fsync FAILS but the data persists anyway; the process
/// "dies"; on reopen the Err'd batch is INVISIBLE and counted in recovery_report.
#[test]
fn crown_jewel_fsync_fails_but_data_persists_is_invisible_and_counted() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path().to_path_buf();
    let vfs = Arc::new(FaultVfs::new());

    {
        let pg = PersistentGraph::create_with_vfs(vfs.clone(), &dir, root_base()).unwrap();
        pg.set_fsync_policy(FsyncPolicy::EveryBatch);
        pg.apply(batch_ops(1, 1).0, None).unwrap(); // committed 1
        pg.apply(batch_ops(2, 1).0, None).unwrap(); // committed 2
        assert_eq!(pg.committed_seq(), 2);

        // The next WAL fsync fails, BUT the frame persists (kernel writeback).
        vfs.fail_next_fsync("wal.log", true);
        let err = pg.apply(batch_ops(3, 1).0, None);
        assert!(err.is_err(), "the fsync-failed apply must return Err");
        assert_eq!(
            pg.committed_seq(),
            2,
            "an Err'd apply must NOT advance the durable marker"
        );

        // Power loss: keep the durable image (which now HOLDS the persisted,
        // never-committed frame). The frame is complete + crc-valid but beyond
        // the marker — exactly the fsync-failure ambiguity.
        vfs.power_cut_dropping_unsynced();
        drop(pg);
    }

    let pg = PersistentGraph::open_with_vfs(
        vfs.clone(),
        &dir,
        false,
        BaseMode::Mmap,
        Validate::Full,
        false,
    )
    .expect("reopen after a pure power cut must never be Corrupt");
    assert_eq!(
        pg.seq(),
        2,
        "the Err'd (fsync-failed) batch must NOT be visible after reopen"
    );
    assert!(
        pg.recovery_report().uncommitted_tail_frames >= 1,
        "the persisted-but-uncommitted frame must be counted in recovery_report: {:?}",
        pg.recovery_report()
    );
    let s = pg.snapshot();
    let acc = DeltaAccessor::new(s.base.as_accessor(), s.delta.as_ref());
    assert!(
        acc.get_node("mark-3-0").is_none(),
        "ambiguity frame leaked in"
    );
    assert!(acc.get_node("mark-1-0").is_some());
    assert!(acc.get_node("mark-2-0").is_some());
}

/// Documents the IRREDUCIBLE single-fsync ambiguity contract (the COMMIT-side
/// sibling of `crown_jewel`): a COMMIT-slot fsync returns Err while the fully
/// written 44-byte slot still PERSISTS (kernel writeback), so the on-disk marker
/// advances and an `apply()`-Err batch CAN be visible on reopen. No on-disk means
/// distinguishes fsync-fails-but-persists from fsync-success, so invisibility is
/// UNACHIEVABLE here — we assert the ACHIEVABLE invariant instead: reopen yields a
/// CONSISTENT committed prefix (committed_seq is EXACTLY the pre-batch OR the batch
/// seq, and the content is exactly that prefix) and no acked batch is lost.
#[test]
fn commit_fsync_persist_makes_apply_err_indeterminate_but_prefix_consistent() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path().to_path_buf();
    let vfs = Arc::new(FaultVfs::new());

    {
        let pg = PersistentGraph::create_with_vfs(vfs.clone(), &dir, root_base()).unwrap();
        pg.set_fsync_policy(FsyncPolicy::EveryBatch);
        pg.apply(batch_ops(1, 1).0, None).unwrap(); // committed 1 (acked)
        pg.apply(batch_ops(2, 1).0, None).unwrap(); // committed 2 (acked)
        assert_eq!(pg.committed_seq(), 2);

        // seq 3: the WAL frame fsyncs fine (durable), but the COMMIT slot's OWN
        // fsync fails AND the slot persists anyway. apply() Errs and the in-memory
        // marker stays at 2 — yet the on-disk slot now reads committed_seq = 3.
        vfs.fail_next_fsync("COMMIT", true);
        let err = pg.apply(batch_ops(3, 1).0, None);
        assert!(
            err.is_err(),
            "the COMMIT-fsync-failed apply must return Err"
        );
        assert!(
            pg.committed_seq() == 2 && pg.seq() == 2,
            "an Err'd apply must NOT advance the in-memory marker/seq (poisoned)"
        );

        // Drop the un-synced tail. The persisted COMMIT slot and the durable WAL
        // frame 3 are already folded into `durable`, so both survive the cut.
        vfs.power_cut_dropping_unsynced();
        drop(pg); // abandon WITHOUT close (a crash)
    }

    let pg = PersistentGraph::open_with_vfs(
        vfs.clone(),
        &dir,
        false,
        BaseMode::Mmap,
        Validate::Full,
        false,
    )
    // (a) open must succeed — never Corrupt — after a pure power cut.
    .expect("reopen after a COMMIT-fsync-persist power cut must never be Corrupt");

    // (b) committed_seq is EITHER the pre-batch seq (2, invisible direction) OR
    //     the batch seq (3, visible direction) — both legal under the indeterminate
    //     contract, and NOTHING else. Recovery opens seq == committed_seq.
    let r = pg.committed_seq();
    assert_eq!(pg.seq(), r, "recovery must open seq == committed_seq");
    assert!(
        r == 2 || r == 3,
        "committed_seq {r} must be exactly 2 (invisible) or 3 (visible) — indeterminate"
    );

    let s = pg.snapshot();
    let acc = DeltaAccessor::new(s.base.as_accessor(), s.delta.as_ref());

    // (d) both acked batches MUST survive regardless of the direction taken.
    assert!(
        acc.get_node("mark-1-0").is_some(),
        "acked batch 1 must survive"
    );
    assert!(
        acc.get_node("mark-2-0").is_some(),
        "acked batch 2 must survive"
    );

    // (c) the content is EXACTLY the committed prefix `r` — no torn/half-applied
    //     batch: batch 3 present iff r == 3, and the counts + the last edge pin it.
    if r == 3 {
        assert!(
            acc.get_node("mark-3-0").is_some(),
            "prefix 3 => the (visible) batch 3 node must be present"
        );
    } else {
        assert!(
            acc.get_node("mark-3-0").is_none(),
            "prefix 2 => batch 3 must be invisible"
        );
    }
    assert_eq!(
        acc.node_count(),
        1 + r as usize,
        "consistent prefix: node_count == root + one node per committed batch"
    );
    assert_eq!(
        acc.edge_count(),
        r as usize,
        "consistent prefix: edge_count == one edge per committed batch (no torn edge)"
    );
    for i in 1..=r {
        assert!(
            acc.outgoing_neighbors(&format!("mark-{i}-0"))
                .iter()
                .any(|nb| nb.target_name == "root"),
            "committed edge for seq {i} must point to root (whole batch materialized)"
        );
    }
}

/// Cut mid-COMMIT-slot write: the slot is torn, so the PREVIOUS slot wins and
/// the frame past the (un-advanced) marker is not replayed.
#[test]
fn cut_mid_commit_slot_previous_slot_wins() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path().to_path_buf();
    let vfs = Arc::new(FaultVfs::new());

    {
        let pg = PersistentGraph::create_with_vfs(vfs.clone(), &dir, root_base()).unwrap();
        pg.set_fsync_policy(FsyncPolicy::EveryBatch);
        pg.apply(batch_ops(1, 1).0, None).unwrap(); // committed 1, COMMIT slot durable
        assert_eq!(pg.committed_seq(), 1);

        // seq 2: the WAL frame fsyncs fine (durable), but the COMMIT slot write's
        // fsync fails and does NOT persist — the slot write is un-synced.
        vfs.fail_next_fsync("COMMIT", false);
        assert!(pg.apply(batch_ops(2, 1).0, None).is_err());
        assert_eq!(pg.committed_seq(), 1, "marker must not advance");

        // Torn the un-synced slot to a 20-byte prefix -> its crc fails.
        vfs.power_cut_torn_keep(20);
        drop(pg);
    }

    let pg = PersistentGraph::open_with_vfs(
        vfs.clone(),
        &dir,
        false,
        BaseMode::Mmap,
        Validate::Full,
        false,
    )
    .expect("torn COMMIT slot must fall back to the previous slot, not Corrupt");
    assert_eq!(pg.seq(), 1, "the previous (intact) COMMIT slot wins");
    assert!(
        pg.recovery_report().uncommitted_tail_frames >= 1,
        "the durable frame past the un-advanced marker must be counted"
    );
    let s = pg.snapshot();
    let acc = DeltaAccessor::new(s.base.as_accessor(), s.delta.as_ref());
    assert!(acc.get_node("mark-2-0").is_none());
    assert!(acc.get_node("mark-1-0").is_some());
}

/// Cut between the snapshot rename and the COMMIT advance during compaction:
/// snapshot + MANIFEST are durable, the marker lags — recovery uses the
/// `snapshot_seq` floor to recover the folded state.
#[test]
fn cut_between_snapshot_and_commit_advance_uses_snapshot_seq_floor() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path().to_path_buf();
    let vfs = Arc::new(FaultVfs::new());

    {
        let pg = PersistentGraph::create_with_vfs(vfs.clone(), &dir, root_base()).unwrap();
        // OnFlush: applies alone never commit, so ONLY the compaction can.
        for i in 1..=5 {
            pg.apply(batch_ops(i, 1).0, None).unwrap();
        }
        assert_eq!(
            pg.committed_seq(),
            0,
            "OnFlush: nothing committed by applies"
        );

        // Compaction writes snapshot + MANIFEST durably, then the COMMIT advance
        // fsync fails: the crash lands in that window.
        vfs.fail_next_fsync("COMMIT", false);
        assert!(pg.compact().is_err(), "the COMMIT advance must fail");
        vfs.power_cut_dropping_unsynced();
        drop(pg);
    }

    let pg = PersistentGraph::open_with_vfs(
        vfs.clone(),
        &dir,
        false,
        BaseMode::Mmap,
        Validate::Full,
        false,
    )
    .expect("snapshot_seq floor must recover the folded state, not Corrupt");
    assert_eq!(
        pg.seq(),
        5,
        "folded state recovered via the snapshot_seq floor"
    );
    assert_eq!(pg.committed_seq(), 5);
    let s = pg.snapshot();
    let acc = DeltaAccessor::new(s.base.as_accessor(), s.delta.as_ref());
    for i in 1..=5 {
        assert!(
            acc.get_node(&format!("mark-{i}-0")).is_some(),
            "seq {i} must be present in the folded snapshot"
        );
    }
    assert_eq!(acc.node_count(), 1 + 5);
}

/// Cut before `fsync_dir` after a rename: the rename must be undone. A
/// compaction whose snapshot rename never reaches the disk leaves the store on
/// its pre-compaction snapshot; the new snapshot file is gone.
#[test]
fn cut_before_fsync_dir_after_rename_undoes_the_rename() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path().to_path_buf();
    let vfs = Arc::new(FaultVfs::new());

    let old_snap = "snapshot-00000000000000000000-0000000000.og"; // create(): seq 0, cid 0
    let new_snap = "snapshot-00000000000000000004-0000000001.og"; // compact: fold 4, cid 1

    {
        let pg = PersistentGraph::create_with_vfs(vfs.clone(), &dir, root_base()).unwrap();
        pg.set_fsync_policy(FsyncPolicy::EveryBatch);
        for i in 1..=4 {
            pg.apply(batch_ops(i, 1).0, None).unwrap(); // committed 1..=4
        }
        assert_eq!(pg.committed_seq(), 4);

        // The compaction's snapshot rename lands, but its fsync_dir fails -> the
        // rename never becomes durable.
        vfs.fail_next_dir_fsync();
        assert!(pg.compact().is_err(), "the snapshot fsync_dir must fail");
        vfs.power_cut_dropping_unsynced(); // undoes the un-dir-synced rename
        drop(pg);
    }

    let pg = PersistentGraph::open_with_vfs(
        vfs.clone(),
        &dir,
        false,
        BaseMode::Mmap,
        Validate::Full,
        false,
    )
    .expect("an undone-rename store must reopen on its old snapshot, not Corrupt");
    assert_eq!(
        pg.seq(),
        4,
        "committed prefix recovered via the old snapshot + WAL"
    );
    let s = pg.snapshot();
    let acc = DeltaAccessor::new(s.base.as_accessor(), s.delta.as_ref());
    for i in 1..=4 {
        assert!(acc.get_node(&format!("mark-{i}-0")).is_some());
    }
    // The rename was undone: the new snapshot file is gone, the old one remains.
    assert!(
        !dir.join(new_snap).exists(),
        "the un-dir-synced new snapshot must have been removed"
    );
    assert!(dir.join(old_snap).exists(), "the old snapshot must survive");
    let og_count = std::fs::read_dir(&dir)
        .unwrap()
        .flatten()
        .filter_map(|e| e.file_name().into_string().ok())
        .filter(|n| n.starts_with("snapshot-") && n.ends_with(".og"))
        .count();
    assert_eq!(
        og_count, 1,
        "exactly one snapshot must remain after the undo"
    );
}
