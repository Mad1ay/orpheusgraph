//! kill -9 crash harness (spec §6 — "the load-bearing one").
//!
//! A child process opens the store and applies monotonic marker batches in a
//! tight loop, appending each acknowledged seq to a ground-truth `acked.log`.
//! The parent spawns it, SIGKILLs it at varied offsets, reopens, and asserts:
//!   (i)   open() after a real kill -9 always succeeds (never panics on bytes a
//!         real interruption produced);
//!   (ii)  the recovered marker set is a clean PREFIX — marker `seq` present,
//!         marker `seq+1` absent (no hole, no phantom beyond the last durable);
//!   (iii) recovered `seq >= max_acked`: process death loses NOTHING that
//!         apply() acked, under ANY fsync policy — the OS page cache survives a
//!         process crash (spec §4.2; power loss is a different, un-covered case);
//!   (iv)  seq is monotonic across incarnations and the epoch re-mints on each
//!         unclean reopen.
//!
//! The child is this same test binary re-exec'd with `OG_CRASH_CHILD=1`.
//! Iterations: `OG_CRASH_ITERS` (default 24/policy; run ×100 for CI).

use orpheusgraph::accessor::GraphAccessor;
use orpheusgraph::types::NodeData;
use orpheusgraph::{
    build_graph, DeltaAccessor, FsyncPolicy, Op, OrpheusGraphInner, PersistentGraph,
};
use std::collections::HashMap;
use std::io::Write;
use std::path::Path;

fn marker_op(n: u64) -> Op {
    Op::UpsertNode(NodeData {
        name: format!("crash-mark-{n}"),
        kind: "marker".into(),
        metadata: HashMap::new(),
        base_weight: 0.5,
        noise_penalty: 0.0,
        pagerank_weight: 0.0,
    })
}

fn child_loop(dir: &Path, policy: FsyncPolicy) -> ! {
    let pg = PersistentGraph::open(dir, false).expect("child: open store");
    pg.set_fsync_policy(policy);
    // Force frequent compaction so the parent's SIGKILLs (1-40ms apart) land
    // inside compact_locked's non-atomic on-disk sequence (snapshot write,
    // MANIFEST rename, WAL truncate, re-mmap) — the riskiest Phase 2b path.
    pg.set_auto_compact_threshold(Some(8));
    let mut acked = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(dir.join("acked.log"))
        .expect("child: open acked.log");
    let mut n = pg.seq() + 1;
    loop {
        match pg.apply(vec![marker_op(n)], None) {
            Ok(seq) => {
                // Ground-truth ledger: record what apply() ACKed, fsync so the
                // parent reads the true high-water mark even after the kill.
                // FIXED-WIDTH 20-digit lines: a kill mid-write leaves a short
                // torn record, and a following append then forms a >20-digit
                // line that overflows u64 parsing — so `max_acked` rejects both
                // torn and concatenated records instead of misreading them.
                let _ = writeln!(acked, "{seq:020}");
                let _ = acked.sync_all();
                n = seq + 1;
            }
            // A poison/CAS/etc. is unexpected here; exit non-zero so a hang or
            // logic error is visible rather than silently spinning.
            Err(_) => std::process::exit(2),
        }
    }
}

fn max_acked(dir: &Path) -> u64 {
    std::fs::read_to_string(dir.join("acked.log"))
        .unwrap_or_default()
        .lines()
        // Only exactly-20-digit records are intact (torn=short, concat=long);
        // parse still guards against a 20-digit value > u64::MAX.
        .filter(|l| l.len() == 20 && l.bytes().all(|b| b.is_ascii_digit()))
        .filter_map(|l| l.parse::<u64>().ok())
        .max()
        .unwrap_or(0)
}

#[test]
fn kill9_crash_harness() {
    // ---- CHILD entry ----
    if std::env::var("OG_CRASH_CHILD").is_ok() {
        let dir = std::env::var("OG_CRASH_DIR").expect("child: OG_CRASH_DIR");
        let policy = match std::env::var("OG_CRASH_POLICY").as_deref() {
            Ok("every") => FsyncPolicy::EveryBatch,
            _ => FsyncPolicy::OnFlush,
        };
        child_loop(Path::new(&dir), policy);
    }

    // ---- PARENT ----
    let iters: u32 = std::env::var("OG_CRASH_ITERS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(24);
    let exe = std::env::current_exe().expect("current_exe");

    for policy_name in ["every", "onflush"] {
        let tmp = tempfile::tempdir().expect("tempdir");
        let dir = tmp.path().to_path_buf();

        // A fresh, cleanly-closed valid store for the first child to open.
        let (g, idx) = build_graph(Vec::new(), Vec::new());
        PersistentGraph::create(&dir, OrpheusGraphInner::new(g, idx))
            .expect("create")
            .close()
            .expect("close");

        let mut last_seq: u64 = 0;
        let mut last_epoch: u128 = 0;

        for i in 0..iters {
            let mut child = std::process::Command::new(&exe)
                .arg("kill9_crash_harness")
                .arg("--exact")
                .env("OG_CRASH_CHILD", "1")
                .env("OG_CRASH_DIR", &dir)
                .env("OG_CRASH_POLICY", policy_name)
                .stdout(std::process::Stdio::null())
                .stderr(std::process::Stdio::null())
                .spawn()
                .expect("spawn child");

            // Varied offset so kills land at different points of the write loop
            // (and occasionally almost immediately, near the child's own open()).
            let ms = 1 + (u64::from(i) * 7) % 40;
            std::thread::sleep(std::time::Duration::from_millis(ms));

            child.kill().expect("SIGKILL child"); // std kill() == SIGKILL on unix
            let _ = child.wait();

            // (i) reopen after a real kill -9 must succeed, never panic on bytes.
            let pg = PersistentGraph::open(&dir, false).unwrap_or_else(|e| {
                panic!("[{policy_name} #{i}] reopen after kill -9 failed: {e}")
            });
            let seq = pg.seq();
            let acked = max_acked(&dir);

            // (ii) clean prefix: boundary marker present, nothing beyond seq.
            let s = pg.snapshot();
            let acc = DeltaAccessor::new(s.base.as_accessor(), s.delta.as_ref());
            if seq > 0 {
                assert!(
                    acc.get_node(&format!("crash-mark-{seq}")).is_some(),
                    "[{policy_name} #{i}] boundary marker crash-mark-{seq} missing (hole in prefix)"
                );
            }
            assert!(
                acc.get_node(&format!("crash-mark-{}", seq + 1)).is_none(),
                "[{policy_name} #{i}] phantom marker crash-mark-{} beyond recovered seq {seq}",
                seq + 1
            );

            // (iii) process death loses nothing acked, under ANY policy.
            assert!(
                seq >= acked,
                "[{policy_name} #{i}] recovered seq {seq} < max_acked {acked}: kill -9 lost an acked batch"
            );

            // (iv) monotonic across incarnations + epoch re-mint on unclean reopen.
            assert!(
                seq >= last_seq,
                "[{policy_name} #{i}] seq regressed {last_seq} -> {seq}"
            );
            if i > 0 {
                assert!(
                    pg.epoch() != last_epoch,
                    "[{policy_name} #{i}] epoch not re-minted on unclean reopen"
                );
            }
            last_seq = seq;
            last_epoch = pg.epoch();

            drop(pg); // release the flock before the next child opens
        }

        assert!(
            last_seq > 0,
            "[{policy_name}] no progress was ever made — child never applied a batch"
        );
        // Prove compaction actually ran (and survived the kills): create() writes
        // cid=0 (snapshot-...-0000000000.og); every compaction increments the cid,
        // so a live/orphan snapshot with cid != 0 means compact_locked ran at
        // least once across the crash-interrupted incarnations.
        let compacted = std::fs::read_dir(&dir)
            .unwrap()
            .flatten()
            .filter_map(|e| e.file_name().into_string().ok())
            .any(|n| {
                n.starts_with("snapshot-") && n.ends_with(".og") && !n.contains("-0000000000.og")
            });
        assert!(
            compacted,
            "[{policy_name}] no compacted snapshot (cid>0) found — compaction never fired under crashes"
        );
    }
}
