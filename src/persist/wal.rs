//! Write-ahead log: framing + poison-aware append writer + a recovery scan that
//! NEVER allocates from an unvalidated length and NEVER panics on corrupt bytes
//! (§4.2). This module is pure framing + I/O — it knows nothing about `ArcSwap`.
//!
//! ## Frame layout (little-endian), one frame per `apply()` batch
//! ```text
//! frame   := [len: u32 LE] [crc32: u32 LE] [payload: len bytes]
//! payload := postcard(WalRecord{ seq: u64, ops: Vec<Op> })
//! ```
//! `len` counts the payload only (the 8-byte header is excluded). The crc32
//! covers `len_le_bytes ‖ payload` — NOT the payload alone — so a torn 4-byte
//! length prefix (a real power-loss artifact that would otherwise decode as an
//! arbitrary length up to 4 GiB) fails the crc instead of driving a giant
//! read/alloc. `seq` strictly increases by exactly 1 per frame.

use std::fs::File;
use std::io::{Read, Seek, SeekFrom, Write};

use crc32fast::Hasher;

use super::FsyncPolicy;
use crate::delta::Op;
use crate::persist::error::PersistError;

/// One durable batch: the commit seq plus the ops that were applied. `Op`
/// already derives `serde`, so this needs no new derive work.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct WalRecord {
    pub seq: u64,
    pub ops: Vec<Op>,
}

/// Encode a record into a length-prefixed, crc-guarded frame.
pub fn encode_frame(rec: &WalRecord) -> Result<Vec<u8>, PersistError> {
    let payload = postcard::to_stdvec(rec)
        .map_err(|e| PersistError::Corrupt(format!("postcard WAL encode failed: {e}")))?;
    if payload.len() > u32::MAX as usize {
        return Err(PersistError::Corrupt(format!(
            "WAL frame payload {} bytes exceeds u32::MAX",
            payload.len()
        )));
    }
    let len = payload.len() as u32;
    let mut h = Hasher::new();
    h.update(&len.to_le_bytes());
    h.update(&payload);
    let crc = h.finalize();

    let mut out = Vec::with_capacity(8 + payload.len());
    out.extend_from_slice(&len.to_le_bytes());
    out.extend_from_slice(&crc.to_le_bytes());
    out.extend_from_slice(&payload);
    Ok(out)
}

/// Outcome of scanning a WAL byte buffer. A torn/short/over-long/crc-bad tail
/// is reported (never errored) so recovery can truncate it; a crc-VALID frame
/// that fails to decode is a hard `Corrupt` (returned as `Err`, not here).
#[derive(Debug)]
pub struct ScanResult {
    /// Every intact, in-order record up to the first bad frame or clean EOF.
    pub records: Vec<WalRecord>,
    /// Byte offset just past the last intact frame — the truncation boundary
    /// and the position at which the next append must land.
    pub valid_end: u64,
    /// True iff trailing bytes were found that do not form an intact frame.
    pub tail_truncated: bool,
    /// Number of trailing frames dropped (0 or 1 — a torn tail is one frame).
    pub dropped: usize,
}

/// Scan a WAL byte buffer per the decode/scan rules of §4.2. Bounds are checked
/// BEFORE any payload allocation. Returns `Err(Corrupt)` only for a crc-valid
/// frame whose payload fails to decode (intact bytes, real corruption).
pub fn scan_wal_bytes(buf: &[u8]) -> Result<ScanResult, PersistError> {
    let total = buf.len();
    let mut offset: usize = 0;
    let mut records: Vec<WalRecord> = Vec::new();

    loop {
        let remaining = total - offset;

        // 1. Not even a full 8-byte header left.
        if remaining == 0 {
            return Ok(ScanResult {
                records,
                valid_end: offset as u64,
                tail_truncated: false,
                dropped: 0,
            });
        }
        if remaining < 8 {
            return Ok(ScanResult {
                records,
                valid_end: offset as u64,
                tail_truncated: true,
                dropped: 1,
            });
        }

        // 2. Read header WITHOUT unwrap (indices are in-bounds: remaining >= 8).
        let len = u32::from_le_bytes([
            buf[offset],
            buf[offset + 1],
            buf[offset + 2],
            buf[offset + 3],
        ]);
        let crc_stored = u32::from_le_bytes([
            buf[offset + 4],
            buf[offset + 5],
            buf[offset + 6],
            buf[offset + 7],
        ]);

        // 3. BOUNDS CHECK FIRST — before touching a `len`-sized buffer. A torn
        //    or over-long length (e.g. 0xFFFF_FFFF) is a tail, never an alloc.
        if len as u64 > (remaining as u64 - 8) {
            return Ok(ScanResult {
                records,
                valid_end: offset as u64,
                tail_truncated: true,
                dropped: 1,
            });
        }

        // 4. Only now slice exactly `len` payload bytes.
        let payload = &buf[offset + 8..offset + 8 + len as usize];

        // 5. Recompute crc over len_le_bytes ‖ payload.
        let mut h = Hasher::new();
        h.update(&len.to_le_bytes());
        h.update(payload);
        if h.finalize() != crc_stored {
            return Ok(ScanResult {
                records,
                valid_end: offset as u64,
                tail_truncated: true,
                dropped: 1,
            });
        }

        // 6. crc passed => bytes are intact; a decode failure is REAL corruption.
        let rec: WalRecord = postcard::from_bytes(payload).map_err(|e| {
            PersistError::Corrupt(format!(
                "WAL frame at offset {offset}: crc-valid but postcard decode failed: {e}"
            ))
        })?;

        // 7. Advance and yield.
        offset += 8 + len as usize;
        records.push(rec);
    }
}

/// Read the whole WAL file and scan it. The WAL is small in 2a; the read is
/// bounded by the real file size (never by an on-disk length field).
pub fn read_and_scan(file: &mut File) -> Result<ScanResult, PersistError> {
    file.seek(SeekFrom::Start(0))?;
    let mut buf = Vec::new();
    file.read_to_end(&mut buf)?;
    scan_wal_bytes(&buf)
}

/// Append-only WAL writer. `poisoned` is a plain bool (touched only under the
/// owning writer Mutex) — any append/fsync failure sets it, which is what makes
/// a torn frame provably the LAST frame (§4.2).
pub struct WalWriter {
    file: File,
    policy: FsyncPolicy,
    since_fsync: u32,
    pub(crate) poisoned: bool,
    pub(crate) len: u64,
    /// Test-only seam: force the next `append` to fail (simulate a short write
    /// / device error) so poison propagation can be exercised deterministically.
    #[cfg(test)]
    pub(crate) fail_next: bool,
}

impl WalWriter {
    /// Wrap an already-opened (append-mode) file positioned at `len` bytes.
    pub fn new(file: File, policy: FsyncPolicy, len: u64) -> Self {
        Self {
            file,
            policy,
            since_fsync: 0,
            poisoned: false,
            len,
            #[cfg(test)]
            fail_next: false,
        }
    }

    /// Set the fsync policy (runtime knob; not persisted).
    pub fn set_policy(&mut self, policy: FsyncPolicy) {
        self.policy = policy;
    }

    /// Append one frame. ANY failure poisons the writer. Append is the commit
    /// point: the caller advances in-memory seq/delta only after this returns
    /// `Ok`. fsync happens here per policy (`EveryBatch` always; `EveryN` on the
    /// counter; `OnFlush` never here — durability then comes from `flush()`).
    ///
    /// **fsync-failure caveat (honest limitation).** If `write_all` succeeds but
    /// the subsequent `fsync` fails, this returns `Err` and the caller treats the
    /// batch as *not committed* — yet a complete, crc-valid frame is already in
    /// the OS page cache and the kernel MAY still write it back. So on the next
    /// `open()` recovery can legitimately replay a frame whose `apply()` returned
    /// `Err`. Consequence: **"apply returned Err" does NOT guarantee "not
    /// durable" when the failure was an fsync error.** A caller that must know
    /// the true committed state after a write error should reopen and read
    /// `(epoch, seq)` to reconcile. Making `apply`-Err strictly equal
    /// not-durable would need a per-frame commit/torn-write marker — deferred
    /// (logged in docs/persistence_impl_log.md and the spec durability section).
    pub fn append(&mut self, frame: &[u8]) -> Result<(), PersistError> {
        #[cfg(test)]
        if self.fail_next {
            // Injected BEFORE any write: nothing hits disk, len unchanged.
            self.fail_next = false;
            self.poisoned = true;
            return Err(PersistError::Io(std::io::Error::new(
                std::io::ErrorKind::Other,
                "injected WAL append failure",
            )));
        }

        if let Err(e) = self.file.write_all(frame) {
            self.poisoned = true;
            return Err(PersistError::Io(e));
        }
        self.len += frame.len() as u64;

        match self.policy {
            FsyncPolicy::EveryBatch => self.fsync_now()?,
            FsyncPolicy::EveryN(n) => {
                self.since_fsync += 1;
                if self.since_fsync >= n.max(1) {
                    self.fsync_now()?;
                }
            }
            FsyncPolicy::OnFlush => {}
        }
        Ok(())
    }

    /// Force the WAL to durable storage now (the durability point under
    /// `OnFlush`). Poison-checked and idempotent.
    pub fn flush(&mut self) -> Result<(), PersistError> {
        if self.poisoned {
            return Err(PersistError::Poisoned);
        }
        self.fsync_now()
    }

    /// Truncate the WAL to empty for the compaction WAL-rotate step (§4.4).
    /// All frames `<= snapshot_seq` have been folded into the new base, so the
    /// log is reset: `set_len(0)` + seek to 0 + reset `len` + fsync. The next
    /// `append` lands at offset 0. Any failure poisons the writer.
    ///
    /// Ordering: this runs AFTER the new MANIFEST is durably renamed (the
    /// compaction commit point). A crash between the MANIFEST rename and this
    /// truncate leaves stale frames `<= snapshot_seq` in the log; recovery skips
    /// them (they are already folded) and the next compaction re-truncates.
    pub fn truncate(&mut self) -> Result<(), PersistError> {
        if self.poisoned {
            return Err(PersistError::Poisoned);
        }
        if let Err(e) = self.file.set_len(0) {
            self.poisoned = true;
            return Err(PersistError::Io(e));
        }
        if let Err(e) = self.file.seek(SeekFrom::Start(0)) {
            self.poisoned = true;
            return Err(PersistError::Io(e));
        }
        self.len = 0;
        self.since_fsync = 0;
        if let Err(e) = self.file.sync_all() {
            self.poisoned = true;
            return Err(PersistError::Io(e));
        }
        Ok(())
    }

    fn fsync_now(&mut self) -> Result<(), PersistError> {
        if let Err(e) = self.file.sync_all() {
            self.poisoned = true;
            return Err(PersistError::Io(e));
        }
        self.since_fsync = 0;
        Ok(())
    }

    #[cfg(test)]
    pub(crate) fn arm_fail(&mut self) {
        self.fail_next = true;
    }
}

// ===========================================================================
// Framing / scan unit tests (pure, precise — no on-disk dirs needed).
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{EdgeData, NodeData};
    use std::collections::HashMap;

    fn node(name: &str) -> NodeData {
        NodeData {
            name: name.into(),
            kind: "model".into(),
            metadata: HashMap::new(),
            base_weight: 0.5,
            noise_penalty: 0.0,
            pagerank_weight: 0.0,
        }
    }

    fn all_four_ops() -> Vec<Op> {
        vec![
            Op::UpsertNode(node("a")),
            Op::AddEdge {
                from: "a".into(),
                to: "b".into(),
                edge: EdgeData {
                    kind: "rel".into(),
                    field_name: Some("f".into()),
                    base_weight: 1.0,
                },
            },
            Op::RemoveNode { name: "c".into() },
            Op::RemoveEdge {
                from: "a".into(),
                to: "b".into(),
                kind: "rel".into(),
            },
        ]
    }

    #[test]
    fn postcard_round_trip_all_op_variants() {
        // Encode -> scan -> re-encode is byte-identical (postcard is
        // deterministic), and the decoded record carries all four variants.
        let rec = WalRecord {
            seq: 7,
            ops: all_four_ops(),
        };
        let frame = encode_frame(&rec).unwrap();
        let scan = scan_wal_bytes(&frame).unwrap();
        assert!(!scan.tail_truncated);
        assert_eq!(scan.records.len(), 1);
        let got = &scan.records[0];
        assert_eq!(got.seq, 7);
        assert_eq!(got.ops.len(), 4);
        assert!(matches!(got.ops[0], Op::UpsertNode(_)));
        assert!(matches!(got.ops[1], Op::AddEdge { .. }));
        assert!(matches!(got.ops[2], Op::RemoveNode { .. }));
        assert!(matches!(got.ops[3], Op::RemoveEdge { .. }));
        // Identity at the wire level.
        let re = encode_frame(got).unwrap();
        assert_eq!(re, frame);
    }

    #[test]
    fn scan_multiple_frames_in_order() {
        let mut buf = Vec::new();
        for s in 1..=3u64 {
            buf.extend_from_slice(
                &encode_frame(&WalRecord {
                    seq: s,
                    ops: vec![Op::UpsertNode(node(&format!("n{s}")))],
                })
                .unwrap(),
            );
        }
        let scan = scan_wal_bytes(&buf).unwrap();
        assert_eq!(scan.records.len(), 3);
        assert_eq!(scan.valid_end, buf.len() as u64);
        assert!(!scan.tail_truncated);
        assert_eq!(
            scan.records.iter().map(|r| r.seq).collect::<Vec<_>>(),
            vec![1, 2, 3]
        );
    }

    #[test]
    fn empty_buffer_is_clean_eof() {
        let scan = scan_wal_bytes(&[]).unwrap();
        assert_eq!(scan.records.len(), 0);
        assert_eq!(scan.valid_end, 0);
        assert!(!scan.tail_truncated);
        assert_eq!(scan.dropped, 0);
    }

    #[test]
    fn short_header_tail_is_truncated_not_errored() {
        let mut buf = encode_frame(&WalRecord {
            seq: 1,
            ops: vec![Op::UpsertNode(node("a"))],
        })
        .unwrap();
        let boundary = buf.len() as u64;
        buf.extend_from_slice(&[0xAB, 0xCD, 0xEF]); // 3 stray bytes < 8-byte header
        let scan = scan_wal_bytes(&buf).unwrap();
        assert_eq!(scan.records.len(), 1);
        assert!(scan.tail_truncated);
        assert_eq!(scan.dropped, 1);
        assert_eq!(scan.valid_end, boundary);
    }

    #[test]
    fn over_long_len_is_torn_tail_without_allocating() {
        // Header claims 0xFFFF_FFFF payload bytes on an 8-byte file. The bounds
        // check must fire BEFORE any Vec::with_capacity(len); this returns
        // immediately on a tiny buffer with no OOM/panic.
        let mut buf = Vec::new();
        buf.extend_from_slice(&0xFFFF_FFFFu32.to_le_bytes());
        buf.extend_from_slice(&0xDEAD_BEEFu32.to_le_bytes()); // garbage crc
        let scan = scan_wal_bytes(&buf).unwrap();
        assert_eq!(scan.records.len(), 0);
        assert!(scan.tail_truncated);
        assert_eq!(scan.valid_end, 0);
    }

    #[test]
    fn flipped_payload_bit_fails_crc_and_truncates() {
        let good = encode_frame(&WalRecord {
            seq: 1,
            ops: vec![Op::UpsertNode(node("a"))],
        })
        .unwrap();
        let mut buf = good.clone();
        // Flip a bit in the payload (byte 8 is the first payload byte).
        let last = buf.len() - 1;
        buf[last] ^= 0x01;
        let scan = scan_wal_bytes(&buf).unwrap();
        // crc mismatch => whole frame is a torn tail, not a decode-level Corrupt.
        assert_eq!(scan.records.len(), 0);
        assert!(scan.tail_truncated);
        assert_eq!(scan.valid_end, 0);
    }

    #[test]
    fn good_frame_then_flipped_tail_keeps_prefix() {
        let f1 = encode_frame(&WalRecord {
            seq: 1,
            ops: vec![Op::UpsertNode(node("a"))],
        })
        .unwrap();
        let mut f2 = encode_frame(&WalRecord {
            seq: 2,
            ops: vec![Op::UpsertNode(node("b"))],
        })
        .unwrap();
        let boundary = f1.len() as u64;
        // Corrupt f2's crc byte.
        f2[4] ^= 0xFF;
        let mut buf = f1;
        buf.extend_from_slice(&f2);
        let scan = scan_wal_bytes(&buf).unwrap();
        assert_eq!(scan.records.len(), 1);
        assert_eq!(scan.records[0].seq, 1);
        assert!(scan.tail_truncated);
        assert_eq!(scan.valid_end, boundary);
    }
}
