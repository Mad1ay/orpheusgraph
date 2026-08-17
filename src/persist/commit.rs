//! COMMIT sidecar — the authoritative durable-commit high-water mark that
//! recovery trusts INSTEAD of "whatever crc-valid WAL frames survived" (§4.2/§4.3).
//!
//! ## Why this exists (the fsync-failure ambiguity — and its irreducible limit)
//! A WAL frame `write_all` can succeed while its `fsync` FAILS. The frame is
//! then a complete, crc-valid record in the OS page cache that the kernel MAY
//! still write back later, so recovery — which replays every crc-valid frame —
//! could otherwise resurrect a batch whose `apply()` returned `Err`. The same is
//! true of any frame that was only write-through and never fsync'd (the
//! `OnFlush` "lucky tail"). This marker is a separate, fsync-gated high-water
//! mark: recovery replays frames only up to `committed_seq` and treats crc-valid
//! frames beyond it as the artifact (truncated + counted). That makes the two
//! WAL-side directions INVISIBLE — a WAL fsync that fails, or a never-fsync'd
//! tail, does NOT advance the marker, so its frame stays above-marker and is
//! discarded (the `crown_jewel` power-loss test).
//!
//! ## What the marker does NOT do: it RELOCATES the ambiguity, not removes it
//! The COMMIT slot's OWN `fsync` (in [`CommitFile::advance`]) has the IDENTICAL
//! fails-but-persists property: it can return `Err` while the fully-written
//! 44-byte slot still reaches stable storage via later writeback. When that
//! happens the on-disk marker reads `committed_seq = N` even though `apply()`
//! returned `Err` and left the in-memory seq at `N-1` — so on reopen the
//! highest-`gen` crc-valid slot wins, `commit_hw = N`, and the Err'd batch's
//! (already durably fsync'd) WAL frame is REPLAYED and VISIBLE. No on-disk means
//! can distinguish fsync-fails-but-persists from fsync-success for ANY single
//! fsync, so this is irreducible — the two slots + crc defend only against a
//! TORN slot, never against a fully-written slot whose fsync merely erred.
//!
//! ## The honest contract
//! "Below-marker" therefore means "acknowledged durable on the `apply()`-`Ok`
//! path", NOT "every write's fsync literally succeeded". Concretely:
//! * `apply()`-`Ok`  ⇒ durable (the WAL frame fsync'd AND a marker covering it
//!   fsync'd). The marker STRENGTHENS this direction and it stays strictly true.
//! * `apply()`-`Err` ⇒ durability INDETERMINATE: the batch may be visible on
//!   reopen (this COMMIT-fsync-persist path) or invisible (the WAL-fsync-fails
//!   path). What recovery ALWAYS guarantees is (1) the store opens to a CONSISTENT
//!   committed prefix (batches are all-or-nothing — no torn/half-applied batch)
//!   and (2) no `Ok`-acked batch is ever lost. A caller reconciles an `Err` by
//!   reading `committed_seq()` / `recovery_report()` after reopen, NOT by blind
//!   retry (re-applying a non-idempotent batch on the persist path double-applies).
//!
//! See the "Honest apply() durability contract" entry in
//! `docs/persistence_impl_log.md` for the trace, theory and resolution.
//!
//! ## On-disk format — torn-write-safe two-slot ping-pong
//! Two fixed-size records at fixed offsets, written ALTERNATELY. A torn write
//! to one slot cannot corrupt the other, so the previous good record always
//! survives; a reader picks the crc-valid slot with the highest `gen` and one
//! valid slot is enough. `gen` (a monotonic per-write counter) is the freshness
//! key rather than `(epoch, committed_seq)`: `epoch` is a random `u128` (not
//! chronologically monotonic), so it cannot order two slots written across an
//! epoch re-mint, and `committed_seq` ties on an open-time re-stamp — `gen`
//! disambiguates both unambiguously (see the impl-log entry for this phase).
//!
//! ```text
//! slot := magic(4) ‖ version(4) ‖ gen(8) ‖ epoch(16) ‖ committed_seq(8) ‖ crc32(4)
//!         = 44 bytes, little-endian; crc32 covers the first 40 bytes.
//! file := slot0 ‖ slot1 = 88 bytes.
//! ```

use std::io::SeekFrom;
use std::path::Path;

use crc32fast::Hasher;

use crate::persist::error::PersistError;
use crate::persist::vfs::{Vfs, VfsFile};

/// `OGCM` = OrpheusGraph CoMmit. Distinguishes a COMMIT slot from stray bytes.
const MAGIC: [u8; 4] = *b"OGCM";
/// Slot encoding version. Bumped independently of the store `format_version`;
/// an unknown version reads as an invalid slot (contributes to a both-slots-bad
/// -> `Corrupt` outcome), never a silent misparse.
const COMMIT_VERSION: u32 = 1;

/// Bytes covered by the crc (everything up to, but excluding, the crc field).
const CONTENT_LEN: usize = 4 + 4 + 8 + 16 + 8; // = 40
/// One slot including its trailing crc.
const SLOT_SIZE: usize = CONTENT_LEN + 4; // = 44
/// The whole file: two slots.
const FILE_SIZE: usize = SLOT_SIZE * 2; // = 88

/// The COMMIT sidecar filename in the store dir.
pub const COMMIT_FILE: &str = "COMMIT";

/// The decoded contents of a COMMIT slot. `committed_seq` is the durable
/// high-water; `epoch` is the incarnation it was stamped in; `gen` is the
/// physical write counter used only for slot selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CommitRecord {
    pub epoch: u128,
    pub committed_seq: u64,
    pub gen: u64,
}

/// Encode one slot: fixed layout + a crc over the content bytes.
fn encode_slot(rec: &CommitRecord) -> [u8; SLOT_SIZE] {
    let mut buf = [0u8; SLOT_SIZE];
    buf[0..4].copy_from_slice(&MAGIC);
    buf[4..8].copy_from_slice(&COMMIT_VERSION.to_le_bytes());
    buf[8..16].copy_from_slice(&rec.gen.to_le_bytes());
    buf[16..32].copy_from_slice(&rec.epoch.to_le_bytes());
    buf[32..40].copy_from_slice(&rec.committed_seq.to_le_bytes());
    let mut h = Hasher::new();
    h.update(&buf[0..CONTENT_LEN]);
    buf[40..44].copy_from_slice(&h.finalize().to_le_bytes());
    buf
}

/// Decode one slot. Returns `None` for ANY defect (short, wrong magic, unknown
/// version, crc mismatch) — a torn or foreign slot is simply "not valid", never
/// a panic. Selection then relies on the OTHER slot.
fn decode_slot(bytes: &[u8]) -> Option<CommitRecord> {
    if bytes.len() < SLOT_SIZE {
        return None;
    }
    if bytes[0..4] != MAGIC {
        return None;
    }
    let version = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
    if version != COMMIT_VERSION {
        return None;
    }
    let stored_crc = u32::from_le_bytes([bytes[40], bytes[41], bytes[42], bytes[43]]);
    let mut h = Hasher::new();
    h.update(&bytes[0..CONTENT_LEN]);
    if h.finalize() != stored_crc {
        return None;
    }
    let gen = u64::from_le_bytes(bytes[8..16].try_into().ok()?);
    let epoch = u128::from_le_bytes(bytes[16..32].try_into().ok()?);
    let committed_seq = u64::from_le_bytes(bytes[32..40].try_into().ok()?);
    Some(CommitRecord {
        epoch,
        committed_seq,
        gen,
    })
}

/// Ping-pong writer over the COMMIT sidecar. Everything here runs under the
/// single writer `Mutex`, so slot alternation needs no extra synchronization.
pub struct CommitFile {
    file: Box<dyn VfsFile>,
    /// Slot index (0 or 1) the NEXT `advance` will overwrite — always the slot
    /// NOT holding the current authoritative record, so a torn write can never
    /// destroy the live marker.
    next_slot: usize,
    /// Monotonic write counter stamped into the next slot; strictly increases
    /// so the freshest slot always wins selection.
    next_gen: u64,
}

impl CommitFile {
    /// Create a fresh COMMIT sidecar for `create()`: write slot 0 with the
    /// initial `(epoch, committed_seq)` at `gen = 0`, fsync the file, then fsync
    /// the dir so the new directory entry is durable (§4.1). The next write goes
    /// to slot 1 at `gen = 1`.
    pub fn create(
        vfs: &dyn Vfs,
        dir: &Path,
        epoch: u128,
        committed_seq: u64,
    ) -> Result<Self, PersistError> {
        let path = dir.join(COMMIT_FILE);
        let mut file = vfs.create(&path)?;
        let rec = CommitRecord {
            epoch,
            committed_seq,
            gen: 0,
        };
        file.seek(SeekFrom::Start(0))?;
        file.write_all(&encode_slot(&rec))?;
        file.sync_all()?;
        vfs.fsync_dir(dir)?;
        Ok(Self {
            file,
            next_slot: 1,
            next_gen: 1,
        })
    }

    /// Open an existing COMMIT sidecar and return the authoritative record plus
    /// a writer positioned to overwrite the OTHER slot next. A missing file, or
    /// a file present with NEITHER slot crc-valid, is `Corrupt` — never "treat
    /// as 0" (would silently discard committed data) and never "treat as
    /// infinity" (would resurrect the very ambiguity this marker removes). The
    /// caller only reaches here for a current-format store, which always has a
    /// COMMIT written by `create()`.
    pub fn open(vfs: &dyn Vfs, dir: &Path) -> Result<(CommitRecord, Self), PersistError> {
        let path = dir.join(COMMIT_FILE);
        let mut file = vfs.open_rw(&path).map_err(|e| {
            if e.kind() == std::io::ErrorKind::NotFound {
                PersistError::Corrupt(format!(
                    "COMMIT sidecar missing at {} (current-format store must have one; \
                         refusing to treat as seq 0 and silently discard committed data)",
                    path.display()
                ))
            } else {
                PersistError::Io(e)
            }
        })?;
        let mut buf = Vec::new();
        file.seek(SeekFrom::Start(0))?;
        file.read_to_end(&mut buf)?;

        let slot0 = buf.get(0..SLOT_SIZE).and_then(decode_slot);
        let slot1 = buf.get(SLOT_SIZE..FILE_SIZE).and_then(decode_slot);

        // Pick the crc-valid slot with the higher `gen`; the next write must go
        // to the OTHER slot so we never clobber the one we just trusted.
        let (record, chosen_slot) = match (slot0, slot1) {
            (Some(a), Some(b)) => {
                if a.gen >= b.gen {
                    (a, 0)
                } else {
                    (b, 1)
                }
            }
            (Some(a), None) => (a, 0),
            (None, Some(b)) => (b, 1),
            (None, None) => {
                return Err(PersistError::Corrupt(format!(
                    "COMMIT sidecar at {} has no crc-valid slot (both torn/corrupt)",
                    path.display()
                )));
            }
        };

        Ok((
            record,
            Self {
                file,
                next_slot: 1 - chosen_slot,
                next_gen: record.gen + 1,
            },
        ))
    }

    /// Advance the marker to `(epoch, committed_seq)`: write the alternate slot,
    /// fsync the file. No dir fsync — the file already exists, only its content
    /// (fixed offsets) changes. ANY failure is returned so the caller poisons.
    pub fn advance(&mut self, epoch: u128, committed_seq: u64) -> Result<(), PersistError> {
        let rec = CommitRecord {
            epoch,
            committed_seq,
            gen: self.next_gen,
        };
        let off = (self.next_slot * SLOT_SIZE) as u64;
        self.file.seek(SeekFrom::Start(off))?;
        self.file.write_all(&encode_slot(&rec))?;
        self.file.sync_all()?;
        self.next_slot = 1 - self.next_slot;
        self.next_gen += 1;
        Ok(())
    }
}

// ===========================================================================
// Slot framing unit tests (pure; no on-disk dir needed for encode/decode).
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::persist::vfs::RealVfs;

    fn rec(gen: u64, epoch: u128, seq: u64) -> CommitRecord {
        CommitRecord {
            epoch,
            committed_seq: seq,
            gen,
        }
    }

    #[test]
    fn encode_decode_round_trip() {
        let r = rec(7, 0xDEAD_BEEF_1234_5678_9ABC_DEF0_1122_3344, 42);
        let bytes = encode_slot(&r);
        assert_eq!(bytes.len(), SLOT_SIZE);
        assert_eq!(decode_slot(&bytes), Some(r));
    }

    #[test]
    fn flipped_bit_fails_crc() {
        let mut bytes = encode_slot(&rec(1, 9, 3));
        bytes[20] ^= 0x01; // corrupt an epoch byte
        assert_eq!(decode_slot(&bytes), None);
    }

    #[test]
    fn wrong_magic_and_short_are_none() {
        let mut bytes = encode_slot(&rec(1, 9, 3));
        bytes[0] ^= 0xFF;
        assert_eq!(decode_slot(&bytes), None);
        assert_eq!(decode_slot(&bytes[0..10]), None);
    }

    #[test]
    fn create_open_round_trip() {
        let dir = tempfile::tempdir().unwrap();
        let cf = CommitFile::create(&RealVfs, dir.path(), 123, 5).unwrap();
        drop(cf);
        let (r, _w) = CommitFile::open(&RealVfs, dir.path()).unwrap();
        assert_eq!(r.epoch, 123);
        assert_eq!(r.committed_seq, 5);
        assert_eq!(r.gen, 0);
    }

    #[test]
    fn advance_ping_pongs_and_picks_highest_gen() {
        let dir = tempfile::tempdir().unwrap();
        let mut cf = CommitFile::create(&RealVfs, dir.path(), 1, 0).unwrap(); // slot0 gen0
        cf.advance(1, 10).unwrap(); // slot1 gen1
        cf.advance(1, 20).unwrap(); // slot0 gen2
        drop(cf);
        let (r, _w) = CommitFile::open(&RealVfs, dir.path()).unwrap();
        assert_eq!(r.committed_seq, 20);
        assert_eq!(r.gen, 2);
    }

    #[test]
    fn one_corrupt_slot_still_recovers_the_other() {
        let dir = tempfile::tempdir().unwrap();
        let mut cf = CommitFile::create(&RealVfs, dir.path(), 1, 0).unwrap(); // slot0 gen0
        cf.advance(1, 99).unwrap(); // slot1 gen1 (the newest, authoritative)
        drop(cf);
        // Corrupt slot1 (the newest) -> reader must fall back to slot0 gen0.
        let path = dir.path().join(COMMIT_FILE);
        let mut bytes = std::fs::read(&path).unwrap();
        bytes[SLOT_SIZE + 20] ^= 0xFF;
        std::fs::write(&path, &bytes).unwrap();
        let (r, _w) = CommitFile::open(&RealVfs, dir.path()).unwrap();
        assert_eq!(r.gen, 0, "must fall back to the older intact slot");
        assert_eq!(r.committed_seq, 0);
    }

    #[test]
    fn both_slots_corrupt_is_corrupt() {
        let dir = tempfile::tempdir().unwrap();
        let cf = CommitFile::create(&RealVfs, dir.path(), 1, 0).unwrap();
        drop(cf);
        let path = dir.path().join(COMMIT_FILE);
        let mut bytes = std::fs::read(&path).unwrap();
        for b in bytes.iter_mut() {
            *b ^= 0xFF;
        }
        std::fs::write(&path, &bytes).unwrap();
        assert!(matches!(
            CommitFile::open(&RealVfs, dir.path()),
            Err(PersistError::Corrupt(_))
        ));
    }

    #[test]
    fn missing_file_is_corrupt() {
        let dir = tempfile::tempdir().unwrap();
        assert!(matches!(
            CommitFile::open(&RealVfs, dir.path()),
            Err(PersistError::Corrupt(_))
        ));
    }
}
