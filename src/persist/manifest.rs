//! MANIFEST (JSON) — the durable root pointer of the store, plus the
//! tmp+fsync+rename+fsync(dir) atomic-write discipline reused for the snapshot,
//! the `format_version` gate, and epoch minting (§4.1/§4.3).

use std::fs::File;
use std::io::Write;
use std::path::Path;

use crate::persist::error::PersistError;

/// Highest on-disk `format_version` this build understands. `0` is the flat
/// rkyv format (Phase 2a, ephemeral/Redis path); `1` is the CSR mmap snapshot
/// (Phase 2b). The gate rejects anything `> FORMAT_VERSION`, so this build reads
/// BOTH 0 and 1. `create()` writes 1 going forward; V0 stores still open (as
/// owned, never mmap-traversed).
pub const FORMAT_VERSION: u32 = 1;

/// `created_by` stamp written into every MANIFEST.
pub const CREATED_BY: &str = concat!("orpheusgraph ", env!("CARGO_PKG_VERSION"));

/// The durable root record. `serde`/JSON so it is human-inspectable and
/// forward-tolerant; the `format_version` gate rejects unknown layouts.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Manifest {
    /// On-disk snapshot encoding version (0: flat rkyv; 1: CSR mmap). Gate rejects `>`.
    pub format_version: u32,
    /// Snapshot filename: `snapshot-{seq:020}.og` (V0) or
    /// `snapshot-{seq:020}-{cid:010}.og` (V1, cid = compaction id).
    pub snapshot_file: String,
    /// Seq folded into the snapshot; WAL frames `<=` this are already durable.
    pub snapshot_seq: u64,
    /// crc32 of the snapshot file bytes (integrity check on load).
    pub snapshot_crc32: u32,
    /// Monotonic compaction id (§4.4b). Bumped on every compaction so successive
    /// snapshot filenames are always distinct even when the fold seq is unchanged;
    /// persisted so it stays monotonic across reopens. `#[serde(default)]` so a
    /// pre-2b (V0) MANIFEST without the field reads as 0.
    #[serde(default)]
    pub compaction_id: u64,
    /// Incarnation id — 16 random bytes, re-minted on any timeline fork (§4.3).
    pub epoch: u128,
    /// Highest seq durably acked as of the last MANIFEST write (close/compaction/
    /// recovery). A LOWER bound on what recovery must find in the WAL: if replay
    /// yields `last_applied < high_seq`, acked data was externally lost (a
    /// truncated/suffix-lost WAL) and open() hard-errors `Corrupt` rather than
    /// silently regressing seq (§7 "never auto-heal by discarding data"). Updated
    /// only at close/compaction/recovery, so it lags live applies by design — the
    /// un-fsynced tail beyond it is the accepted power-loss window. `#[serde(default)]`
    /// so a MANIFEST written before this field reads as 0 (check never fires).
    #[serde(default)]
    pub high_seq: u64,
    /// `true` only after a successful `close()`; absence/false => crash.
    pub clean_shutdown: bool,
    /// Provenance stamp, e.g. "orpheusgraph 0.1.0".
    pub created_by: String,
    /// crc32 over the integrity-critical fields (set by `write_manifest_atomic`,
    /// verified by `read_manifest`). `#[serde(default)]` = 0 for a MANIFEST
    /// written before this field; a 0 checksum skips verification.
    #[serde(default)]
    pub checksum: u32,
}

/// Mint a fresh 128-bit epoch from 16 CSPRNG bytes (§4.3).
pub fn mint_epoch() -> Result<u128, PersistError> {
    let mut buf = [0u8; 16];
    getrandom::getrandom(&mut buf).map_err(|e| {
        PersistError::Io(std::io::Error::other(
            format!("getrandom failed while minting epoch: {e}"),
        ))
    })?;
    Ok(u128::from_le_bytes(buf))
}

/// crc32 over the integrity-critical MANIFEST fields (everything that steers
/// recovery), computed with `checksum` itself excluded. Guards against bit-rot
/// of e.g. `snapshot_seq`/`high_seq` to a valid-but-wrong value that would
/// silently skip WAL frames — the snapshot bytes have their own crc, the
/// MANIFEST did not. `created_by` is provenance and excluded.
fn manifest_checksum(m: &Manifest) -> u32 {
    let mut h = crc32fast::Hasher::new();
    h.update(&m.format_version.to_le_bytes());
    h.update(m.snapshot_file.as_bytes());
    h.update(&m.snapshot_seq.to_le_bytes());
    h.update(&m.snapshot_crc32.to_le_bytes());
    h.update(&m.compaction_id.to_le_bytes());
    h.update(&m.epoch.to_le_bytes());
    h.update(&m.high_seq.to_le_bytes());
    h.update(&[m.clean_shutdown as u8]);
    h.finalize()
}

/// Read and parse `MANIFEST.json`. A parse error is `Corrupt` (not a torn tail).
/// If a `checksum` is present (non-zero), it is verified — a mismatch is
/// `Corrupt` rather than a silently-trusted bit-rotted pointer. A `0` checksum
/// (a MANIFEST written before the field existed) skips verification.
pub fn read_manifest(path: &Path) -> Result<Manifest, PersistError> {
    let bytes = std::fs::read(path)?;
    let m: Manifest = serde_json::from_slice(&bytes)
        .map_err(|e| PersistError::Corrupt(format!("MANIFEST parse failed: {e}")))?;
    if m.checksum != 0 {
        let expect = manifest_checksum(&m);
        if m.checksum != expect {
            return Err(PersistError::Corrupt(format!(
                "MANIFEST checksum mismatch: stored {:#010x}, computed {:#010x} (bit-rot?)",
                m.checksum, expect
            )));
        }
    }
    Ok(m)
}

/// Atomically (re)write `MANIFEST.json`: tmp -> fsync -> rename -> fsync(dir).
/// The directory fsync is mandatory so the rename survives power loss (§4.1).
/// Stamps the integrity `checksum` before serializing.
pub fn write_manifest_atomic(dir: &Path, manifest: &Manifest) -> Result<(), PersistError> {
    let mut m = manifest.clone();
    m.checksum = manifest_checksum(&m);
    let json = serde_json::to_vec_pretty(&m)
        .map_err(|e| PersistError::Corrupt(format!("MANIFEST serialize failed: {e}")))?;
    write_file_atomic(dir, "MANIFEST.json", &json)
}

/// Generic tmp+fsync+rename+fsync(dir) atomic file write. Used for MANIFEST and
/// for the snapshot in `create()` — identical durability discipline (§4.1).
pub fn write_file_atomic(dir: &Path, name: &str, bytes: &[u8]) -> Result<(), PersistError> {
    let tmp = dir.join(format!("{name}.tmp"));
    let final_path = dir.join(name);
    {
        let mut f = File::create(&tmp)?;
        f.write_all(bytes)?;
        f.sync_all()?;
    }
    std::fs::rename(&tmp, &final_path)?;
    fsync_dir(dir)?;
    Ok(())
}

/// fsync a directory fd so a preceding `rename`/`create` entry is durable.
pub fn fsync_dir(dir: &Path) -> Result<(), PersistError> {
    let f = File::open(dir)?;
    f.sync_all()?;
    Ok(())
}
