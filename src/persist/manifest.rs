//! MANIFEST (JSON) — the durable root pointer of the store, plus the
//! tmp+fsync+rename+fsync(dir) atomic-write discipline reused for the snapshot,
//! the `format_version` gate, and epoch minting (§4.1/§4.3).

use std::fs::File;
use std::io::Write;
use std::path::Path;

use crate::persist::error::PersistError;

/// Highest on-disk `format_version` this build understands. Phase 2a is the
/// flat rkyv format (`0`); anything greater is rejected as `UnsupportedVersion`.
pub const FORMAT_VERSION: u32 = 0;

/// `created_by` stamp written into every MANIFEST.
pub const CREATED_BY: &str = concat!("orpheusgraph ", env!("CARGO_PKG_VERSION"));

/// The durable root record. `serde`/JSON so it is human-inspectable and
/// forward-tolerant; the `format_version` gate rejects unknown layouts.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Manifest {
    /// On-disk snapshot encoding version (0 in 2a: flat rkyv). Gate rejects `>`.
    pub format_version: u32,
    /// Snapshot filename, `snapshot-{seq:020}.og`.
    pub snapshot_file: String,
    /// Seq folded into the snapshot; WAL frames `<=` this are already durable.
    pub snapshot_seq: u64,
    /// crc32 of the snapshot file bytes (integrity check on load).
    pub snapshot_crc32: u32,
    /// Incarnation id — 16 random bytes, re-minted on any timeline fork (§4.3).
    pub epoch: u128,
    /// `true` only after a successful `close()`; absence/false => crash.
    pub clean_shutdown: bool,
    /// Provenance stamp, e.g. "orpheusgraph 0.1.0".
    pub created_by: String,
}

/// Mint a fresh 128-bit epoch from 16 CSPRNG bytes (§4.3).
pub fn mint_epoch() -> Result<u128, PersistError> {
    let mut buf = [0u8; 16];
    getrandom::getrandom(&mut buf).map_err(|e| {
        PersistError::Io(std::io::Error::new(
            std::io::ErrorKind::Other,
            format!("getrandom failed while minting epoch: {e}"),
        ))
    })?;
    Ok(u128::from_le_bytes(buf))
}

/// Read and parse `MANIFEST.json`. A parse error is `Corrupt` (not a torn tail).
pub fn read_manifest(path: &Path) -> Result<Manifest, PersistError> {
    let bytes = std::fs::read(path)?;
    serde_json::from_slice(&bytes)
        .map_err(|e| PersistError::Corrupt(format!("MANIFEST parse failed: {e}")))
}

/// Atomically (re)write `MANIFEST.json`: tmp -> fsync -> rename -> fsync(dir).
/// The directory fsync is mandatory so the rename survives power loss (§4.1).
pub fn write_manifest_atomic(dir: &Path, manifest: &Manifest) -> Result<(), PersistError> {
    let json = serde_json::to_vec_pretty(manifest)
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
