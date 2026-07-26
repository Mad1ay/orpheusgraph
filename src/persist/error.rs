//! Error type for the persistence layer.
//!
//! Hand-written `Display`/`Error`/`From` impls, matching the crate's existing
//! [`crate::delta::DeltaError`] style (no `thiserror` proc-macro dependency).

use std::path::PathBuf;

use crate::delta::DeltaError;

/// Every failure the persistent store can surface. Recovery paths NEVER panic
/// on disk bytes — corruption is always one of these typed variants.
#[derive(Debug)]
pub enum PersistError {
    /// An underlying I/O error; the original `io::Error` kind is preserved.
    Io(std::io::Error),
    /// A previous WAL append/fsync failed, poisoning the writer. The store is
    /// read-only until reopened (§4.2). Checked first thing in `apply()`.
    Poisoned,
    /// Optimistic-concurrency (CAS) mismatch: the caller's `expected_seq` did
    /// not match the durable commit counter at apply time (§3.3.6).
    Conflict { expected: u64, actual: u64 },
    /// The batch was rejected during validation on the delta clone and therefore
    /// NEVER reached the WAL (§3.3 rule 4).
    Delta(DeltaError),
    /// Structural corruption of MANIFEST/snapshot/WAL that is not a torn tail:
    /// crc mismatch, decode failure on a crc-valid frame, or a WAL seq gap.
    Corrupt(String),
    /// The on-disk `format_version` is newer than this build supports (§4.3).
    UnsupportedVersion { found: u32, max: u32 },
    /// Another writer (this process or another) holds the exclusive `LOCK`.
    LockHeld(PathBuf),
    /// `open(create=false)` on a directory with no store.
    NotFound(PathBuf),
}

impl std::fmt::Display for PersistError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PersistError::Io(e) => write!(f, "persist I/O error: {e}"),
            PersistError::Poisoned => write!(
                f,
                "persist writer is poisoned: a prior WAL append failed; the store \
                 is read-only until reopened"
            ),
            PersistError::Conflict { expected, actual } => {
                write!(f, "CAS conflict: expected seq {expected}, actual {actual}")
            }
            PersistError::Delta(e) => write!(f, "batch rejected at validation: {e}"),
            PersistError::Corrupt(s) => write!(f, "corrupt persistent store: {s}"),
            PersistError::UnsupportedVersion { found, max } => write!(
                f,
                "unsupported store format_version {found} (this build supports up to {max})"
            ),
            PersistError::LockHeld(p) => {
                write!(f, "another writer holds the lock on {}", p.display())
            }
            PersistError::NotFound(p) => {
                write!(f, "no persistent store found at {}", p.display())
            }
        }
    }
}

impl std::error::Error for PersistError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            PersistError::Io(e) => Some(e),
            PersistError::Delta(e) => Some(e),
            _ => None,
        }
    }
}

impl From<std::io::Error> for PersistError {
    fn from(e: std::io::Error) -> Self {
        PersistError::Io(e)
    }
}

impl From<DeltaError> for PersistError {
    fn from(e: DeltaError) -> Self {
        PersistError::Delta(e)
    }
}
