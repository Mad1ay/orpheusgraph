//! `DirLock` — an advisory exclusive `flock` on the store's `LOCK` file,
//! enforcing single-writer across processes (§4.1). Held for the process
//! lifetime; released by dropping the `File` (kernel also releases on crash, so
//! there is no stale-lock dance).

use std::fs::OpenOptions;
use std::path::Path;

use crate::persist::error::PersistError;

/// RAII holder of the exclusive `flock`. Dropping it closes the fd, which the
/// kernel treats as unlocking. There is deliberately no explicit `unlock` —
/// `close()`/drop is the single release path.
pub struct DirLock {
    // Kept only for its Drop side effect (closing the fd releases the flock).
    _file: std::fs::File,
}

impl DirLock {
    /// Take the exclusive lock on `dir/LOCK` without blocking. A contended lock
    /// (another open file description already holds it, in this process or any
    /// other) maps to [`PersistError::LockHeld`]; a genuine I/O failure to
    /// [`PersistError::Io`].
    pub fn acquire(dir: &Path) -> Result<Self, PersistError> {
        let lock_path = dir.join("LOCK");
        let file = OpenOptions::new()
            .create(true)
            .read(true)
            .write(true)
            .truncate(false)
            .open(&lock_path)?;

        // Fully-qualified fs2 call so it is unambiguous w.r.t. any inherent
        // std `File::try_lock_exclusive` on newer toolchains.
        match fs2::FileExt::try_lock_exclusive(&file) {
            Ok(()) => Ok(Self { _file: file }),
            Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                Err(PersistError::LockHeld(dir.to_path_buf()))
            }
            Err(e) => Err(PersistError::Io(e)),
        }
    }
}
