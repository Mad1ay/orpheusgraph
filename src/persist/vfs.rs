//! `Vfs` — a thin virtual-filesystem seam over the persist layer's
//! WRITE-AND-DURABILITY operations, so a deterministic userspace
//! fault-injection filesystem (the power-loss harness) can model torn writes
//! and the fsync-failure ambiguity WITHOUT root, `dm-flakey` or a VM — the
//! RocksDB `FaultInjectionTestFS` pattern.
//!
//! ## What is and is NOT routed here (invariant)
//! * Prod always uses [`RealVfs`], a zero-cost `std::fs` passthrough. The hot
//!   READ path (traversal) NEVER touches this trait — it runs entirely against
//!   an `ArcSwap`-published `GraphState`, so `RealVfs` adds no measurable
//!   regression (an `apply` reaches the trait only through the already-open WAL
//!   / COMMIT handles it holds, i.e. one `Box<dyn VfsFile>` deref).
//! * The snapshot mmap read stays a DIRECT `std::fs`/`memmap2` open (not routed
//!   here). This is sound because a fault FS reconstructs the crash image onto
//!   the REAL directory before reopen, so a direct mmap sees exactly the
//!   post-power-cut bytes — routing the read would add nothing.
//! * Pure recovery reads (`read_manifest`, existence checks, `read_dir`) also
//!   stay direct for the same reason: after `power_cut` the real directory IS
//!   the crash image, so a passthrough read is already correct.
//!
//! Only durability writes are routed: WAL append/fsync, the COMMIT slot
//! ping-pong, the MANIFEST/snapshot `tmp+fsync+rename+fsync_dir` discipline,
//! the recovery WAL truncate, and snapshot GC removals.
//!
//! ## Public surface — a deliberate extension point
//! [`Vfs`], [`VfsFile`] and [`RealVfs`] are exported (not feature-gated), and
//! [`PersistentGraph::create_with_vfs`]/[`open_with_vfs`] accept any `Vfs` (they
//! are `#[doc(hidden)]`, so not in rustdoc, but reachable). This is intentional,
//! following RocksDB's `Env`/`FileSystem` seam: an embedding host can supply its
//! own filesystem (an encrypting/instrumented/in-memory backend, or a fault FS in
//! its own tests). The trait is the ONLY supported way to do so; the default
//! `create`/`open` use [`RealVfs`] and callers who don't need the seam ignore it.

use std::fs::{File, OpenOptions};
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::Path;

/// An open file handle. The persist layer needs exactly this set: sequential /
/// positioned writes, whole-file reads (WAL scan, COMMIT slots), `sync_all`
/// (the durability point), `set_len` (WAL rotate / recovery truncate) and a
/// length query (the recovery boundary math). Deliberately minimal so a fault
/// impl has a small, auditable surface to shadow.
// `len` is the on-disk byte length (a mutable I/O query, like `File::metadata`),
// not a container size — an `is_empty` companion would be meaningless here.
#[allow(clippy::len_without_is_empty)]
pub trait VfsFile: Send {
    fn read_to_end(&mut self, buf: &mut Vec<u8>) -> io::Result<usize>;
    fn write_all(&mut self, buf: &[u8]) -> io::Result<()>;
    fn seek(&mut self, pos: SeekFrom) -> io::Result<u64>;
    /// The durability point: force this file's bytes to stable storage. A fault
    /// impl may return `Err` here yet still persist the bytes (the fsync-failure
    /// ambiguity the COMMIT marker exists for).
    fn sync_all(&mut self) -> io::Result<()>;
    fn set_len(&mut self, size: u64) -> io::Result<()>;
    fn len(&mut self) -> io::Result<u64>;
}

/// The durability-relevant filesystem operations. A fault impl tracks per-file
/// shadow state on top of a real passthrough; [`RealVfs`] adds nothing.
pub trait Vfs: Send + Sync {
    /// Create/truncate for read+write (tmp files, a fresh WAL, a fresh COMMIT).
    fn create(&self, path: &Path) -> io::Result<Box<dyn VfsFile>>;
    /// Open an existing file read+write, no truncate (COMMIT re-open).
    fn open_rw(&self, path: &Path) -> io::Result<Box<dyn VfsFile>>;
    /// Open an existing file read + append (the WAL).
    fn open_append(&self, path: &Path) -> io::Result<Box<dyn VfsFile>>;
    fn rename(&self, from: &Path, to: &Path) -> io::Result<()>;
    fn remove_file(&self, path: &Path) -> io::Result<()>;
    /// fsync a directory fd so a preceding rename/create entry is durable.
    fn fsync_dir(&self, dir: &Path) -> io::Result<()>;
}

/// Zero-overhead production impl: every call is a direct `std::fs` passthrough.
/// Unit struct so callers can pass `&RealVfs` inline with no allocation.
pub struct RealVfs;

/// A real file handle wrapping `std::fs::File`.
struct RealFile(File);

impl VfsFile for RealFile {
    fn read_to_end(&mut self, buf: &mut Vec<u8>) -> io::Result<usize> {
        self.0.read_to_end(buf)
    }
    fn write_all(&mut self, buf: &[u8]) -> io::Result<()> {
        self.0.write_all(buf)
    }
    fn seek(&mut self, pos: SeekFrom) -> io::Result<u64> {
        self.0.seek(pos)
    }
    fn sync_all(&mut self) -> io::Result<()> {
        self.0.sync_all()
    }
    fn set_len(&mut self, size: u64) -> io::Result<()> {
        self.0.set_len(size)
    }
    fn len(&mut self) -> io::Result<u64> {
        Ok(self.0.metadata()?.len())
    }
}

impl Vfs for RealVfs {
    fn create(&self, path: &Path) -> io::Result<Box<dyn VfsFile>> {
        // read+write+create+truncate: a superset of what any create site needs
        // (tmp/WAL/COMMIT), so the passthrough behavior is identical to the
        // original `File::create` / `OpenOptions` calls it replaces.
        let f = OpenOptions::new()
            .create(true)
            .read(true)
            .write(true)
            .truncate(true)
            .open(path)?;
        Ok(Box::new(RealFile(f)))
    }
    fn open_rw(&self, path: &Path) -> io::Result<Box<dyn VfsFile>> {
        let f = OpenOptions::new().read(true).write(true).open(path)?;
        Ok(Box::new(RealFile(f)))
    }
    fn open_append(&self, path: &Path) -> io::Result<Box<dyn VfsFile>> {
        let f = OpenOptions::new().read(true).append(true).open(path)?;
        Ok(Box::new(RealFile(f)))
    }
    fn rename(&self, from: &Path, to: &Path) -> io::Result<()> {
        std::fs::rename(from, to)
    }
    fn remove_file(&self, path: &Path) -> io::Result<()> {
        std::fs::remove_file(path)
    }
    fn fsync_dir(&self, dir: &Path) -> io::Result<()> {
        // fsync a directory fd so a preceding `rename`/`create` entry is durable.
        File::open(dir)?.sync_all()
    }
}
