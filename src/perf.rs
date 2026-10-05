//! Live memory stats through a counting global allocator.
//!
//! The overlay reads `allocated_bytes` and `peak_bytes` each frame.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

static LIVE: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);

/// Bytes currently allocated by the program.
#[must_use]
pub fn allocated_bytes() -> usize {
    LIVE.load(Ordering::Relaxed)
}

/// Highest value `allocated_bytes` ever reached.
#[must_use]
pub fn peak_bytes() -> usize {
    PEAK.load(Ordering::Relaxed)
}

/// Wraps [`System`] and tracks live allocation size.
pub struct Counting;

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // SAFETY: the caller upholds the GlobalAlloc contract. System gets the
        // same size and alignment, so the pointer stays valid for dealloc.
        let ptr = unsafe { System.alloc(layout) };
        if !ptr.is_null() {
            let live = LIVE.fetch_add(layout.size(), Ordering::Relaxed) + layout.size();
            PEAK.fetch_max(live, Ordering::Relaxed);
        }
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: ptr came from System.alloc with this exact layout (see alloc).
        unsafe { System.dealloc(ptr, layout) };
        LIVE.fetch_sub(layout.size(), Ordering::Relaxed);
    }
}

#[global_allocator]
static GLOBAL_ALLOC: Counting = Counting;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allocation_is_tracked() {
        // Large enough that parallel test threads cannot mask it.
        let before = allocated_bytes();
        let keep: Vec<u64> = vec![0; 4 * 1024 * 1024]; // 32 MiB
        assert!(allocated_bytes() >= before + 32 * 1024 * 1024);
        drop(keep);
        assert!(peak_bytes() >= allocated_bytes());
    }
}
