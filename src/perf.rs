//! Live memory stats through a counting global allocator.
//!
//! The overlay reads `allocated_bytes` and `peak_bytes` each frame.
//! ponytail: the counting wrapper wraps any allocator; mimalloc is the swap
//! when libmimalloc-sys compiles again (v3 sources fail on VS 18 BuildTools)

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

static LIVE: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);

/// Bytes currently allocated by the program.
#[must_use]
pub fn allocated_bytes() -> usize {
    LIVE.load(Ordering::Relaxed)
}

/// Highest value `allocated_bytes` has reached.
///
/// An allocation updates the two counters as two separate atomic steps. This
/// read combines both counters, so the peak never reads below the live total.
#[must_use]
pub fn peak_bytes() -> usize {
    let peak = PEAK.load(Ordering::Relaxed);
    let live = LIVE.load(Ordering::Relaxed);
    peak.max(live)
}

/// Wraps an allocator and tracks live allocation size.
pub struct Counting<A>(pub A);

unsafe impl<A: GlobalAlloc> GlobalAlloc for Counting<A> {
    /// Allocates memory as described by `layout`.
    ///
    /// # Safety
    ///
    /// The caller must pass a `Layout` with a nonzero size.
    /// The caller must free the returned pointer with `dealloc` and this layout.
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // SAFETY: the caller upholds the GlobalAlloc contract. The inner
        // allocator gets the same size and alignment, so the pointer stays
        // valid for dealloc.
        let ptr = unsafe { self.0.alloc(layout) };
        if !ptr.is_null() {
            let live = LIVE.fetch_add(layout.size(), Ordering::Relaxed) + layout.size();
            PEAK.fetch_max(live, Ordering::Relaxed);
        }
        ptr
    }

    /// Deallocates memory that `alloc` returned.
    ///
    /// # Safety
    ///
    /// The caller must pass a pointer that this allocator returned.
    /// The layout must match the layout used at allocation time.
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: ptr came from the inner allocator with this exact layout
        // (see alloc).
        unsafe { self.0.dealloc(ptr, layout) };
        LIVE.fetch_sub(layout.size(), Ordering::Relaxed);
    }
}

#[global_allocator]
static GLOBAL_ALLOC: Counting<System> = Counting(System);

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::sync::atomic::AtomicBool;
    use std::time::{Duration, Instant};

    /// Blocks this large stand out against other tests that run in parallel.
    const BIG_BLOCK: usize = 128 * 1024 * 1024;

    /// Serializes the counter tests. Other tests allocate in parallel and
    /// pollute the global counters between two reads.
    static COUNTER_TESTS: std::sync::Mutex<()> = std::sync::Mutex::new(());

    #[test]
    fn allocation_is_tracked() {
        let _guard = COUNTER_TESTS.lock().unwrap();
        // Allocate. In-use bytes must rise by the block size. Parallel
        // tests free memory between two reads, so one window can read
        // low. Retry until a clean window shows the full rise. The old
        // block drops before the window opens, or its drop would cancel
        // the new allocation inside it.
        let mut keep: Option<Vec<u64>> = None;
        let mut clean = false;
        let mut held = 0;
        for _ in 0..16 {
            drop(keep.take());
            let before = allocated_bytes();
            keep = Some(vec![0; BIG_BLOCK / 8]); // 128 MiB
            held = allocated_bytes();
            clean = held >= before + BIG_BLOCK;
            if clean {
                break;
            }
        }
        assert!(clean, "alloc must raise in-use bytes by the block size");
        // Peak must absorb the new allocation while it is still live.
        assert!(peak_bytes() >= held, "peak must absorb the live allocation");
        // Free. Parallel tests can allocate between the two reads, so the
        // exact drop is not observable. In-use must still fall overall.
        drop(keep);
        let after = allocated_bytes();
        assert!(after < held, "free must lower in-use bytes");
        // Peak never falls. It must stay at or above in-use at every point.
        assert!(peak_bytes() >= after, "peak must stay at or above in-use");
    }

    #[test]
    fn peak_stays_above_in_use_after_big_alloc_then_small_storm() {
        let _guard = COUNTER_TESTS.lock().unwrap();
        // Reported pattern: one big block, then a storm of small ones.
        // Small allocations must never push in-use above the recorded peak.
        let big: Vec<u8> = vec![0; BIG_BLOCK];
        assert!(peak_bytes() >= allocated_bytes(), "peak must lead in-use");
        drop(big);

        let mut smalls: Vec<Vec<u8>> = Vec::with_capacity(1024);
        for _ in 0..1024 {
            smalls.push(vec![0u8; 4096]);
            assert!(
                peak_bytes() >= allocated_bytes(),
                "in-use rose above peak during the small storm"
            );
        }
        drop(smalls);
        assert!(peak_bytes() >= allocated_bytes());
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn peak_stays_above_in_use_under_thread_churn() {
        // Workers churn the heap. The reader samples both counters, like the
        // overlay does each frame. One sample may skew while workers run.
        // The next sample must be clean, and the pair must never stay inverted.
        let stop = Arc::new(AtomicBool::new(false));
        let mut workers = Vec::new();
        for _ in 0..4 {
            let stop = Arc::clone(&stop);
            workers.push(std::thread::spawn(move || {
                let mut blocks: Vec<Vec<u8>> = Vec::new();
                let mut seed = 0x1234_5678_u32;
                while !stop.load(Ordering::Relaxed) {
                    // LCG step. `wrapping_*` makes the overflow explicit.
                    seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                    let size = ((seed >> 16) % 64) as usize * 1024 + 1024;
                    if blocks.len() >= 64 {
                        blocks.clear();
                    }
                    blocks.push(vec![0u8; size]);
                }
            }));
        }

        // Track the highest in-use sample. Peak must hold this high-water mark.
        let mut high_water = 0usize;
        let start = Instant::now();
        while start.elapsed() < Duration::from_millis(200) {
            let in_use = allocated_bytes();
            let peak = peak_bytes();
            high_water = high_water.max(in_use);
            if peak < in_use {
                // One skewed sample is the known read race.
                assert!(
                    peak_bytes() >= allocated_bytes(),
                    "in-use stayed above peak after a skewed sample"
                );
            }
        }

        stop.store(true, Ordering::Relaxed);
        for worker in workers {
            worker.join().expect("churn worker must not panic");
        }
        // Workers stopped. The pair must sit clean and hold the high-water mark.
        assert!(peak_bytes() >= allocated_bytes(), "pair inverted at rest");
        assert!(
            peak_bytes() >= high_water,
            "peak lost the churn high-water mark"
        );
    }
}
