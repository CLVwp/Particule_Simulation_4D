//! The world container: bodies, scratch buffers, and pool helpers.

use std::sync::atomic::AtomicU32;
use std::time::Instant;

use rayon::prelude::*;

use self::contacts::Contact;
use self::resolve::ContactDelta;
use crate::engine::body::Body;
use crate::engine::config::{BODY_RADIUS, SimSettings};

mod contacts;
mod grid;
mod resolve;
mod spawn;
mod step;

/// Logical cores visible to this process.
#[must_use]
pub fn thread_count() -> usize {
    std::thread::available_parallelism()
        .map(usize::from)
        .unwrap_or(1)
}

/// Particle positions and velocities in x, y, z (time lives in `step(dt)`).
///
/// # Examples
///
/// ```
/// use particule_simulation_4d::engine::World;
///
/// let mut world = World::new();
/// world.spawn_wave(100, [0.0, 5.0, 0.0], 4.0);
/// world.step(1.0 / 60.0);
/// assert_eq!(world.bodies.len(), 100);
/// ```
#[derive(Clone, Debug)]
pub struct World {
    /// Every body in the world. Callers may edit them between steps.
    pub bodies: Vec<Body>,
    /// Laws of motion. Change them freely between steps.
    pub settings: SimSettings,
    /// Grid cell edge. Grows to fit the biggest spawned radius.
    pub(super) cell_size: f32,
    /// Candidate contacts found this step.
    pub(super) contacts: Vec<Contact>,
    /// Per-contact solve results for the current round.
    pub(super) deltas: Vec<ContactDelta>,
    /// Cell key + body index, sorted by key. Replaces the hash grid.
    pub(super) cell_sort: Vec<(u64, u32)>,
    /// Unique cell keys of `cell_sort`, ascending. Runs of equal key.
    pub(super) cell_keys: Vec<u64>,
    /// Start offset of each key's run in `cell_sort`, plus the end. One entry
    /// per key, last entry is `cell_sort.len()`.
    pub(super) cell_start: Vec<u32>,
    /// Per-body contact index as flat arrays: offsets into `bc_items`.
    pub(super) bc_start: Vec<u32>,
    /// Fill cursor for `bc_items`, kept across steps to avoid reallocation.
    /// Plain u32 slots. The fill takes an atomic view for the fetch-add rank.
    pub(super) bc_cursor: Vec<u32>,
    /// Contact tags in body order. Bit `J_SIDE` marks the `j` side.
    /// Plain u32 slots. The fill takes an atomic view to write from the pool.
    pub(super) bc_items: Vec<u32>,
    /// Wall time of the last step per phase, in ms:
    /// integrate, grid, contacts, resolve, floor. Read by the debug overlay.
    pub phase_ms: [f32; 5],
}

impl Default for World {
    fn default() -> Self {
        World {
            bodies: Vec::new(),
            settings: SimSettings::default(),
            cell_size: 2.0 * BODY_RADIUS,
            contacts: Vec::new(),
            deltas: Vec::new(),
            cell_sort: Vec::new(),
            cell_keys: Vec::new(),
            cell_start: Vec::new(),
            bc_start: Vec::new(),
            bc_cursor: Vec::new(),
            bc_items: Vec::new(),
            phase_ms: [0.0; 5],
        }
    }
}

impl World {
    /// Creates an empty world with default settings.
    pub fn new() -> Self {
        World::default()
    }

    /// Candidate contacts found in the last step.
    #[must_use]
    pub fn contact_count(&self) -> usize {
        self.contacts.len()
    }

    /// Removes every body, resets the grid, and frees the scratch buffers.
    /// A big spawn transient leaves gigabytes of retained capacity behind;
    /// `clear` is the explicit boundary where that memory must go back.
    pub fn clear(&mut self) {
        fn free<T>(buffer: &mut Vec<T>) {
            buffer.clear();
            buffer.shrink_to_fit();
        }
        self.bodies.clear();
        self.cell_size = 2.0 * BODY_RADIUS;
        free(&mut self.contacts);
        free(&mut self.deltas);
        free(&mut self.cell_sort);
        free(&mut self.cell_keys);
        free(&mut self.cell_start);
        free(&mut self.bc_start);
        free(&mut self.bc_cursor);
        free(&mut self.bc_items);
    }
}

/// Wall time since `t`, in milliseconds.
fn ms_since(t: Instant) -> f32 {
    t.elapsed().as_secs_f32() * 1000.0
}

/// Runs `f` on every element, on the pool above `par_min`, inline below.
fn par_each<T: Send>(slice: &mut [T], par_min: usize, f: impl Fn(&mut T) + Sync + Send) {
    if slice.len() < par_min {
        slice.iter_mut().for_each(f);
    } else {
        slice.par_iter_mut().for_each(f);
    }
}

/// Atomic view over the plain u32 scratch. The contact fill ranks slots
/// with fetch-adds from the pool; the apply loop reads them back.
///
/// The caller borrows the slice before the view and drops the view before
/// any plain write, so the two never overlap.
pub(super) fn atomic_u32s(v: &mut [u32]) -> &[AtomicU32] {
    const _: () = assert!(size_of::<AtomicU32>() == size_of::<u32>());
    // SAFETY: u32 and AtomicU32 share size, alignment, and bit validity, so
    // the cast preserves every value. The atomic view borrows `v`, so the
    // memory stays alive and unaliased for the view's whole life.
    unsafe { &*(std::ptr::from_ref(v) as *const [AtomicU32]) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clear_releases_the_scratch_capacity() {
        let mut w = World::new();
        w.spawn_wave(3000, [0.0, 5.0, 0.0], 4.0);
        w.step(1.0 / 60.0);
        w.clear();
        assert_eq!(w.contacts.capacity(), 0, "contacts kept its capacity");
        assert_eq!(w.bc_items.capacity(), 0, "bc_items kept its capacity");
        assert_eq!(w.cell_sort.capacity(), 0, "cell_sort kept its capacity");
    }
}
