//! Broad phase: candidate contact pairs from the sorted cell grid.

use std::sync::atomic::{AtomicU32, Ordering};

use rayon::prelude::*;

use super::World;
use super::body::Body;
use super::grid::{STENCIL, key_cell, pack_cell};

/// A pair of bodies that may overlap. Masses are stored once; they never change.
#[derive(Clone, Copy, Debug)]
pub(super) struct Contact {
    pub(super) i: u32,
    pub(super) j: u32,
    pub(super) mi: f32,
    pub(super) mj: f32,
}

/// High bit of a `body_contacts` tag. Set when the body is the `j` side.
pub(super) const J_SIDE: u32 = 1 << 31;

// Layout guards. These types fill the hot arrays, so their size must not drift.
const _: () = assert!(std::mem::size_of::<Contact>() == 16);

/// The overlap test of one body pair. Same test as `contact_delta`, so no
/// live pair is lost.
fn overlap(bodies: &[Body], prune: bool, bi: u32, bj: u32) -> Option<Contact> {
    let (pi, pj) = (&bodies[bi as usize], &bodies[bj as usize]);
    if prune {
        let d = [
            pj.pos[0] - pi.pos[0],
            pj.pos[1] - pi.pos[1],
            pj.pos[2] - pi.pos[2],
        ];
        let dist2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
        let min_d = pi.radius + pj.radius;
        if dist2 >= min_d * min_d || dist2 < 1e-12 {
            return None;
        }
    }
    Some(Contact {
        i: bi,
        j: bj,
        mi: pi.mass(),
        mj: pj.mass(),
    })
}

/// All pairs of one cell: same-cell pairs first, `j > i`, then the 13
/// cross-cell directions in stencil order.
fn emit_cell<'a>(
    cell_sort: &'a [(u64, u32)],
    cell_start: &'a [u32],
    bodies: &'a [Body],
    prune: bool,
    c: usize,
    hits: [u32; 13],
    n_cells: u32,
) -> impl Iterator<Item = Contact> + 'a {
    let own = &cell_sort[cell_start[c] as usize..cell_start[c + 1] as usize];
    let same = own.iter().flat_map(move |&(_, bj)| {
        own.iter().filter_map(move |&(_, bi)| {
            if bi >= bj {
                return None;
            }
            overlap(bodies, prune, bi, bj)
        })
    });
    let cross = STENCIL[1..]
        .iter()
        .zip(hits)
        .filter_map(move |(_, h)| (h != n_cells).then_some(h as usize))
        .flat_map(move |lo| {
            let from = cell_start[lo] as usize;
            let to = cell_start[lo + 1] as usize;
            cell_sort[from..to].iter().flat_map(move |&(_, bj)| {
                own.iter()
                    .filter_map(move |&(_, bi)| overlap(bodies, prune, bi, bj))
            })
        });
    same.chain(cross)
}

/// Merge walk over one chunk of cells. Yields each cell with the neighbor
/// cell index per stencil offset, or `n_cells` when absent.
struct Walk<'a> {
    keys: &'a [u64],
    n_cells: u32,
    cell: usize,
    end: usize,
    ptrs: [u32; 13],
}

impl Iterator for Walk<'_> {
    type Item = (usize, [u32; 13]);
    fn next(&mut self) -> Option<Self::Item> {
        let c = self.cell;
        if c >= self.end {
            return None;
        }
        self.cell += 1;
        let [cx, cy, cz] = key_cell(self.keys[c]);
        let mut hits = [self.n_cells; 13];
        for (k, &(dx, dy, dz)) in STENCIL[1..].iter().enumerate() {
            let key = pack_cell(cx as i64 + dx, cy as i64 + dy, cz as i64 + dz);
            let p = &mut self.ptrs[k];
            while *p < self.n_cells && self.keys[*p as usize] < key {
                *p += 1;
            }
            if *p < self.n_cells && self.keys[*p as usize] == key {
                hits[k] = *p;
            }
        }
        Some((c, hits))
    }
}

/// Pairs of one chunk of cells. One binary search per offset positions the
/// merge pointers at the chunk start. The walk only advances them, so the
/// total advance per pass stays bounded by the cell count.
fn scan_chunk<'a>(
    cell_keys: &'a [u64],
    cell_sort: &'a [(u64, u32)],
    cell_start: &'a [u32],
    bodies: &'a [Body],
    prune: bool,
    n_cells: u32,
    range: [usize; 2],
) -> impl Iterator<Item = Contact> + 'a {
    let [from, to] = range;
    let [cx, cy, cz] = key_cell(cell_keys[from]);
    let ptrs = std::array::from_fn(|k| {
        let (dx, dy, dz) = STENCIL[k + 1];
        let key = pack_cell(cx as i64 + dx, cy as i64 + dy, cz as i64 + dz);
        cell_keys.partition_point(|&k| k < key) as u32
    });
    Walk {
        keys: cell_keys,
        n_cells,
        cell: from,
        end: to,
        ptrs,
    }
    .flat_map(move |(c, hits)| emit_cell(cell_sort, cell_start, bodies, prune, c, hits, n_cells))
}

impl World {
    /// Finds overlapping pairs cell by cell, then emits body pairs. A pair is
    /// kept only when the two spheres overlap.
    ///
    /// The stencil holds self plus the 13 lex-positive offsets of the 27-cell
    /// neighborhood. Every pair of cells within reach meets in exactly one
    /// stencil direction, so cross-cell pairs need no filter. Same-cell pairs
    /// keep `j > i`.
    ///
    /// The scan walks cells in ascending key order. A translated cell key is
    /// monotone in the own key, so each stencil offset owns one merge pointer
    /// that only moves forward. One binary search per offset and chunk
    /// replaces one binary search per cell and offset.
    ///
    /// The per-body index fill runs on the pool above `par_min`. Each body's
    /// list ends in ascending contact order, so the Jacobi sum keeps its
    /// exact order and stays deterministic. Below `par_min` the fill is
    /// sequential and does the same work inline.
    pub(super) fn build_contacts(&mut self) {
        let World {
            cell_sort,
            cell_keys,
            cell_start,
            bodies,
            contacts,
            bc_start,
            bc_cursor,
            bc_items,
            settings,
            ..
        } = self;
        let cell_sort = &*cell_sort;
        let cell_keys = &*cell_keys;
        let cell_start = &*cell_start;
        let bodies = &*bodies;
        let par_min = settings.par_min;
        let prune = settings.prune_dead_pairs;
        // Guards the `as u32` cell-count and body-index casts below.
        debug_assert!(bodies.len() <= u32::MAX as usize);
        let n_cells = cell_keys.len();
        let n_cells = n_cells as u32;
        let cells = n_cells as usize;
        // Two chunks per pool thread keeps the load balanced without tiny
        // per-chunk overhead.
        let chunk = (cells / (2 * rayon::current_num_threads())).max(512).max(1);
        let mut bounds: Vec<usize> = (0..cells).step_by(chunk).collect();
        bounds.push(cells);
        // Extend in place, so the buffer survives from one step to the next.
        contacts.clear();
        let scan = |range: [usize; 2]| {
            scan_chunk(
                cell_keys, cell_sort, cell_start, bodies, prune, n_cells, range,
            )
        };
        if cells < par_min {
            contacts.extend(bounds.windows(2).flat_map(|w| scan([w[0], w[1]])));
        } else {
            contacts.par_extend(bounds.par_windows(2).flat_map_iter(|w| scan([w[0], w[1]])));
        }
        // Guards the `as u32` contact-index cast in the CSR fill below.
        debug_assert!(contacts.len() <= u32::MAX as usize);

        // Per-body contact index as CSR: counts, prefix sum, then the fill.
        // The count and the prefix stay sequential. The fill claims each
        // contact's two slots with an atomic fetch-add, so a body's slots land
        // in thread order. The sort pass below restores the ascending contact
        // order, which keeps the Jacobi sum deterministic. One code path
        // serves both drivers: sequential fill already produces sorted lists.
        let n = bodies.len();
        bc_start.clear();
        bc_start.resize(n + 1, 0);
        for c in contacts.iter() {
            bc_start[c.i as usize + 1] += 1;
            bc_start[c.j as usize + 1] += 1;
        }
        for k in 1..=n {
            bc_start[k] += bc_start[k - 1];
        }
        bc_items.clear();
        // Every slot is written by the fill below; only the length matters.
        bc_items.resize_with(contacts.len() * 2, || AtomicU32::new(0));
        bc_cursor.clear();
        bc_cursor.resize_with(n, || AtomicU32::new(0));
        let starts = &*bc_start;
        let cursors = &*bc_cursor;
        let items = &*bc_items;
        let fill = |(ci, c): (usize, &Contact)| {
            let ci = ci as u32;
            let rank = cursors[c.i as usize].fetch_add(1, Ordering::Relaxed);
            let slot = starts[c.i as usize] as usize + rank as usize;
            items[slot].store(ci, Ordering::Relaxed);
            let rank = cursors[c.j as usize].fetch_add(1, Ordering::Relaxed);
            let slot = starts[c.j as usize] as usize + rank as usize;
            items[slot].store(ci | J_SIDE, Ordering::Relaxed);
        };
        if contacts.len() < par_min {
            contacts.iter().enumerate().for_each(fill);
        } else {
            contacts.par_iter().enumerate().for_each(fill);
        }
        // Back to the exact order: insertion sort of each short body list by
        // contact index. The average list holds a few entries.
        (0..n).into_par_iter().for_each(|b| {
            let from = starts[b] as usize;
            let to = starts[b + 1] as usize;
            let seg = &items[from..to];
            for p in 1..seg.len() {
                let v = seg[p].load(Ordering::Relaxed);
                let key = v & !J_SIDE;
                let mut q = p;
                while q > 0 && (seg[q - 1].load(Ordering::Relaxed) & !J_SIDE) > key {
                    q -= 1;
                }
                for s in (q..p).rev() {
                    let t = seg[s].load(Ordering::Relaxed);
                    seg[s + 1].store(t, Ordering::Relaxed);
                }
                seg[q].store(v, Ordering::Relaxed);
            }
        });
        // Reclaim capacity after a spawn transient. The hysteresis keeps the
        // steady state from shrinking and re-growing every step.
        if contacts.capacity() > contacts.len() * 2 + (1 << 16) {
            contacts.shrink_to_fit();
            bc_items.shrink_to_fit();
        }
    }
}
