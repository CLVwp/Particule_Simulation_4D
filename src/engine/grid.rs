//! Cell keys, the body sort, and the neighbor stencil of the broad phase.

use rayon::prelude::*;

use super::World;
use super::config::PAR_MIN;

fn cell_of(p: [f32; 3], cell_size: f32) -> [i32; 3] {
    [
        (p[0] / cell_size).floor() as i32,
        (p[1] / cell_size).floor() as i32,
        (p[2] / cell_size).floor() as i32,
    ]
}

/// One axis of a packed cell key: 21 bits, offset to be unsigned.
pub(super) const KEY_OFF: i64 = 1 << 20;
/// Values per key field. Cells past the edge clamp onto it.
pub(super) const KEY_SPAN: i64 = 1 << 21;

/// Clamps one axis into its key field.
pub(super) fn key_part(v: i64) -> u64 {
    (v + KEY_OFF).clamp(0, KEY_SPAN - 1) as u64
}

/// Packs a cell into a sortable u64: 21 bits per axis, x highest.
/// ponytail: axes clamp at +/-1M cells (~ +/-200 km of scene); past that far
/// bodies share edge cells and the distance test rejects the fake pairs
pub(super) fn cell_key(p: [f32; 3], cell_size: f32) -> u64 {
    let [x, y, z] = cell_of(p, cell_size);
    key_part(x as i64) << 42 | key_part(y as i64) << 21 | key_part(z as i64)
}

/// Packs already-computed cell coordinates.
pub(super) fn pack_cell(x: i64, y: i64, z: i64) -> u64 {
    key_part(x) << 42 | key_part(y) << 21 | key_part(z)
}

/// The 21-bit fields back into cell coordinates.
pub(super) fn key_cell(key: u64) -> [i32; 3] {
    let mask = (KEY_SPAN - 1) as u64;
    [
        ((key >> 42) as i64 - KEY_OFF) as i32,
        ((key >> 21 & mask) as i64 - KEY_OFF) as i32,
        ((key & mask) as i64 - KEY_OFF) as i32,
    ]
}

/// Self plus the 13 lex-positive offsets of the 27-cell neighborhood.
/// For every nonzero cell delta, exactly one of `d` and `-d` is in this list.
pub(super) const STENCIL: [(i64, i64, i64); 14] = [
    (0, 0, 0),
    (0, 0, 1),
    (0, 1, -1),
    (0, 1, 0),
    (0, 1, 1),
    (1, -1, -1),
    (1, -1, 0),
    (1, -1, 1),
    (1, 0, -1),
    (1, 0, 0),
    (1, 0, 1),
    (1, 1, -1),
    (1, 1, 0),
    (1, 1, 1),
];

impl World {
    /// Packs every body's cell into a key, sorts the pairs, and records the
    /// runs. One cell holds one body diameter. Neighbor cells become binary
    /// searches over the unique keys, so there is no map and no allocation.
    pub(super) fn sort_cells(&mut self) {
        let World {
            cell_sort,
            cell_keys,
            cell_start,
            bodies,
            cell_size,
            ..
        } = self;
        let cs = *cell_size;
        let bodies = &*bodies;
        // Guards the `as u32` body-index casts below.
        debug_assert!(bodies.len() <= u32::MAX as usize);
        cell_sort.clear();
        cell_sort.resize(bodies.len(), (0, 0));
        let fill = |(i, slot): (usize, &mut (u64, u32))| {
            *slot = (cell_key(bodies[i].pos, cs), i as u32);
        };
        if cell_sort.len() < PAR_MIN {
            cell_sort.iter_mut().enumerate().for_each(fill);
            cell_sort.sort_unstable();
        } else {
            cell_sort.par_iter_mut().enumerate().for_each(fill);
            cell_sort.par_sort_unstable();
        }
        // Runs of equal key. ponytail: O(n) sequential; hide it if it ever shows
        cell_keys.clear();
        cell_start.clear();
        cell_start.push(0);
        for (pos, &(k, _)) in cell_sort.iter().enumerate() {
            if cell_keys.last() != Some(&k) {
                cell_keys.push(k);
                // The leading 0 already covers the first key's start.
                if pos > 0 {
                    cell_start.push(pos as u32);
                }
            }
        }
        cell_start.push(cell_sort.len() as u32);
    }
}
