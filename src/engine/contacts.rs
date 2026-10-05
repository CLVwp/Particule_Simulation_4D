//! Broad phase: candidate contact pairs from the sorted cell grid.

use rayon::prelude::*;

use super::World;
use super::config::PAR_MIN;
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

impl World {
    /// Finds overlapping pairs cell by cell. Each cell binary-searches its
    /// stencil neighbors in the unique keys, then emits body pairs. A pair is
    /// kept only when the two spheres overlap.
    ///
    /// The stencil holds self plus the 13 lex-positive offsets of the 27-cell
    /// neighborhood. Every pair of cells within reach meets in exactly one
    /// stencil direction, so cross-cell pairs need no filter. Same-cell pairs
    /// keep `j > i`.
    ///
    /// The per-body index fill stays sequential on purpose: each body's list
    /// must keep contact order, or the Jacobi sum stops being deterministic.
    /// ponytail: a parallel fill needs per-contact ranks; more passes than it saves below ~100k contacts
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
            ..
        } = self;
        let cell_sort = &*cell_sort;
        let cell_keys = &*cell_keys;
        let cell_start = &*cell_start;
        let bodies = &*bodies;
        // Guards the `as u32` cell-count cast in the scan below.
        debug_assert!(bodies.len() <= u32::MAX as usize);
        let n_cells = cell_keys.len();
        let cell_scan = move |c: u32| {
            let c = c as usize;
            let [cx, cy, cz] = key_cell(cell_keys[c]);
            let own_lo = cell_start[c] as usize;
            let own_hi = cell_start[c + 1] as usize;
            STENCIL
                .iter()
                .filter_map(move |&(dx, dy, dz)| {
                    let key = pack_cell(cx as i64 + dx, cy as i64 + dy, cz as i64 + dz);
                    let lo = cell_keys.partition_point(|&k| k < key);
                    if lo >= n_cells || cell_keys[lo] != key {
                        return None;
                    }
                    let from = cell_start[lo] as usize;
                    let to = cell_start[lo + 1] as usize;
                    // The found run is the own cell only for (0, 0, 0).
                    let same = lo == c;
                    Some(cell_sort[from..to].iter().flat_map(move |&(_, bj)| {
                        cell_sort[own_lo..own_hi]
                            .iter()
                            .filter_map(move |&(_, bi)| {
                                if same && bi >= bj {
                                    return None;
                                }
                                let (pi, pj) = (&bodies[bi as usize], &bodies[bj as usize]);
                                let d = [
                                    pj.pos[0] - pi.pos[0],
                                    pj.pos[1] - pi.pos[1],
                                    pj.pos[2] - pi.pos[2],
                                ];
                                let dist2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
                                let min_d = pi.radius + pj.radius;
                                // Same test as `contact_delta`, so no live pair is lost.
                                if dist2 >= min_d * min_d || dist2 < 1e-12 {
                                    return None;
                                }
                                Some(Contact {
                                    i: bi,
                                    j: bj,
                                    mi: pi.mass(),
                                    mj: pj.mass(),
                                })
                            })
                    }))
                })
                .flatten()
        };
        // Extend in place, so the buffer survives from one step to the next.
        contacts.clear();
        if n_cells < PAR_MIN {
            contacts.extend((0..n_cells as u32).flat_map(cell_scan));
        } else {
            contacts.par_extend((0..n_cells as u32).into_par_iter().flat_map_iter(cell_scan));
        }
        // Guards the `as u32` contact-index cast in the CSR fill below.
        debug_assert!(contacts.len() <= u32::MAX as usize);

        // Per-body contact index as CSR: counts, prefix sum, then the fill.
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
        bc_items.resize(contacts.len() * 2, 0);
        bc_cursor.clear();
        bc_cursor.extend_from_slice(&bc_start[..n]);
        for (ci, c) in contacts.iter().enumerate() {
            let ci = ci as u32;
            bc_items[bc_cursor[c.i as usize] as usize] = ci;
            bc_cursor[c.i as usize] += 1;
            bc_items[bc_cursor[c.j as usize] as usize] = ci | J_SIDE;
            bc_cursor[c.j as usize] += 1;
        }
        // Reclaim capacity after a spawn transient. The hysteresis keeps the
        // steady state from shrinking and re-growing every step.
        if contacts.capacity() > contacts.len() * 2 + (1 << 16) {
            contacts.shrink_to_fit();
            bc_items.shrink_to_fit();
        }
    }
}
