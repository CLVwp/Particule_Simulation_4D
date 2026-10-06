//! Contact solver: impulses and position correction in Jacobi rounds.

use rayon::prelude::*;

use super::World;
use super::contacts::{Contact, J_SIDE};
use crate::engine::body::Body;

const SLOP: f32 = 0.001; // allowed penetration
const CORRECTION: f32 = 0.8; // share of overlap removed each step

/// Impulse and correction data for one contact, recomputed every round.
#[derive(Clone, Copy, Debug)]
pub(in crate::engine) struct ContactDelta {
    /// Contact normal, from `i` to `j`.
    n: [f32; 3],
    push: f32,
    impulse: f32,
    /// Unit tangent along the relative tangential speed. Zero vector when
    /// the pair does not slide.
    t: [f32; 3],
    /// Friction impulse along `t`. Zero when `pair_friction` is zero.
    jt: f32,
}

impl ContactDelta {
    const ZERO: Self = Self {
        n: [0.0; 3],
        push: 0.0,
        impulse: 0.0,
        t: [0.0; 3],
        jt: 0.0,
    };
}

// Layout guards. These types fill the hot arrays, so their size must not drift.
const _: () = assert!(std::mem::size_of::<ContactDelta>() == 36);
// ponytail: 36 bytes, not 20. The apply pass reads no velocities, so it
// cannot rebuild the tangent; the direction must ride with the delta. The
// precomputed-shares item in TODO.md grows this struct further by design.

/// Impulse + positional correction between two spheres, from their current state.
fn contact_delta(a: &Body, b: &Body, c: &Contact, rest: f32, friction: f32) -> ContactDelta {
    let d = [
        b.pos[0] - a.pos[0],
        b.pos[1] - a.pos[1],
        b.pos[2] - a.pos[2],
    ];
    let dist2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
    let min_d = a.radius + b.radius;
    if dist2 >= min_d * min_d || dist2 < 1e-12 {
        return ContactDelta::ZERO;
    }
    let dist = dist2.sqrt();
    let n = [d[0] / dist, d[1] / dist, d[2] / dist]; // from i to j

    // Relative velocity along the contact normal. Negative = closing.
    let rv = [
        b.vel[0] - a.vel[0],
        b.vel[1] - a.vel[1],
        b.vel[2] - a.vel[2],
    ];
    let vn = rv[0] * n[0] + rv[1] * n[1] + rv[2] * n[2];
    let impulse = if vn < 0.0 {
        -(1.0 + rest) * vn / (1.0 / c.mi + 1.0 / c.mj)
    } else {
        0.0
    };
    // Tangential share of the relative velocity, and the friction impulse
    // that removes the configured share of it. Zero friction costs one dot.
    let vt = [
        rv[0] - vn * n[0],
        rv[1] - vn * n[1],
        rv[2] - vn * n[2],
    ];
    let vt2 = vt[0] * vt[0] + vt[1] * vt[1] + vt[2] * vt[2];
    let (t, jt) = if friction > 0.0 && vt2 > 1e-18 {
        let vt_len = vt2.sqrt();
        let t = [vt[0] / vt_len, vt[1] / vt_len, vt[2] / vt_len];
        let jt = -friction * vt_len / (1.0 / c.mi + 1.0 / c.mj);
        (t, jt)
    } else {
        ([0.0; 3], 0.0)
    };
    let push = (min_d - dist - SLOP).max(0.0) * CORRECTION / (c.mi + c.mj);
    ContactDelta {
        n,
        push,
        impulse,
        t,
        jt,
    }
}

/// Contribution of one contact to one of its bodies. `j_side` picks the side.
fn contact_share(d: &ContactDelta, c: &Contact, j_side: bool) -> ([f32; 3], [f32; 3]) {
    // Body i moves against the normal and pays with its own mass for velocity.
    let (sign, m_pos, m_vel) = if j_side {
        (1.0, c.mi, c.mj)
    } else {
        (-1.0, c.mj, c.mi)
    };
    (
        [
            sign * d.n[0] * d.push * m_pos,
            sign * d.n[1] * d.push * m_pos,
            sign * d.n[2] * d.push * m_pos,
        ],
        [
            sign * (d.n[0] * d.impulse + d.t[0] * d.jt) / m_vel,
            sign * (d.n[1] * d.impulse + d.t[1] * d.jt) / m_vel,
            sign * (d.n[2] * d.impulse + d.t[2] * d.jt) / m_vel,
        ],
    )
}

impl World {
    /// Solves the contacts in `settings.resolve_rounds` Jacobi rounds.
    /// Each round reads one snapshot, computes all impulses in parallel, then
    /// applies them in parallel. No body is written by two threads at once.
    /// A positive `resolve_epsilon` skips the remaining rounds when the mean
    /// delta motion drops below it.
    /// ponytail: Jacobi is softer than sequential Gauss-Seidel; two rounds make up for it
    pub(super) fn resolve(&mut self) {
        let World {
            bodies,
            contacts,
            deltas,
            bc_start,
            bc_items,
            settings,
            ..
        } = self;
        let rest = settings.pair_restitution;
        let friction = settings.pair_friction;
        let par_min = settings.par_min;
        let rounds = settings.resolve_rounds.max(1);
        let epsilon = settings.resolve_epsilon;
        let items = super::atomic_u32s(bc_items);
        for _ in 0..rounds {
            // Every slot is rewritten below, so only growth needs a write.
            deltas.resize(contacts.len(), ContactDelta::ZERO);
            // Reclaim capacity after a spawn transient, with hysteresis.
            if deltas.capacity() > deltas.len() * 2 + (1 << 16) {
                deltas.shrink_to_fit();
            }
            let delta_of = |(d, c): (&mut ContactDelta, &Contact)| {
                *d = contact_delta(
                    &bodies[c.i as usize],
                    &bodies[c.j as usize],
                    c,
                    rest,
                    friction,
                );
            };
            if deltas.len() < par_min {
                deltas.iter_mut().zip(contacts.iter()).for_each(delta_of);
            } else {
                deltas
                    .par_iter_mut()
                    .zip(contacts.par_iter())
                    .for_each(delta_of);
            }
            // Resting piles produce near-zero deltas. Skip the rest of the
            // rounds. The mean normalizes the sum over the contact count.
            if epsilon > 0.0 && !deltas.is_empty() {
                let motion = deltas
                    .par_iter()
                    .map(|d| d.impulse.abs() + d.push)
                    .sum::<f32>()
                    / deltas.len() as f32;
                if motion < epsilon {
                    break;
                }
            }

            let apply = |(b, body): (usize, &mut Body)| {
                let (mut dpos, mut dvel) = ([0.0f32; 3], [0.0f32; 3]);
                let from = bc_start[b] as usize;
                let to = bc_start[b + 1] as usize;
                for tag in &items[from..to] {
                    let tag = tag.load(std::sync::atomic::Ordering::Relaxed);
                    let idx = (tag & !J_SIDE) as usize;
                    let (dp, dv) = contact_share(&deltas[idx], &contacts[idx], tag & J_SIDE != 0);
                    for k in 0..3 {
                        dpos[k] += dp[k];
                        dvel[k] += dv[k];
                    }
                }
                for k in 0..3 {
                    body.pos[k] += dpos[k];
                    body.vel[k] += dvel[k];
                }
            };
            if bodies.len() < par_min {
                bodies.iter_mut().enumerate().for_each(apply);
            } else {
                bodies.par_iter_mut().enumerate().for_each(apply);
            }
        }
    }
}
