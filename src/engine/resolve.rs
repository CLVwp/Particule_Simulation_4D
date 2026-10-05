//! Contact solver: impulses and position correction in Jacobi rounds.

use rayon::prelude::*;

use super::World;
use super::body::Body;
use super::config::PAR_MIN;
use super::contacts::{Contact, J_SIDE};

pub(super) const PAIR_RESTITUTION: f32 = 0.6;
const SLOP: f32 = 0.001; // allowed penetration
const CORRECTION: f32 = 0.8; // share of overlap removed each step
/// Parallel solve rounds per step. Two rounds keep piles stiff enough.
const RESOLVE_ROUNDS: usize = 2;

/// Impulse and correction data for one contact, recomputed every round.
#[derive(Clone, Copy, Debug)]
pub(super) struct ContactDelta {
    /// Contact normal, from `i` to `j`.
    n: [f32; 3],
    push: f32,
    impulse: f32,
}

impl ContactDelta {
    const ZERO: Self = Self {
        n: [0.0; 3],
        push: 0.0,
        impulse: 0.0,
    };
}

// Layout guards. These types fill the hot arrays, so their size must not drift.
const _: () = assert!(std::mem::size_of::<ContactDelta>() == 20);

/// Impulse + positional correction between two spheres, from their current state.
fn contact_delta(a: &Body, b: &Body, c: &Contact, rest: f32) -> ContactDelta {
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
    let push = (min_d - dist - SLOP).max(0.0) * CORRECTION / (c.mi + c.mj);
    ContactDelta { n, push, impulse }
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
            sign * d.n[0] * d.impulse / m_vel,
            sign * d.n[1] * d.impulse / m_vel,
            sign * d.n[2] * d.impulse / m_vel,
        ],
    )
}

impl World {
    /// Solves the contacts in `RESOLVE_ROUNDS` Jacobi rounds.
    /// Each round reads one snapshot, computes all impulses in parallel, then
    /// applies them in parallel. No body is written by two threads at once.
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
        for _ in 0..RESOLVE_ROUNDS {
            // Every slot is rewritten below, so only growth needs a write.
            deltas.resize(contacts.len(), ContactDelta::ZERO);
            // Reclaim capacity after a spawn transient, with hysteresis.
            if deltas.capacity() > deltas.len() * 2 + (1 << 16) {
                deltas.shrink_to_fit();
            }
            let delta_of = |(d, c): (&mut ContactDelta, &Contact)| {
                *d = contact_delta(&bodies[c.i as usize], &bodies[c.j as usize], c, rest);
            };
            if deltas.len() < PAR_MIN {
                deltas.iter_mut().zip(contacts.iter()).for_each(delta_of);
            } else {
                deltas
                    .par_iter_mut()
                    .zip(contacts.par_iter())
                    .for_each(delta_of);
            }

            let apply = |(b, body): (usize, &mut Body)| {
                let (mut dpos, mut dvel) = ([0.0f32; 3], [0.0f32; 3]);
                let from = bc_start[b] as usize;
                let to = bc_start[b + 1] as usize;
                for &tag in &bc_items[from..to] {
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
            if bodies.len() < PAR_MIN {
                bodies.iter_mut().enumerate().for_each(apply);
            } else {
                bodies.par_iter_mut().enumerate().for_each(apply);
            }
        }
    }
}
