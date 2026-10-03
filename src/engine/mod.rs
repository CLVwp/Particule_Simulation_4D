//! Physics core — no GPUI here, testable standalone.
//!
//! `step` splits its work over the rayon pool. The pool has one worker per
//! logical core, detected from the CPU at startup.

use rayon::prelude::*;
use rustc_hash::FxHashMap;

// ponytail: hand-rolled LCG instead of the `rand` crate; swap if we need real distributions
struct Rng(u64);

impl Rng {
    /// Uniform in [-1, 1].
    fn next_f32(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 33) as f32 / u32::MAX as f32) * 2.0 - 1.0
    }
}

/// Downward acceleration, in units per second squared.
pub const GRAVITY: f32 = -9.81;
/// Height of the floor plane.
pub const FLOOR_Y: f32 = 0.0;
/// Fraction of vertical speed kept after a floor bounce.
pub const FLOOR_RESTITUTION: f32 = 0.75;
/// Fraction of horizontal speed kept while a body touches the floor.
pub const GROUND_FRICTION: f32 = 0.9;
/// Radius of every spawned body.
pub const BODY_RADIUS: f32 = 0.1;

const PAIR_RESTITUTION: f32 = 0.6;
const SLOP: f32 = 0.001; // allowed penetration
const CORRECTION: f32 = 0.8; // share of overlap removed each step
const CELL_SIZE: f32 = 2.0 * BODY_RADIUS; // grid cell: one body diameter
/// Parallel solve rounds per step. Two rounds keep piles stiff enough.
const RESOLVE_ROUNDS: usize = 2;
/// High bit of a `body_contacts` tag. Set when the body is the `j` side.
const J_SIDE: u32 = 1 << 31;
/// Pool work starts above this body count.
/// ponytail: bench shows 16-thread sync is a net loss at 1000 bodies, a win at 4000
const PAR_MIN: usize = 2048;

/// Logical cores visible to this process.
pub fn thread_count() -> usize {
    std::thread::available_parallelism()
        .map(usize::from)
        .unwrap_or(1)
}

/// One particle: a sphere with a position and a velocity.
#[derive(Clone, Copy, Debug)]
pub struct Body {
    pub pos: [f32; 3],
    pub vel: [f32; 3],
    pub radius: f32,
}

impl Body {
    /// Uniform density: mass grows with the volume.
    fn mass(&self) -> f32 {
        self.radius * self.radius * self.radius
    }
}

/// A pair of bodies that may overlap. Masses are stored once; they never change.
#[derive(Clone, Copy, Debug)]
struct Contact {
    i: u32,
    j: u32,
    mi: f32,
    mj: f32,
}

/// Impulse and correction data for one contact, recomputed every round.
#[derive(Clone, Copy, Debug)]
struct ContactDelta {
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

/// Particle positions and velocities in x, y, z (time lives in `step(dt)`).
#[derive(Debug, Default)]
pub struct World {
    pub bodies: Vec<Body>,
    /// Spatial hash grid. Cleared each step; the cell buffers stay allocated.
    grid: FxHashMap<[i32; 3], Vec<u32>>,
    /// Candidate contacts found in the grid this step.
    contacts: Vec<Contact>,
    /// Per-contact solve results for the current round.
    deltas: Vec<ContactDelta>,
    /// For each body: tags of the contacts that touch it. Bit `J_SIDE` marks the `j` side.
    body_contacts: Vec<Vec<u32>>,
}

impl World {
    pub fn new() -> Self {
        World::default()
    }

    /// Spawns `n` bodies at `origin` with fountain-like velocities.
    pub fn spawn_wave(&mut self, n: usize, origin: [f32; 3], speed: f32) {
        let mut rng = Rng(0x2545F4914F6CDD1D ^ n as u64);
        for _ in 0..n {
            self.bodies.push(Body {
                pos: [
                    origin[0] + rng.next_f32() * 0.2,
                    origin[1] + rng.next_f32() * 0.2,
                    origin[2] + rng.next_f32() * 0.2,
                ],
                vel: [
                    rng.next_f32() * speed * 0.4,
                    speed * (0.7 + 0.3 * rng.next_f32()),
                    rng.next_f32() * speed * 0.4,
                ],
                radius: BODY_RADIUS,
            });
        }
    }

    pub fn clear(&mut self) {
        self.bodies.clear();
    }

    pub fn step(&mut self, dt: f32) {
        self.integrate(dt);
        self.build_grid();
        self.build_contacts();
        self.resolve();
        self.collide_floor();
    }

    /// Explicit Euler integration, split across the pool.
    fn integrate(&mut self, dt: f32) {
        par_each(&mut self.bodies, |b| {
            b.vel[1] += GRAVITY * dt;
            b.pos[0] += b.vel[0] * dt;
            b.pos[1] += b.vel[1] * dt;
            b.pos[2] += b.vel[2] * dt;
        });
    }

    /// Fills the spatial hash grid. One cell holds one body diameter.
    /// ponytail: O(n) grid with 27-cell neighborhoods; a BVH only pays off past ~50k bodies
    fn build_grid(&mut self) {
        let World { grid, bodies, .. } = self;
        for cell in grid.values_mut() {
            cell.clear();
        }
        for (i, b) in bodies.iter().enumerate() {
            grid.entry(cell_of(b.pos)).or_default().push(i as u32);
        }
    }

    /// Finds candidate pairs: body `i` scans its 27 neighbor cells and keeps `j > i`.
    /// Runs on the pool; the grid and the bodies are read-only here.
    fn build_contacts(&mut self) {
        let World {
            grid,
            bodies,
            contacts,
            body_contacts,
            ..
        } = self;
        let grid = &*grid;
        let bodies = &*bodies;
        let scan = |i: u32| {
            let c = cell_of(bodies[i as usize].pos);
            let mi = bodies[i as usize].mass();
            (-1..=1i32)
                .flat_map(move |dx| {
                    (-1..=1i32).flat_map(move |dy| {
                        (-1..=1i32)
                            .filter_map(move |dz| grid.get(&[c[0] + dx, c[1] + dy, c[2] + dz]))
                    })
                })
                .flat_map(|cell| cell.iter().copied())
                .filter(move |&j| j > i)
                .map(move |j| Contact {
                    i,
                    j,
                    mi,
                    mj: bodies[j as usize].mass(),
                })
        };
        *contacts = if bodies.len() < PAR_MIN {
            (0..bodies.len() as u32).flat_map(scan).collect()
        } else {
            (0..bodies.len() as u32)
                .into_par_iter()
                .flat_map_iter(scan)
                .collect()
        };

        body_contacts.clear();
        body_contacts.resize(bodies.len(), Vec::new());
        for (ci, c) in contacts.iter().enumerate() {
            body_contacts[c.i as usize].push(ci as u32);
            body_contacts[c.j as usize].push(ci as u32 | J_SIDE);
        }
    }

    /// Solves the contacts in `RESOLVE_ROUNDS` Jacobi rounds.
    /// Each round reads one snapshot, computes all impulses in parallel, then
    /// applies them in parallel. No body is written by two threads at once.
    /// ponytail: Jacobi is softer than sequential Gauss-Seidel; two rounds make up for it
    fn resolve(&mut self) {
        let World {
            bodies,
            contacts,
            deltas,
            body_contacts,
            ..
        } = self;
        for _ in 0..RESOLVE_ROUNDS {
            deltas.clear();
            deltas.resize(contacts.len(), ContactDelta::ZERO);
            let delta_of = |(d, c): (&mut ContactDelta, &Contact)| {
                *d = contact_delta(&bodies[c.i as usize], &bodies[c.j as usize], c);
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
                for &tag in &body_contacts[b] {
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

    /// Floor plane at `FLOOR_Y`, spheres rest on top of it.
    fn collide_floor(&mut self) {
        par_each(&mut self.bodies, |b| {
            if b.pos[1] - b.radius < FLOOR_Y && b.vel[1] < 0.0 {
                b.pos[1] = FLOOR_Y + b.radius;
                b.vel[1] = -b.vel[1] * FLOOR_RESTITUTION;
                b.vel[0] *= GROUND_FRICTION;
                b.vel[2] *= GROUND_FRICTION;
            }
        });
    }
}

/// Runs `f` on every element, on the pool above `PAR_MIN`, inline below.
fn par_each<T: Send>(slice: &mut [T], f: impl Fn(&mut T) + Sync + Send) {
    if slice.len() < PAR_MIN {
        slice.iter_mut().for_each(f);
    } else {
        slice.par_iter_mut().for_each(f);
    }
}

/// Impulse + positional correction between two spheres, from their current state.
fn contact_delta(a: &Body, b: &Body, c: &Contact) -> ContactDelta {
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
        -(1.0 + PAIR_RESTITUTION) * vn / (1.0 / c.mi + 1.0 / c.mj)
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

fn cell_of(p: [f32; 3]) -> [i32; 3] {
    [
        (p[0] / CELL_SIZE).floor() as i32,
        (p[1] / CELL_SIZE).floor() as i32,
        (p[2] / CELL_SIZE).floor() as i32,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn falls_and_bounces_never_below_floor() {
        let mut w = World::new();
        w.spawn_wave(1, [0.0, 5.0, 0.0], 0.0);
        let mut bounced = false;
        for _ in 0..600 {
            w.step(1.0 / 60.0);
            assert!(
                w.bodies[0].pos[1] >= FLOOR_Y - 1e-4,
                "sank through the floor"
            );
            if w.bodies[0].pos[1] <= FLOOR_Y + 2.0 * BODY_RADIUS && w.bodies[0].vel[1] > 0.0 {
                bounced = true;
            }
        }
        assert!(bounced, "never bounced off the floor");
    }

    #[test]
    fn head_on_collision_pushes_bodies_apart() {
        let mut w = World::new();
        // Mid-air meeting point: floor friction never touches the test.
        w.bodies.push(Body {
            pos: [-0.5, 2.0, 0.0],
            vel: [0.5, 0.0, 0.0],
            radius: BODY_RADIUS,
        });
        w.bodies.push(Body {
            pos: [0.5, 2.0, 0.0],
            vel: [-0.5, 0.0, 0.0],
            radius: BODY_RADIUS,
        });
        let min_d = 2.0 * BODY_RADIUS;
        let mut bounced = false;
        for _ in 0..600 {
            w.step(1.0 / 60.0);
            let d = (w.bodies[0].pos[0] - w.bodies[1].pos[0]).abs()
                + (w.bodies[0].pos[1] - w.bodies[1].pos[1]).abs()
                + (w.bodies[0].pos[2] - w.bodies[1].pos[2]).abs();
            // Positional correction leaves SLOP plus a small remainder. It is by design.
            assert!(d >= min_d - 0.005, "bodies overlap");
            if d < min_d + 0.1 && w.bodies[0].vel[0] < 0.0 && w.bodies[1].vel[0] > 0.0 {
                bounced = true;
            }
        }
        assert!(bounced, "bodies never bounced off each other");
    }

    #[test]
    fn result_does_not_depend_on_thread_count() {
        // 3000 bodies sit above PAR_MIN, so the pooled run takes the parallel path.
        let mut solo = World::new();
        let mut pooled = World::new();
        solo.spawn_wave(3000, [0.0, 5.0, 0.0], 4.0);
        pooled.spawn_wave(3000, [0.0, 5.0, 0.0], 4.0);
        let one_thread = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap();
        for _ in 0..120 {
            one_thread.install(|| solo.step(1.0 / 60.0));
            pooled.step(1.0 / 60.0);
        }
        for (a, b) in solo.bodies.iter().zip(&pooled.bodies) {
            for k in 0..3 {
                assert!(
                    (a.pos[k] - b.pos[k]).abs() < 1e-4,
                    "1-thread and N-thread runs diverged"
                );
            }
        }
    }
}
