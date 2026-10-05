//! Physics core — no GPUI here, testable standalone.
//!
//! `step` splits its work over the rayon pool. The pool has one worker per
//! logical core, detected from the CPU at startup.

use std::time::Instant;

use rayon::prelude::*;

pub mod fluid;

// ponytail: hand-rolled LCG instead of the `rand` crate; swap if we need real distributions
struct Rng(u64);

/// Multiplier of the Knuth MMIX linear congruential generator.
const LCG_MULT: u64 = 6364136223846793005;
/// Increment of the Knuth MMIX linear congruential generator.
const LCG_INC: u64 = 1442695040888963407;

impl Rng {
    /// Uniform in [-1, 1].
    fn next_f32(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(LCG_MULT).wrapping_add(LCG_INC);
        // The top 32 bits fill a full u32, so the fraction spans [0, 1].
        ((self.0 >> 32) as u32 as f32 / u32::MAX as f32) * 2.0 - 1.0
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
/// Smallest radius `spawn` accepts. Thinner bodies divide by zero in the solver.
pub const MIN_RADIUS: f32 = 0.05;

const PAIR_RESTITUTION: f32 = 0.6;
const SLOP: f32 = 0.001; // allowed penetration
const CORRECTION: f32 = 0.8; // share of overlap removed each step
/// Parallel solve rounds per step. Two rounds keep piles stiff enough.
const RESOLVE_ROUNDS: usize = 2;
/// High bit of a `body_contacts` tag. Set when the body is the `j` side.
const J_SIDE: u32 = 1 << 31;
/// Pool work starts above this body count.
/// ponytail: bench shows 16-thread sync is a net loss at 1000 bodies, a win at 4000
pub const PAR_MIN: usize = 2048;

/// Tunable laws of motion. `Default` matches the constants above.
#[derive(Clone, Copy, Debug)]
pub struct SimSettings {
    /// Downward acceleration, in units per second squared.
    pub gravity: f32,
    /// Fraction of vertical speed kept after a floor bounce.
    pub floor_restitution: f32,
    /// Fraction of horizontal speed kept while a body touches the floor.
    pub ground_friction: f32,
    /// Fraction of relative speed kept when two bodies collide.
    pub pair_restitution: f32,
}

impl Default for SimSettings {
    fn default() -> Self {
        SimSettings {
            gravity: GRAVITY,
            floor_restitution: FLOOR_RESTITUTION,
            ground_friction: GROUND_FRICTION,
            pair_restitution: PAIR_RESTITUTION,
        }
    }
}

/// Visual shape of a body. Physics always uses a sphere of the same radius.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Shape {
    /// Drawn as a sphere.
    Sphere,
    /// Drawn as a box. Physics still uses the sphere.
    Cube,
}

/// Logical cores visible to this process.
#[must_use]
pub fn thread_count() -> usize {
    std::thread::available_parallelism()
        .map(usize::from)
        .unwrap_or(1)
}

/// One particle: a sphere with a position and a velocity.
#[derive(Clone, Copy, Debug)]
pub struct Body {
    /// Position in world units, as x, y, z.
    pub pos: [f32; 3],
    /// Velocity in units per second.
    pub vel: [f32; 3],
    /// Sphere radius in world units.
    pub radius: f32,
    /// Visual shape. Physics always uses the sphere.
    pub shape: Shape,
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

// Layout guards. These types fill the hot arrays, so their size must not drift.
// Body holds 7 floats and a 1-byte tag. Alignment pads it to 32 bytes.
const _: () = assert!(std::mem::size_of::<Body>() == 32);
const _: () = assert!(std::mem::size_of::<Contact>() == 16);
const _: () = assert!(std::mem::size_of::<ContactDelta>() == 20);

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
#[derive(Debug)]
pub struct World {
    /// Every body in the world. Callers may edit them between steps.
    pub bodies: Vec<Body>,
    /// Laws of motion. Change them freely between steps.
    pub settings: SimSettings,
    /// Grid cell edge. Grows to fit the biggest spawned radius.
    cell_size: f32,
    /// Candidate contacts found this step.
    contacts: Vec<Contact>,
    /// Per-contact solve results for the current round.
    deltas: Vec<ContactDelta>,
    /// Cell key + body index, sorted by key. Replaces the hash grid.
    cell_sort: Vec<(u64, u32)>,
    /// Unique cell keys of `cell_sort`, ascending. Runs of equal key.
    cell_keys: Vec<u64>,
    /// Start offset of each key's run in `cell_sort`, plus the end. One entry
    /// per key, last entry is `cell_sort.len()`.
    cell_start: Vec<u32>,
    /// Per-body contact index as flat arrays: offsets into `bc_items`.
    bc_start: Vec<u32>,
    /// Fill cursor for `bc_items`, kept across steps to avoid reallocation.
    bc_cursor: Vec<u32>,
    /// Contact tags in body order. Bit `J_SIDE` marks the `j` side.
    bc_items: Vec<u32>,
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

    /// Spawns `n` bodies at `origin` with fountain-like velocities.
    pub fn spawn(&mut self, n: usize, origin: [f32; 3], speed: f32, shape: Shape, radius: f32) {
        let radius = radius.max(MIN_RADIUS);
        self.cell_size = self.cell_size.max(2.0 * radius);
        self.bodies.reserve(n);
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
                radius,
                shape,
            });
        }
    }

    /// Spawns spheres at the default radius.
    pub fn spawn_wave(&mut self, n: usize, origin: [f32; 3], speed: f32) {
        self.spawn(n, origin, speed, Shape::Sphere, BODY_RADIUS);
    }

    /// Removes every body and resets the grid to its base size.
    pub fn clear(&mut self) {
        self.bodies.clear();
        self.cell_size = 2.0 * BODY_RADIUS;
    }

    /// Advances the world by `dt` seconds.
    pub fn step(&mut self, dt: f32) {
        let t = Instant::now();
        self.integrate(dt);
        self.phase_ms[0] = ms_since(t);
        let t = Instant::now();
        self.sort_cells();
        self.phase_ms[1] = ms_since(t);
        let t = Instant::now();
        self.build_contacts();
        self.phase_ms[2] = ms_since(t);
        let t = Instant::now();
        self.resolve();
        self.phase_ms[3] = ms_since(t);
        let t = Instant::now();
        self.collide_floor();
        self.phase_ms[4] = ms_since(t);
    }

    /// Explicit Euler integration, split across the pool.
    fn integrate(&mut self, dt: f32) {
        let g = self.settings.gravity;
        par_each(&mut self.bodies, |b| {
            b.vel[1] += g * dt;
            b.pos[0] += b.vel[0] * dt;
            b.pos[1] += b.vel[1] * dt;
            b.pos[2] += b.vel[2] * dt;
        });
    }

    /// Packs every body's cell into a key, sorts the pairs, and records the
    /// runs. One cell holds one body diameter. Neighbor cells become binary
    /// searches over the unique keys, so there is no map and no allocation.
    fn sort_cells(&mut self) {
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

    /// Finds candidate pairs cell by cell. Each cell binary-searches its
    /// stencil neighbors in the unique keys, then emits body pairs.
    ///
    /// The stencil holds self plus the 13 lex-positive offsets of the 27-cell
    /// neighborhood. Every pair of cells within reach meets in exactly one
    /// stencil direction, so cross-cell pairs need no filter. Same-cell pairs
    /// keep `j > i`.
    ///
    /// The per-body index fill stays sequential on purpose: each body's list
    /// must keep contact order, or the Jacobi sum stops being deterministic.
    /// ponytail: a parallel fill needs per-contact ranks; more passes than it saves below ~100k contacts
    fn build_contacts(&mut self) {
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
                                Some(Contact {
                                    i: bi,
                                    j: bj,
                                    mi: bodies[bi as usize].mass(),
                                    mj: bodies[bj as usize].mass(),
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
            bc_start,
            bc_items,
            settings,
            ..
        } = self;
        let rest = settings.pair_restitution;
        for _ in 0..RESOLVE_ROUNDS {
            // Every slot is rewritten below, so only growth needs a write.
            deltas.resize(contacts.len(), ContactDelta::ZERO);
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

    /// Floor plane at `FLOOR_Y`, spheres rest on top of it.
    fn collide_floor(&mut self) {
        let rest = self.settings.floor_restitution;
        let friction = self.settings.ground_friction;
        par_each(&mut self.bodies, |b| {
            if b.pos[1] - b.radius < FLOOR_Y && b.vel[1] < 0.0 {
                b.pos[1] = FLOOR_Y + b.radius;
                b.vel[1] = -b.vel[1] * rest;
                b.vel[0] *= friction;
                b.vel[2] *= friction;
            }
        });
    }
}

/// Wall time since `t`, in milliseconds.
fn ms_since(t: Instant) -> f32 {
    t.elapsed().as_secs_f32() * 1000.0
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

fn cell_of(p: [f32; 3], cell_size: f32) -> [i32; 3] {
    [
        (p[0] / cell_size).floor() as i32,
        (p[1] / cell_size).floor() as i32,
        (p[2] / cell_size).floor() as i32,
    ]
}

/// One axis of a packed cell key: 21 bits, offset to be unsigned.
const KEY_OFF: i64 = 1 << 20;
/// Values per key field. Cells past the edge clamp onto it.
const KEY_SPAN: i64 = 1 << 21;

/// Clamps one axis into its key field.
fn key_part(v: i64) -> u64 {
    (v + KEY_OFF).clamp(0, KEY_SPAN - 1) as u64
}

/// Packs a cell into a sortable u64: 21 bits per axis, x highest.
/// ponytail: axes clamp at +/-1M cells (~ +/-200 km of scene); past that far
/// bodies share edge cells and the distance test rejects the fake pairs
fn cell_key(p: [f32; 3], cell_size: f32) -> u64 {
    let [x, y, z] = cell_of(p, cell_size);
    key_part(x as i64) << 42 | key_part(y as i64) << 21 | key_part(z as i64)
}

/// Packs already-computed cell coordinates.
fn pack_cell(x: i64, y: i64, z: i64) -> u64 {
    key_part(x) << 42 | key_part(y) << 21 | key_part(z)
}

/// The 21-bit fields back into cell coordinates.
fn key_cell(key: u64) -> [i32; 3] {
    let mask = (KEY_SPAN - 1) as u64;
    [
        ((key >> 42) as i64 - KEY_OFF) as i32,
        ((key >> 21 & mask) as i64 - KEY_OFF) as i32,
        ((key & mask) as i64 - KEY_OFF) as i32,
    ]
}

/// Self plus the 13 lex-positive offsets of the 27-cell neighborhood.
/// For every nonzero cell delta, exactly one of `d` and `-d` is in this list.
const STENCIL: [(i64, i64, i64); 14] = [
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
            shape: Shape::Sphere,
        });
        w.bodies.push(Body {
            pos: [0.5, 2.0, 0.0],
            vel: [-0.5, 0.0, 0.0],
            radius: BODY_RADIUS,
            shape: Shape::Sphere,
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

    #[test]
    fn gravity_setting_controls_fall_speed() {
        // Same spawn seed, so the only difference is the law of motion.
        let mut light = World::new();
        light.settings.gravity = -2.0;
        let mut heavy = World::new();
        heavy.settings.gravity = -20.0;
        light.spawn_wave(1, [0.0, 3.0, 0.0], 0.0);
        heavy.spawn_wave(1, [0.0, 3.0, 0.0], 0.0);
        for _ in 0..30 {
            light.step(1.0 / 60.0);
            heavy.step(1.0 / 60.0);
        }
        assert!(
            heavy.bodies[0].pos[1] < light.bodies[0].pos[1],
            "stronger gravity must fall faster"
        );
    }

    #[test]
    fn custom_size_and_shape_bodies_collide() {
        let mut w = World::new();
        w.spawn(2, [0.0, 2.0, 0.0], 0.0, Shape::Cube, 0.5);
        assert_eq!(w.bodies[0].shape, Shape::Cube);
        assert_eq!(w.bodies[0].radius, 0.5);
        // Deterministic head-on setup: overwrite the fountain jitter.
        w.bodies[0].pos = [-1.0, 2.0, 0.0];
        w.bodies[0].vel = [0.5, 0.0, 0.0];
        w.bodies[1].pos = [1.0, 2.0, 0.0];
        w.bodies[1].vel = [-0.5, 0.0, 0.0];
        let min_d = 1.0;
        let mut bounced = false;
        for _ in 0..600 {
            w.step(1.0 / 60.0);
            let d = (w.bodies[0].pos[0] - w.bodies[1].pos[0]).abs()
                + (w.bodies[0].pos[1] - w.bodies[1].pos[1]).abs()
                + (w.bodies[0].pos[2] - w.bodies[1].pos[2]).abs();
            assert!(d >= min_d - 0.005, "big bodies overlap");
            if d < min_d + 0.1 && w.bodies[0].vel[0] < 0.0 && w.bodies[1].vel[0] > 0.0 {
                bounced = true;
            }
        }
        assert!(bounced, "big bodies never bounced off each other");
    }

    #[test]
    fn cells_are_sorted_and_match_bodies() {
        let mut w = World::new();
        w.spawn_wave(200, [0.0, 5.0, 0.0], 4.0);
        w.step(1.0 / 60.0);
        // step() moves bodies after the sort (contacts push them), so re-sync
        // the array with the final positions before checking.
        w.sort_cells();
        assert_eq!(w.cell_sort.len(), w.bodies.len(), "lost bodies in the sort");
        assert!(
            w.cell_sort.windows(2).all(|pair| pair[0] <= pair[1]),
            "cell keys not sorted"
        );
        // Every body finds its own (key, index) pair in the sorted array.
        let cs = 2.0 * BODY_RADIUS;
        for (i, b) in w.bodies.iter().enumerate() {
            let entry = (cell_key(b.pos, cs), i as u32);
            assert!(
                w.cell_sort.binary_search(&entry).is_ok(),
                "body {i} missing from the sorted cells"
            );
        }
    }

    #[test]
    fn next_f32_spans_full_range() {
        // Fresh generator: this must not disturb the spawn seeds of other tests.
        let mut rng = Rng(0x853C_49E6_748F_EA9B);
        let (mut lo, mut hi) = (f32::MAX, f32::MIN);
        for _ in 0..10_000 {
            let v = rng.next_f32();
            lo = lo.min(v);
            hi = hi.max(v);
        }
        assert!(lo < -0.9, "never sampled below {lo}");
        assert!(hi > 0.9, "never sampled above {hi}");
    }

    #[test]
    fn key_part_clamps_extreme_cells_into_the_field() {
        // No input may escape the 21-bit key field.
        let cases = [
            0,
            1,
            -1,
            KEY_OFF - 1,
            KEY_OFF,
            -KEY_OFF,
            -KEY_OFF - 1,
            1 << 62,
            -(1 << 62),
        ];
        for &v in &cases {
            assert!(
                key_part(v) < KEY_SPAN as u64,
                "key_part({v}) left the field"
            );
        }
        // Values past each edge fold onto the first and last field value.
        assert_eq!(key_part(-KEY_OFF - 1), 0);
        assert_eq!(key_part(-KEY_OFF), 0);
        assert_eq!(key_part(KEY_OFF - 1), (KEY_SPAN - 1) as u64);
        assert_eq!(key_part(KEY_OFF), (KEY_SPAN - 1) as u64);
    }
}
