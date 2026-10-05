//! The world container: bodies, the step pipeline, and pool helpers.

use std::time::Instant;

use rayon::prelude::*;

use super::body::{Body, Shape};
use super::config::{BODY_RADIUS, FLOOR_Y, MIN_RADIUS, PAR_MIN, SimSettings};
use super::contacts::Contact;
use super::resolve::ContactDelta;
use super::rng::Rng;

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
#[derive(Debug)]
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
    pub(super) bc_cursor: Vec<u32>,
    /// Contact tags in body order. Bit `J_SIDE` marks the `j` side.
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

    /// Spawns `n` bodies around `origin` with fountain-like velocities. The
    /// spread grows with `n`, so one big spawn stays a cloud, not a point.
    pub fn spawn(&mut self, n: usize, origin: [f32; 3], speed: f32, shape: Shape, radius: f32) {
        let radius = radius.max(MIN_RADIUS);
        self.cell_size = self.cell_size.max(2.0 * radius);
        self.bodies.reserve(n);
        let mut rng = Rng(0x2545F4914F6CDD1D ^ n as u64);
        // A dense point makes the first step pair every body with every
        // neighbor. Spread over a body-proportional volume instead: at 21 %
        // packing the candidate pairs stay bounded by the neighborhood.
        let spread = (radius * (n as f32 / 0.4).cbrt()).max(0.2);
        for _ in 0..n {
            self.bodies.push(Body {
                pos: [
                    origin[0] + rng.next_f32() * spread,
                    origin[1] + rng.next_f32() * spread,
                    origin[2] + rng.next_f32() * spread,
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
