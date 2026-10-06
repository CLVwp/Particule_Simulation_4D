//! Spawning: body creation with a fountain-like spread, or on a lattice.

use super::World;
use crate::engine::body::{Body, Shape};
use crate::engine::config::{BODY_RADIUS, MIN_RADIUS};
use crate::engine::rng::Rng;

/// Root seed of the default spawn stream. Same stream per count `n`, so one
/// count always builds the same cloud.
const SPAWN_SEED: u64 = 0x2545_F491_4F6C_DD1D;

impl World {
    /// Spawns `n` bodies around `origin` with fountain-like velocities. The
    /// spread grows with `n`, so one big spawn stays a cloud, not a point.
    /// The seed derives from the count alone.
    pub fn spawn(&mut self, n: usize, origin: [f32; 3], speed: f32, shape: Shape, radius: f32) {
        self.spawn_seeded(n, origin, speed, shape, radius, SPAWN_SEED ^ n as u64);
    }

    /// Spawns `n` bodies like [`World::spawn`], from an explicit seed. Two
    /// calls with the same seed build the same cloud; different seeds keep
    /// same-size spawns from landing on the same spots.
    pub fn spawn_seeded(
        &mut self,
        n: usize,
        origin: [f32; 3],
        speed: f32,
        shape: Shape,
        radius: f32,
        seed: u64,
    ) {
        let radius = radius.max(MIN_RADIUS);
        self.cell_size = self.cell_size.max(2.0 * radius);
        self.bodies.reserve(n);
        let mut rng = Rng(seed);
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

    /// Spawns `side` cubed bodies on a lattice with no velocity. `spacing`
    /// sets the lattice step; at or above `2 * radius` the pile starts out
    /// of overlap. No randomness: identical calls build identical piles.
    pub fn spawn_grid(
        &mut self,
        side: usize,
        origin: [f32; 3],
        spacing: f32,
        shape: Shape,
        radius: f32,
    ) {
        let radius = radius.max(MIN_RADIUS);
        self.cell_size = self.cell_size.max(2.0 * radius);
        let count = side
            .checked_mul(side)
            .and_then(|s| s.checked_mul(side))
            .expect("lattice count overflows");
        // Guards the `as u32` body-index casts in the scan.
        debug_assert!(count <= u32::MAX as usize);
        self.bodies.reserve(count);
        for k in 0..side {
            for j in 0..side {
                for i in 0..side {
                    self.bodies.push(Body {
                        pos: [
                            origin[0] + i as f32 * spacing,
                            origin[1] + j as f32 * spacing,
                            origin[2] + k as f32 * spacing,
                        ],
                        vel: [0.0; 3],
                        radius,
                        shape,
                    });
                }
            }
        }
    }

    /// Spawns spheres at the default radius.
    pub fn spawn_wave(&mut self, n: usize, origin: [f32; 3], speed: f32) {
        self.spawn(n, origin, speed, Shape::Sphere, BODY_RADIUS);
    }
}

#[cfg(test)]
mod tests {
    use super::World;
    use crate::engine::body::Shape;
    use crate::engine::config::BODY_RADIUS;

    #[test]
    fn seeds_change_the_cloud() {
        // Two same-size spawns from one origin must not land on the same
        // spots, or the 1e-12 near-coincidence elides every pair.
        let mut a = World::new();
        let mut b = World::new();
        a.spawn_seeded(100, [0.0, 5.0, 0.0], 0.0, Shape::Sphere, BODY_RADIUS, 1);
        b.spawn_seeded(100, [0.0, 5.0, 0.0], 0.0, Shape::Sphere, BODY_RADIUS, 2);
        let same = a
            .bodies
            .iter()
            .zip(&b.bodies)
            .filter(|(x, y)| x.pos == y.pos)
            .count();
        assert_eq!(same, 0, "{same} bodies landed on the same spots");
    }

    #[test]
    fn default_spawn_stays_reproducible() {
        // The seed derives from the count alone. Tests and benches rely on
        // the replay, so the default stream must not drift.
        let mut a = World::new();
        let mut b = World::new();
        a.spawn_wave(50, [0.0, 5.0, 0.0], 2.0);
        b.spawn_wave(50, [0.0, 5.0, 0.0], 2.0);
        assert!(
            a.bodies
                .iter()
                .zip(&b.bodies)
                .all(|(x, y)| x.pos == y.pos && x.vel == y.vel)
        );
    }

    #[test]
    fn lattice_spawns_exact_counts_and_spots() {
        let mut w = World::new();
        w.spawn_grid(4, [1.0, 2.0, 3.0], 0.5, Shape::Cube, BODY_RADIUS);
        assert_eq!(w.bodies.len(), 64);
        assert_eq!(w.bodies[0].pos, [1.0, 2.0, 3.0]);
        assert_eq!(w.bodies[63].pos, [1.0 + 1.5, 2.0 + 1.5, 3.0 + 1.5]);
        assert!(w.bodies.iter().all(|b| b.vel == [0.0; 3]));
    }

    #[test]
    fn big_spawns_spread_proportionally() {
        // A 40 000-body spawn must not collapse into one dense cell.
        let extent = |n: usize| {
            let mut w = World::new();
            w.spawn_wave(n, [0.0, 30.0, 0.0], 0.0);
            let xs = w.bodies.iter().map(|b| b.pos[0]);
            let (lo, hi) = (
                xs.clone().fold(f32::MAX, f32::min),
                xs.fold(f32::MIN, f32::max),
            );
            hi - lo
        };
        let (small, big) = (extent(100), extent(40_000));
        // The spread scales as the cube root of the count: ~6.8x here.
        assert!(
            big > small * 5.0,
            "the big spawn stayed dense: {big} vs {small}"
        );
    }
}
