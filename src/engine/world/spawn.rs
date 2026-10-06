//! Spawning: body creation with a fountain-like spread.

use super::World;
use crate::engine::body::{Body, Shape};
use crate::engine::config::{BODY_RADIUS, MIN_RADIUS};
use crate::engine::rng::Rng;

impl World {
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
}

#[cfg(test)]
mod tests {
    use super::World;

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
