//! Physics core — no GPUI here, testable standalone.

use std::collections::HashMap;

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

pub const GRAVITY: f32 = -9.81;
pub const FLOOR_Y: f32 = 0.0;
pub const FLOOR_RESTITUTION: f32 = 0.75;
pub const GROUND_FRICTION: f32 = 0.9;
pub const BODY_RADIUS: f32 = 0.1;

const PAIR_RESTITUTION: f32 = 0.6;
const SLOP: f32 = 0.001; // allowed penetration
const CORRECTION: f32 = 0.8; // share of overlap removed each step
const CELL_SIZE: f32 = 2.0 * BODY_RADIUS; // grid cell: one body diameter

/// One particle: a sphere with a position and a velocity.
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

/// Particle positions and velocities in x, y, z (time lives in `step(dt)`).
pub struct World {
    pub bodies: Vec<Body>,
}

impl World {
    pub fn new() -> Self {
        World { bodies: Vec::new() }
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
        self.collide_bodies();
        self.collide_floor();
    }

    /// Explicit Euler integration.
    fn integrate(&mut self, dt: f32) {
        for b in &mut self.bodies {
            b.vel[1] += GRAVITY * dt;
            b.pos[0] += b.vel[0] * dt;
            b.pos[1] += b.vel[1] * dt;
            b.pos[2] += b.vel[2] * dt;
        }
    }

    /// Sphere-sphere collisions through a uniform spatial hash grid.
    /// ponytail: O(n) grid with 27-cell neighborhoods; a BVH only pays off past ~50k bodies
    fn collide_bodies(&mut self) {
        let mut grid: HashMap<[i32; 3], Vec<u32>> =
            HashMap::with_capacity(self.bodies.len());
        for (i, b) in self.bodies.iter().enumerate() {
            grid.entry(cell_of(b.pos)).or_default().push(i as u32);
        }
        for i in 0..self.bodies.len() {
            let c = cell_of(self.bodies[i].pos);
            for dx in -1..=1i32 {
                for dy in -1..=1i32 {
                    for dz in -1..=1i32 {
                        let Some(cell) = grid.get(&[c[0] + dx, c[1] + dy, c[2] + dz])
                        else {
                            continue;
                        };
                        for &j in cell {
                            if j as usize > i {
                                self.resolve_pair(i, j as usize);
                            }
                        }
                    }
                }
            }
        }
    }

    /// Impulse + positional correction between two spheres.
    fn resolve_pair(&mut self, i: usize, j: usize) {
        let (pi, vi, ri, mi) = {
            let b = &self.bodies[i];
            (b.pos, b.vel, b.radius, b.mass())
        };
        let (pj, vj, rj, mj) = {
            let b = &self.bodies[j];
            (b.pos, b.vel, b.radius, b.mass())
        };

        let d = [pj[0] - pi[0], pj[1] - pi[1], pj[2] - pi[2]];
        let dist2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
        let min_d = ri + rj;
        if dist2 >= min_d * min_d || dist2 < 1e-12 {
            return;
        }
        let dist = dist2.sqrt();
        let n = [d[0] / dist, d[1] / dist, d[2] / dist]; // from i to j

        // Relative velocity along the contact normal. Negative = closing.
        let rv = [vj[0] - vi[0], vj[1] - vi[1], vj[2] - vi[2]];
        let vn = rv[0] * n[0] + rv[1] * n[1] + rv[2] * n[2];
        let impulse = if vn < 0.0 {
            -(1.0 + PAIR_RESTITUTION) * vn / (1.0 / mi + 1.0 / mj)
        } else {
            0.0
        };
        let push = (min_d - dist - SLOP).max(0.0) * CORRECTION / (mi + mj);

        let (left, right) = self.bodies.split_at_mut(j);
        let bi = &mut left[i];
        let bj = &mut right[0];
        for k in 0..3 {
            bi.pos[k] -= n[k] * push * mj;
            bj.pos[k] += n[k] * push * mi;
            bi.vel[k] -= n[k] * impulse / mi;
            bj.vel[k] += n[k] * impulse / mj;
        }
    }

    /// Floor plane at `FLOOR_Y`, spheres rest on top of it.
    fn collide_floor(&mut self) {
        for b in &mut self.bodies {
            if b.pos[1] - b.radius < FLOOR_Y && b.vel[1] < 0.0 {
                b.pos[1] = FLOOR_Y + b.radius;
                b.vel[1] = -b.vel[1] * FLOOR_RESTITUTION;
                b.vel[0] *= GROUND_FRICTION;
                b.vel[2] *= GROUND_FRICTION;
            }
        }
    }
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
            if w.bodies[0].pos[1] <= FLOOR_Y + 2.0 * BODY_RADIUS
                && w.bodies[0].vel[1] > 0.0
            {
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
            if d < min_d + 0.1 && w.bodies[0].vel[0] < 0.0 && w.bodies[1].vel[0] > 0.0
            {
                bounced = true;
            }
        }
        assert!(bounced, "bodies never bounced off each other");
    }
}
