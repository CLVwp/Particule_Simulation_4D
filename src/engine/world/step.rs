//! The step pipeline: integrate, sort, contacts, solve, floor.

use std::time::Instant;

use super::World;
use crate::engine::config::FLOOR_Y;

impl World {
    /// Advances the world by `dt` seconds.
    ///
    /// # Panics
    /// Debug builds panic when `dt` is zero, negative, or not finite. A
    /// bad `dt` poisons every position, so the guard fires at the call.
    pub fn step(&mut self, dt: f32) {
        debug_assert!(
            dt.is_finite() && dt > 0.0,
            "dt must be finite and above zero"
        );
        let t = Instant::now();
        self.integrate(dt);
        self.phase_ms[0] = super::ms_since(t);
        let t = Instant::now();
        self.sort_cells();
        self.phase_ms[1] = super::ms_since(t);
        let t = Instant::now();
        self.build_contacts();
        self.phase_ms[2] = super::ms_since(t);
        let t = Instant::now();
        self.resolve();
        self.phase_ms[3] = super::ms_since(t);
        let t = Instant::now();
        self.collide_floor();
        self.phase_ms[4] = super::ms_since(t);
    }

    /// Explicit Euler integration, split across the pool.
    fn integrate(&mut self, dt: f32) {
        let g = self.settings.gravity;
        let cap = self.settings.max_speed;
        let par_min = self.settings.par_min;
        super::par_each(&mut self.bodies, par_min, |b| {
            b.vel[1] += g * dt;
            // The clamp runs before the position update, so one step can
            // never carry a body past its contact reach.
            if cap > 0.0 {
                let s2 = b.vel[0] * b.vel[0] + b.vel[1] * b.vel[1] + b.vel[2] * b.vel[2];
                let cap2 = cap * cap;
                if s2 > cap2 {
                    let k = cap / s2.sqrt();
                    b.vel[0] *= k;
                    b.vel[1] *= k;
                    b.vel[2] *= k;
                }
            }
            b.pos[0] += b.vel[0] * dt;
            b.pos[1] += b.vel[1] * dt;
            b.pos[2] += b.vel[2] * dt;
        });
    }

    /// Floor plane at `FLOOR_Y`, spheres rest on top of it.
    fn collide_floor(&mut self) {
        let rest = self.settings.floor_restitution;
        let friction = self.settings.ground_friction;
        let par_min = self.settings.par_min;
        super::par_each(&mut self.bodies, par_min, |b| {
            if b.pos[1] - b.radius < FLOOR_Y && b.vel[1] < 0.0 {
                b.pos[1] = FLOOR_Y + b.radius;
                b.vel[1] = -b.vel[1] * rest;
                b.vel[0] *= friction;
                b.vel[2] *= friction;
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::World;
    use crate::engine::body::{Body, Shape};
    use crate::engine::config::{BODY_RADIUS, FLOOR_Y};

    /// Two worlds from one spawn. The seed derives from the count alone, so
    /// both start bit-identical.
    fn twin_worlds(n: usize) -> (World, World) {
        let mut a = World::new();
        let mut b = World::new();
        a.spawn_wave(n, [0.0, 5.0, 0.0], 4.0);
        b.spawn_wave(n, [0.0, 5.0, 0.0], 4.0);
        (a, b)
    }

    /// Biggest per-axis position gap between two worlds.
    fn max_pos_diff(a: &World, b: &World) -> f32 {
        a.bodies
            .iter()
            .zip(&b.bodies)
            .map(|(x, y)| {
                (x.pos[0] - y.pos[0])
                    .abs()
                    .max((x.pos[1] - y.pos[1]).abs())
                    .max((x.pos[2] - y.pos[2]).abs())
            })
            .fold(0.0, f32::max)
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn same_spawn_replays_bit_exact() {
        // The spawn seed is fixed, so two identical runs must agree to the
        // last bit. This pins the determinism the 1e-4 tolerance test hides.
        let (mut a, mut b) = twin_worlds(1000);
        for _ in 0..60 {
            a.step(1.0 / 60.0);
            b.step(1.0 / 60.0);
        }
        assert!(
            max_pos_diff(&a, &b) == 0.0,
            "same spawn replayed with a gap"
        );
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn prune_dead_pairs_keeps_the_physics_one_round() {
        // The scan drops pairs that add zero in that round. One solve round
        // must therefore replay bit-exact without the prune. Two rounds do
        // not hold this: round two can see overlaps that round one's pushes
        // created, and those pairs are absent from the pruned list. That
        // drift is the price of the knob, and the README states it.
        let (mut a, mut b) = twin_worlds(1000);
        a.settings.resolve_rounds = 1;
        b.settings.resolve_rounds = 1;
        b.settings.prune_dead_pairs = false;
        for _ in 0..60 {
            a.step(1.0 / 60.0);
            b.step(1.0 / 60.0);
        }
        assert!(
            max_pos_diff(&a, &b) == 0.0,
            "one-round prune replay moved the pile"
        );
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn resolve_epsilon_keeps_the_physics() {
        // TODO: epsilon zero disables the early exit. A small epsilon only
        // skips rounds the solve calls motionless.
        let (mut a, mut b) = twin_worlds(1000);
        b.settings.resolve_epsilon = 1e-4;
        for _ in 0..60 {
            a.step(1.0 / 60.0);
            b.step(1.0 / 60.0);
        }
        let diff = max_pos_diff(&a, &b);
        assert!(diff < 1e-5, "epsilon moved the pile by {diff}");
    }

    #[test]
    fn shape_tag_never_changes_the_trajectory() {
        // The bench step_shape asserts equal timings across shape mixes.
        // This test asserts the stronger fact: the physics reads no shape.
        let mut a = World::new();
        let mut b = World::new();
        a.spawn(500, [0.0, 5.0, 0.0], 4.0, Shape::Sphere, BODY_RADIUS);
        b.spawn(500, [0.0, 5.0, 0.0], 4.0, Shape::Cube, BODY_RADIUS);
        for _ in 0..60 {
            a.step(1.0 / 60.0);
            b.step(1.0 / 60.0);
        }
        assert!(max_pos_diff(&a, &b) == 0.0, "shape changed the trajectory");
    }

    #[test]
    // ponytail: miri runs these 600-step loops thousands of times slower.
    // `cargo test` still runs them on every machine. Revisit when miri gets faster.
    #[cfg_attr(miri, ignore)]
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
    #[cfg_attr(miri, ignore)]
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
    #[cfg_attr(miri, ignore)]
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
    fn max_speed_caps_the_step_length() {
        // One body crosses one cell of 0.2 at cell_size per step at most.
        // Above the cap the motion must clip to cap * dt.
        let mut w = World::new();
        w.settings.gravity = 0.0;
        w.settings.max_speed = 24.0;
        w.bodies.push(Body {
            pos: [0.0; 3],
            vel: [1000.0, 0.0, 0.0],
            radius: BODY_RADIUS,
            shape: Shape::Sphere,
        });
        w.step(1.0 / 60.0);
        let moved = w.bodies[0].pos[0];
        assert!(
            moved <= 24.0 / 60.0 + 1e-4,
            "body moved past the speed cap: {moved}"
        );
        assert!(moved > 20.0 / 60.0, "cap behaved like a stop: {moved}");
    }

    #[test]
    fn zero_max_speed_leaves_motion_free() {
        // Default setting. The cap must not touch an uncapped body.
        let mut w = World::new();
        w.settings.gravity = 0.0;
        w.bodies.push(Body {
            pos: [0.0; 3],
            vel: [1000.0, 0.0, 0.0],
            radius: BODY_RADIUS,
            shape: Shape::Sphere,
        });
        w.step(1.0 / 60.0);
        let moved = w.bodies[0].pos[0];
        assert!((moved - 1000.0 / 60.0).abs() < 1e-3, "free motion changed");
    }

    #[test]
    fn pair_friction_damps_tangential_slide() {
        // Two overlapping bodies slide past each other along x. The contact
        // normal is vertical, so the slide is tangential. Friction must slow
        // it; zero friction must leave it alone.
        let slide = |friction: f32| {
            let mut w = World::new();
            w.settings.gravity = 0.0;
            w.settings.pair_friction = friction;
            w.bodies.push(Body {
                pos: [0.0, 2.0, 0.0],
                vel: [1.0, 0.0, 0.0],
                radius: BODY_RADIUS,
                shape: Shape::Sphere,
            });
            w.bodies.push(Body {
                pos: [0.0, 2.05, 0.0],
                vel: [-1.0, 0.0, 0.0],
                radius: BODY_RADIUS,
                shape: Shape::Sphere,
            });
            for _ in 0..30 {
                w.step(1.0 / 60.0);
            }
            (w.bodies[0].vel[0] - w.bodies[1].vel[0]).abs()
        };
        let free = slide(0.0);
        let gripped = slide(0.5);
        assert!(free > 1.5, "frictionless slide must persist: {free}");
        assert!(
            gripped < free,
            "friction must slow the slide: {gripped} vs {free}"
        );
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
    #[cfg_attr(miri, ignore)]
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
}
