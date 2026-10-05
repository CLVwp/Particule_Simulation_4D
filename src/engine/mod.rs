//! Physics core — no GPUI here, testable standalone.
//!
//! `step` splits its work over the rayon pool. The pool has one worker per
//! logical core, detected from the CPU at startup.

pub mod fluid;

mod body;
mod config;
mod contacts;
mod grid;
mod resolve;
mod rng;
mod world;

pub use body::{Body, Shape};
pub use config::{
    BODY_RADIUS, FLOOR_RESTITUTION, FLOOR_Y, GRAVITY, GROUND_FRICTION, MIN_RADIUS, PAR_MIN,
    SimSettings,
};
pub use world::{World, thread_count};

#[cfg(test)]
mod tests {
    use super::grid::{KEY_OFF, KEY_SPAN, cell_key, key_part};
    use super::rng::Rng;
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
