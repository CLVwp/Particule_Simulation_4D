//! Public API checks for the physics engine.

use particule_simulation_4d::engine::{FLOOR_Y, World};

const DT: f32 = 1.0 / 60.0;

#[test]
fn single_body_falls_under_gravity() {
    let mut w = World::new();
    w.spawn_wave(1, [0.0, 5.0, 0.0], 0.0);
    for _ in 0..30 {
        w.step(DT);
        assert!(w.bodies[0].pos[1] >= FLOOR_Y - 1e-4, "sank below the floor");
    }
    assert!(w.bodies[0].pos[1] < 5.0, "body did not fall");
    assert_eq!(w.contact_count(), 0, "a lone body has no contacts");
}

#[test]
fn sparse_scene_stays_contact_free() {
    let mut w = World::new();
    w.spawn_wave(1000, [0.0, 5.0, 0.0], 0.0);
    // One body per unit of x. The cell edge is 0.2, so no two bodies share
    // a cell or a neighbor cell.
    w.settings.gravity = 0.0;
    for (i, b) in w.bodies.iter_mut().enumerate() {
        b.pos = [i as f32, 5.0, 0.0];
        b.vel = [0.0, 0.0, 0.0];
    }
    for _ in 0..10 {
        w.step(DT);
        assert_eq!(w.contact_count(), 0, "bodies one unit apart touched");
    }
}

#[test]
fn step_without_bodies_does_not_panic() {
    let mut w = World::new();
    w.step(DT);
    assert_eq!(w.contact_count(), 0);
    assert_eq!(w.bodies.len(), 0);
}
