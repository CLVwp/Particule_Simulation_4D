//! Prints the per-phase step cost at a body count, like the F1 table,
//! plus one scene build line. Run: `cargo run --release --example phase_table [count]`

use std::time::Instant;

use particule_simulation_4d::engine::{World, thread_count};
use particule_simulation_4d::ui::{Camera, SceneOut, TileCache, Tuning, emit_scene_cpu};

fn main() {
    let n: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(500_000);
    let dt = 1.0 / 60.0;
    let mut w = World::new();
    w.spawn_wave(n, [0.0, 8.0, 0.0], 4.0);
    // Settle the pile. Sixty frames, same depth as the bench `settled`
    // helper, so the example and the bench start from the same rest state.
    for _ in 0..60 {
        w.step(dt);
    }
    let frames = 60;
    let mut sums = [0.0f32; 5];
    // The step phases run alone: interleaving the scene build inflates
    // them through cache pressure, and the benches step alone too.
    for _ in 0..frames {
        w.step(dt);
        for (s, ms) in sums.iter_mut().zip(w.phase_ms) {
            *s += ms;
        }
    }
    let f = frames as f32;
    let names = ["integrate", "grid", "contacts", "resolve", "floor"];
    println!(
        "bodies {n}, contacts {}, {} threads",
        w.contact_count(),
        thread_count()
    );
    for (name, s) in names.iter().zip(sums) {
        println!("{name:>10}: {:7.2} ms", s / f);
    }
    println!("{:>10}: {:7.2} ms", "step", sums.iter().sum::<f32>() / f);
    // The scene build runs its own loop, like the `scene` bench group.
    let cam = Camera::default();
    let tuning = Tuning::default();
    let mut tiles = TileCache::default();
    let mut out = SceneOut::default();
    let mut scene_sum = 0.0f32;
    for _ in 0..frames {
        let scene = Instant::now();
        emit_scene_cpu(
            &cam,
            &w.bodies,
            1920.0,
            1080.0,
            cam.dist,
            &tuning,
            w.settings.par_min,
            &mut tiles,
            &mut out,
        );
        scene_sum += scene.elapsed().as_secs_f32() * 1000.0;
    }
    println!("{:>10}: {:7.2} ms", "scene", scene_sum / f);
}
