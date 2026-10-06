//! Prints the per-phase step cost at a body count, like the F1 table.
//! Run: `cargo run --release --example phase_table [count]`

use particule_simulation_4d::engine::World;

fn main() {
    let n: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(500_000);
    let dt = 1.0 / 60.0;
    let mut w = World::new();
    w.spawn_wave(n, [0.0, 8.0, 0.0], 4.0);
    // Settle the pile. The measurement starts on a resting scene.
    for _ in 0..30 {
        w.step(dt);
    }
    let frames = 60;
    let mut sums = [0.0f32; 5];
    for _ in 0..frames {
        w.step(dt);
        for (s, ms) in sums.iter_mut().zip(w.phase_ms) {
            *s += ms;
        }
    }
    let f = frames as f32;
    let names = ["integrate", "grid", "contacts", "resolve", "floor"];
    println!("bodies {n}, contacts {}, thread pool", w.contact_count());
    for (name, s) in names.iter().zip(sums) {
        println!("{name:>10}: {:7.2} ms", s / f);
    }
    println!("{:>10}: {:7.2} ms", "step", sums.iter().sum::<f32>() / f);
}
