//! Starts the particle simulation window.

mod ui;

use particule_simulation_4d::engine;

fn main() {
    // One worker per logical core, detected from the CPU.
    rayon::ThreadPoolBuilder::new()
        .num_threads(engine::thread_count())
        .build_global()
        .ok(); // ponytail: fails only if a pool exists already; that pool is fine then

    env_logger::init();

    ui::app::run();
}
