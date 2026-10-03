mod ui;

use gpui_kit::*;

use crate::ui::SimView;
use particule_simulation_4d::engine;

fn main() {
    // One worker per logical core, detected from the CPU.
    rayon::ThreadPoolBuilder::new()
        .num_threads(engine::thread_count())
        .build_global()
        .ok(); // ponytail: fails only if a pool exists already; that pool is fine then

    application().with_assets(assets::Assets).run(|cx| {
        init(cx);

        open_window(WindowOptions::default(), cx, |_, cx| cx.new(SimView::new))
            .expect("Failed to open window");
    });
}
