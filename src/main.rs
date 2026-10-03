mod engine;
mod ui;

use gpui_kit::*;

use crate::ui::SimView;

fn main() {
    application()
        .with_assets(assets::Assets)
        .run(|cx| {
            init(cx);

            open_window(WindowOptions::default(), cx, |_, cx| {
                cx.new(SimView::new)
            })
            .expect("Failed to open window");
        });
}
