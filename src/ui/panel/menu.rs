//! Menu page. Title plus the entry buttons.

use gpui_kit::component::button::{Button, ButtonVariants};
use gpui_kit::*;

use crate::ui::theme::{BG, FG};
use crate::ui::{Page, SimView};

impl SimView {
    /// Builds the menu page. Title plus the enter and settings buttons.
    pub(crate) fn render_menu(&mut self, cx: &mut Context<Self>) -> Div {
        div()
            .size_full()
            .bg(rgb(BG))
            .flex()
            .flex_col()
            .items_center()
            .justify_center()
            .gap_4()
            .track_focus(&self.focus)
            .on_mouse_down(
                MouseButton::Left,
                cx.listener(|this, _: &MouseDownEvent, window, cx| {
                    window.focus(&this.focus, cx);
                }),
            )
            .on_key_down(cx.listener(|this, ev: &KeyDownEvent, _w, _cx| {
                this.handle_key_down(ev);
            }))
            .on_key_up(cx.listener(|this, ev: &KeyUpEvent, _w, _cx| {
                this.keys.remove(&ev.keystroke.key);
            }))
            .child(
                div()
                    .text_size(px(28.0))
                    .text_color(FG)
                    .child("Particule Simulation 4D"),
            )
            .child(
                Button::new("enter-sim")
                    .primary()
                    .label("Enter simulation")
                    .on_click(cx.listener(|this, _: &ClickEvent, _, _| {
                        this.page = Page::Sim;
                    })),
            )
            .child(
                Button::new("open-settings")
                    .label("Settings")
                    .on_click(cx.listener(|this, _: &ClickEvent, _, _| {
                        this.page = Page::Settings;
                    })),
            )
    }
}
