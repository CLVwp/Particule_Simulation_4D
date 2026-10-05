//! Settings page. Layout preset and the move bindings.

use gpui_kit::component::button::{Button, ButtonVariants};
use gpui_kit::*;

use crate::ui::input::{KeyLayout, MoveAction};
use crate::ui::theme::{BG, FAINT, FG};
use crate::ui::{Page, SimView};

impl SimView {
    /// Builds the settings page. Layout buttons, binding rows, and back.
    pub(crate) fn render_settings(&mut self, cx: &mut Context<Self>) -> Div {
        let (qwerty, azerty) = (
            Button::new("layout-qwerty").label(KeyLayout::Qwerty.label()),
            Button::new("layout-azerty").label(KeyLayout::Azerty.label()),
        );
        let (qwerty, azerty) = (
            if self.layout == KeyLayout::Qwerty {
                qwerty.primary()
            } else {
                qwerty
            },
            if self.layout == KeyLayout::Azerty {
                azerty.primary()
            } else {
                azerty
            },
        );

        let mut rows = div().flex().flex_col().gap_2();
        for action in MoveAction::all() {
            let listening = self.rebinding == Some(action);
            let key = if listening {
                "press a key...".to_string()
            } else {
                self.bindings[action as usize].to_uppercase()
            };
            rows = rows.child(
                div()
                    .flex()
                    .items_center()
                    .gap_6()
                    .child(div().w(px(110.0)).text_color(FAINT).child(action.label()))
                    .child(Button::new(("bind", action as usize)).label(key).on_click(
                        cx.listener(move |this, _: &ClickEvent, window, cx| {
                            this.rebinding = Some(action);
                            window.focus(&this.focus, cx);
                        }),
                    )),
            );
        }

        div()
            .size_full()
            .bg(rgb(BG))
            .flex()
            .items_center()
            .justify_center()
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
                    .flex()
                    .flex_col()
                    .gap_4()
                    .p_8()
                    .rounded_lg()
                    .bg(rgb(0x151b23))
                    .child(div().text_size(px(20.0)).text_color(FG).child("Settings"))
                    .child(
                        div()
                            .text_color(FAINT)
                            .text_size(px(12.0))
                            .child("Pick an action. Then press the new key. Escape cancels."),
                    )
                    .child(
                        div()
                            .flex()
                            .items_center()
                            .gap_6()
                            .child(div().w(px(110.0)).text_color(FAINT).child("Key layout"))
                            .child(qwerty.on_click(cx.listener(|this, _: &ClickEvent, _, _| {
                                this.layout = KeyLayout::Qwerty;
                                this.apply_layout();
                            })))
                            .child(azerty.on_click(cx.listener(|this, _: &ClickEvent, _, _| {
                                this.layout = KeyLayout::Azerty;
                                this.apply_layout();
                            }))),
                    )
                    .child(rows)
                    .child(Button::new("back").label("Back").on_click(cx.listener(
                        |this, _: &ClickEvent, _, _| {
                            this.page = Page::Menu;
                            this.rebinding = None;
                        },
                    ))),
            )
    }
}
