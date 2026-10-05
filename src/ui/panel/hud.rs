//! Top-left status HUD and the quick spawn toolbar.

use gpui_kit::component::button::Button;
use gpui_kit::*;
use particule_simulation_4d::engine::thread_count;

use crate::ui::input::MoveAction;
use crate::ui::theme::FAINT;
use crate::ui::{Page, SimView};

impl SimView {
    /// Builds the HUD. Body count, frame stats, camera state, and help lines.
    pub(crate) fn render_hud(&mut self, cx: &mut Context<Self>) -> Div {
        let (fwd, back, left, right) = (
            self.bindings[MoveAction::Forward as usize].to_uppercase(),
            self.bindings[MoveAction::Back as usize].to_uppercase(),
            self.bindings[MoveAction::Left as usize].to_uppercase(),
            self.bindings[MoveAction::Right as usize].to_uppercase(),
        );
        let (rise, sink) = (
            self.bindings[MoveAction::Rise as usize].to_uppercase(),
            self.bindings[MoveAction::Sink as usize].to_uppercase(),
        );

        div()
            .absolute()
            .top_2()
            .left_2()
            .flex()
            .flex_col()
            .gap_1()
            .text_size(px(12.0))
            .text_color(FAINT)
            .child(format!("Bodies: {}", self.world.bodies.len()))
            .child(format!("Threads: {}", thread_count()))
            .child(format!("FPS: {:.0}", self.fps))
            .child(format!(
                "Camera: dist {:.1}  yaw {:.2}  pitch {:.2}",
                self.dist, self.yaw, self.pitch
            ))
            .child(format!(
                "Slide: {fwd} {back} {left} {right}.  Rise: {rise}.  Sink: {sink}."
            ))
            .child("Drag: orbit. Shift+drag or middle-drag: pan. Wheel: zoom.")
            .child("Movement follows the camera.")
            .child("F1: engine stats.")
            .child(Button::new("menu").label("Menu").on_click(cx.listener(
                |this, _: &ClickEvent, _, _| {
                    this.page = Page::Menu;
                    this.drag = None;
                },
            )))
    }

    /// Builds the toolbar. Quick spawn buttons and the clear button.
    pub(crate) fn render_toolbar(&mut self, cx: &mut Context<Self>) -> Div {
        div()
            .absolute()
            .top_2()
            .left_1_2()
            .flex()
            .gap_2()
            .child(Button::new("add-100").label("+100").on_click(cx.listener(
                |this, _: &ClickEvent, _, _| {
                    this.spawn_from_panel(100);
                },
            )))
            .child(Button::new("add-1000").label("+1000").on_click(cx.listener(
                |this, _: &ClickEvent, _, _| {
                    this.spawn_from_panel(1000);
                },
            )))
            .child(Button::new("clear").label("Clear").on_click(cx.listener(
                |this, _: &ClickEvent, _, _| {
                    this.world.clear();
                },
            )))
    }
}
