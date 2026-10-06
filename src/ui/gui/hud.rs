//! HUD, toolbar, and the axis tip labels.

use egui::{Align2, Area, Button, Color32, Context, FontId, Id, LayerId, Vec2, pos2};

use particule_simulation_4d::engine::thread_count;

use crate::ui::input::MoveAction;
use crate::ui::scene::SceneOut;
use crate::ui::widgets::faint_px;
use crate::ui::{App, Page, PhysicsMode};

/// HUD and toolbar. Status lines, the menu button, and the quick spawn row.
pub(crate) fn hud(ctx: &Context, app: &mut App) {
    Area::new(Id::new("hud"))
        .anchor(Align2::LEFT_TOP, Vec2::splat(8.0))
        .show(ctx, |ui| {
            ui.vertical(|ui| {
                faint_px(ui, format!("Bodies: {}", app.world.bodies.len()), 12.0);
                faint_px(ui, format!("Threads: {}", thread_count()), 12.0);
                faint_px(ui, format!("FPS: {:.0}", app.fps), 12.0);
                faint_px(
                    ui,
                    format!(
                        "Camera: dist {:.1}  yaw {:.2}  pitch {:.2}",
                        app.cam.dist, app.cam.yaw, app.cam.pitch
                    ),
                    12.0,
                );
                let [fwd, back, left, right, rise, sink] = MoveAction::all()
                    .map(|action| app.input.bindings[action as usize].to_uppercase());
                faint_px(
                    ui,
                    format!("Slide: {fwd} {back} {left} {right}.  Rise: {rise}.  Sink: {sink}."),
                    12.0,
                );
                for help in [
                    "Drag: orbit. Shift+drag or middle-drag: pan. Wheel: zoom.",
                    "Movement follows the camera.",
                    "Space: pause.  F1: engine stats.",
                ] {
                    faint_px(ui, help, 12.0);
                }
                if app.paused {
                    faint_px(ui, "PAUSED", 14.0);
                }
                if ui.button("Menu").clicked() {
                    app.page = Page::Menu;
                    app.drag = None;
                }
                ui.separator();
                // The fluid mode draws no bodies, so its world edits would
                // stay invisible. Disable the body buttons while it runs.
                let newton = app.mode == PhysicsMode::Newton;
                let spawn100 = ui
                    .add_enabled(newton, Button::new("+100"))
                    .on_disabled_hover_text("Bodies stay hidden in fluid mode");
                if spawn100.clicked() {
                    app.spawn_from_panel(100);
                }
                let spawn1000 = ui
                    .add_enabled(newton, Button::new("+1000"))
                    .on_disabled_hover_text("Bodies stay hidden in fluid mode");
                if spawn1000.clicked() {
                    app.spawn_from_panel(1000);
                }
                if ui.add_enabled(newton, Button::new("Clear")).clicked() {
                    app.world.clear();
                }
            });
        });
}

/// Draws one axis tip label at its screen position.
pub(crate) fn axis_labels(ctx: &Context, scene: &SceneOut) {
    if scene.axis_labels.is_empty() {
        return;
    }
    let painter = egui::Painter::new(
        ctx.clone(),
        // Background order. A Middle layer would intercept every pointer
        // event across the viewport and block the camera drags.
        LayerId::background(),
        ctx.viewport_rect(),
    );
    // The scene stores label positions in physical pixels. egui painters
    // take logical points, so the scaling factor divides here.
    let ppp = ctx.pixels_per_point();
    for (x, y, rgba, label) in &scene.axis_labels {
        let color = Color32::from_rgba_unmultiplied(
            channel(rgba[0]),
            channel(rgba[1]),
            channel(rgba[2]),
            channel(rgba[3]),
        );
        painter.text(
            pos2(x / ppp, y / ppp),
            Align2::CENTER_TOP,
            *label,
            FontId::proportional(11.0),
            color,
        );
    }
}

/// Converts one straight 0..1 channel to a byte. Rounds and clamps.
fn channel(x: f32) -> u8 {
    (x * 255.0).round().clamp(0.0, 255.0) as u8
}
