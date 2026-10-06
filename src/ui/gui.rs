//! egui panels: menu, settings, HUD, toolbar, spawn, physics, and the overlay.

use std::mem::size_of;

use egui::{
    Align2, Area, CentralPanel, Color32, Context, DragValue, FontId, Frame, Grid, Id, LayerId,
    RichText, Slider, Ui, UiBuilder, Vec2, Window, pos2,
};

use particule_simulation_4d::engine::{Body, Shape, thread_count};
use particule_simulation_4d::perf::{allocated_bytes, peak_bytes};

use crate::ui::input::{KeyLayout, MoveAction};
use crate::ui::scene::{SceneOut, Tuning};
use crate::ui::theme::{BG, FAINT, FG};
use crate::ui::{App, Page, PhysicsMode};

/// Panel and window fill. Matches the render clear color.
const PANEL: Color32 = Color32::from_rgb(
    ((BG >> 16) & 0xff) as u8,
    ((BG >> 8) & 0xff) as u8,
    (BG & 0xff) as u8,
);
/// Card fill for the settings page and the side panels.
const CARD: Color32 = Color32::from_rgb(0x15, 0x1b, 0x23);
/// Fill of the debug overlay.
const OVERLAY: Color32 = Color32::from_rgb(0x10, 0x14, 0x1b);

/// Draws every panel for one frame.
pub(crate) fn show(ctx: &Context, app: &mut App, scene: &SceneOut) {
    apply_theme(ctx);
    match app.page {
        Page::Menu => menu(ctx, app),
        Page::Settings => settings(ctx, app),
        Page::Sim => {
            let below_spawn = spawn_window(ctx, app);
            physics_window(ctx, app, below_spawn);
            hud(ctx, app);
            overlay(ctx, app, scene);
            axis_labels(ctx, scene);
        }
    }
}

/// Applies the dark theme and the app colors. Cheap to run every frame.
fn apply_theme(ctx: &Context) {
    ctx.all_styles_mut(|style| {
        style.visuals = egui::Visuals::dark();
        style.visuals.panel_fill = PANEL;
        style.visuals.window_fill = PANEL;
    });
}

/// Builds the full-window ui that hosts the page panels.
fn root(ctx: &Context) -> Ui {
    Ui::new(
        ctx.clone(),
        Id::new("ps4d root"),
        UiBuilder::new()
            .layer_id(LayerId::background())
            .max_rect(ctx.viewport_rect()),
    )
}

/// Menu page. Big title and the entry buttons, centered.
fn menu(ctx: &Context, app: &mut App) {
    let mut ui = root(ctx);
    CentralPanel::default().show(&mut ui, |ui| {
        ui.centered_and_justified(|ui| {
            // Plain top-down flow. The parent centers this block once; a
            // Center main-align here would re-center every widget instead.
            ui.with_layout(egui::Layout::top_down(egui::Align::Center), |ui| {
                ui.vertical_centered(|ui| {
                    ui.label(
                        RichText::new("Particule Simulation 4D")
                            .size(28.0)
                            .color(FG),
                    );
                    ui.add_space(16.0);
                    if ui
                        .button(RichText::new("Enter simulation").strong())
                        .clicked()
                    {
                        app.page = Page::Sim;
                    }
                    if ui.button("Settings").clicked() {
                        app.page = Page::Settings;
                    }
                });
            });
        });
    });
}

/// Settings page. Layout preset, one binding row per move action, and back.
fn settings(ctx: &Context, app: &mut App) {
    let mut ui = root(ctx);
    CentralPanel::default().show(&mut ui, |ui| {
        ui.centered_and_justified(|ui| {
            Frame::default()
                .fill(CARD)
                .inner_margin(32.0)
                .corner_radius(8.0)
                .show(ui, |ui| {
                    // Plain top-down flow inside the card. A Center main-align
                    // inherited from the parent re-centers every widget in the
                    // remaining space and pushes rows off-screen.
                    ui.with_layout(egui::Layout::top_down(egui::Align::Center), |ui| {
                        ui.set_min_width(340.0);
                        ui.label(RichText::new("Settings").size(20.0).color(FG));
                        ui.label(
                            RichText::new(
                                "Pick an action. Then press the new key. Escape cancels.",
                            )
                            .size(12.0)
                            .color(FAINT),
                        );
                        ui.add_space(8.0);
                        ui.horizontal(|ui| {
                            ui.label(RichText::new("Key layout").color(FAINT));
                            for layout in [KeyLayout::Qwerty, KeyLayout::Azerty] {
                                if ui
                                    .selectable_label(app.input.layout == layout, layout.label())
                                    .clicked()
                                {
                                    app.input.layout = layout;
                                    app.input.apply_layout();
                                }
                            }
                        });
                        Grid::new("bind rows")
                            .num_columns(2)
                            .spacing([24.0, 8.0])
                            .show(ui, |ui| {
                                for action in MoveAction::all() {
                                    let listening = app.input.rebinding == Some(action);
                                    let key = if listening {
                                        "press a key...".to_string()
                                    } else {
                                        app.input.bindings[action as usize].to_uppercase()
                                    };
                                    ui.label(RichText::new(action.label()).color(FAINT));
                                    if ui.button(key).clicked() {
                                        app.input.rebinding = Some(action);
                                    }
                                    ui.end_row();
                                }
                            });
                        if ui.button("Back").clicked() {
                            app.page = Page::Menu;
                            app.input.rebinding = None;
                        }
                    });
                });
        });
    });
}

/// HUD and toolbar. Status lines, the menu button, and the quick spawn row.
fn hud(ctx: &Context, app: &mut App) {
    Area::new(Id::new("hud"))
        .anchor(Align2::LEFT_TOP, Vec2::splat(8.0))
        .show(ctx, |ui| {
            ui.vertical(|ui| {
                ui.label(
                    RichText::new(format!("Bodies: {}", app.world.bodies.len()))
                        .size(12.0)
                        .color(FAINT),
                );
                ui.label(
                    RichText::new(format!("Threads: {}", thread_count()))
                        .size(12.0)
                        .color(FAINT),
                );
                ui.label(
                    RichText::new(format!("FPS: {:.0}", app.fps))
                        .size(12.0)
                        .color(FAINT),
                );
                ui.label(
                    RichText::new(format!(
                        "Camera: dist {:.1}  yaw {:.2}  pitch {:.2}",
                        app.cam.dist, app.cam.yaw, app.cam.pitch
                    ))
                    .size(12.0)
                    .color(FAINT),
                );
                let [fwd, back, left, right, rise, sink] = [
                    MoveAction::Forward,
                    MoveAction::Back,
                    MoveAction::Left,
                    MoveAction::Right,
                    MoveAction::Rise,
                    MoveAction::Sink,
                ]
                .map(|action| app.input.bindings[action as usize].to_uppercase());
                ui.label(
                    RichText::new(format!(
                        "Slide: {fwd} {back} {left} {right}.  Rise: {rise}.  Sink: {sink}."
                    ))
                    .size(12.0)
                    .color(FAINT),
                );
                for help in [
                    "Drag: orbit. Shift+drag or middle-drag: pan. Wheel: zoom.",
                    "Movement follows the camera.",
                    "F1: engine stats.",
                ] {
                    ui.label(RichText::new(help).size(12.0).color(FAINT));
                }
                if ui.button("Menu").clicked() {
                    app.page = Page::Menu;
                    app.drag = None;
                }
                ui.separator();
                ui.horizontal(|ui| {
                    if ui.button("+100").clicked() {
                        app.spawn_from_panel(100);
                    }
                    if ui.button("+1000").clicked() {
                        app.spawn_from_panel(1000);
                    }
                    if ui.button("Clear").clicked() {
                        app.world.clear();
                    }
                });
            });
        });
}

/// Spawn window. Shape, count, size, speed, and the spawn button.
/// Returns the y offset that sits below the window.
fn spawn_window(ctx: &Context, app: &mut App) -> f32 {
    let response = Window::new("Spawn")
        .anchor(Align2::RIGHT_TOP, Vec2::splat(8.0))
        .default_width(250.0)
        .resizable(false)
        .collapsible(false)
        .show(ctx, |ui| {
            Grid::new("spawn rows").num_columns(2).show(ui, |ui| {
                ui.label(RichText::new("Shape").color(FAINT));
                ui.horizontal(|ui| {
                    if ui
                        .selectable_label(app.spawn_shape == Shape::Sphere, "Sphere")
                        .clicked()
                    {
                        app.spawn_shape = Shape::Sphere;
                    }
                    if ui
                        .selectable_label(app.spawn_shape == Shape::Cube, "Cube")
                        .clicked()
                    {
                        app.spawn_shape = Shape::Cube;
                    }
                });
                ui.end_row();
                ui.label(RichText::new("Count").color(FAINT));
                ui.add(Slider::new(&mut app.spawn_count, 0..=100_000).step_by(100.0));
                ui.end_row();
                ui.label(RichText::new("Size").color(FAINT));
                ui.add(Slider::new(&mut app.spawn_radius, 0.05..=2.0).step_by(0.05));
                ui.end_row();
                ui.label(RichText::new("Speed").color(FAINT));
                ui.add(Slider::new(&mut app.spawn_speed, 0.0..=30.0).step_by(1.0));
                ui.end_row();
            });
            if ui.button("Spawn").clicked() {
                app.spawn_from_panel(app.spawn_count);
            }
        });
    let height = response
        .map(|inner| inner.response.rect.height())
        .unwrap_or(0.0);
    8.0 + height + 8.0
}

/// Physics window. Law toggle, then the sliders of the active law set.
fn physics_window(ctx: &Context, app: &mut App, below: f32) {
    Window::new("Physics")
        .anchor(Align2::RIGHT_TOP, Vec2::new(8.0, below))
        .default_width(250.0)
        .resizable(false)
        .collapsible(false)
        .show(ctx, |ui| {
            Grid::new("physics rows").num_columns(2).show(ui, |ui| {
                ui.label(RichText::new("Laws").color(FAINT));
                ui.horizontal(|ui| {
                    if ui
                        .selectable_label(app.mode == PhysicsMode::Newton, "Newton")
                        .clicked()
                    {
                        app.mode = PhysicsMode::Newton;
                    }
                    if ui
                        .selectable_label(app.mode == PhysicsMode::Fluid, "Fluid (N-S)")
                        .clicked()
                    {
                        app.mode = PhysicsMode::Fluid;
                    }
                });
                ui.end_row();
                match app.mode {
                    PhysicsMode::Newton => {
                        ui.label(RichText::new("Gravity").color(FAINT));
                        ui.add(
                            Slider::new(&mut app.world.settings.gravity, -40.0..=0.0).step_by(1.0),
                        );
                        ui.end_row();
                        ui.label(RichText::new("Bounce").color(FAINT));
                        ui.add(
                            Slider::new(&mut app.world.settings.floor_restitution, 0.0..=1.0)
                                .step_by(0.05),
                        );
                        ui.end_row();
                        ui.label(RichText::new("Friction").color(FAINT));
                        ui.add(
                            Slider::new(&mut app.world.settings.ground_friction, 0.0..=1.0)
                                .step_by(0.05),
                        );
                        ui.end_row();
                    }
                    PhysicsMode::Fluid => {
                        ui.label(RichText::new("Viscosity").color(FAINT));
                        ui.add(
                            DragValue::new(&mut app.fluid.viscosity)
                                .speed(0.05)
                                .custom_formatter(|v, _| fmt_rate(v as f32)),
                        );
                        ui.end_row();
                        ui.label(RichText::new("Diffusion").color(FAINT));
                        ui.add(
                            DragValue::new(&mut app.fluid.diffusion)
                                .speed(0.05)
                                .custom_formatter(|v, _| fmt_rate(v as f32)),
                        );
                        ui.end_row();
                        ui.label(RichText::new("Emit").color(FAINT));
                        ui.add(Slider::new(&mut app.fluid.emit, 0.0..=10.0).step_by(0.5));
                        ui.end_row();
                    }
                }
            });
        });
}

/// F1 overlay. CPU, GPU, and memory stats in the bottom-left corner.
fn overlay(ctx: &Context, app: &mut App, scene: &SceneOut) {
    if !app.debug {
        return;
    }
    // The scene works in points. egui paints in points. Show device pixels.
    let view = ctx.viewport_rect();
    let scale = ctx.pixels_per_point();
    let (w, h) = (view.width() * scale, view.height() * scale);
    let p = app.world.phase_ms;
    let step_total: f32 = p.iter().sum();
    let par_min = app.world.settings.par_min;
    let path = if app.mode == PhysicsMode::Fluid {
        format!("fluid grid {} x {}", app.fluid.n, app.fluid.n)
    } else if app.world.bodies.len() >= par_min {
        "parallel".to_string()
    } else {
        "inline".to_string()
    };

    Area::new(Id::new("overlay"))
        .anchor(Align2::LEFT_BOTTOM, Vec2::new(8.0, -8.0))
        .show(ctx, |ui| {
            Frame::default()
                .fill(OVERLAY)
                .inner_margin(12.0)
                .corner_radius(8.0)
                .show(ui, |ui| {
                    ui.vertical(|ui| {
                        head(ui, "CPU");
                        line(
                            ui,
                            format!(
                                "FPS {:.0}   frame {:.1} ms   step {:.3} ms",
                                app.fps,
                                1000.0 / app.fps,
                                app.step_ms
                            ),
                        );
                        line(
                            ui,
                            format!(
                                "bodies {}   contacts {}   threads {}",
                                app.world.bodies.len(),
                                app.world.contact_count(),
                                thread_count()
                            ),
                        );
                        line(ui, format!("path: {path}   PAR_MIN {par_min}"));
                        let names = ["integrate", "grid", "contacts", "resolve", "floor"];
                        for (name, ms) in names.iter().zip(p.iter()) {
                            let share = if step_total > 0.0 {
                                100.0 * ms / step_total
                            } else {
                                0.0
                            };
                            line(ui, format!("{name:<9} {ms:7.3} ms {share:5.1} %"));
                        }
                        head(ui, "GPU");
                        line(ui, format!("viewport {w:.0} x {h:.0} px"));
                        line(ui, format!("instances painted {}", scene.instance_count));
                        line(ui, format!("scene (project + sort) {:.3} ms", app.scene_ms));
                        line(ui, format!("adapter: {}", app.adapter_info));
                        head(ui, "Memory");
                        line(
                            ui,
                            format!(
                                "in use {:.1} MB   peak {:.1} MB",
                                allocated_bytes() as f32 / 1048576.0,
                                peak_bytes() as f32 / 1048576.0
                            ),
                        );
                        line(
                            ui,
                            format!(
                                "bodies array {:.2} MB",
                                (size_of::<Body>() * app.world.bodies.len()) as f32 / 1048576.0
                            ),
                        );
                        tuning_panel(ui, app);
                    });
                });
        });
}

/// One labeled slider row inside the tuning panel.
fn slider_row(
    ui: &mut Ui,
    label: &str,
    value: &mut f32,
    range: std::ops::RangeInclusive<f32>,
    step: f64,
) {
    ui.horizontal(|ui| {
        ui.label(RichText::new(label).color(FAINT));
        ui.add(Slider::new(value, range).step_by(step));
    });
}

/// The live optimization controls. Every knob edits a running system; the
/// stats above answer the "did it help" question at a glance.
fn tuning_panel(ui: &mut Ui, app: &mut App) {
    head(ui, "Tuning");
    ui.checkbox(&mut app.tuning.lod_merge, "Tile merge (LOD)");
    ui.add_enabled_ui(app.tuning.lod_merge, |ui| {
        slider_row(
            ui,
            "merge px",
            &mut app.tuning.merge_radius_px,
            1.5..=8.0,
            0.5,
        );
        slider_row(
            ui,
            "tile px",
            &mut app.tuning.merge_tile_px,
            4.0..=32.0,
            1.0,
        );
        slider_row(
            ui,
            "tile fill",
            &mut app.tuning.merge_tile_fill,
            0.5..=1.0,
            0.05,
        );
    });
    ui.checkbox(&mut app.tuning.cull_offscreen, "Off-screen cull");
    slider_row(
        ui,
        "fluid cutoff",
        &mut app.tuning.density_cutoff,
        0.0..=0.2,
        0.01,
    );
    ui.checkbox(&mut app.world.settings.prune_dead_pairs, "Prune dead pairs");
    let mut par = app.world.settings.par_min as f32;
    slider_row(ui, "PAR_MIN", &mut par, 0.0..=16384.0, 256.0);
    app.world.settings.par_min = par as usize;
    let mut rounds = app.world.settings.resolve_rounds as f32;
    slider_row(ui, "resolve rounds", &mut rounds, 1.0..=8.0, 1.0);
    app.world.settings.resolve_rounds = rounds as usize;
    slider_row(
        ui,
        "resolve epsilon",
        &mut app.world.settings.resolve_epsilon,
        0.0..=0.01,
        0.001,
    );
    if ui.button("Reset tuning").clicked() {
        app.tuning = Tuning::default();
        let defaults = crate::ui::App::new();
        app.world.settings.par_min = defaults.world.settings.par_min;
        app.world.settings.prune_dead_pairs = defaults.world.settings.prune_dead_pairs;
        app.world.settings.resolve_rounds = defaults.world.settings.resolve_rounds;
        app.world.settings.resolve_epsilon = defaults.world.settings.resolve_epsilon;
    }
}

/// Draws one axis tip label at its screen position.
fn axis_labels(ctx: &Context, scene: &SceneOut) {
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
    for (x, y, rgba, label) in &scene.axis_labels {
        let color = Color32::from_rgba_unmultiplied(
            channel(rgba[0]),
            channel(rgba[1]),
            channel(rgba[2]),
            channel(rgba[3]),
        );
        painter.text(
            pos2(*x, *y),
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

/// Adds one faint monospace line at 11 pt.
fn line(ui: &mut Ui, text: String) {
    ui.label(RichText::new(text).size(11.0).monospace().color(FAINT));
}

/// Adds one small bright heading.
fn head(ui: &mut Ui, text: &str) {
    ui.label(RichText::new(text.to_string()).size(12.0).color(FG));
}

/// Formats a diffusion-style rate. Zero prints plain, other values print
/// scientific.
pub(crate) fn fmt_rate(x: f32) -> String {
    if x <= 0.0 {
        "0".to_string()
    } else {
        format!("{x:.1e}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ui::App;
    use egui::{RawInput, Rect, Vec2};

    /// Runs one egui pass for one page. Returns the drawn pixel bounds.
    fn drawn_bounds(page: Page, w: f32, h: f32) -> Rect {
        let ctx = Context::default();
        let raw = RawInput {
            screen_rect: Some(Rect::from_min_size(pos2(0.0, 0.0), Vec2::new(w, h))),
            ..RawInput::default()
        };
        let mut app = App::new();
        app.page = page;
        // The first pass loads fonts and draws nothing. Run a second pass.
        let mut bounds: Option<Rect> = None;
        for _ in 0..2 {
            ctx.begin_pass(raw.clone());
            show(&ctx, &mut app, &SceneOut::default());
            let mut output = ctx.end_pass();
            // No renderer here. Drop the font deltas on purpose.
            output.textures_delta.set.clear();
            output.textures_delta.free.clear();
            let prims = ctx.tessellate(output.shapes, output.pixels_per_point);
            bounds = mesh_bounds(&prims, bounds);
        }
        bounds.expect("the page drew nothing")
    }

    /// Unions the clipped mesh bounds into `acc`.
    fn mesh_bounds(prims: &[egui::ClippedPrimitive], acc: Option<Rect>) -> Option<Rect> {
        let mut acc = acc;
        for prim in prims {
            let egui::epaint::Primitive::Mesh(mesh) = &prim.primitive else {
                continue;
            };
            let mut r = Rect::NOTHING;
            for v in &mesh.vertices {
                r = r.union(Rect::from_pos(v.pos));
            }
            let r = r.intersect(prim.clip_rect);
            if !r.is_positive() {
                continue;
            }
            acc = Some(match acc {
                Some(b) => b.union(r),
                None => r,
            });
        }
        acc
    }

    /// Every page must draw inside the viewport at common sizes.
    #[test]
    fn every_page_draws_inside_the_viewport() {
        let sizes = [(1920.0, 1009.0), (1280.0, 720.0)];
        for page in [Page::Menu, Page::Settings, Page::Sim] {
            for &(w, h) in &sizes {
                let bounds = drawn_bounds(page, w, h);
                assert!(
                    bounds.min.x >= -1.0 && bounds.min.y >= -1.0,
                    "{page:?} at {w}x{h}: content starts off-screen: {bounds:?}"
                );
                assert!(
                    bounds.max.x <= w + 1.0 && bounds.max.y <= h + 1.0,
                    "{page:?} at {w}x{h}: content ends off-screen: {bounds:?}"
                );
            }
        }
    }
}
