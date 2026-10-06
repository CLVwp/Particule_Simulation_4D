//! F1 debug overlay and the live tuning controls.

use std::mem::size_of;

use egui::{Align2, Area, Color32, Context, Frame, Id, Ui, Vec2};

use particule_simulation_4d::engine::{Body, SimSettings, thread_count};
use particule_simulation_4d::perf::{allocated_bytes, peak_bytes};

use crate::ui::scene::{SceneOut, Tuning};
use crate::ui::widgets::{head, line, slider_row};
use crate::ui::{App, PhysicsMode};

/// Fill of the debug overlay.
const OVERLAY: Color32 = Color32::from_rgb(0x10, 0x14, 0x1b);

/// F1 overlay. CPU, GPU, and memory stats in the bottom-left corner.
pub(crate) fn overlay(ctx: &Context, app: &mut App, scene: &SceneOut) {
    if !app.debug {
        return;
    }
    // The scene works in physical pixels. egui paints in points.
    // Show device pixels.
    let view = ctx.viewport_rect();
    let scale = ctx.pixels_per_point();
    let (w, h) = (view.width() * scale, view.height() * scale);
    // One law set fills the phase table. The rows switch with the mode.
    let (names, p): (&[&str], &[f32]) = match app.mode {
        PhysicsMode::Newton => (
            &["integrate", "grid", "contacts", "resolve", "floor"],
            &app.world.phase_ms,
        ),
        PhysicsMode::Fluid => (
            &["emit", "buoyancy", "velocity", "density"],
            &app.fluid.phase_ms,
        ),
    };
    let step_total: f32 = p.iter().sum();
    let par_min = app.world.settings.par_min;
    let gpu_render = app.mode == PhysicsMode::Newton && app.tuning.gpu_render;
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
                        match app.mode {
                            PhysicsMode::Newton => line(
                                ui,
                                format!(
                                    "bodies {}   contacts {}   threads {}",
                                    app.world.bodies.len(),
                                    app.world.contact_count(),
                                    thread_count()
                                ),
                            ),
                            PhysicsMode::Fluid => line(
                                ui,
                                format!(
                                    "fluid {} x {}   emit {:.1}   threads {}",
                                    app.fluid.n,
                                    app.fluid.n,
                                    app.fluid.emit,
                                    thread_count()
                                ),
                            ),
                        }
                        line(ui, format!("path: {path}   PAR_MIN {par_min}"));
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
                        if gpu_render {
                            line(
                                ui,
                                format!(
                                    "bodies drawn {} (vertex pull, merge off)",
                                    app.world.bodies.len()
                                ),
                            );
                        } else {
                            line(ui, format!("instances painted {}", scene.instances.len()));
                        }
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

/// The live optimization controls. Every knob edits a running system; the
/// stats above answer the "did it help" question at a glance.
fn tuning_panel(ui: &mut Ui, app: &mut App) {
    head(ui, "Tuning");
    ui.checkbox(&mut app.tuning.gpu_render, "GPU vertex pull");
    // The vertex-pull path skips the CPU scene build, so these knobs do
    // nothing while it runs. Grey them out.
    ui.add_enabled_ui(!app.tuning.gpu_render, |ui| {
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
    });
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
        let defaults = SimSettings::default();
        app.world.settings.par_min = defaults.par_min;
        app.world.settings.prune_dead_pairs = defaults.prune_dead_pairs;
        app.world.settings.resolve_rounds = defaults.resolve_rounds;
        app.world.settings.resolve_epsilon = defaults.resolve_epsilon;
    }
}
