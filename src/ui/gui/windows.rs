//! Spawn window and the physics window.

use egui::{Context, Grid, Vec2};

use particule_simulation_4d::engine::Shape;

use crate::ui::widgets::{enum_toggle, faint, grid_slider, rate_drag, side_window};
use crate::ui::{App, PhysicsMode};

/// Spawn window. Shape, count, size, speed, and the spawn button.
/// Returns the y offset that sits below the window.
pub(crate) fn spawn_window(ctx: &Context, app: &mut App) -> f32 {
    let response = side_window("Spawn", Vec2::splat(8.0)).show(ctx, |ui| {
        Grid::new("spawn rows").num_columns(2).show(ui, |ui| {
            faint(ui, "Shape");
            ui.horizontal(|ui| {
                enum_toggle(
                    ui,
                    &mut app.spawn_shape,
                    &[(Shape::Sphere, "Sphere"), (Shape::Cube, "Cube")],
                );
            });
            ui.end_row();
            grid_slider(ui, "Count", &mut app.spawn_count, 0..=100_000, 100.0);
            grid_slider(ui, "Size", &mut app.spawn_radius, 0.05..=2.0, 0.05);
            grid_slider(ui, "Speed", &mut app.spawn_speed, 0.0..=30.0, 1.0);
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
pub(crate) fn physics_window(ctx: &Context, app: &mut App, below: f32) {
    side_window("Physics", Vec2::new(8.0, below)).show(ctx, |ui| {
        Grid::new("physics rows").num_columns(2).show(ui, |ui| {
            faint(ui, "Laws");
            ui.horizontal(|ui| {
                enum_toggle(
                    ui,
                    &mut app.mode,
                    &[
                        (PhysicsMode::Newton, "Newton"),
                        (PhysicsMode::Fluid, "Fluid (N-S)"),
                    ],
                );
            });
            ui.end_row();
            grid_slider(
                ui,
                "Time scale",
                &mut app.time_scale,
                0.1..=4.0,
                0.1,
            );
            match app.mode {
                PhysicsMode::Newton => {
                    grid_slider(
                        ui,
                        "Gravity",
                        &mut app.world.settings.gravity,
                        -40.0..=0.0,
                        1.0,
                    );
                    grid_slider(
                        ui,
                        "Bounce",
                        &mut app.world.settings.floor_restitution,
                        0.0..=1.0,
                        0.05,
                    );
                    grid_slider(
                        ui,
                        "Friction",
                        &mut app.world.settings.ground_friction,
                        0.0..=1.0,
                        0.05,
                    );
                    grid_slider(
                        ui,
                        "Pair bounce",
                        &mut app.world.settings.pair_restitution,
                        0.0..=1.0,
                        0.05,
                    );
                    grid_slider(
                        ui,
                        "Pair friction",
                        &mut app.world.settings.pair_friction,
                        0.0..=1.0,
                        0.05,
                    );
                    grid_slider(
                        ui,
                        "Speed cap",
                        &mut app.world.settings.max_speed,
                        0.0..=100.0,
                        5.0,
                    );
                }
                PhysicsMode::Fluid => {
                    rate_drag(ui, &mut app.fluid.viscosity);
                    ui.end_row();
                    rate_drag(ui, &mut app.fluid.diffusion);
                    ui.end_row();
                    grid_slider(ui, "Emit", &mut app.fluid.emit, 0.0..=10.0, 0.5);
                }
            }
        });
    });
}
