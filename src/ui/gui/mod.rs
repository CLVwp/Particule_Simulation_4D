//! egui panels: menu, settings, HUD, toolbar, spawn, physics, and the overlay.

mod hud;
mod menu;
mod overlay;
mod windows;

use egui::{Context, Id, LayerId, Ui, UiBuilder};

use hud::{axis_labels, hud};
use menu::{menu, settings};
use overlay::overlay;
use windows::{physics_window, spawn_window};

use crate::ui::scene::SceneOut;
use crate::ui::theme::BG;
use crate::ui::{App, Page};

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
        style.visuals.panel_fill = BG;
        style.visuals.window_fill = BG;
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ui::App;
    use egui::{RawInput, Rect, Vec2, pos2};

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
