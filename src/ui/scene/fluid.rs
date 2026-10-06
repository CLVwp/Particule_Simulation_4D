//! Fluid instances. Projects one quad per lit fluid cell.

use particule_simulation_4d::engine::fluid::Fluid;

use super::{Instance, NEAR, SceneOut};
use crate::ui::camera::Camera;
use crate::ui::theme::hsla_to_rgba;

/// The fluid plane spans `FLUID_SPAN` world units and starts at `FLUID_LEFT`.
const FLUID_LEFT: f32 = -8.0;
/// Width of the fluid plane in world units.
pub(crate) const FLUID_SPAN: f32 = 16.0;

/// One projected fluid cell quad. Held only during a build, then emitted.
struct SortCell {
    x: f32,
    y: f32,
    radius_px: f32,
    density: f32,
    depth: f32,
}

/// Emits one instance per lit fluid cell. Cells never merge into tiles.
/// Sorts far to near, so draw order follows the painter's algorithm.
pub(crate) fn emit_fluid(
    cam: &Camera,
    fluid: &Fluid,
    w: f32,
    h: f32,
    cutoff: f32,
    out: &mut SceneOut,
) {
    let n = fluid.n;
    let cell = FLUID_SPAN / n as f32;
    // The loop visits `n * n` cells, so each push fills one sorted quad.
    let mut quads: Vec<SortCell> = Vec::with_capacity(n * n);
    for j in 1..=n {
        for i in 1..=n {
            let density = fluid.dens[i + (n + 2) * j];
            if density <= cutoff {
                continue;
            }
            let xw = FLUID_LEFT + (i as f32 - 0.5) * cell;
            let yw = (j as f32 - 0.5) * cell;
            let (x, y, focal, depth) = cam.project([xw, yw, 0.0], w, h);
            if depth <= NEAR {
                continue;
            }
            quads.push(SortCell {
                x,
                y,
                radius_px: (cell * focal / depth * 0.5).max(1.0),
                density,
                depth,
            });
        }
    }
    quads.sort_unstable_by(|a, b| b.depth.total_cmp(&a.depth));
    // Every cell becomes one square instance. Colors keep the old formulas.
    for q in &quads {
        let alpha = (0.15 + 0.75 * q.density).clamp(0.15, 0.9);
        let light = (0.45 + 0.2 * q.density).clamp(0.4, 0.75);
        out.instances.push(Instance {
            x: q.x,
            y: q.y,
            radius: q.radius_px,
            shape: 1.0,
            color: hsla_to_rgba(0.55, 0.85, light, alpha),
        });
    }
}
