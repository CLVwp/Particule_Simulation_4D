//! Scene projection. Turns bodies or fluid cells into sorted screen quads.

use std::mem::size_of;
use std::time::Instant;

use gpui_kit::*;
use particule_simulation_4d::engine::{Body, PAR_MIN, Shape};
use rayon::prelude::*;

use crate::ui::theme::{SMOOTH_KEEP, SMOOTH_NEW};
use crate::ui::{PhysicsMode, SimView};

/// Cells at or below this density do not draw.
pub(crate) const DENSITY_CUTOFF: f32 = 0.02;
/// The fluid plane spans `FLUID_SPAN` world units and starts at `FLUID_LEFT`.
pub(crate) const FLUID_LEFT: f32 = -8.0;
/// Width of the fluid plane in world units.
pub(crate) const FLUID_SPAN: f32 = 16.0;
/// Depths at or below this sit at or behind the camera.
pub(crate) const NEAR: f32 = 0.5;
/// Bodies below this screen radius merge into tiles. Bigger bodies paint alone.
pub(crate) const MERGE_RADIUS_PX: f32 = 3.0;
/// Side length of one merge tile, in screen pixels.
pub(crate) const MERGE_TILE_PX: f32 = 8.0;
/// The merged quad radius spans this share of the tile. It covers the whole tile.
pub(crate) const MERGE_TILE_FILL: f32 = 0.75;
/// Spill tile rows and columns on each screen edge.
/// A merged dot reaches less than one tile past an edge.
const MERGE_TILE_SPILL: i32 = 1;
/// Cap for tile rows and columns on one axis. It keeps tile indexes inside `u32`.
const MERGE_TILES_MAX: i32 = 1 << 15;

/// One projected point: screen x, y, focal scale, camera depth.
pub(crate) type ProjectedPoint = (f32, f32, f32, f32);
/// A projected line segment: two screen points and a color.
pub(crate) type ProjectedLine = (ProjectedPoint, ProjectedPoint, Hsla);
/// A projected segment with no color. The painter picks the color.
pub(crate) type ProjectedSeg = (ProjectedPoint, ProjectedPoint);
/// One projected axis label: screen x, y, color, text.
pub(crate) type AxisLabel = (f32, f32, Hsla, &'static str);

/// A body drawn as one screen quad at `x`, `y`.
pub(crate) struct ProjectedBody {
    pub(crate) x: f32,
    pub(crate) y: f32,
    pub(crate) radius_px: f32,
    pub(crate) depth: f32,
    pub(crate) shape: Shape,
}

/// One lit fluid cell drawn as one screen quad at `x`, `y`.
pub(crate) struct ProjectedCell {
    pub(crate) x: f32,
    pub(crate) y: f32,
    pub(crate) radius_px: f32,
    pub(crate) density: f32,
    pub(crate) depth: f32,
}

/// Sums one merge tile. Each field totals every member of the tile.
/// A zero count marks an empty tile.
#[derive(Clone, Copy, Default)]
pub(crate) struct TileMerge {
    count: u32,
    x: f32,
    y: f32,
    depth: f32,
    spheres: u32,
}

// The bins store this type by value, one slot per tile. Size growth would
// waste cache on every dot.
const _: () = assert!(size_of::<TileMerge>() == 20);

/// Tile rows or columns for one screen span. The result is at least one tile.
fn merge_tile_axis(span: f32) -> i32 {
    ((span / MERGE_TILE_PX).ceil().max(1.0) as i32).min(MERGE_TILES_MAX)
}

impl SimView {
    /// Projects and sorts one quad list. The active physics mode picks the list.
    /// Small far bodies merge into screen tiles. The Fluid branch skips the merge.
    pub(crate) fn build_scene(
        &mut self,
        w: f32,
        h: f32,
    ) -> (Vec<ProjectedBody>, Vec<ProjectedCell>) {
        // Project everything up-front; the paint closure only draws.
        // The projection divides by the camera depth, so anything at or behind
        // the camera (depth <= NEAR) projects to garbage. Cull it, or it shows
        // up as mirrored "ghost" shapes.
        let scene = Instant::now();
        let (points, fluid_quads): (Vec<ProjectedBody>, Vec<ProjectedCell>) = if self.mode
            == PhysicsMode::Newton
        {
            let project = |b: &Body| {
                let (x, y, focal, depth) = self.project(b.pos, w, h);
                ProjectedBody {
                    x,
                    y,
                    radius_px: (b.radius * focal / depth).max(1.5),
                    depth,
                    shape: b.shape,
                }
            };
            let bodies = &self.world.bodies;
            let mut points: Vec<ProjectedBody> = if bodies.len() >= PAR_MIN {
                bodies
                    .par_iter()
                    .map(project)
                    .filter(|p| p.depth > NEAR)
                    .collect()
            } else {
                bodies
                    .iter()
                    .map(project)
                    .filter(|p| p.depth > NEAR)
                    .collect()
            };
            // Off-screen quads waste paint time. Cull them before the sort.
            // Direct-indexed bins replace a hash map. No hash, no per-frame alloc.
            let bins = &mut self.tile_bins;
            let touched = &mut self.tile_touched;
            let across = merge_tile_axis(w);
            let down = merge_tile_axis(h);
            let spill = 2 * MERGE_TILE_SPILL;
            // Grid width. One spill column sits on each screen edge.
            let stride = across + spill;
            let tile_total = (stride * (down + spill)) as usize;
            // MERGE_TILES_MAX keeps this product inside `u32`.
            debug_assert!(tile_total <= u32::MAX as usize);
            if bins.len() < tile_total {
                bins.resize(tile_total, TileMerge::default());
            }
            points.retain_mut(|p| {
                if p.x + p.radius_px < 0.0
                    || p.x - p.radius_px > w
                    || p.y + p.radius_px < 0.0
                    || p.y - p.radius_px > h
                {
                    return false;
                }
                if p.radius_px >= MERGE_RADIUS_PX {
                    return true;
                }
                // Small dots overdraw the same few pixels. Merge them per tile.
                // Same key math as before. Spill tiles hold dots past an edge.
                let tx = ((p.x / MERGE_TILE_PX).floor() as i32).clamp(-MERGE_TILE_SPILL, across);
                let ty = ((p.y / MERGE_TILE_PX).floor() as i32).clamp(-MERGE_TILE_SPILL, down);
                let idx = ((ty + MERGE_TILE_SPILL) * stride + (tx + MERGE_TILE_SPILL)) as usize;
                debug_assert!(idx < tile_total);
                let t = &mut bins[idx];
                if t.count == 0 {
                    touched.push(idx as u32);
                }
                t.count += 1;
                t.x += p.x;
                t.y += p.y;
                t.depth += p.depth;
                t.spheres += u32::from(p.shape == Shape::Sphere);
                false
            });
            // Painter's algorithm: far bodies draw first.
            if points.len() >= PAR_MIN {
                points.par_sort_unstable_by(|a, b| b.depth.total_cmp(&a.depth));
            } else {
                points.sort_unstable_by(|a, b| b.depth.total_cmp(&a.depth));
            }
            // ponytail: per-tile depth flattening; per-tile depth sort is the upgrade path
            // One haze quad replaces the dots of one tile.
            // `points` moves into the list, so the capacity covers both parts.
            let mut merged: Vec<ProjectedBody> = Vec::with_capacity(touched.len() + points.len());
            for &idx in touched.iter() {
                let t = &bins[idx as usize];
                let n = t.count as f32;
                merged.push(ProjectedBody {
                    x: t.x / n,
                    y: t.y / n,
                    radius_px: MERGE_TILE_PX * MERGE_TILE_FILL,
                    depth: t.depth / n,
                    // The majority shape picks the tile color.
                    shape: if t.spheres * 2 >= t.count {
                        Shape::Sphere
                    } else {
                        Shape::Cube
                    },
                });
            }
            // Reset only the tiles this build used. The next build starts clean.
            for &idx in touched.iter() {
                bins[idx as usize] = TileMerge::default();
            }
            touched.clear();
            // Far haze paints first. Sharp bodies paint after it.
            merged.sort_unstable_by(|a, b| b.depth.total_cmp(&a.depth));
            merged.extend(points);
            (merged, Vec::new())
        } else {
            let n = self.fluid.n;
            let cell = FLUID_SPAN / n as f32;
            // The loop visits `n * n` cells, so each pushes at most one quad.
            let mut quads: Vec<ProjectedCell> = Vec::with_capacity(n * n);
            for j in 1..=n {
                for i in 1..=n {
                    let density = self.fluid.dens[i + (n + 2) * j];
                    if density <= DENSITY_CUTOFF {
                        continue;
                    }
                    let xw = FLUID_LEFT + (i as f32 - 0.5) * cell;
                    let yw = (j as f32 - 0.5) * cell;
                    let (x, y, focal, depth) = self.project([xw, yw, 0.0], w, h);
                    if depth <= NEAR {
                        continue;
                    }
                    let radius_px = (cell * focal / depth * 0.5).max(1.0);
                    quads.push(ProjectedCell {
                        x,
                        y,
                        radius_px,
                        density,
                        depth,
                    });
                }
            }
            quads.sort_unstable_by(|a, b| b.depth.total_cmp(&a.depth));
            (Vec::new(), quads)
        };
        let ms = (scene.elapsed().as_secs_f32() * 1000.0).min(1000.0);
        self.scene_ms = self.scene_ms * SMOOTH_KEEP + ms * SMOOTH_NEW;
        (points, fluid_quads)
    }
}
