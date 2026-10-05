//! Scene output. Turns world objects into GPU instances and line vertices.

use std::mem::size_of;
use std::time::Instant;

use bytemuck::{Pod, Zeroable};
use particule_simulation_4d::engine::fluid::Fluid;
use particule_simulation_4d::engine::{Body, PAR_MIN, Shape};
use rayon::prelude::*;

use crate::ui::camera::Camera;
use crate::ui::theme::{SMOOTH_KEEP, SMOOTH_NEW, hsla_to_rgba};
use crate::ui::{App, PhysicsMode};

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

/// Floor grid half span, in grid steps. Covers -10..=10 world units.
const GRID_HALF: i32 = 10;
/// Distance between two grid lines, in world units.
const GRID_STEP: i32 = 1;
/// Axis arm length, in world units.
const AXIS_LEN: f32 = 2.0;

/// One GPU quad instance. Exactly 32 bytes. Flat fields only: a `vec4` field
/// would force align 16 and pad the vertex stride to 48 bytes.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(crate) struct Instance {
    /// Center x, in physical pixels.
    pub(crate) x: f32,
    /// Center y, in physical pixels.
    pub(crate) y: f32,
    /// Radius, in physical pixels.
    pub(crate) radius: f32,
    /// 0.0 draws a circle. 1.0 draws a square.
    pub(crate) shape: f32,
    /// Straight-alpha color.
    pub(crate) color: [f32; 4],
}

// One instance feeds one GPU vertex step. Size growth would break the layout.
const _: () = assert!(size_of::<Instance>() == 32);

/// One line vertex. Exactly 24 bytes. Six vertices build one segment as two
/// triangles, with the corner offsets computed on the CPU.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(crate) struct LineVert {
    /// Position x, in physical pixels.
    pub(crate) x: f32,
    /// Position y, in physical pixels.
    pub(crate) y: f32,
    /// Straight-alpha color.
    pub(crate) color: [f32; 4],
}

// Two floats plus one color. Size growth would break the layout.
const _: () = assert!(size_of::<LineVert>() == 24);

/// Scene output. The renderer uploads these vectors once per frame.
/// All fields reset each build. Both vectors stay allocated between frames.
#[derive(Default)]
pub(crate) struct SceneOut {
    /// One quad per drawn body or fluid cell.
    pub(crate) instances: Vec<Instance>,
    /// Six vertices per drawn grid or axis segment.
    pub(crate) lines: Vec<LineVert>,
    /// Number of valid instances. Set to `instances.len()` at each build.
    pub(crate) instance_count: usize,
    /// One entry per visible axis tip: screen x, y, color, text.
    pub(crate) axis_labels: Vec<(f32, f32, [f32; 4], &'static str)>,
}

/// One projected body quad. Held only during a build, then emitted.
struct SortBody {
    x: f32,
    y: f32,
    radius_px: f32,
    depth: f32,
    shape: Shape,
}

/// One projected fluid cell quad. Held only during a build, then emitted.
struct SortCell {
    x: f32,
    y: f32,
    radius_px: f32,
    density: f32,
    depth: f32,
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

/// Reused tile merge bins across frames.
// ponytail: per-tile depth flattening; per-tile depth sort is the upgrade path
#[derive(Default)]
pub(crate) struct TileCache {
    /// One sum set per tile. Each build resets only its used slots.
    pub(crate) bins: Vec<TileMerge>,
    /// Indexes of the tiles the last build used.
    pub(crate) touched: Vec<u32>,
}

/// Tile rows or columns for one screen span. The result is at least one tile.
fn merge_tile_axis(span: f32) -> i32 {
    ((span / MERGE_TILE_PX).ceil().max(1.0) as i32).min(MERGE_TILES_MAX)
}

/// Appends one screen segment as six vertices. They form two triangles of a
/// quad with the given thickness. Corner math comes from the old `paint_line`.
pub(crate) fn push_segment(
    out: &mut SceneOut,
    x1: f32,
    y1: f32,
    x2: f32,
    y2: f32,
    color: [f32; 4],
    thickness: f32,
) {
    let (dx, dy) = (x2 - x1, y2 - y1);
    let len = (dx * dx + dy * dy).sqrt().max(1e-6);
    let (nx, ny) = (-dy / len * thickness, dx / len * thickness);
    let corners = [
        (x1 + nx, y1 + ny),
        (x2 + nx, y2 + ny),
        (x2 - nx, y2 - ny),
        (x1 - nx, y1 - ny),
    ];
    for tri in [[0usize, 1, 2], [0, 2, 3]] {
        for i in tri {
            let (x, y) = corners[i];
            out.lines.push(LineVert { x, y, color });
        }
    }
}

/// Emits one instance per visible Newton body. Small dots merge into tiles.
/// Sorts far to near, so draw order follows the painter's algorithm.
pub(crate) fn emit_bodies(
    cam: &Camera,
    bodies: &[Body],
    w: f32,
    h: f32,
    dist: f32,
    tiles: &mut TileCache,
    out: &mut SceneOut,
) {
    // The projection divides by the camera depth. Anything at or behind the
    // camera (depth <= NEAR) projects to garbage. Cull it, or it shows up as
    // mirrored ghost shapes.
    let project = |b: &Body| {
        let (x, y, focal, depth) = cam.project(b.pos, w, h);
        SortBody {
            x,
            y,
            radius_px: (b.radius * focal / depth).max(1.5),
            depth,
            shape: b.shape,
        }
    };
    let mut points: Vec<SortBody> = if bodies.len() >= PAR_MIN {
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
    let bins = &mut tiles.bins;
    let touched = &mut tiles.touched;
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
        // Spill tiles hold dots past a screen edge.
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
    // Far bodies draw first.
    if points.len() >= PAR_MIN {
        points.par_sort_unstable_by(|a, b| b.depth.total_cmp(&a.depth));
    } else {
        points.sort_unstable_by(|a, b| b.depth.total_cmp(&a.depth));
    }
    // One haze quad replaces the dots of one tile.
    // `points` moves into the list, so the capacity covers both parts.
    let mut merged: Vec<SortBody> = Vec::with_capacity(touched.len() + points.len());
    for &idx in touched.iter() {
        let t = &bins[idx as usize];
        let n = t.count as f32;
        merged.push(SortBody {
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
    // Every quad becomes one instance. Colors keep the old paint formulas.
    for p in &merged {
        let alpha = (1.5 - p.depth / dist).clamp(0.25, 1.0);
        let sphere = p.shape == Shape::Sphere;
        out.instances.push(Instance {
            x: p.x,
            y: p.y,
            radius: p.radius_px,
            shape: if sphere { 0.0 } else { 1.0 },
            color: hsla_to_rgba(if sphere { 0.53 } else { 0.08 }, 0.9, 0.6, alpha),
        });
    }
}

/// Emits one instance per lit fluid cell. Cells never merge into tiles.
/// Sorts far to near, so draw order follows the painter's algorithm.
pub(crate) fn emit_fluid(cam: &Camera, fluid: &Fluid, w: f32, h: f32, out: &mut SceneOut) {
    let n = fluid.n;
    let cell = FLUID_SPAN / n as f32;
    // The loop visits `n * n` cells, so each push fills one sorted quad.
    let mut quads: Vec<SortCell> = Vec::with_capacity(n * n);
    for j in 1..=n {
        for i in 1..=n {
            let density = fluid.dens[i + (n + 2) * j];
            if density <= DENSITY_CUTOFF {
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

/// Emits the floor grid and the axes. Both modes draw them.
/// Each segment that touches the camera plane drops whole.
pub(crate) fn emit_lines(cam: &Camera, w: f32, h: f32, out: &mut SceneOut) {
    // Floor grid. A segment with one end behind the camera warps too, so
    // the whole segment goes.
    let span = (GRID_HALF * GRID_STEP) as f32;
    let grid_color = hsla_to_rgba(0.55, 0.4, 0.5, 0.35);
    for gi in -GRID_HALF..=GRID_HALF {
        let g = (gi * GRID_STEP) as f32;
        let segs = [
            ([g, 0.0, -span], [g, 0.0, span]),
            ([-span, 0.0, g], [span, 0.0, g]),
        ];
        for (a, b) in segs {
            let pa = cam.project(a, w, h);
            let pb = cam.project(b, w, h);
            if pa.3 > NEAR && pb.3 > NEAR {
                push_segment(out, pa.0, pa.1, pb.0, pb.1, grid_color, 0.7);
            }
        }
    }
    // Orthonormal frame: one colored arm per axis, plus a text label at each tip.
    let axes = [
        ([AXIS_LEN, 0.0, 0.0], hsla_to_rgba(0.0, 0.8, 0.55, 1.0), "X"),
        ([0.0, AXIS_LEN, 0.0], hsla_to_rgba(0.33, 0.8, 0.5, 1.0), "Y"),
        ([0.0, 0.0, AXIS_LEN], hsla_to_rgba(0.58, 0.8, 0.6, 1.0), "Z"),
    ];
    let origin = cam.project([0.0, 0.0, 0.0], w, h);
    for (tip, color, label) in axes {
        let end = cam.project(tip, w, h);
        if origin.3 > NEAR && end.3 > NEAR {
            push_segment(out, origin.0, origin.1, end.0, end.1, color, 1.5);
            out.axis_labels.push((end.0, end.1, color, label));
        }
    }
}

impl App {
    /// Fills `out` with this frame's instances and lines. Empties every list
    /// first, so their capacity stays reused between frames.
    pub(crate) fn build_scene(&mut self, w: f32, h: f32, out: &mut SceneOut) {
        let scene = Instant::now();
        out.instances.clear();
        out.lines.clear();
        out.axis_labels.clear();
        let dist = self.cam.dist;
        // One law set fills the instance stream. Grid and axes always draw.
        match self.mode {
            PhysicsMode::Newton => {
                emit_bodies(
                    &self.cam,
                    &self.world.bodies,
                    w,
                    h,
                    dist,
                    &mut self.tiles,
                    out,
                );
            }
            PhysicsMode::Fluid => emit_fluid(&self.cam, &self.fluid, w, h, out),
        }
        emit_lines(&self.cam, w, h, out);
        out.instance_count = out.instances.len();
        let ms = (scene.elapsed().as_secs_f32() * 1000.0).min(1000.0);
        self.scene_ms = self.scene_ms * SMOOTH_KEEP + ms * SMOOTH_NEW;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const W: f32 = 800.0;
    const H: f32 = 600.0;

    /// Camera with yaw 0 and pitch 0. Screen x maps to world x, screen y to
    /// world y, and depth to world z plus `dist`.
    fn flat_cam() -> Camera {
        Camera {
            target: [0.0; 3],
            yaw: 0.0,
            pitch: 0.0,
            dist: 12.0,
        }
    }

    #[test]
    fn body_at_target_lands_at_screen_center() {
        let cam = flat_cam();
        // Radius 0.1 projects to 5 px: past the merge floor of 3 px.
        let body = Body {
            pos: [0.0; 3],
            vel: [0.0; 3],
            radius: 0.1,
            shape: Shape::Sphere,
        };
        let mut tiles = TileCache::default();
        let mut out = SceneOut::default();
        emit_bodies(&cam, &[body], W, H, cam.dist, &mut tiles, &mut out);
        assert_eq!(out.instances.len(), 1);
        let q = out.instances[0];
        assert_eq!((q.x, q.y), (W / 2.0, H / 2.0));
        // Radius = 0.1 * focal 600 / depth 12 = 5 px.
        assert!((q.radius - 5.0).abs() < 1e-4);
        // Alpha = 1.5 - depth/dist = 0.5. This proves depth equals `dist`.
        assert_eq!(q.color[3], 0.5);
        assert_eq!(q.shape, 0.0);
    }

    #[test]
    fn far_offscreen_body_culls() {
        let cam = flat_cam();
        // World x 1e6 projects past the right edge. Depth stays 12, above NEAR,
        // so only the off-screen cull can drop it.
        let body = Body {
            pos: [1.0e6, 0.0, 0.0],
            vel: [0.0; 3],
            radius: 0.1,
            shape: Shape::Sphere,
        };
        let mut tiles = TileCache::default();
        let mut out = SceneOut::default();
        emit_bodies(&cam, &[body], W, H, cam.dist, &mut tiles, &mut out);
        assert!(out.instances.is_empty());
    }
}
