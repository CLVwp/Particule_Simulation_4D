//! Body instances. Projects Newton bodies and merges small dots into tiles.

use particule_simulation_4d::engine::{Body, Shape};
use rayon::prelude::*;

use super::{Instance, NEAR, SceneOut, TileCache, TileMerge, Tuning};
use crate::ui::camera::Camera;
use crate::ui::theme::hsla_to_rgba;

/// Bodies below this screen radius merge into tiles. Bigger bodies paint alone.
pub(crate) const MERGE_RADIUS_PX: f32 = 3.0;
/// Side length of one merge tile, in screen pixels.
pub(crate) const MERGE_TILE_PX: f32 = 8.0;
/// The merged quad radius spans this share of the tile. It covers the whole tile.
pub(crate) const MERGE_TILE_FILL: f32 = 0.75;
/// Cap for tile rows and columns on one axis. It keeps tile indexes inside `u32`.
const MERGE_TILES_MAX: i32 = 1 << 15;

/// One projected body quad. Held only during a build, then emitted.
struct SortBody {
    x: f32,
    y: f32,
    radius_px: f32,
    depth: f32,
    shape: Shape,
}

/// Tile rows or columns for one screen span. The result is at least one tile.
fn merge_tile_axis(span: f32, tile_px: f32) -> i32 {
    ((span / tile_px).ceil().max(1.0) as i32).min(MERGE_TILES_MAX)
}

/// Emits one instance per visible Newton body. Small dots merge into tiles.
/// Sorts far to near, so draw order follows the painter's algorithm.
#[expect(clippy::too_many_arguments)]
pub(crate) fn emit_bodies(
    cam: &Camera,
    bodies: &[Body],
    w: f32,
    h: f32,
    dist: f32,
    tuning: &Tuning,
    par_min: usize,
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
    let mut points: Vec<SortBody> = if bodies.len() >= par_min {
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
    let across = merge_tile_axis(w, tuning.merge_tile_px);
    let down = merge_tile_axis(h, tuning.merge_tile_px);
    // Dots can reach one tile past an edge per merge radius, so the spill
    // ring follows the tuning values.
    let edge = ((tuning.merge_radius_px / tuning.merge_tile_px).ceil() as i32).max(1);
    let spill = 2 * edge;
    // Grid width. One spill column sits on each screen edge.
    let stride = across + spill;
    let tile_total = (stride * (down + spill)) as usize;
    // MERGE_TILES_MAX keeps this product inside `u32`.
    debug_assert!(tile_total <= u32::MAX as usize);
    if bins.len() < tile_total {
        bins.resize(tile_total, TileMerge::default());
    }
    points.retain_mut(|p| {
        if tuning.cull_offscreen
            && (p.x + p.radius_px < 0.0
                || p.x - p.radius_px > w
                || p.y + p.radius_px < 0.0
                || p.y - p.radius_px > h)
        {
            return false;
        }
        if !tuning.lod_merge || p.radius_px >= tuning.merge_radius_px {
            return true;
        }
        // Small dots overdraw the same few pixels. Merge them per tile.
        // Spill tiles hold dots past a screen edge.
        let tx = ((p.x / tuning.merge_tile_px).floor() as i32).clamp(-edge, across);
        let ty = ((p.y / tuning.merge_tile_px).floor() as i32).clamp(-edge, down);
        let idx = ((ty + edge) * stride + (tx + edge)) as usize;
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
    if points.len() >= par_min {
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
            radius_px: tuning.merge_tile_px * tuning.merge_tile_fill,
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
        emit_bodies(
            &cam,
            &[body],
            W,
            H,
            cam.dist,
            &Tuning::default(),
            usize::MAX,
            &mut tiles,
            &mut out,
        );
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
        emit_bodies(
            &cam,
            &[body],
            W,
            H,
            cam.dist,
            &Tuning::default(),
            usize::MAX,
            &mut tiles,
            &mut out,
        );
        assert!(out.instances.is_empty());
    }
}
