//! Scene output. Turns world objects into GPU instances and line vertices.

use std::mem::size_of;
use std::time::Instant;

use bytemuck::{Pod, Zeroable};

use crate::ui::theme::{SMOOTH_KEEP, SMOOTH_NEW};
use crate::ui::{App, PhysicsMode};

mod bodies;
mod fluid;
mod lines;

use self::bodies::{MERGE_RADIUS_PX, MERGE_TILE_FILL, MERGE_TILE_PX, emit_bodies};
use self::fluid::emit_fluid;
use self::lines::emit_lines;

/// Depths at or below this sit at or behind the camera.
const NEAR: f32 = 0.5;
/// Cells at or below this density do not draw.
pub(crate) const DENSITY_CUTOFF: f32 = 0.02;

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
    /// One entry per visible axis tip: screen x, y, color, text.
    pub(crate) axis_labels: Vec<(f32, f32, [f32; 4], &'static str)>,
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

/// Live tuning for the render-side optimizations. `Default` matches the
/// constants above. The F1 panel edits these at runtime.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Tuning {
    /// Merge small dots into screen tiles.
    pub(crate) lod_merge: bool,
    /// Bodies below this screen radius merge into tiles.
    pub(crate) merge_radius_px: f32,
    /// Side length of one merge tile, in screen pixels.
    pub(crate) merge_tile_px: f32,
    /// The merged quad radius spans this share of the tile.
    pub(crate) merge_tile_fill: f32,
    /// Drop quads that lie fully outside the viewport.
    pub(crate) cull_offscreen: bool,
    /// Fluid cells at or below this density do not draw.
    pub(crate) density_cutoff: f32,
}

impl Default for Tuning {
    fn default() -> Self {
        Tuning {
            lod_merge: true,
            merge_radius_px: MERGE_RADIUS_PX,
            merge_tile_px: MERGE_TILE_PX,
            merge_tile_fill: MERGE_TILE_FILL,
            cull_offscreen: true,
            density_cutoff: DENSITY_CUTOFF,
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
        let par_min = self.world.settings.par_min;
        // One law set fills the instance stream. Grid and axes always draw.
        match self.mode {
            PhysicsMode::Newton => {
                emit_bodies(
                    &self.cam,
                    &self.world.bodies,
                    w,
                    h,
                    dist,
                    &self.tuning,
                    par_min,
                    &mut self.tiles,
                    out,
                );
            }
            PhysicsMode::Fluid => emit_fluid(
                &self.cam,
                &self.fluid,
                w,
                h,
                self.tuning.density_cutoff,
                out,
            ),
        }
        emit_lines(&self.cam, w, h, out);
        let ms = (scene.elapsed().as_secs_f32() * 1000.0).min(1000.0);
        self.scene_ms = self.scene_ms * SMOOTH_KEEP + ms * SMOOTH_NEW;
    }
}
