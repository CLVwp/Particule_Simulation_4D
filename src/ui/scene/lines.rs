//! Line geometry. Emits the floor grid and the axes.

use super::{LineVert, NEAR, SceneOut};
use crate::ui::camera::Camera;
use crate::ui::theme::hsla_to_rgba;

/// Floor grid half span, in grid steps. Covers -10..=10 world units.
const GRID_HALF: i32 = 10;
/// Distance between two grid lines, in world units.
const GRID_STEP: i32 = 1;
/// Axis arm length, in world units.
const AXIS_LEN: f32 = 2.0;

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
