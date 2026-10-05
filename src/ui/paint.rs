//! Scene painting. Builds the floor grid and axes, then paints the quads.

use gpui_kit::*;
use particule_simulation_4d::engine::Shape;

use crate::ui::SimView;
use crate::ui::scene::{
    AxisLabel, NEAR, ProjectedBody, ProjectedCell, ProjectedLine, ProjectedPoint, ProjectedSeg,
};

const GRID_HALF: i32 = 10; // floor grid spans -10..=10 units
const GRID_STEP: i32 = 1;
const AXIS_LEN: f32 = 2.0;

impl SimView {
    /// Projects the floor grid. Culls each segment that touches the camera plane.
    pub(crate) fn grid_lines(&self, w: f32, h: f32) -> Vec<ProjectedSeg> {
        // Floor grid. A segment with one end behind the camera warps too, so
        // the whole segment goes.
        let span = (GRID_HALF * GRID_STEP) as f32;
        let mut grid: Vec<([f32; 3], [f32; 3])> = Vec::with_capacity(42);
        for gi in -GRID_HALF..=GRID_HALF {
            let g = (gi * GRID_STEP) as f32;
            grid.push(([g, 0.0, -span], [g, 0.0, span]));
            grid.push(([-span, 0.0, g], [span, 0.0, g]));
        }
        grid.into_iter()
            .map(|(a, b)| (self.project(a, w, h), self.project(b, w, h)))
            .filter(|(a, b)| a.3 > NEAR && b.3 > NEAR)
            .collect::<Vec<_>>()
    }

    /// Builds one colored line per axis and one label anchor at each tip.
    pub(crate) fn axes(&self, w: f32, h: f32) -> (Vec<ProjectedLine>, Vec<AxisLabel>) {
        // Orthonormal frame: one colored arm per axis, plus a text label at each tip.
        const AXES: [([f32; 3], Hsla, &str); 3] = [
            ([AXIS_LEN, 0.0, 0.0], hsla(0.0, 0.8, 0.55, 1.0), "X"),
            ([0.0, AXIS_LEN, 0.0], hsla(0.33, 0.8, 0.5, 1.0), "Y"),
            ([0.0, 0.0, AXIS_LEN], hsla(0.58, 0.8, 0.6, 1.0), "Z"),
        ];
        let mut axis_lines: Vec<ProjectedLine> = Vec::with_capacity(3);
        let mut axis_labels: Vec<AxisLabel> = Vec::with_capacity(3);
        for (tip, color, label) in AXES {
            let a = self.project([0.0, 0.0, 0.0], w, h);
            let b = self.project(tip, w, h);
            if a.3 > NEAR && b.3 > NEAR {
                axis_lines.push((a, b, color));
                axis_labels.push((b.0, b.1, color, label));
            }
        }
        (axis_lines, axis_labels)
    }
}

/// Paints the projected scene. Consumes the projected lists.
pub(crate) fn paint_scene(
    window: &mut Window,
    dist: f32,
    points: Vec<ProjectedBody>,
    fluid_quads: Vec<ProjectedCell>,
    grid: Vec<ProjectedSeg>,
    axis_lines: Vec<ProjectedLine>,
) {
    for (a, b, color) in axis_lines {
        paint_line(window, a, b, color, 1.5);
    }
    for (a, b) in grid {
        paint_line(window, a, b, hsla(0.55, 0.4, 0.5, 0.35), 0.7);
    }
    for q in &fluid_quads {
        let alpha = (0.15 + 0.75 * q.density).clamp(0.15, 0.9);
        let light = (0.45 + 0.2 * q.density).clamp(0.4, 0.75);
        window.paint_quad(fill(
            Bounds::new(
                point(px(q.x - q.radius_px), px(q.y - q.radius_px)),
                size(px(q.radius_px * 2.0), px(q.radius_px * 2.0)),
            ),
            hsla(0.55, 0.85, light, alpha),
        ));
    }
    for p in &points {
        let alpha = (1.5 - p.depth / dist).clamp(0.25, 1.0);
        let color = match p.shape {
            Shape::Sphere => hsla(0.53, 0.9, 0.6, alpha),
            Shape::Cube => hsla(0.08, 0.9, 0.6, alpha),
        };
        let round = match p.shape {
            Shape::Sphere => px(p.radius_px),
            Shape::Cube => px(0.0),
        };
        window.paint_quad(
            fill(
                Bounds::new(
                    point(px(p.x - p.radius_px), px(p.y - p.radius_px)),
                    size(px(p.radius_px * 2.0), px(p.radius_px * 2.0)),
                ),
                color,
            )
            .corner_radii(round),
        );
    }
}

/// Builds one positioned text div per axis label.
pub(crate) fn axis_label_divs(labels: Vec<AxisLabel>) -> impl Iterator<Item = Div> {
    labels.into_iter().map(|(x, y, color, label)| {
        div()
            .absolute()
            .left(px(x - 4.0))
            .top(px(y - 14.0))
            .text_size(px(11.0))
            .text_color(color)
            .child(label)
    })
}

/// Paints a world-space line segment as a thin filled quad.
fn paint_line(
    window: &mut Window,
    a: ProjectedPoint,
    b: ProjectedPoint,
    color: Hsla,
    thickness: f32,
) {
    let (x1, y1) = (a.0, a.1);
    let (x2, y2) = (b.0, b.1);
    let dx = x2 - x1;
    let dy = y2 - y1;
    let len = (dx * dx + dy * dy).sqrt().max(1e-6);
    let nx = -dy / len * thickness;
    let ny = dx / len * thickness;
    let mut path = PathBuilder::default();
    path.move_to(point(px(x1 + nx), px(y1 + ny)));
    path.line_to(point(px(x2 + nx), px(y2 + ny)));
    path.line_to(point(px(x2 - nx), px(y2 - ny)));
    path.line_to(point(px(x1 - nx), px(y1 - ny)));
    if let Ok(p) = path.build() {
        window.paint_path(p, color);
    }
}
