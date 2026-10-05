//! Debug overlay. CPU, GPU, and memory stats in the bottom-left corner.

use std::mem::size_of;

use gpui_kit::*;
use particule_simulation_4d::engine::{Body, PAR_MIN, thread_count};
use particule_simulation_4d::perf::{allocated_bytes, peak_bytes};

use crate::ui::theme::{FAINT, FG};
use crate::ui::{PhysicsMode, SimView};

impl SimView {
    /// Builds the overlay. CPU, GPU, and memory stats.
    // ponytail: per-thread OS load needs Win32 FFI; the phase split stands in for it
    pub(crate) fn render_debug(&self, quads: usize, w: f32, h: f32) -> Div {
        let head = |t: &str| {
            div()
                .text_color(FG)
                .text_size(px(12.0))
                .child(t.to_string())
        };
        let line = |t: String| div().text_color(FAINT).child(t);
        let p = self.world.phase_ms;
        let step_total: f32 = p.iter().sum();
        let phase = |name: &str, ms: f32| {
            let share = if step_total > 0.0 {
                100.0 * ms / step_total
            } else {
                0.0
            };
            format!("{name:<9} {ms:7.3} ms {share:5.1} %")
        };
        let path = if self.mode == PhysicsMode::Fluid {
            format!("fluid grid {} x {}", self.fluid.n, self.fluid.n)
        } else if self.world.bodies.len() >= PAR_MIN {
            "parallel".to_string()
        } else {
            "inline".to_string()
        };

        div()
            .absolute()
            .bottom_2()
            .left_2()
            .flex()
            .flex_col()
            .gap_3()
            .p_3()
            .rounded_lg()
            .bg(rgb(0x10141b))
            .text_size(px(11.0))
            .child(head("CPU"))
            .child(line(format!(
                "FPS {:.0}   frame {:.1} ms   step {:.3} ms",
                self.fps,
                1000.0 / self.fps,
                self.step_ms
            )))
            .child(line(format!(
                "bodies {}   contacts {}   threads {}",
                self.world.bodies.len(),
                self.world.contact_count(),
                thread_count()
            )))
            .child(line(format!("path: {path}   PAR_MIN {PAR_MIN}")))
            .child(line(phase("integrate", p[0])))
            .child(line(phase("grid", p[1])))
            .child(line(phase("contacts", p[2])))
            .child(line(phase("resolve", p[3])))
            .child(line(phase("floor", p[4])))
            .child(head("GPU"))
            .child(line(format!("viewport {:.0} x {:.0} px", w, h)))
            .child(line(format!("quads painted {quads}")))
            .child(line(format!(
                "scene (project + sort) {:.3} ms",
                self.scene_ms
            )))
            .child(line("gpui does not expose GPU timers.".to_string()))
            .child(head("Memory"))
            .child(line(format!(
                "in use {:.1} MB   peak {:.1} MB",
                allocated_bytes() as f32 / 1048576.0,
                peak_bytes() as f32 / 1048576.0
            )))
            .child(line(format!(
                "bodies array {:.2} MB",
                (size_of::<Body>() * self.world.bodies.len()) as f32 / 1048576.0
            )))
    }
}

/// Formats a diffusion-style rate. Zero prints plain, other values print scientific.
pub(crate) fn fmt_rate(x: f32) -> String {
    if x <= 0.0 {
        "0".to_string()
    } else {
        format!("{x:.1e}")
    }
}
