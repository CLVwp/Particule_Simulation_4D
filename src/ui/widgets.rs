//! Reusable egui widgets. Every fn takes the ui. No fn takes the app.

use std::ops::RangeInclusive;

use egui::{Align2, DragValue, RichText, Slider, Ui, Vec2, Window};

use crate::ui::theme::{FAINT, FG};

/// Adds one faint label at the default body size.
pub(crate) fn faint(ui: &mut Ui, text: impl Into<String>) {
    ui.label(RichText::new(text).color(FAINT));
}

/// Adds one faint label at an explicit point size.
pub(crate) fn faint_px(ui: &mut Ui, text: impl Into<String>, px: f32) {
    ui.label(RichText::new(text).size(px).color(FAINT));
}

/// Adds one faint monospace line at 11 pt.
pub(crate) fn line(ui: &mut Ui, text: String) {
    ui.label(RichText::new(text).size(11.0).monospace().color(FAINT));
}

/// Adds one small bright heading.
pub(crate) fn head(ui: &mut Ui, text: &str) {
    ui.label(RichText::new(text.to_string()).size(12.0).color(FG));
}

/// Adds one faint label, one slider, and the grid row end.
pub(crate) fn grid_slider<T: egui::emath::Numeric>(
    ui: &mut Ui,
    label: &str,
    value: &mut T,
    range: RangeInclusive<T>,
    step: f64,
) {
    faint(ui, label);
    ui.add(Slider::new(value, range).step_by(step));
    ui.end_row();
}

/// Adds one selectable per option. A click assigns the option. Returns
/// whether any option was clicked this frame, even the active one. The
/// re-click lets the caller replay a side effect, as the layout picker
/// did before the split.
pub(crate) fn enum_toggle<T: PartialEq + Copy>(
    ui: &mut Ui,
    current: &mut T,
    options: &[(T, &str)],
) -> bool {
    let mut clicked = false;
    for (value, name) in options {
        if ui.selectable_label(*current == *value, *name).clicked() {
            *current = *value;
            clicked = true;
        }
    }
    clicked
}

/// Adds one drag box for a diffusion-style rate.
pub(crate) fn rate_drag(ui: &mut Ui, value: &mut f32) {
    ui.add(
        DragValue::new(value)
            .speed(0.05)
            .custom_formatter(|v, _| fmt_rate(v as f32)),
    );
}

/// One labeled slider row inside the tuning panel.
pub(crate) fn slider_row(
    ui: &mut Ui,
    label: &str,
    value: &mut f32,
    range: RangeInclusive<f32>,
    step: f64,
) {
    ui.horizontal(|ui| {
        faint(ui, label);
        ui.add(Slider::new(value, range).step_by(step));
    });
}

/// Builds one non-resizable side window. It anchors top right at `offset`.
pub(crate) fn side_window(title: &str, offset: Vec2) -> Window<'_> {
    Window::new(title)
        .anchor(Align2::RIGHT_TOP, offset)
        .default_width(250.0)
        .resizable(false)
        .collapsible(false)
}

/// Formats a diffusion-style rate. Zero prints plain, other values print
/// scientific.
fn fmt_rate(x: f32) -> String {
    if x <= 0.0 {
        "0".to_string()
    } else {
        format!("{x:.1e}")
    }
}
