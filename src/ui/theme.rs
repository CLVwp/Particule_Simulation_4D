//! Shared colors, smoothing weights, and the HSL conversion.

use egui::Color32;

/// Window background color. Single source for the panel fill and the render clear.
pub(crate) const BG: Color32 = Color32::from_rgb(0x0b, 0x0e, 0x14);
/// Main text color. From `hsla(0.58, 0.15, 0.9, 1.0)`. egui stores premultiplied.
pub(crate) const FG: Color32 = Color32::from_rgba_premultiplied(226, 230, 233, 255);
/// Secondary text color. From `hsla(0.58, 0.15, 0.85, 0.9)`. Premultiplied.
pub(crate) const FAINT: Color32 = Color32::from_rgba_premultiplied(190, 196, 200, 230);
/// Each frame-stat average keeps this share of its old value.
pub(crate) const SMOOTH_KEEP: f32 = 0.9;
/// Each frame-stat average takes this share of its new value.
pub(crate) const SMOOTH_NEW: f32 = 0.1;

/// Converts HSL to straight-alpha RGBA. `h` is one turn (0..1).
/// `s`, `l`, and `a` are 0..1. Matches the old `Hsla` conversion.
pub(crate) fn hsla_to_rgba(h: f32, s: f32, l: f32, a: f32) -> [f32; 4] {
    let c = (1.0 - (2.0 * l - 1.0).abs()) * s;
    let hp = h.rem_euclid(1.0) * 6.0;
    let x = c * (1.0 - (hp.rem_euclid(2.0) - 1.0).abs());
    let (r1, g1, b1) = match hp as u32 {
        0 => (c, x, 0.0),
        1 => (x, c, 0.0),
        2 => (0.0, c, x),
        3 => (0.0, x, c),
        4 => (x, 0.0, c),
        _ => (c, 0.0, x),
    };
    let m = l - c / 2.0;
    [r1 + m, g1 + m, b1 + m, a]
}
