//! Shared colors and smoothing weights.

use gpui_kit::*;

/// Window background color.
pub(crate) const BG: u32 = 0x0b0e14;
/// Main text color.
pub(crate) const FG: Hsla = hsla(0.58, 0.15, 0.9, 1.0);
/// Secondary text color.
pub(crate) const FAINT: Hsla = hsla(0.58, 0.15, 0.85, 0.9);
/// Each frame-stat average keeps this share of its old value.
pub(crate) const SMOOTH_KEEP: f32 = 0.9;
/// Each frame-stat average takes this share of its new value.
pub(crate) const SMOOTH_NEW: f32 = 0.1;
