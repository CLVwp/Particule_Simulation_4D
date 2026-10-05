//! The density and jet emitter at the bottom center of the fluid grid.

use super::Fluid;
use super::ix;

/// Emitter width divisor. The half-width is the grid side divided by this.
const EMITTER_SPAN: usize = 16;
/// Emitter height, in cells, counted up from the bottom wall.
const EMITTER_ROWS: usize = 3;
/// Upward speed the emitter adds to each cell, in units per second.
const JET_SPEED: f32 = 2.0;

impl Fluid {
    /// Injects density and an upward jet at the bottom center.
    pub(super) fn emit_density(&mut self) {
        if self.emit <= 0.0 {
            return;
        }
        let n = self.n;
        let cx = n / 2;
        let half = (n / EMITTER_SPAN).max(1);
        for j in 1..=EMITTER_ROWS.min(n) {
            for i in cx.saturating_sub(half)..=(cx + half) {
                let k = ix(i, j, n);
                self.dens[k] += self.emit;
                self.v[k] += JET_SPEED;
            }
        }
    }
}
