//! Wall boundary passes for the fluid fields.

use super::ix;

/// The field a wall pass reflects.
#[derive(Clone, Copy)]
pub(super) enum Bnd {
    /// Density or pressure. Walls mirror the neighbor value.
    Scalar,
    /// Horizontal velocity. Vertical walls flip its sign.
    U,
    /// Vertical velocity. Horizontal walls flip its sign.
    V,
}

/// Solid walls on every side. The field picks the reflected component.
pub(super) fn set_bnd(b: Bnd, x: &mut [f32], n: usize) {
    // Vertical walls flip the horizontal part.
    // Horizontal walls flip the vertical part.
    let (flip_x, flip_y) = match b {
        Bnd::Scalar => (false, false),
        Bnd::U => (true, false),
        Bnd::V => (false, true),
    };
    for i in 1..=n {
        x[ix(0, i, n)] = if flip_x {
            -x[ix(1, i, n)]
        } else {
            x[ix(1, i, n)]
        };
        x[ix(n + 1, i, n)] = if flip_x {
            -x[ix(n, i, n)]
        } else {
            x[ix(n, i, n)]
        };
        x[ix(i, 0, n)] = if flip_y {
            -x[ix(i, 1, n)]
        } else {
            x[ix(i, 1, n)]
        };
        x[ix(i, n + 1, n)] = if flip_y {
            -x[ix(i, n, n)]
        } else {
            x[ix(i, n, n)]
        };
    }
    let (first, last) = (ix(0, 0, n), ix(n + 1, n + 1, n));
    x[first] = 0.5 * (x[ix(1, 0, n)] + x[ix(0, 1, n)]);
    x[ix(n + 1, 0, n)] = 0.5 * (x[ix(n, 0, n)] + x[ix(n + 1, 1, n)]);
    x[ix(0, n + 1, n)] = 0.5 * (x[ix(0, n, n)] + x[ix(1, n + 1, n)]);
    x[last] = 0.5 * (x[ix(n, n + 1, n)] + x[ix(n + 1, n, n)]);
}
