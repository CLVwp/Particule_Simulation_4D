//! The diffusion solve, advection, and pressure projection of the fluid.

use super::Fluid;
use super::bounds::{Bnd, set_bnd};
use super::ix;

// ponytail: 4 Gauss-Seidel iterations per solve; raise to 20 if swirls look wrong
const SOLVE_ITERS: usize = 4;

/// Gauss-Seidel relaxation for diffusion and the pressure solve.
pub(super) fn lin_solve(b: Bnd, x: &mut [f32], x0: &[f32], a: f32, c: f32, n: usize) {
    let row = n + 2;
    assert_eq!(x.len(), row * row, "fluid array does not match the grid");
    let inv = c.recip();
    for _ in 0..SOLVE_ITERS {
        for j in 1..=n {
            // Split the three rows apart, so the neighbor loads need no checks.
            let mid = j * row;
            let (head, next) = x.split_at_mut(mid + row);
            let (_, pair) = head.split_at_mut(mid - row);
            let (prev, cur) = pair.split_at_mut(row);
            let src = &x0[mid..mid + row];
            for i in 1..row - 1 {
                cur[i] = (a * (cur[i - 1] + cur[i + 1] + prev[i] + next[i]) + src[i]) * inv;
            }
        }
        set_bnd(b, x, n);
    }
}

/// Moves `d0` along the velocity field into `d` (semi-Lagrangian advection).
pub(super) fn advect(b: Bnd, d: &mut [f32], d0: &[f32], u: &[f32], v: &[f32], dt: f32, n: usize) {
    let row = n + 2;
    assert_eq!(d.len(), row * row, "fluid array does not match the grid");
    let dt0 = dt * n as f32;
    let top = n as f32 + 0.5;
    for j in 1..=n {
        let mid = j * row;
        let u_row = &u[mid..mid + row];
        let v_row = &v[mid..mid + row];
        let d_row = &mut d[mid..mid + row];
        for i in 1..row - 1 {
            let x = (i as f32 - dt0 * u_row[i]).clamp(0.5, top);
            let y = (j as f32 - dt0 * v_row[i]).clamp(0.5, top);
            let i0 = x as usize;
            let j0 = y as usize;
            let s1 = x - i0 as f32;
            let t1 = y - j0 as f32;
            let (s0, t0) = (1.0 - s1, 1.0 - t1);
            d_row[i] = s0 * (t0 * d0[ix(i0, j0, n)] + t1 * d0[ix(i0, j0 + 1, n)])
                + s1 * (t0 * d0[ix(i0 + 1, j0, n)] + t1 * d0[ix(i0 + 1, j0 + 1, n)]);
        }
    }
    set_bnd(b, d, n);
}

impl Fluid {
    /// Makes the velocity field divergence-free: the pressure projection.
    pub(super) fn project(&mut self) {
        let n = self.n;
        let row = n + 2;
        let h = 1.0 / n as f32;
        // Scratch: divergence in `v0`, pressure in `u0`.
        for j in 1..=n {
            let mid = j * row;
            let u_row = &self.u[mid..mid + row];
            let v_up = &self.v[mid + row..mid + 2 * row];
            let v_dn = &self.v[mid - row..mid];
            let div = &mut self.v0[mid..mid + row];
            let prs = &mut self.u0[mid..mid + row];
            for i in 1..row - 1 {
                div[i] = -0.5 * h * (u_row[i + 1] - u_row[i - 1] + v_up[i] - v_dn[i]);
                prs[i] = 0.0;
            }
        }
        set_bnd(Bnd::Scalar, &mut self.v0, n);
        set_bnd(Bnd::Scalar, &mut self.u0, n);
        lin_solve(Bnd::Scalar, &mut self.u0, &self.v0, 1.0, 4.0, n);
        for j in 1..=n {
            let mid = j * row;
            let p_row = &self.u0[mid..mid + row];
            let p_up = &self.u0[mid + row..mid + 2 * row];
            let p_dn = &self.u0[mid - row..mid];
            let u_row = &mut self.u[mid..mid + row];
            let v_row = &mut self.v[mid..mid + row];
            for i in 1..row - 1 {
                u_row[i] -= 0.5 * (p_row[i + 1] - p_row[i - 1]) / h;
                v_row[i] -= 0.5 * (p_up[i] - p_dn[i]) / h;
            }
        }
        set_bnd(Bnd::U, &mut self.u, n);
        set_bnd(Bnd::V, &mut self.v, n);
    }
}
