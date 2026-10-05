//! Grid fluid solver (Navier-Stokes, stable-fluids scheme) for visualization.
//!
//! The grid lives in a vertical plane. Grid x maps to world x, grid y maps to
//! world y. Advection is semi-Lagrangian, so the step is unconditionally stable.

// ponytail: 4 Gauss-Seidel iterations per solve; raise to 20 if swirls look wrong
const SOLVE_ITERS: usize = 4;
/// Upward push on dense cells, in units per second squared per unit density.
const BUOYANCY: f32 = 6.0;
/// Fraction of density kept after one step.
const DENSITY_KEEP: f32 = 0.995;
/// Emitter width divisor. The half-width is the grid side divided by this.
const EMITTER_SPAN: usize = 16;
/// Emitter height, in cells, counted up from the bottom wall.
const EMITTER_ROWS: usize = 3;
/// Upward speed the emitter adds to each cell, in units per second.
const JET_SPEED: f32 = 2.0;

/// The field a wall pass reflects.
#[derive(Clone, Copy)]
enum Bnd {
    /// Density or pressure. Walls mirror the neighbor value.
    Scalar,
    /// Horizontal velocity. Vertical walls flip its sign.
    U,
    /// Vertical velocity. Horizontal walls flip its sign.
    V,
}

/// A 2-D fluid in the x-y plane. Navier-Stokes with viscosity and diffusion.
#[derive(Debug)]
pub struct Fluid {
    /// Interior cells per side. Every array holds `n + 2` per side, walls included.
    pub n: usize,
    /// Kinematic viscosity. 0 turns velocity diffusion off.
    pub viscosity: f32,
    /// Diffusion of density. 0 turns it off.
    pub diffusion: f32,
    /// Density added per step at the emitter, at the bottom center.
    pub emit: f32,
    /// Horizontal velocity, walls included.
    pub u: Vec<f32>,
    /// Vertical velocity, walls included.
    pub v: Vec<f32>,
    /// Density, walls included. Read this to draw the fluid.
    pub dens: Vec<f32>,
    /// Scratch arrays: advected velocity and density, pressure and divergence.
    u0: Vec<f32>,
    v0: Vec<f32>,
    dens0: Vec<f32>,
}

impl Fluid {
    /// Creates a zeroed `n` by `n` fluid grid.
    ///
    /// # Panics
    ///
    /// Panics when the padded grid size overflows `usize`.
    pub fn new(n: usize) -> Self {
        let side = n.checked_add(2).expect("fluid grid size overflows");
        let size = side.checked_mul(side).expect("fluid grid size overflows");
        Fluid {
            n,
            viscosity: 0.0,
            diffusion: 0.0,
            emit: 1.0,
            u: vec![0.0; size],
            v: vec![0.0; size],
            dens: vec![0.0; size],
            u0: vec![0.0; size],
            v0: vec![0.0; size],
            dens0: vec![0.0; size],
        }
    }

    /// One simulation step.
    pub fn step(&mut self, dt: f32) {
        self.emit_density();
        self.buoyancy(dt);
        self.vel_step(dt);
        self.dens_step(dt);
    }

    /// Injects density and an upward jet at the bottom center.
    fn emit_density(&mut self) {
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

    /// Dense cells rise: the buoyancy term of Navier-Stokes.
    fn buoyancy(&mut self, dt: f32) {
        let n = self.n;
        for j in 1..=n {
            for i in 1..=n {
                self.v[ix(i, j, n)] += BUOYANCY * self.dens[ix(i, j, n)] * dt;
            }
        }
    }

    fn vel_step(&mut self, dt: f32) {
        let n = self.n;
        if self.viscosity > 0.0 {
            let a = dt * self.viscosity * (n * n) as f32;
            self.u0.clone_from(&self.u);
            self.v0.clone_from(&self.v);
            lin_solve(Bnd::U, &mut self.u, &self.u0, a, 1.0 + 4.0 * a, n);
            lin_solve(Bnd::V, &mut self.v, &self.v0, a, 1.0 + 4.0 * a, n);
        }
        self.project();
        self.u0.clone_from(&self.u);
        self.v0.clone_from(&self.v);
        advect(Bnd::U, &mut self.u, &self.u0, &self.u0, &self.v0, dt, n);
        advect(Bnd::V, &mut self.v, &self.v0, &self.u0, &self.v0, dt, n);
        self.project();
    }

    fn dens_step(&mut self, dt: f32) {
        let n = self.n;
        if self.diffusion > 0.0 {
            let a = dt * self.diffusion * (n * n) as f32;
            self.dens0.clone_from(&self.dens);
            lin_solve(
                Bnd::Scalar,
                &mut self.dens,
                &self.dens0,
                a,
                1.0 + 4.0 * a,
                n,
            );
        }
        self.dens0.clone_from(&self.dens);
        advect(
            Bnd::Scalar,
            &mut self.dens,
            &self.dens0,
            &self.u,
            &self.v,
            dt,
            n,
        );
        for d in &mut self.dens {
            *d *= DENSITY_KEEP;
        }
    }

    /// Makes the velocity field divergence-free: the pressure projection.
    fn project(&mut self) {
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

/// Row-major index of cell (`i`, `j`) in an `n + 2` square grid.
fn ix(i: usize, j: usize, n: usize) -> usize {
    i + (n + 2) * j
}

/// Gauss-Seidel relaxation for diffusion and the pressure solve.
fn lin_solve(b: Bnd, x: &mut [f32], x0: &[f32], a: f32, c: f32, n: usize) {
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
fn advect(b: Bnd, d: &mut [f32], d0: &[f32], u: &[f32], v: &[f32], dt: f32, n: usize) {
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

/// Solid walls on every side. The field picks the reflected component.
fn set_bnd(b: Bnd, x: &mut [f32], n: usize) {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tiny_grid_steps_without_panic() {
        // n = 1 puts the emitter range at the array edge.
        let mut f = Fluid::new(1);
        f.step(1.0 / 60.0);
    }

    #[test]
    fn emitted_fluid_spreads_and_stays_finite() {
        let mut f = Fluid::new(32);
        f.emit = 1.0;
        for _ in 0..90 {
            f.step(1.0 / 60.0);
        }
        assert!(f.dens.iter().all(|&d| d.is_finite()), "density diverged");
        assert!(f.u.iter().all(|&x| x.is_finite()), "velocity diverged");
        assert!(f.v.iter().all(|&x| x.is_finite()), "velocity diverged");
        let mut sum = 0.0;
        let mut peak = 0.0f32;
        let mut lit = 0;
        for &d in &f.dens {
            sum += d;
            peak = peak.max(d);
            if d > 0.1 {
                lit += 1;
            }
        }
        assert!(sum > 1.0, "density vanished");
        assert!(lit >= 8, "density never spread");
        assert!(peak < 200.0, "density piled up without limit");
    }
}
