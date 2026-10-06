//! Grid fluid solver (Navier-Stokes, stable-fluids scheme) for visualization.
//!
//! The grid lives in a vertical plane. Grid x maps to world x, grid y maps to
//! world y. Advection is semi-Lagrangian, so the step is unconditionally stable.

mod bounds;
mod emitter;
mod solver;

use bounds::Bnd;
use solver::{advect, lin_solve};

/// Upward push on dense cells, in units per second squared per unit density.
const BUOYANCY: f32 = 6.0;
/// Fraction of density kept after one step.
const DENSITY_KEEP: f32 = 0.995;

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
    u: Vec<f32>,
    /// Vertical velocity, walls included.
    v: Vec<f32>,
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
}

/// Row-major index of cell (`i`, `j`) in an `n + 2` square grid.
fn ix(i: usize, j: usize, n: usize) -> usize {
    i + (n + 2) * j
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
