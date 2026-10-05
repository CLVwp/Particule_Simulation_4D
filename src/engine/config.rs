//! Shared tuning constants and the per-simulation settings.

use super::resolve::PAIR_RESTITUTION;

/// Downward acceleration, in units per second squared.
pub const GRAVITY: f32 = -9.81;
/// Height of the floor plane.
pub const FLOOR_Y: f32 = 0.0;
/// Fraction of vertical speed kept after a floor bounce.
pub const FLOOR_RESTITUTION: f32 = 0.75;
/// Fraction of horizontal speed kept while a body touches the floor.
pub const GROUND_FRICTION: f32 = 0.9;
/// Radius of every spawned body.
pub const BODY_RADIUS: f32 = 0.1;
/// Smallest radius `spawn` accepts. Thinner bodies divide by zero in the solver.
pub const MIN_RADIUS: f32 = 0.05;
/// Pool work starts above this body count.
/// ponytail: bench shows 16-thread sync is a net loss at 1000 bodies, a win at 4000
pub const PAR_MIN: usize = 2048;

/// Tunable laws of motion. `Default` matches the constants above.
#[derive(Clone, Copy, Debug)]
pub struct SimSettings {
    /// Downward acceleration, in units per second squared.
    pub gravity: f32,
    /// Fraction of vertical speed kept after a floor bounce.
    pub floor_restitution: f32,
    /// Fraction of horizontal speed kept while a body touches the floor.
    pub ground_friction: f32,
    /// Fraction of relative speed kept when two bodies collide.
    pub pair_restitution: f32,
}

impl Default for SimSettings {
    fn default() -> Self {
        SimSettings {
            gravity: GRAVITY,
            floor_restitution: FLOOR_RESTITUTION,
            ground_friction: GROUND_FRICTION,
            pair_restitution: PAIR_RESTITUTION,
        }
    }
}
