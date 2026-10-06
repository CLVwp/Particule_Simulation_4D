//! Shared tuning constants and the per-simulation settings.

/// Fraction of relative speed kept when two bodies collide.
const PAIR_RESTITUTION: f32 = 0.6;
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
/// Default solve rounds per step. Two rounds keep piles stiff enough.
const RESOLVE_ROUNDS: usize = 2;

/// Tunable laws of motion and solver tuning. `Default` matches the constants
/// above.
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
    /// Pool work starts above this body count. Live view of [`PAR_MIN`].
    pub par_min: usize,
    /// Keep only overlapping pairs in the contact list. The pairs this drops
    /// contribute zero to the solve, so the physics is unchanged either way.
    pub prune_dead_pairs: bool,
    /// Jacobi solve rounds per step.
    pub resolve_rounds: usize,
    /// Skip the remaining rounds when the mean delta motion drops below
    /// this. Zero disables the early exit.
    pub resolve_epsilon: f32,
}

impl Default for SimSettings {
    fn default() -> Self {
        SimSettings {
            gravity: GRAVITY,
            floor_restitution: FLOOR_RESTITUTION,
            ground_friction: GROUND_FRICTION,
            pair_restitution: PAIR_RESTITUTION,
            par_min: PAR_MIN,
            prune_dead_pairs: true,
            resolve_rounds: RESOLVE_ROUNDS,
            resolve_epsilon: 0.0,
        }
    }
}
