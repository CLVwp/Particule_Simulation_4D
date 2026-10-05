//! A small linear congruential generator for spawn jitter.

// ponytail: hand-rolled LCG instead of the `rand` crate; swap if we need real distributions
pub(super) struct Rng(pub(super) u64);

/// Multiplier of the Knuth MMIX linear congruential generator.
const LCG_MULT: u64 = 6364136223846793005;
/// Increment of the Knuth MMIX linear congruential generator.
const LCG_INC: u64 = 1442695040888963407;

impl Rng {
    /// Uniform in [-1, 1].
    pub(super) fn next_f32(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(LCG_MULT).wrapping_add(LCG_INC);
        // The top 32 bits fill a full u32, so the fraction spans [0, 1].
        ((self.0 >> 32) as u32 as f32 / u32::MAX as f32) * 2.0 - 1.0
    }
}
