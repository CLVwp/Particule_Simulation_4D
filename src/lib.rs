//! Particle physics simulation.
//!
//! [`engine`] holds the physics. The `main` binary wires it to the UI.
//!
//! Lints: correctness bugs fail the build; perf issues warn. Set in Cargo.toml
//! so every target, including the binary and the bench, gets the same policy.

pub mod engine;
pub mod perf;
