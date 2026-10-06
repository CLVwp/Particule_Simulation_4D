//! Particle physics simulation.
//!
//! [`engine`] holds the physics. [`ui`] wires it to the window, and keeps
//! the scene build reachable for benches and examples. The engine reads no
//! UI type, so it stays testable and reusable on its own.
//!
//! Lints: correctness bugs fail the build; perf issues warn. Set in Cargo.toml
//! so every target, including the binary and the bench, gets the same policy.

pub mod engine;
pub mod perf;
pub mod ui;
