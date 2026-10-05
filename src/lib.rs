//! Particle physics simulation.
//!
//! [`engine`] holds the physics. The `main` binary wires it to the GPUI UI.
//!
//! Lints: correctness bugs fail the build; perf issues warn.

#![deny(clippy::correctness)]
#![warn(clippy::perf)]

pub mod engine;
pub mod perf;
