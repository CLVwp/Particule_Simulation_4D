//! Physics core — no UI code here, testable standalone.
//!
//! `step` splits its work over the rayon pool. The pool has one worker per
//! logical core, detected from the CPU at startup.

pub mod fluid;

mod body;
mod config;
mod rng;
mod world;

pub use body::{Body, Shape};
pub use config::{BODY_RADIUS, CONTACT_CORRECTION, CONTACT_SLOP, FLOOR_Y, MIN_RADIUS, SimSettings};
pub use world::{World, thread_count};
