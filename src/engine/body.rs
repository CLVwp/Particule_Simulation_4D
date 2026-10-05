//! The body type and its visual shape.

/// Visual shape of a body. Physics always uses a sphere of the same radius.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Shape {
    /// Drawn as a sphere.
    Sphere,
    /// Drawn as a box. Physics still uses the sphere.
    Cube,
}

/// One particle: a sphere with a position and a velocity.
#[derive(Clone, Copy, Debug)]
pub struct Body {
    /// Position in world units, as x, y, z.
    pub pos: [f32; 3],
    /// Velocity in units per second.
    pub vel: [f32; 3],
    /// Sphere radius in world units.
    pub radius: f32,
    /// Visual shape. Physics always uses the sphere.
    pub shape: Shape,
}

impl Body {
    /// Uniform density: mass grows with the volume.
    pub(super) fn mass(&self) -> f32 {
        self.radius * self.radius * self.radius
    }
}

// Layout guards. These types fill the hot arrays, so their size must not drift.
// Body holds 7 floats and a 1-byte tag. Alignment pads it to 32 bytes.
const _: () = assert!(std::mem::size_of::<Body>() == 32);
