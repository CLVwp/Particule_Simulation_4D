//! Camera math. Projects world points to screen space.

/// Orbit camera. Looks at `target` from `dist` along the rotated +z axis.
#[derive(Clone, Copy)]
pub(crate) struct Camera {
    /// Look-at point, in world units.
    pub(crate) target: [f32; 3],
    /// Horizontal angle, in radians.
    pub(crate) yaw: f32,
    /// Vertical angle, in radians. Clamped to -1.4..=1.4.
    pub(crate) pitch: f32,
    /// Distance from the target, in world units.
    pub(crate) dist: f32,
}

impl Default for Camera {
    /// Defaults match the old `SimView::new`.
    fn default() -> Self {
        Self {
            target: [0.0, 1.0, 0.0],
            yaw: 0.6,
            pitch: 0.35,
            dist: 12.0,
        }
    }
}

impl Camera {
    /// Projects a world point. Returns screen x, y, focal scale, camera depth.
    pub(crate) fn project(&self, p: [f32; 3], w: f32, h: f32) -> (f32, f32, f32, f32) {
        let p = [
            p[0] - self.target[0],
            p[1] - self.target[1],
            p[2] - self.target[2],
        ];
        let (cy, sy) = (self.yaw.cos(), self.yaw.sin());
        let (cp, sp) = (self.pitch.cos(), self.pitch.sin());
        let x = cy * p[0] + sy * p[2];
        let z1 = -sy * p[0] + cy * p[2];
        let y = cp * p[1] - sp * z1;
        let z = sp * p[1] + cp * z1 + self.dist;
        let focal = w.min(h);
        (w / 2.0 + x * focal / z, h / 2.0 - y * focal / z, focal, z)
    }

    /// Camera right axis in world space.
    pub(crate) fn right(&self) -> [f32; 3] {
        [self.yaw.cos(), 0.0, self.yaw.sin()]
    }

    /// Camera up axis in world space.
    pub(crate) fn up(&self) -> [f32; 3] {
        let (cp, sp) = (self.pitch.cos(), self.pitch.sin());
        [self.yaw.sin() * sp, cp, -self.yaw.cos() * sp]
    }

    /// Horizontal look direction in world space. The camera looks toward +depth,
    /// and depth 0 sits at the camera, so this is the positive depth axis.
    pub(crate) fn forward(&self) -> [f32; 3] {
        [-self.yaw.sin(), 0.0, self.yaw.cos()]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn project_centers_the_target() {
        let cam = Camera {
            target: [1.0, 2.0, 3.0],
            yaw: 0.6,
            pitch: 0.35,
            dist: 12.0,
        };
        let (x, y, focal, depth) = cam.project([1.0, 2.0, 3.0], 800.0, 600.0);
        assert_eq!((x, y, focal, depth), (400.0, 300.0, 600.0, 12.0));
    }

    #[test]
    fn basis_axes_stay_orthonormal() {
        let cam = Camera {
            target: [0.0, 0.0, 0.0],
            yaw: 1.1,
            pitch: -0.4,
            dist: 7.0,
        };
        let (r, u, f) = (cam.right(), cam.up(), cam.forward());
        let dot = |a: [f32; 3], b: [f32; 3]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
        assert!(dot(r, u).abs() < 1e-5);
        assert!(dot(r, f).abs() < 1e-5);
        assert!((dot(r, r).sqrt() - 1.0).abs() < 1e-5);
        assert!((dot(u, u).sqrt() - 1.0).abs() < 1e-5);
    }
}
