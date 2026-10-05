//! Camera math. Projects world points to screen space.

use crate::ui::SimView;
use crate::ui::scene::ProjectedPoint;

impl SimView {
    /// Projects a world point. Returns one [`ProjectedPoint`].
    pub(crate) fn project(&self, p: [f32; 3], w: f32, h: f32) -> ProjectedPoint {
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
