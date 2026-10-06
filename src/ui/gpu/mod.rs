//! GPU body storage for the vertex-pull render path.

use std::mem::size_of;

use bytemuck::{Pod, Zeroable};

use crate::engine::{Body, Shape};

use crate::ui::camera::Camera;
use crate::ui::scene::{BODY_LIGHT, BODY_SAT, CUBE_HUE, NEAR, SPHERE_HUE};
use crate::ui::theme::hsla_to_rgba;

#[cfg(test)]
mod grid;
#[cfg(test)]
mod pairs;
#[cfg(test)]
mod physics;

/// Builds one offscreen device for the GPU test modules. Returns `None`
/// without an adapter, so machines without Vulkan or DirectX skip.
#[cfg(test)]
pub(crate) fn headless_device() -> Option<(wgpu::Device, wgpu::Queue)> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::all(),
        ..wgpu::InstanceDescriptor::new_without_display_handle()
    });
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        ..Default::default()
    }))
    .ok()?;
    Some(
        pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: None,
            required_features: wgpu::Features::default(),
            required_limits: wgpu::Limits::default(),
            experimental_features: wgpu::ExperimentalFeatures::disabled(),
            memory_hints: wgpu::MemoryHints::default(),
            trace: wgpu::Trace::Off,
        }))
        .expect("device request fails"),
    )
}

/// One body in the GPU storage buffer. Exactly 32 bytes. Flat arrays: a
/// `vec3` would force align 16 and grow the stride to 48 bytes.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
pub(crate) struct GpuBody {
    /// Position in world units, as x, y, z.
    pub(crate) pos: [f32; 3],
    /// Velocity in units per second. The GPU physics will write it here.
    pub(crate) vel: [f32; 3],
    /// Sphere radius in world units.
    pub(crate) radius: f32,
    /// Below 0.5 draws a circle, at or above draws a square.
    pub(crate) shape: f32,
}

// Seven floats and one shape flag. Size growth would break the WGSL layout.
const _: () = assert!(size_of::<GpuBody>() == 32);

impl GpuBody {
    /// Copies one body. The shape maps to the draw flag.
    pub(crate) fn from_body(body: &Body) -> Self {
        GpuBody {
            pos: body.pos,
            vel: body.vel,
            radius: body.radius,
            shape: if body.shape == Shape::Sphere {
                0.0
            } else {
                1.0
            },
        }
    }
}

/// Camera constants for the vertex-pull shader. Exactly 64 bytes: sixteen
/// flat floats, so the Rust and WGSL offsets agree by construction.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(crate) struct CamUniforms {
    /// Viewport width, in logical points.
    pub(crate) vp_w: f32,
    /// Viewport height, in logical points.
    pub(crate) vp_h: f32,
    /// Focal scale. `min(vp_w, vp_h)`, like `Camera::project`.
    pub(crate) focal: f32,
    /// Camera distance from the target.
    pub(crate) dist: f32,
    /// Look-at point x, in world units.
    pub(crate) tx: f32,
    /// Look-at point y, in world units.
    pub(crate) ty: f32,
    /// Look-at point z, in world units.
    pub(crate) tz: f32,
    /// Horizontal angle, in radians.
    pub(crate) yaw: f32,
    /// Vertical angle, in radians.
    pub(crate) pitch: f32,
    /// Depths at or below this sit at or behind the camera.
    pub(crate) near: f32,
    /// Sphere color, red channel.
    pub(crate) sphere_r: f32,
    /// Sphere color, green channel.
    pub(crate) sphere_g: f32,
    /// Sphere color, blue channel.
    pub(crate) sphere_b: f32,
    /// Cube color, red channel.
    pub(crate) cube_r: f32,
    /// Cube color, green channel.
    pub(crate) cube_g: f32,
    /// Cube color, blue channel.
    pub(crate) cube_b: f32,
}

// Sixteen floats. Size growth would break the WGSL layout.
const _: () = assert!(size_of::<CamUniforms>() == 64);

impl CamUniforms {
    /// Packs the camera and the two body colors. The colors come from the
    /// same HSL constants the instance path paints with.
    pub(crate) fn new(cam: &Camera, w: f32, h: f32) -> Self {
        let sphere = hsla_to_rgba(SPHERE_HUE, BODY_SAT, BODY_LIGHT, 1.0);
        let cube = hsla_to_rgba(CUBE_HUE, BODY_SAT, BODY_LIGHT, 1.0);
        CamUniforms {
            vp_w: w,
            vp_h: h,
            focal: w.min(h),
            dist: cam.dist,
            tx: cam.target[0],
            ty: cam.target[1],
            tz: cam.target[2],
            yaw: cam.yaw,
            pitch: cam.pitch,
            near: NEAR,
            sphere_r: sphere[0],
            sphere_g: sphere[1],
            sphere_b: sphere[2],
            cube_r: cube[0],
            cube_g: cube[1],
            cube_b: cube[2],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_body_maps_the_shape_flag() {
        let sphere = Body {
            pos: [1.0, 2.0, 3.0],
            vel: [0.1, 0.2, 0.3],
            radius: 0.5,
            shape: Shape::Sphere,
        };
        let cube = Body {
            shape: Shape::Cube,
            ..sphere
        };
        let gs = GpuBody::from_body(&sphere);
        let gc = GpuBody::from_body(&cube);
        assert_eq!((gs.pos, gs.vel, gs.radius), (sphere.pos, sphere.vel, 0.5));
        assert_eq!(gs.shape, 0.0);
        assert_eq!(gc.shape, 1.0);
    }

    #[test]
    fn cam_uniforms_match_the_camera() {
        let cam = Camera {
            target: [1.0, 2.0, 3.0],
            yaw: 0.6,
            pitch: 0.35,
            dist: 12.0,
        };
        let u = CamUniforms::new(&cam, 800.0, 600.0);
        assert_eq!((u.vp_w, u.vp_h, u.focal), (800.0, 600.0, 600.0));
        assert_eq!(
            (u.tx, u.ty, u.tz, u.yaw, u.pitch, u.dist, u.near),
            (1.0, 2.0, 3.0, 0.6, 0.35, 12.0, NEAR)
        );
        // The packed colors stay inside the unit cube.
        assert!((0.0..=1.0).contains(&u.sphere_r));
        assert!((0.0..=1.0).contains(&u.cube_b));
    }
}
