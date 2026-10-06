// Vertex-pull body shader. One quad per body, read from a storage buffer.
// All coordinates are logical points, like the camera projection.

// One body in the storage buffer. Mirrors ui::gpu::GpuBody. Flat arrays:
// a vec3 would force align 16 and grow the stride to 48 bytes.
struct GpuBody {
    pos: array<f32, 3>,
    vel: array<f32, 3>,
    radius: f32,
    shape: f32,
}

// Camera constants. Mirrors ui::gpu::CamUniforms. Sixteen flat floats, so
// both sides agree on every offset.
struct Cam {
    vp_w: f32,
    vp_h: f32,
    focal: f32,
    dist: f32,
    tx: f32,
    ty: f32,
    tz: f32,
    yaw: f32,
    pitch: f32,
    near: f32,
    sphere_r: f32,
    sphere_g: f32,
    sphere_b: f32,
    cube_r: f32,
    cube_g: f32,
    cube_b: f32,
}

@group(0) @binding(0) var<uniform> c: Cam;
@group(0) @binding(1) var<storage, read> bodies: array<GpuBody>;

struct QuadOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) color: vec4<f32>,
    @location(2) shape: f32,
}

// One of six quad corners. Two triangles, no index buffer.
fn corner(vi: u32) -> vec2<f32> {
    var corners = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(1.0, -1.0),
        vec2<f32>(1.0, 1.0),
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(1.0, 1.0),
        vec2<f32>(-1.0, 1.0),
    );
    return corners[vi];
}

@vertex
fn vs_main(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> QuadOut {
    let b = bodies[ii];
    // Same math as Camera::project: translate, rotate yaw then pitch, add
    // the camera distance. Depth 0 sits at the camera.
    let rel = vec3<f32>(b.pos[0], b.pos[1], b.pos[2]) - vec3<f32>(c.tx, c.ty, c.tz);
    let cy = cos(c.yaw);
    let sy = sin(c.yaw);
    let cp = cos(c.pitch);
    let sp = sin(c.pitch);
    let x = cy * rel.x + sy * rel.z;
    let z1 = -sy * rel.x + cy * rel.z;
    let y = cp * rel.y - sp * z1;
    let depth = sp * rel.y + cp * z1 + c.dist;

    var out: QuadOut;
    if (depth <= c.near) {
        // Behind the camera the divide explodes. Push every corner outside
        // the clip volume, so the whole quad disappears.
        out.pos = vec4<f32>(4.0, 4.0, 4.0, 1.0);
        out.uv = vec2<f32>(0.0, 0.0);
        out.color = vec4<f32>(0.0, 0.0, 0.0, 0.0);
        out.shape = 0.0;
        return out;
    }
    let sx = c.vp_w * 0.5 + x * c.focal / depth;
    let syp = c.vp_h * 0.5 - y * c.focal / depth;
    let radius_px = max(b.radius * c.focal / depth, 1.5);
    let co = corner(vi);
    let qx = sx + co.x * radius_px;
    let qy = syp + co.y * radius_px;
    // Pixel to NDC. y points down in pixels and up in NDC.
    out.pos = vec4<f32>(
        qx / c.vp_w * 2.0 - 1.0,
        1.0 - qy / c.vp_h * 2.0,
        0.0,
        1.0,
    );
    out.uv = co;
    // Depth haze. Matches the instance path: near 1.0, far floor 0.25.
    let alpha = clamp(1.5 - depth / c.dist, 0.25, 1.0);
    let rgb = select(
        vec3<f32>(c.cube_r, c.cube_g, c.cube_b),
        vec3<f32>(c.sphere_r, c.sphere_g, c.sphere_b),
        b.shape < 0.5,
    );
    out.color = vec4<f32>(rgb, alpha);
    out.shape = b.shape;
    return out;
}
