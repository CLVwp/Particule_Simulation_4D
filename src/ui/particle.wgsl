// Particle and line shaders. All coordinates are physical pixels.
// One module holds two pipelines: instanced quads and flat colored lines.

struct Uniforms {
    vp_w: f32,
    vp_h: f32,
    pad: vec2<f32>,
}

@group(0) @binding(0) var<uniform> u: Uniforms;

struct InstanceIn {
    @location(0) pos_radius: vec4<f32>, // x, y, radius, shape
    @location(1) color: vec4<f32>,
}

struct QuadOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) color: vec4<f32>,
    @location(2) shape: f32,
}

@vertex
fn vs_main(@builtin(vertex_index) vi: u32, inst: InstanceIn) -> QuadOut {
    // Two triangles per quad, six corners, no index buffer.
    var corners = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(1.0, -1.0),
        vec2<f32>(1.0, 1.0),
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(1.0, 1.0),
        vec2<f32>(-1.0, 1.0),
    );
    let c = corners[vi];
    let x = inst.pos_radius.x + c.x * inst.pos_radius.z;
    let y = inst.pos_radius.y + c.y * inst.pos_radius.z;
    var out: QuadOut;
    // Pixel to NDC. y points down in pixels and up in NDC.
    out.pos = vec4<f32>(
        x / u.vp_w * 2.0 - 1.0,
        1.0 - y / u.vp_h * 2.0,
        0.0,
        1.0,
    );
    out.uv = c;
    out.color = inst.color;
    out.shape = inst.pos_radius.w;
    return out;
}

@fragment
fn fs_main(in: QuadOut) -> @location(0) vec4<f32> {
    // Signed distance to the edge. Circle: length. Square: box distance.
    let d = select(length(in.uv), max(abs(in.uv.x), abs(in.uv.y)), in.shape > 0.5);
    // One pixel of anti-aliasing along the edge.
    let aa = max(fwidth(d), 1e-6);
    let alpha = clamp((1.0 - d) / aa, 0.0, 1.0);
    // Straight alpha. Blending does the mix with the target.
    return vec4<f32>(in.color.rgb, in.color.a * alpha);
}

struct LineIn {
    @location(0) xy: vec2<f32>,
    @location(1) color: vec4<f32>,
}

struct LineOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) color: vec4<f32>,
}

@vertex
fn vs_line(v: LineIn) -> LineOut {
    var out: LineOut;
    out.pos = vec4<f32>(
        v.xy.x / u.vp_w * 2.0 - 1.0,
        1.0 - v.xy.y / u.vp_h * 2.0,
        0.0,
        1.0,
    );
    out.color = v.color;
    return out;
}

@fragment
fn fs_line(in: LineOut) -> @location(0) vec4<f32> {
    return in.color;
}
