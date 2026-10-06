// Contact pair pass over the 14-offset stencil: count, scan, fill.
// The scan turns per-body counts into one CSR range per body, so the flat
// pair order is body ascending, then stencil order, then bucket order.
// The bucket sort made the bucket order ascending, so two runs replay
// byte-identical.

struct GpuBody {
    pos: array<f32, 3>,
    vel: array<f32, 3>,
    radius: f32,
    shape: f32,
}

struct Sim {
    dt: f32,
    gravity: f32,
    cell_size: f32,
    floor_restitution: f32,
    ground_friction: f32,
    pair_restitution: f32,
    pair_friction: f32,
    slop: f32,
    correction: f32,
    max_speed: f32,
    floor_y: f32,
    prune: f32,
    n: u32,
    rounds: u32,
    table_mask: u32,
    pair_cap: u32,
}

struct Pair {
    i: u32,
    j: u32,
    mi: f32,
    mj: f32,
}

// Shared grid bindings. The cursor is unused here but stays in the layout.
@group(0) @binding(0) var<storage, read_write> bodies: array<GpuBody>;
@group(0) @binding(1) var<uniform> sim: Sim;
@group(0) @binding(2) var<storage, read_write> table: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> cursor: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read_write> body_ids: array<u32>;
@group(0) @binding(5) var<storage, read_write> pair_start: array<u32>;
@group(0) @binding(6) var<storage, read_write> pairs: array<Pair>;

// Self plus the 13 lex-positive offsets of the 27-cell neighborhood. Every
// unordered cell pair appears exactly once, like the CPU STENCIL.
const STENCIL = array<vec3<i32>, 14>(
    vec3<i32>(0i, 0i, 0i),
    vec3<i32>(0i, 0i, 1i),
    vec3<i32>(0i, 1i, -1i),
    vec3<i32>(0i, 1i, 0i),
    vec3<i32>(0i, 1i, 1i),
    vec3<i32>(1i, -1i, -1i),
    vec3<i32>(1i, -1i, 0i),
    vec3<i32>(1i, -1i, 1i),
    vec3<i32>(1i, 0i, -1i),
    vec3<i32>(1i, 0i, 0i),
    vec3<i32>(1i, 0i, 1i),
    vec3<i32>(1i, 1i, -1i),
    vec3<i32>(1i, 1i, 0i),
    vec3<i32>(1i, 1i, 1i),
);

// Clamped cell coordinates. Same floor, offset, and clamp as the CPU.
fn cell_coords(p: vec3<f32>, cell_size: f32) -> vec3<i32> {
    let q = floor(p / cell_size);
    let off = vec3<i32>(1 << 20);
    return clamp(vec3<i32>(q) + off, vec3<i32>(0i), vec3<i32>((1 << 21) - 1));
}

// Murmur-style 32-bit mix of the cell triple, masked to the table.
fn bucket_of(c: vec3<i32>, mask: u32) -> u32 {
    var h = bitcast<u32>(c.x) * 0x9E3779B1u;
    h = h ^ (bitcast<u32>(c.y) * 0x85EBCA77u);
    h = h ^ (bitcast<u32>(c.z) * 0xC2B2AE3Du);
    h = h ^ (h >> 16u);
    h = h * 0x7FEB352Du;
    h = h ^ (h >> 15u);
    h = h * 0x846CA68Bu;
    h = h ^ (h >> 16u);
    return h & mask;
}

fn at(b: GpuBody) -> vec3<f32> {
    return vec3<f32>(b.pos[0], b.pos[1], b.pos[2]);
}

// Bucket range after the scan. Branch, never select: the last bucket would
// read b + 1 out of bounds and trip the driver.
fn bucket_range(b: u32) -> vec2<u32> {
    let start = atomicLoad(&table[b]);
    var end = sim.n;
    if (b < sim.table_mask) {
        end = atomicLoad(&table[b + 1u]);
    }
    return vec2<u32>(start, end);
}

// The overlap test. Mirrors the CPU `overlap`: d is j minus i, and the
// reject test is dist2 >= min_d*min_d or dist2 < 1e-12.
fn overlaps(pi: vec3<f32>, pj: vec3<f32>, ri: f32, rj: f32) -> bool {
    if (sim.prune < 0.5) {
        return true;
    }
    let d = pj - pi;
    let dist2 = d.x * d.x + d.y * d.y + d.z * d.z;
    let min_d = ri + rj;
    return dist2 < min_d * min_d && dist2 >= 1e-12;
}

// Walks the 14 offsets for body `i`. Returns the accepted pair count and,
// when `fill` is true, writes them at base plus k, clipped at pair_cap.
fn walk_pairs(i: u32, base: u32, fill: bool) -> u32 {
    let body_i = bodies[i];
    let ci = cell_coords(at(body_i), sim.cell_size);
    var k = 0u;
    for (var s = 0u; s < 14u; s = s + 1u) {
        let want = ci + STENCIL[s];
        let range = bucket_range(bucket_of(want, sim.table_mask));
        for (var idx = range.x; idx < range.y; idx = idx + 1u) {
            let j = body_ids[idx];
            // Same cell: each pair once, low index first.
            if (s == 0u && j <= i) {
                continue;
            }
            let body_j = bodies[j];
            let cj = cell_coords(at(body_j), sim.cell_size);
            // Exact cell check: bucket collisions end here.
            if (cj.x != want.x || cj.y != want.y || cj.z != want.z) {
                continue;
            }
            if (overlaps(at(body_i), at(body_j), body_i.radius, body_j.radius)) {
                if (fill && base + k < sim.pair_cap) {
                    pairs[base + k] = Pair(
                        i,
                        j,
                        body_i.radius * body_i.radius * body_i.radius,
                        body_j.radius * body_j.radius * body_j.radius,
                    );
                }
                k = k + 1u;
            }
        }
    }
    return k;
}

// One thread per body. Writes the accepted pair count into the CSR slot.
@compute @workgroup_size(64)
fn pair_count(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= sim.n) {
        return;
    }
    pair_start[i] = walk_pairs(i, 0u, false);
}

// One thread per body. Fills its CSR slice of the flat pair list.
@compute @workgroup_size(64)
fn pair_fill(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= sim.n) {
        return;
    }
    let base = pair_start[i];
    if (base < sim.pair_cap) {
        walk_pairs(i, base, true);
    }
}
