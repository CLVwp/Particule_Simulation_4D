// Broad-phase grid build: histogram, scatter, bucket sort.
// The table is a spatial hash. WGSL has no 64-bit integers, so the bucket
// hashes the cell triple in u32 space. Exact cell checks in the pair pass
// reject hash collisions, like the CPU distance test rejects folded cells.
//
// Each kernel owns one bind group: the layouts carry exactly the buffers
// the entry point uses.

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

// Grid kernels: bodies, constants, table, cursor, ids.
@group(0) @binding(0) var<storage, read_write> bodies: array<GpuBody>;
@group(0) @binding(1) var<uniform> sim: Sim;
@group(0) @binding(2) var<storage, read_write> table: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> cursor: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read_write> body_ids: array<u32>;
// Cell triple per body, cached for the pair and solve passes.
@group(0) @binding(7) var<storage, read_write> cells: array<vec3<i32>>;

// Clamped cell coordinates: the same floor, offset, and clamp as the CPU
// `key_part`. One axis spans 21 bits, so cells reach +/-1M.
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

// One thread per body. Adds one count to the body's bucket.
@compute @workgroup_size(64)
fn grid_hist(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= sim.n) {
        return;
    }
    let b = bucket_of(cell_coords(at(bodies[i]), sim.cell_size), sim.table_mask);
    atomicAdd(&table[b], 1u);
}

// One thread per body. Lands the body in its bucket slice. The slot order
// inside one bucket is nondeterministic; the sort kernel fixes it next.
@compute @workgroup_size(64)
fn grid_scatter(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= sim.n) {
        return;
    }
    let c = cell_coords(at(bodies[i]), sim.cell_size);
    let b = bucket_of(c, sim.table_mask);
    let slot = atomicLoad(&table[b]) + atomicAdd(&cursor[b], 1u);
    body_ids[slot] = i;
    cells[i] = c;
}

// One thread per bucket. Insertion sort by body index, ascending. Bucket
// sizes sit near 1 to 8, so the quadratic walk stays trivial. 256 threads
// per group keep the dispatch under the 65535 group limit.
@compute @workgroup_size(256)
fn grid_bucket_sort(@builtin(global_invocation_id) gid: vec3<u32>) {
    let b = gid.x;
    if (b > sim.table_mask) {
        return;
    }
    let start = atomicLoad(&table[b]);
    // select would evaluate the atomic load on b + 1 even for the last
    // bucket, and that index is out of bounds. Branch instead.
    var end = sim.n;
    if (b < sim.table_mask) {
        end = atomicLoad(&table[b + 1u]);
    }
    var i = start + 1u;
    loop {
        if (i >= end) {
            break;
        }
        let key = body_ids[i];
        var j = i;
        loop {
            if (j == start) {
                break;
            }
            let prev = body_ids[j - 1u];
            if (prev <= key) {
                break;
            }
            body_ids[j] = prev;
            j = j - 1u;
        }
        body_ids[j] = key;
        i = i + 1u;
    }
}
