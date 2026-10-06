// Contact solver: the Jacobi rounds on device. Four kernels build the
// per-body contact lists once per frame, then one delta kernel and one
// apply kernel run per round. The math mirrors the CPU `contact_delta`
// and `contact_share`, friction tangent included.

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

// One delta per pair, recomputed every round. 36 bytes: two flat triples
// and three scalars.
struct Delta {
    n: array<f32, 3>,
    t: array<f32, 3>,
    push: f32,
    impulse: f32,
    jt: f32,
}

// High bit of a body-contact tag. Set when the body is the `j` side.
const J_SIDE: u32 = 2147483648u;

// Solve bindings: eight storage buffers, the WebGPU stage limit. The
// solve never touches the cell cache, so it gets its own layout.
@group(0) @binding(0) var<storage, read_write> bodies: array<GpuBody>;
@group(0) @binding(1) var<uniform> sim: Sim;
@group(0) @binding(2) var<storage, read_write> pairs: array<Pair>;
@group(0) @binding(3) var<storage, read_write> pair_start: array<u32>;
@group(0) @binding(4) var<storage, read_write> deltas: array<Delta>;
@group(0) @binding(5) var<storage, read_write> bc_start: array<atomic<u32>>;
@group(0) @binding(6) var<storage, read_write> bc_cursor: array<atomic<u32>>;
@group(0) @binding(7) var<storage, read_write> bc_items: array<atomic<u32>>;

// One thread per pair. Two contact-list counts, one per side.
@compute @workgroup_size(64)
fn solve_count(@builtin(global_invocation_id) gid: vec3<u32>) {
    let p = gid.x;
    if (p >= sim.pair_cap) {
        return;
    }
    // Past the pair total the slots hold last frame's bytes. Skip them.
    if (p >= pair_start[sim.n]) {
        return;
    }
    let pair = pairs[p];
    atomicAdd(&bc_start[pair.i], 1u);
    atomicAdd(&bc_start[pair.j], 1u);
}

// One thread per pair. Lands the pair index in both body lists, with the
// side bit on the `j` entry. The order inside one list is nondeterministic
// until the sort kernel fixes it.
@compute @workgroup_size(64)
fn solve_fill(@builtin(global_invocation_id) gid: vec3<u32>) {
    let p = gid.x;
    if (p >= sim.pair_cap) {
        return;
    }
    if (p >= pair_start[sim.n]) {
        return;
    }
    let pair = pairs[p];
    let ri = atomicAdd(&bc_cursor[pair.i], 1u);
    let base_i = atomicLoad(&bc_start[pair.i]);
    atomicStore(&bc_items[base_i + ri], p);
    let rj = atomicAdd(&bc_cursor[pair.j], 1u);
    let base_j = atomicLoad(&bc_start[pair.j]);
    atomicStore(&bc_items[base_j + rj], p | J_SIDE);
}

// One thread per body. Insertion sort by pair index, ascending. The sum
// order per body becomes fixed, which pins the round result per device.
@compute @workgroup_size(64)
fn solve_sort(@builtin(global_invocation_id) gid: vec3<u32>) {
    let b = gid.x;
    if (b >= sim.n) {
        return;
    }
    let start = atomicLoad(&bc_start[b]);
    // The scan leaves the total in slot n, so b + 1 is always in bounds.
    let end = atomicLoad(&bc_start[b + 1u]);
    var i = start + 1u;
    loop {
        if (i >= end) {
            break;
        }
        let key_tag = atomicLoad(&bc_items[i]);
        let key = key_tag & ~J_SIDE;
        var j = i;
        loop {
            if (j == start) {
                break;
            }
            let prev = atomicLoad(&bc_items[j - 1u]) & ~J_SIDE;
            if (prev <= key) {
                break;
            }
            // The moved entry keeps its own side bit.
            atomicStore(&bc_items[j], atomicLoad(&bc_items[j - 1u]));
            j = j - 1u;
        }
        atomicStore(&bc_items[j], key_tag);
        i = i + 1u;
    }
}

// One thread per pair. Recomputes the delta. Mirrors the CPU
// `contact_delta`: same re-test, same normal, same friction tangent.
@compute @workgroup_size(64)
fn solve_delta(@builtin(global_invocation_id) gid: vec3<u32>) {
    let p = gid.x;
    if (p >= sim.pair_cap) {
        return;
    }
    if (p >= pair_start[sim.n]) {
        return;
    }
    let pair = pairs[p];
    let a = bodies[pair.i];
    let b = bodies[pair.j];
    var out: Delta;
    out.n = array<f32, 3>(0.0, 0.0, 0.0);
    out.t = array<f32, 3>(0.0, 0.0, 0.0);
    out.push = 0.0;
    out.impulse = 0.0;
    out.jt = 0.0;
    let pv = vec3<f32>(b.pos[0] - a.pos[0], b.pos[1] - a.pos[1], b.pos[2] - a.pos[2]);
    let dist2 = dot(pv, pv);
    let min_d = a.radius + b.radius;
    if (dist2 >= min_d * min_d || dist2 < 1e-12) {
        deltas[p] = out;
        return;
    }
    let dist = sqrt(dist2);
    let nv = pv / dist;
    let rv = vec3<f32>(
        b.vel[0] - a.vel[0],
        b.vel[1] - a.vel[1],
        b.vel[2] - a.vel[2],
    );
    let vn = dot(rv, nv);
    let impulse = select(0.0, -(1.0 + sim.pair_restitution) * vn / (1.0 / pair.mi + 1.0 / pair.mj), vn < 0.0);
    let tv = rv - vn * nv;
    let vt2 = dot(tv, tv);
    var tvn = vec3<f32>(0.0, 0.0, 0.0);
    var jt = 0.0;
    if (sim.pair_friction > 0.0 && vt2 > 1e-18) {
        let vt_len = sqrt(vt2);
        tvn = tv / vt_len;
        jt = -sim.pair_friction * vt_len / (1.0 / pair.mi + 1.0 / pair.mj);
    }
    out.n = array<f32, 3>(nv.x, nv.y, nv.z);
    out.t = array<f32, 3>(tvn.x, tvn.y, tvn.z);
    out.push = max(min_d - dist - sim.slop, 0.0) * sim.correction / (pair.mi + pair.mj);
    out.impulse = impulse;
    out.jt = jt;
    deltas[p] = out;
}

// One thread per body. Accumulates the shares of its contact list in
// list order, then applies. Mirrors the CPU `contact_share`.
@compute @workgroup_size(64)
fn solve_apply(@builtin(global_invocation_id) gid: vec3<u32>) {
    let b = gid.x;
    if (b >= sim.n) {
        return;
    }
    let start = atomicLoad(&bc_start[b]);
    let end = atomicLoad(&bc_start[b + 1u]);
    var dpos = vec3<f32>(0.0, 0.0, 0.0);
    var dvel = vec3<f32>(0.0, 0.0, 0.0);
    for (var idx = start; idx < end; idx = idx + 1u) {
        let tag = atomicLoad(&bc_items[idx]);
        let ci = tag & ~J_SIDE;
        let j_side = (tag & J_SIDE) != 0u;
        let d = deltas[ci];
        let pair = pairs[ci];
        // Pack the sign and the two masses into one vector. No tuple
        // destructuring in WGSL.
        let side = select(
            vec3<f32>(-1.0, pair.mj, pair.mi),
            vec3<f32>(1.0, pair.mi, pair.mj),
            j_side,
        );
        let sign = side.x;
        let m_pos = side.y;
        let m_vel = side.z;
        let dp = sign * vec3<f32>(d.n[0], d.n[1], d.n[2]) * d.push * m_pos;
        let dv = sign * (vec3<f32>(d.n[0], d.n[1], d.n[2]) * d.impulse
            + vec3<f32>(d.t[0], d.t[1], d.t[2]) * d.jt) / m_vel;
        dpos = dpos + dp;
        dvel = dvel + dv;
    }
    bodies[b].pos[0] = bodies[b].pos[0] + dpos.x;
    bodies[b].pos[1] = bodies[b].pos[1] + dpos.y;
    bodies[b].pos[2] = bodies[b].pos[2] + dpos.z;
    bodies[b].vel[0] = bodies[b].vel[0] + dvel.x;
    bodies[b].vel[1] = bodies[b].vel[1] + dvel.y;
    bodies[b].vel[2] = bodies[b].vel[2] + dvel.z;
}
