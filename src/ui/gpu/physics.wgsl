// Compute kernels for the physics phases. One kernel per compute pass:
// the pass boundary is the memory barrier.
// Mirrors ui::gpu::physics::{GpuBody, SimUniforms}.

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
    pad: f32,
    n: u32,
    rounds: u32,
    table_mask: u32,
    pair_cap: u32,
}

@group(0) @binding(0) var<storage, read_write> bodies: array<GpuBody>;
@group(0) @binding(1) var<uniform> sim: Sim;

// Explicit Euler, then the optional speed cap, then the position update.
// Same order as World::integrate: the clamp sees the new velocity, so one
// step can never carry a body past its contact reach.
@compute @workgroup_size(64)
fn integrate(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= sim.n) {
        return;
    }
    var v = vec3<f32>(bodies[i].vel[0], bodies[i].vel[1], bodies[i].vel[2]);
    v.y = v.y + sim.gravity * sim.dt;
    if (sim.max_speed > 0.0) {
        let s2 = v.x * v.x + v.y * v.y + v.z * v.z;
        if (s2 > sim.max_speed * sim.max_speed) {
            v = v * (sim.max_speed / sqrt(s2));
        }
    }
    bodies[i].vel[0] = v.x;
    bodies[i].vel[1] = v.y;
    bodies[i].vel[2] = v.z;
    bodies[i].pos[0] = bodies[i].pos[0] + v.x * sim.dt;
    bodies[i].pos[1] = bodies[i].pos[1] + v.y * sim.dt;
    bodies[i].pos[2] = bodies[i].pos[2] + v.z * sim.dt;
}

// Floor plane at sim.floor_y. Spheres rest on top of it. Port of
// World::collide_floor, friction and restitution included.
@compute @workgroup_size(64)
fn floor_collide(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= sim.n) {
        return;
    }
    let b = bodies[i];
    if (b.pos[1] - b.radius < sim.floor_y && b.vel[1] < 0.0) {
        bodies[i].pos[1] = sim.floor_y + b.radius;
        bodies[i].vel[1] = -b.vel[1] * sim.floor_restitution;
        bodies[i].vel[0] = b.vel[0] * sim.ground_friction;
        bodies[i].vel[2] = b.vel[2] * sim.ground_friction;
    }
}
