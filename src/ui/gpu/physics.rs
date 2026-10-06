//! Compute physics for the integrate and floor phases.
//!
//! Test-only until the app wiring lands: the GPU path stays byte-for-byte
//! inert for the binary, and the parity tests below gate every further
//! phase. The wiring wave removes the `#[cfg(test)]` gate on this module.

use std::mem::size_of;
use std::num::NonZeroU64;

use bytemuck::{Pod, Zeroable};

use crate::engine::{FLOOR_Y, SimSettings};

use super::GpuBody;

/// One body per 32-byte slot, same as [`GpuBody`].
const GPU_BODY_SIZE: u64 = size_of::<GpuBody>() as u64;
/// Threads per compute workgroup.
const WORKGROUP: u32 = 64;

/// Solver and motion constants for the kernels. Exactly 64 bytes: sixteen
/// flat floats, so the Rust and WGSL offsets agree by construction.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(crate) struct SimUniforms {
    /// Step length, in seconds.
    pub(crate) dt: f32,
    /// Downward acceleration.
    pub(crate) gravity: f32,
    /// Broad-phase cell edge. Unused by these two kernels.
    pub(crate) cell_size: f32,
    /// Fraction of vertical speed kept after a floor bounce.
    pub(crate) floor_restitution: f32,
    /// Fraction of horizontal speed kept while a body touches the floor.
    pub(crate) ground_friction: f32,
    /// Fraction of relative speed kept at a contact. Unused here.
    pub(crate) pair_restitution: f32,
    /// Share of tangential speed removed per contact round. Unused here.
    pub(crate) pair_friction: f32,
    /// Contact rest offset. Unused here. Filled by the pair pass.
    pub(crate) slop: f32,
    /// Position correction share. Unused here. Filled by the pair pass.
    pub(crate) correction: f32,
    /// Speed cap per body. Zero disables it, like the CPU path.
    pub(crate) max_speed: f32,
    /// Height of the floor plane.
    pub(crate) floor_y: f32,
    /// Never read.
    pub(crate) pad: f32,
    /// Body count the kernels run over.
    pub(crate) n: u32,
    /// Solve rounds per step. Unused here.
    pub(crate) rounds: u32,
    /// Hash table size mask. Unused here.
    pub(crate) table_mask: u32,
    /// Pair buffer capacity. Unused here.
    pub(crate) pair_cap: u32,
}

// Sixteen floats. Size growth would break the WGSL layout.
const _: () = assert!(size_of::<SimUniforms>() == 64);

impl SimUniforms {
    /// Packs the settings one step needs. The pair-pass fields stay zero
    /// until that phase lands on the GPU.
    pub(crate) fn new(settings: &SimSettings, dt: f32, n: usize) -> Self {
        SimUniforms {
            dt,
            gravity: settings.gravity,
            cell_size: 0.0,
            floor_restitution: settings.floor_restitution,
            ground_friction: settings.ground_friction,
            pair_restitution: settings.pair_restitution,
            pair_friction: settings.pair_friction,
            slop: 0.0,
            correction: 0.0,
            max_speed: settings.max_speed,
            floor_y: FLOOR_Y,
            pad: 0.0,
            n: n as u32,
            rounds: settings.resolve_rounds as u32,
            table_mask: 0,
            pair_cap: 0,
        }
    }
}

/// Compute pipelines and buffers for the GPU physics phases.
pub(crate) struct GpuState {
    pipeline_integrate: wgpu::ComputePipeline,
    pipeline_floor: wgpu::ComputePipeline,
    /// Solver constants, rewritten once per step.
    sim_buf: wgpu::Buffer,
    /// Body storage. Grown only when a step needs more space.
    bodies_buf: wgpu::Buffer,
    /// Body slots the storage buffer holds.
    bodies_capacity: u32,
    bind_group: wgpu::BindGroup,
}

impl GpuState {
    /// Builds the pipelines and seeds the buffers with one body slot.
    pub(crate) fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("physics.wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("physics.wgsl").into()),
        });
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("physics layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: Some(
                            NonZeroU64::new(GPU_BODY_SIZE).expect("body size is not zero"),
                        ),
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: Some(
                            NonZeroU64::new(size_of::<SimUniforms>() as u64)
                                .expect("sim size is not zero"),
                        ),
                    },
                    count: None,
                },
            ],
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("physics pipeline layout"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });
        let make_pipeline = |label: &'static str, entry: &'static str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: Some(&layout),
                module: &shader,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let pipeline_integrate = make_pipeline("integrate", "integrate");
        let pipeline_floor = make_pipeline("floor collide", "floor_collide");

        let sim_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("physics sim uniform"),
            size: size_of::<SimUniforms>() as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::UNIFORM,
            mapped_at_creation: false,
        });
        let bodies_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("physics bodies"),
            size: GPU_BODY_SIZE,
            usage: wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("physics bind group"),
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &bodies_buf,
                        offset: 0,
                        size: None,
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &sim_buf,
                        offset: 0,
                        size: None,
                    }),
                },
            ],
        });
        GpuState {
            pipeline_integrate,
            pipeline_floor,
            sim_buf,
            bodies_buf,
            bodies_capacity: 1,
            bind_group,
        }
    }

    /// Writes the step constants.
    pub(crate) fn write_sim(&self, queue: &wgpu::Queue, sim: &SimUniforms) {
        queue.write_buffer(&self.sim_buf, 0, bytemuck::bytes_of(sim));
    }

    /// Uploads the bodies. Recreates the storage only on growth.
    pub(crate) fn upload_bodies(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        bodies: &[GpuBody],
    ) {
        if bodies.len() as u64 > self.bodies_capacity as u64 {
            self.bodies_buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("physics bodies"),
                size: bodies.len() as u64 * GPU_BODY_SIZE,
                usage: wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            });
            self.bodies_capacity = bodies.len() as u32;
            // The old bind group points at the dropped buffer. Rebuild it.
            self.bind_group = self.make_bind_group(device);
        }
        if !bodies.is_empty() {
            queue.write_buffer(&self.bodies_buf, 0, bytemuck::cast_slice(bodies));
        }
    }

    /// Rebuilds the bind group against the current buffers.
    fn make_bind_group(&self, device: &wgpu::Device) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("physics bind group"),
            layout: &self.pipeline_integrate.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &self.bodies_buf,
                        offset: 0,
                        size: None,
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &self.sim_buf,
                        offset: 0,
                        size: None,
                    }),
                },
            ],
        })
    }

    /// Records the integrate pass and the floor pass. Two passes, so the
    /// pass boundary orders the memory between the kernels.
    pub(crate) fn record_integrate_floor(&self, encoder: &mut wgpu::CommandEncoder, n: usize) {
        if n == 0 {
            return;
        }
        let groups = (n as u32).div_ceil(WORKGROUP);
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("integrate"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline_integrate);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("floor"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline_floor);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.dispatch_workgroups(groups, 1, 1);
        }
    }

    /// Reads the bodies back. One staging buffer per call; the frame loop
    /// never calls this, only the parity tests do.
    // ponytail: per-call staging; a ring of two when the F1 count readback lands
    pub(crate) fn download_bodies(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        n: usize,
    ) -> Vec<GpuBody> {
        if n == 0 {
            return Vec::new();
        }
        let size = n as u64 * GPU_BODY_SIZE;
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("physics bodies staging"),
            size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        encoder.copy_buffer_to_buffer(&self.bodies_buf, 0, &staging, 0, size);
        queue.submit([encoder.finish()]);
        device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("copy poll fails");
        staging.slice(..).map_async(wgpu::MapMode::Read, |_| {});
        device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("map poll fails");
        // The view owns a mapped slice, so it drops before the unmap.
        let out = {
            let view = staging.get_mapped_range(..).expect("map fails");
            bytemuck::cast_slice(&view).to_vec()
        };
        staging.unmap();
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::{BODY_RADIUS, Shape, World};

    /// Builds one offscreen device. Returns `None` without an adapter.
    fn headless_device() -> Option<(wgpu::Device, wgpu::Queue)> {
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

    /// One lattice drifting sideways in weightlessness. Spacing stays at
    /// 1.0 forever, so the CPU step reduces to integrate — exactly what
    /// the GPU runs.
    fn free_flight_world(side: usize) -> World {
        let mut w = World::new();
        w.spawn_grid(side, [0.0, 5.0, 0.0], 1.0, Shape::Sphere, BODY_RADIUS);
        for b in &mut w.bodies {
            b.vel = [0.5, 0.0, 0.0];
        }
        w.settings.gravity = 0.0;
        w
    }

    /// One body falling onto the floor. No pair can ever form, so the CPU
    /// step reduces to integrate plus floor. The bounces keep the floor
    /// branch live for the whole run.
    fn falling_world() -> World {
        let mut w = World::new();
        w.spawn_grid(1, [0.0, 5.0, 0.0], 1.0, Shape::Sphere, BODY_RADIUS);
        w
    }

    /// Steps `frames` frames on the GPU in one submission.
    fn run_gpu(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        state: &GpuState,
        settings: &SimSettings,
        n: usize,
        frames: usize,
    ) {
        state.write_sim(queue, &SimUniforms::new(settings, 1.0 / 60.0, n));
        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        for _ in 0..frames {
            state.record_integrate_floor(&mut encoder, n);
        }
        queue.submit([encoder.finish()]);
        device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("step poll fails");
    }

    /// Biggest per-axis position gap between two body lists.
    fn max_pos_diff(a: &[GpuBody], b: &[GpuBody]) -> f32 {
        a.iter()
            .zip(b)
            .map(|(x, y)| {
                (x.pos[0] - y.pos[0])
                    .abs()
                    .max((x.pos[1] - y.pos[1]).abs())
                    .max((x.pos[2] - y.pos[2]).abs())
            })
            .fold(0.0, f32::max)
    }

    /// Runs one contact-free scene on both paths and checks the gaps at
    /// 1, 10, and 120 steps. Prints the gaps for the release acceptance run.
    fn check_parity(cpu: World, max_speed: f32, label: &str) {
        let Some((device, queue)) = headless_device() else {
            eprintln!("no wgpu adapter; skipping");
            return;
        };
        let mut cpu = cpu;
        cpu.settings.max_speed = max_speed;
        let gpu_bodies: Vec<GpuBody> = cpu.bodies.iter().map(GpuBody::from_body).collect();
        let n = gpu_bodies.len();
        let settings = cpu.settings;

        let mut state = GpuState::new(&device);
        state.upload_bodies(&device, &queue, &gpu_bodies);

        for frames in [1, 9, 110] {
            for _ in 0..frames {
                cpu.step(1.0 / 60.0);
            }
            run_gpu(&device, &queue, &state, &settings, n, frames);
            let back = state.download_bodies(&device, &queue, n);
            let diff = max_pos_diff(&back, &gpu_bodies_of(&cpu));
            println!("{label} cap {max_speed}: +{frames} frames, max gap {diff:.3e}");
            assert!(diff < 1e-4, "GPU and CPU drifted by {diff}");
        }
    }

    /// Rebuilds the GPU mirror of a CPU world.
    fn gpu_bodies_of(w: &World) -> Vec<GpuBody> {
        w.bodies.iter().map(GpuBody::from_body).collect()
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn integrate_matches_cpu_in_free_flight() {
        check_parity(free_flight_world(8), 0.0, "512 free");
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn floor_bounce_matches_cpu() {
        check_parity(falling_world(), 0.0, "1 fall");
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn speed_cap_matches_cpu() {
        // The cap branch is live, so the cap math runs on both paths.
        check_parity(falling_world(), 5.0, "1 fall");
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn gpu_replays_bit_identical() {
        let Some((device, queue)) = headless_device() else {
            eprintln!("no wgpu adapter; skipping");
            return;
        };
        let cpu = free_flight_world(8);
        let mirror = gpu_bodies_of(&cpu);
        let n = mirror.len();
        let settings = cpu.settings;

        let mut first = GpuState::new(&device);
        let mut second = GpuState::new(&device);
        first.upload_bodies(&device, &queue, &mirror);
        second.upload_bodies(&device, &queue, &mirror);
        run_gpu(&device, &queue, &first, &settings, n, 120);
        run_gpu(&device, &queue, &second, &settings, n, 120);
        let a = first.download_bodies(&device, &queue, n);
        let b = second.download_bodies(&device, &queue, n);
        assert_eq!(a, b, "two GPU runs diverged");
    }
}
