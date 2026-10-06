//! Broad-phase grid build on the GPU: histogram, scan, scatter, bucket sort.

use super::physics::GpuState;
use super::physics::{SCAN_SLICES, TABLE_SIZE, WORKGROUP};

/// Threads per bucket-sort workgroup. 256 keeps the table-wide dispatch
/// under the 65535 group limit.
const SORT_WORKGROUP: u32 = 256;

/// The eight grid and pair kernels, built once per device.
pub(crate) struct GridKernels {
    pub(super) hist: wgpu::ComputePipeline,
    pub(super) scatter: wgpu::ComputePipeline,
    pub(super) sort: wgpu::ComputePipeline,
    pub(super) scan_level1: wgpu::ComputePipeline,
    pub(super) scan_sums: wgpu::ComputePipeline,
    pub(super) scan_apply: wgpu::ComputePipeline,
    pub(super) pair_count: wgpu::ComputePipeline,
    pub(super) pair_fill: wgpu::ComputePipeline,
    pub(super) solve_count: wgpu::ComputePipeline,
    pub(super) solve_fill: wgpu::ComputePipeline,
    pub(super) solve_sort: wgpu::ComputePipeline,
    pub(super) solve_delta: wgpu::ComputePipeline,
    pub(super) solve_apply: wgpu::ComputePipeline,
}

impl GridKernels {
    /// Builds the pipelines. `bgl_grid` carries the seven grid and pair
    /// bindings, `bgl_scan` the data and sums pair.
    pub(crate) fn new(
        device: &wgpu::Device,
        bgl_grid: &wgpu::BindGroupLayout,
        bgl_scan: &wgpu::BindGroupLayout,
        bgl_solve: &wgpu::BindGroupLayout,
    ) -> Self {
        let grid_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("grid.wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("grid.wgsl").into()),
        });
        let scan_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("scan.wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("scan.wgsl").into()),
        });
        let pairs_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("pairs.wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("pairs.wgsl").into()),
        });
        let solve_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("solve.wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("solve.wgsl").into()),
        });
        let layout = |label: &'static str, bgl: &wgpu::BindGroupLayout| {
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(label),
                bind_group_layouts: &[Some(bgl)],
                immediate_size: 0,
            })
        };
        let grid_layout = layout("grid pipeline layout", bgl_grid);
        let scan_layout = layout("scan pipeline layout", bgl_scan);
        let solve_layout = layout("solve pipeline layout", bgl_solve);
        let kernel = |label: &'static str,
                      entry: &'static str,
                      module: &wgpu::ShaderModule,
                      shader_layout: &wgpu::PipelineLayout| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: Some(shader_layout),
                module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        GridKernels {
            hist: kernel("grid hist", "grid_hist", &grid_shader, &grid_layout),
            scatter: kernel("grid scatter", "grid_scatter", &grid_shader, &grid_layout),
            sort: kernel(
                "grid bucket sort",
                "grid_bucket_sort",
                &grid_shader,
                &grid_layout,
            ),
            scan_level1: kernel("scan level 1", "scan_level1", &scan_shader, &scan_layout),
            scan_sums: kernel("scan sums", "scan_sums", &scan_shader, &scan_layout),
            scan_apply: kernel("scan apply", "scan_apply", &scan_shader, &scan_layout),
            pair_count: kernel("pair count", "pair_count", &pairs_shader, &grid_layout),
            pair_fill: kernel("pair fill", "pair_fill", &pairs_shader, &grid_layout),
            solve_count: kernel("solve count", "solve_count", &solve_shader, &solve_layout),
            solve_fill: kernel("solve fill", "solve_fill", &solve_shader, &solve_layout),
            solve_sort: kernel("solve sort", "solve_sort", &solve_shader, &solve_layout),
            solve_delta: kernel("solve delta", "solve_delta", &solve_shader, &solve_layout),
            solve_apply: kernel("solve apply", "solve_apply", &solve_shader, &solve_layout),
        }
    }
}

impl GpuState {
    /// Records the full grid build: clear, histogram, scan, scatter, sort.
    /// After this, `table` holds exclusive bucket starts and `body_ids`
    /// lists each bucket's bodies in ascending index order.
    pub(crate) fn record_grid_build(&self, encoder: &mut wgpu::CommandEncoder, n: usize) {
        if n == 0 {
            return;
        }
        encoder.clear_buffer(&self.table_buf, 0, None);
        encoder.clear_buffer(&self.cursor_buf, 0, None);
        let groups = (n as u32).div_ceil(WORKGROUP);
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("grid hist"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.kernels.hist);
            pass.set_bind_group(0, &self.bind_grid, &[]);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("grid scan"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.kernels.scan_level1);
            pass.set_bind_group(0, &self.bind_scan, &[]);
            pass.dispatch_workgroups(SCAN_SLICES, 1, 1);
            pass.set_pipeline(&self.kernels.scan_sums);
            pass.dispatch_workgroups(1, 1, 1);
            pass.set_pipeline(&self.kernels.scan_apply);
            pass.dispatch_workgroups(SCAN_SLICES, 1, 1);
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("grid scatter"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.kernels.scatter);
            pass.set_bind_group(0, &self.bind_grid, &[]);
            pass.dispatch_workgroups(groups, 1, 1);
            pass.set_pipeline(&self.kernels.sort);
            pass.dispatch_workgroups(TABLE_SIZE / SORT_WORKGROUP, 1, 1);
        }
    }

    /// Reads the table and the body ids back. Test-only.
    #[cfg(test)]
    pub(crate) fn download_grid(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        n: usize,
    ) -> (Vec<u32>, Vec<u32>) {
        let table_size = TABLE_SIZE as u64 * 4;
        let ids_size = n as u64 * 4;
        let table_stage = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("grid table staging"),
            size: table_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let ids_stage = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("grid ids staging"),
            size: ids_size.max(4),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        encoder.copy_buffer_to_buffer(&self.table_buf, 0, &table_stage, 0, table_size);
        if n > 0 {
            encoder.copy_buffer_to_buffer(&self.body_ids_buf, 0, &ids_stage, 0, ids_size);
        }
        queue.submit([encoder.finish()]);
        device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("grid copy poll fails");
        let read = |stage: &wgpu::Buffer| {
            stage.slice(..).map_async(wgpu::MapMode::Read, |_| {});
            device
                .poll(wgpu::PollType::wait_indefinitely())
                .expect("grid map poll fails");
            // The view owns a mapped slice, so it drops before the unmap.
            let out = {
                let view = stage.get_mapped_range(..).expect("map fails");
                bytemuck::cast_slice(&view).to_vec()
            };
            stage.unmap();
            out
        };
        let table = read(&table_stage);
        let ids = if n > 0 { read(&ids_stage) } else { Vec::new() };
        (table, ids)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::{SimSettings, World};
    use crate::ui::gpu::GpuBody;
    use crate::ui::gpu::headless_device;
    use crate::ui::gpu::physics::{SimUniforms, TABLE_MASK};

    /// Steps the grid build once on the GPU and returns table and ids.
    fn build_grid(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        state: &GpuState,
        sim: &SimUniforms,
        n: usize,
    ) -> (Vec<u32>, Vec<u32>) {
        state.write_sim(queue, sim);
        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        state.record_grid_build(&mut encoder, n);
        queue.submit([encoder.finish()]);
        device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("grid build poll fails");
        state.download_grid(device, queue, n)
    }

    /// Clamped cell coordinates. Mirrors `cell_coords` in grid.wgsl and
    /// the CPU `cell_of` plus `key_part`.
    fn cell_of(p: [f32; 3], cell_size: f32) -> [i32; 3] {
        p.map(|v| ((v / cell_size).floor() as i32 + (1 << 20)).clamp(0, (1 << 21) - 1))
    }

    /// Murmur-style bucket fold. Mirrors `bucket_of` in grid.wgsl.
    fn bucket_of(c: [i32; 3], mask: u32) -> u32 {
        let mut h = (c[0] as u32).wrapping_mul(0x9E37_79B1);
        h ^= (c[1] as u32).wrapping_mul(0x85EB_CA77);
        h ^= (c[2] as u32).wrapping_mul(0xC2B2_AE3D);
        h ^= h >> 16;
        h = h.wrapping_mul(0x7FEB_352D);
        h ^= h >> 15;
        h = h.wrapping_mul(0x846C_A68B);
        h ^= h >> 16;
        h & mask
    }

    /// The expected bucket lists, computed on the CPU: body indices per
    /// bucket, ascending.
    fn expected_buckets(bodies: &[GpuBody], cell_size: f32, mask: u32) -> Vec<Vec<u32>> {
        let mut buckets = vec![Vec::new(); mask as usize + 1];
        for (i, b) in bodies.iter().enumerate() {
            let c = cell_of(b.pos, cell_size);
            buckets[bucket_of(c, mask) as usize].push(i as u32);
        }
        for list in &mut buckets {
            list.sort_unstable();
        }
        buckets
    }

    /// One GPU grid build vs the CPU mirror. `mask` shrinks the table to
    /// force cell collisions.
    fn check_buckets(n: usize, mask: u32) {
        let Some((device, queue)) = headless_device() else {
            eprintln!("no wgpu adapter; skipping");
            return;
        };
        let mut world = World::new();
        world.spawn_wave(n, [0.0, 5.0, 0.0], 4.0);
        let cell_size = world.cell_size();
        let bodies: Vec<crate::ui::gpu::GpuBody> =
            world.bodies.iter().map(GpuBody::from_body).collect();

        let mut state = GpuState::new(&device);
        state.upload_bodies(&device, &queue, &bodies);
        let sim = SimUniforms::new(&SimSettings::default(), 1.0 / 60.0, cell_size, mask, n);
        let (table, ids) = build_grid(&device, &queue, &state, &sim, n);

        // Every bucket's slice must hold exactly the expected bodies.
        let want = expected_buckets(&bodies, cell_size, mask);
        let mut seen = 0usize;
        for b in 0..=mask as usize {
            let start = table[b] as usize;
            let end = if b == mask as usize {
                n
            } else {
                table[b + 1] as usize
            };
            assert!(start <= end, "bucket {b} runs backward");
            let got = &ids[start..end];
            assert_eq!(got, want[b], "bucket {b} holds the wrong bodies");
            seen += got.len();
        }
        assert_eq!(seen, n, "the buckets lost bodies");
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn grid_buckets_match_cpu() {
        check_buckets(4000, TABLE_MASK);
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn grid_hash_collisions_stay_correct() {
        // 256 buckets for ~4000 cells: many cells share one bucket, and
        // the union must still list every body exactly once.
        check_buckets(4000, 255);
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn grid_replays_bit_identical() {
        let Some((device, queue)) = headless_device() else {
            eprintln!("no wgpu adapter; skipping");
            return;
        };
        let mut world = World::new();
        world.spawn_wave(2000, [0.0, 5.0, 0.0], 4.0);
        let bodies: Vec<crate::ui::gpu::GpuBody> =
            world.bodies.iter().map(GpuBody::from_body).collect();
        let n = bodies.len();
        let sim = SimUniforms::new(
            &SimSettings::default(),
            1.0 / 60.0,
            world.cell_size(),
            TABLE_MASK,
            n,
        );

        let mut a = GpuState::new(&device);
        let mut b = GpuState::new(&device);
        a.upload_bodies(&device, &queue, &bodies);
        b.upload_bodies(&device, &queue, &bodies);
        let (ta, ia) = build_grid(&device, &queue, &a, &sim, n);
        let (tb, ib) = build_grid(&device, &queue, &b, &sim, n);
        assert_eq!(ta, tb, "two grid runs diverged on the table");
        assert_eq!(ia, ib, "two grid runs diverged on the ids");
    }

    /// Records one grid section for the timing test. The clears always run:
    /// scatter without them reuses a stale cursor and writes out of bounds.
    /// Bits: 1 histogram, 2 scan, 4 scatter plus sort.
    fn record_parts(state: &GpuState, encoder: &mut wgpu::CommandEncoder, n: usize, parts: u32) {
        if n == 0 {
            return;
        }
        encoder.clear_buffer(&state.table_buf, 0, None);
        encoder.clear_buffer(&state.cursor_buf, 0, None);
        let groups = (n as u32).div_ceil(WORKGROUP);
        if parts & 1 != 0 {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("grid hist"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&state.kernels.hist);
            pass.set_bind_group(0, &state.bind_grid, &[]);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        if parts & 2 != 0 {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("grid scan"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&state.kernels.scan_level1);
            pass.set_bind_group(0, &state.bind_scan, &[]);
            pass.dispatch_workgroups(SCAN_SLICES, 1, 1);
            pass.set_pipeline(&state.kernels.scan_sums);
            pass.dispatch_workgroups(1, 1, 1);
            pass.set_pipeline(&state.kernels.scan_apply);
            pass.dispatch_workgroups(SCAN_SLICES, 1, 1);
        }
        if parts & 4 != 0 {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("grid scatter"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&state.kernels.scatter);
            pass.set_bind_group(0, &state.bind_grid, &[]);
            pass.dispatch_workgroups(groups, 1, 1);
            pass.set_pipeline(&state.kernels.sort);
            pass.dispatch_workgroups(TABLE_SIZE / SORT_WORKGROUP, 1, 1);
        }
    }

    /// Mean wall time of one section, over `frames` submissions.
    fn time_parts(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        state: &GpuState,
        sim: &SimUniforms,
        n: usize,
        parts: u32,
        frames: usize,
    ) -> f32 {
        state.write_sim(queue, sim);
        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        record_parts(state, &mut encoder, n, parts);
        queue.submit([encoder.finish()]);
        device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("warm poll fails");
        let t = std::time::Instant::now();
        for _ in 0..frames {
            let mut encoder =
                device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            record_parts(state, &mut encoder, n, parts);
            queue.submit([encoder.finish()]);
            device
                .poll(wgpu::PollType::wait_indefinitely())
                .expect("time poll fails");
        }
        t.elapsed().as_secs_f32() * 1000.0 / frames as f32
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn grid_build_times_at_500k() {
        let Some((device, queue)) = headless_device() else {
            eprintln!("no wgpu adapter; skipping");
            return;
        };
        let mut world = World::new();
        world.spawn_wave(500_000, [0.0, 8.0, 0.0], 4.0);
        let bodies: Vec<crate::ui::gpu::GpuBody> =
            world.bodies.iter().map(GpuBody::from_body).collect();
        let n = bodies.len();
        let sim = SimUniforms::new(
            &SimSettings::default(),
            1.0 / 60.0,
            world.cell_size(),
            TABLE_MASK,
            n,
        );
        let mut state = GpuState::new(&device);
        state.upload_bodies(&device, &queue, &bodies);

        let hist = time_parts(&device, &queue, &state, &sim, n, 1, 30);
        let scan = time_parts(&device, &queue, &state, &sim, n, 2, 30);
        let full = time_parts(&device, &queue, &state, &sim, n, 7, 30);
        // Scatter and sort cannot run alone: without the histogram counts
        // the sort degenerates into one thread sorting the whole array.
        let scatter = full - hist - scan;
        println!("grid sections at {n} bodies, ms per frame:");
        println!("  clears + hist    {hist:.3}");
        println!("  scan             {scan:.3}");
        println!("  scatter + sort   {scatter:.3} (derived)");
        println!("  total            {full:.3}");
    }
}
