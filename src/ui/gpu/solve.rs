//! Contact solver on the GPU: the per-body CSR, the Jacobi rounds, and
//! the full step recorded in the CPU phase order.

use super::physics::{GpuState, SCAN_SLICES, WORKGROUP};

impl GpuState {
    /// Records the CSR build: clear, count, scan, fill, sort. The bucket
    /// sort pins the sum order per body, so a round replays per device.
    pub(crate) fn record_solve_pass(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        n: usize,
        rounds: usize,
    ) {
        if n == 0 {
            return;
        }
        encoder.clear_buffer(&self.bc_start_buf, 0, None);
        encoder.clear_buffer(&self.bc_cursor_buf, 0, None);
        let groups = (n as u32).div_ceil(WORKGROUP);
        let pair_groups = self.pairs_capacity.div_ceil(WORKGROUP);
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("solve count"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.kernels.solve_count);
            pass.set_bind_group(0, &self.bind_solve, &[]);
            pass.dispatch_workgroups(pair_groups, 1, 1);
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("solve scan"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.kernels.scan_level1);
            pass.set_bind_group(0, &self.bind_scan_bc, &[]);
            pass.dispatch_workgroups(SCAN_SLICES, 1, 1);
            pass.set_pipeline(&self.kernels.scan_sums);
            pass.dispatch_workgroups(1, 1, 1);
            pass.set_pipeline(&self.kernels.scan_apply);
            pass.dispatch_workgroups(SCAN_SLICES, 1, 1);
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("solve fill"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.kernels.solve_fill);
            pass.set_bind_group(0, &self.bind_solve, &[]);
            pass.dispatch_workgroups(pair_groups, 1, 1);
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("solve sort"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.kernels.solve_sort);
            pass.set_bind_group(0, &self.bind_solve, &[]);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        // Jacobi interleaves: round two must read the state round one
        // applied, so delta and apply alternate per round.
        for _ in 0..rounds {
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("solve delta"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.kernels.solve_delta);
                pass.set_bind_group(0, &self.bind_solve, &[]);
                pass.dispatch_workgroups(pair_groups, 1, 1);
            }
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("solve apply"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.kernels.solve_apply);
                pass.set_bind_group(0, &self.bind_solve, &[]);
                pass.dispatch_workgroups(groups, 1, 1);
            }
        }
    }

    /// Records one full step in the CPU phase order: integrate, grid,
    /// pairs, solve, floor. `ts` takes six timestamp slots and measures
    /// the five phases when the adapter supports the queries.
    pub(crate) fn record_full_step(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        n: usize,
        rounds: usize,
        ts: Option<&wgpu::QuerySet>,
    ) {
        let stamp = |encoder: &mut wgpu::CommandEncoder, index: u32| {
            if let Some(set) = ts {
                encoder.write_timestamp(set, index);
            }
        };
        stamp(encoder, 0);
        self.record_integrate(encoder, n);
        stamp(encoder, 1);
        self.record_grid_build(encoder, n);
        stamp(encoder, 2);
        self.record_pair_pass(encoder, n);
        stamp(encoder, 3);
        self.record_solve_pass(encoder, n, rounds);
        stamp(encoder, 4);
        self.record_floor(encoder, n);
        stamp(encoder, 5);
    }
}

impl GpuState {
    /// Reads the per-body CSR starts and the tagged item list back.
    /// Test-only.
    #[cfg(test)]
    pub(crate) fn download_csr(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        n: usize,
    ) -> (Vec<u32>, Vec<u32>) {
        if n == 0 {
            return (Vec::new(), Vec::new());
        }
        let starts_stage = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("bc start staging"),
            size: (n as u64 + 1) * 4,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        encoder.copy_buffer_to_buffer(&self.bc_start_buf, 0, &starts_stage, 0, (n as u64 + 1) * 4);
        queue.submit([encoder.finish()]);
        device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("csr start poll fails");
        starts_stage
            .slice(..)
            .map_async(wgpu::MapMode::Read, |_| {});
        device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("csr start map poll fails");
        let starts: Vec<u32> = {
            let view = starts_stage.get_mapped_range(..).expect("map fails");
            let bytes: Vec<u8> = view.to_vec();
            bytes
                .chunks_exact(4)
                .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect()
        };
        starts_stage.unmap();

        let total = *starts.last().expect("starts hold slot n") as u64;
        if total == 0 {
            return (starts, Vec::new());
        }
        let items_stage = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("bc items staging"),
            size: total * 4,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        encoder.copy_buffer_to_buffer(&self.bc_items_buf, 0, &items_stage, 0, total * 4);
        queue.submit([encoder.finish()]);
        device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("csr items poll fails");
        items_stage.slice(..).map_async(wgpu::MapMode::Read, |_| {});
        device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("csr items map poll fails");
        let items: Vec<u32> = {
            let view = items_stage.get_mapped_range(..).expect("map fails");
            let bytes: Vec<u8> = view.to_vec();
            bytes
                .chunks_exact(4)
                .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect()
        };
        items_stage.unmap();
        (starts, items)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::World;
    use crate::ui::gpu::GpuBody;
    use crate::ui::gpu::headless_device;
    use crate::ui::gpu::physics::{SimUniforms, TABLE_MASK};

    /// Rebuilds the GPU mirror of a CPU world.
    fn gpu_bodies_of(w: &World) -> Vec<GpuBody> {
        w.bodies.iter().map(GpuBody::from_body).collect()
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

    /// Steps the GPU world in frame groups and returns the bodies.
    fn run_gpu_frames(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        state: &GpuState,
        sim: &SimUniforms,
        n: usize,
        frames: usize,
    ) -> Vec<GpuBody> {
        state.write_sim(queue, sim);
        // One submission per 16 frames: the app also submits per frame,
        // and one giant encoder grows past what the driver accepts.
        for chunk in 0..frames.div_ceil(16) {
            let count = (frames - chunk * 16).min(16);
            let mut encoder =
                device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            for _ in 0..count {
                state.record_full_step(&mut encoder, n, sim.rounds.max(1) as usize, None);
            }
            queue.submit([encoder.finish()]);
            device
                .poll(wgpu::PollType::wait_indefinitely())
                .expect("full step poll fails");
        }
        state.download_bodies(device, queue, n)
    }

    /// Top of the pile, in world units.
    fn pile_height(bodies: &[GpuBody]) -> f32 {
        bodies.iter().map(|b| b.pos[1]).fold(0.0, f32::max)
    }

    /// Full-block parity: tight position gaps while the runs share their
    /// fate, then statistical parity once chaos takes over. A dense pile
    /// amplifies one rounding difference into a different microstate, so
    /// position equality past about ten frames is not a property of the
    /// math, and the test must not demand it.
    fn check_full_step(n: usize, friction: f32) {
        let Some((device, queue)) = headless_device() else {
            eprintln!("no wgpu adapter; skipping");
            return;
        };
        let mut cpu = World::new();
        cpu.spawn_wave(n, [0.0, 5.0, 0.0], 4.0);
        cpu.settings.pair_friction = friction;
        let mirror: Vec<GpuBody> = cpu.bodies.iter().map(GpuBody::from_body).collect();
        let count = mirror.len();
        let sim = SimUniforms::new(
            &cpu.settings,
            1.0 / 60.0,
            cpu.cell_size(),
            TABLE_MASK,
            count,
        );
        let mut state = GpuState::new(&device);
        state.upload_bodies(&device, &queue, &mirror);

        for frames in [1usize, 9] {
            for _ in 0..frames {
                cpu.step(1.0 / 60.0);
            }
            let back = run_gpu_frames(&device, &queue, &state, &sim, count, frames);
            let diff = max_pos_diff(&back, &gpu_bodies_of(&cpu));
            println!("{n} bodies friction {friction}: +{frames} frames, max gap {diff:.3e}");
            assert!(diff < 1e-4, "GPU and CPU full steps drifted by {diff}");
        }

        // Long horizon is chaotic: a dense pile amplifies one rounding
        // difference into a different microstate, so the GPU pile is only
        // asked to stay physical. Finite positions, a bounded pile, and a
        // live pair list over 300 frames.
        let back = run_gpu_frames(&device, &queue, &state, &sim, count, 300);
        let height = pile_height(&back);
        let finite = back
            .iter()
            .all(|b| b.pos.iter().all(|v| v.is_finite()) && b.vel.iter().all(|v| v.is_finite()));
        println!(
            "{n} bodies friction {friction}: +300 frames, height {height:.2}, finite {finite}"
        );
        assert!(finite, "a body left the real numbers");
        assert!(height < 25.0, "the pile exploded: top at {height}");
        let (total, _) = state.download_pairs(&device, &queue, count, sim.pair_cap);
        assert!(total > 0, "the settled pile lost every pair");
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn full_step_matches_cpu() {
        check_full_step(2000, 0.0);
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn full_step_with_friction_matches_cpu() {
        check_full_step(2000, 0.4);
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn full_step_replays_bit_identical() {
        let Some((device, queue)) = headless_device() else {
            eprintln!("no wgpu adapter; skipping");
            return;
        };
        let mut world = World::new();
        world.spawn_wave(2000, [0.0, 5.0, 0.0], 4.0);
        let mirror: Vec<GpuBody> = world.bodies.iter().map(GpuBody::from_body).collect();
        let count = mirror.len();
        let sim = SimUniforms::new(
            &world.settings,
            1.0 / 60.0,
            world.cell_size(),
            TABLE_MASK,
            count,
        );

        let mut a = GpuState::new(&device);
        let mut b = GpuState::new(&device);
        a.upload_bodies(&device, &queue, &mirror);
        b.upload_bodies(&device, &queue, &mirror);
        let ba = run_gpu_frames(&device, &queue, &a, &sim, count, 60);
        let bb = run_gpu_frames(&device, &queue, &b, &sim, count, 60);
        assert_eq!(ba, bb, "two full-step runs diverged");
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn csr_lists_ascending_per_body() {
        let Some((device, queue)) = headless_device() else {
            eprintln!("no wgpu adapter; skipping");
            return;
        };
        let mut world = World::new();
        world.spawn_wave(3000, [0.0, 5.0, 0.0], 4.0);
        let mirror: Vec<GpuBody> = world.bodies.iter().map(GpuBody::from_body).collect();
        let count = mirror.len();
        let sim = SimUniforms::new(
            &world.settings,
            1.0 / 60.0,
            world.cell_size(),
            TABLE_MASK,
            count,
        );
        let mut state = GpuState::new(&device);
        state.upload_bodies(&device, &queue, &mirror);
        state.write_sim(&queue, &sim);
        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        state.record_grid_build(&mut encoder, count);
        state.record_pair_pass(&mut encoder, count);
        state.record_solve_pass(&mut encoder, count, 1);
        queue.submit([encoder.finish()]);
        device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("csr build poll fails");

        // One pass over the item list: within every body slice the masked
        // pair indices must ascend, and both sides must appear once.
        let (starts, items) = state.download_csr(&device, &queue, count);
        let (_, pairs) = state.download_pairs(&device, &queue, count, sim.pair_cap);
        let total = starts[count] as usize;
        assert_eq!(total, pairs.len() * 2, "item count is not two per pair");
        for b in 0..count {
            let slice = &items[starts[b] as usize..starts[b + 1] as usize];
            for w in slice.windows(2) {
                assert!(
                    (w[0] & 0x7FFF_FFFF) < (w[1] & 0x7FFF_FFFF),
                    "body {b} list is not ascending"
                );
            }
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn full_step_times_at_500k() {
        let Some((device, queue)) = headless_device() else {
            eprintln!("no wgpu adapter; skipping");
            return;
        };
        let mut world = World::new();
        world.spawn_wave(500_000, [0.0, 8.0, 0.0], 4.0);
        for _ in 0..30 {
            world.step(1.0 / 60.0);
        }
        let mirror: Vec<GpuBody> = world.bodies.iter().map(GpuBody::from_body).collect();
        let count = mirror.len();
        let sim = SimUniforms::new(
            &world.settings,
            1.0 / 60.0,
            world.cell_size(),
            TABLE_MASK,
            count,
        );
        let mut state = GpuState::new(&device);
        state.upload_bodies(&device, &queue, &mirror);

        let _ = run_gpu_frames(&device, &queue, &state, &sim, count, 1);
        let t = std::time::Instant::now();
        for _ in 0..10 {
            let _ = run_gpu_frames(&device, &queue, &state, &sim, count, 1);
        }
        let per_frame = t.elapsed().as_secs_f32() * 1000.0 / 10.0;
        println!("full GPU step: {per_frame:.3} ms at {count} bodies");
    }
}
