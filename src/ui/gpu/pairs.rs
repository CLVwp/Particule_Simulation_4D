//! Contact pair pass on the GPU: count, scan, fill over the 14-offset
//! stencil.

use std::mem::size_of;

use bytemuck::{Pod, Zeroable};

use super::GpuBody;
use super::physics::{GpuState, SimUniforms, WORKGROUP};

/// One contact pair. Mirrors the CPU `Contact`: two indices and two
/// masses. Exactly 16 bytes.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
pub(crate) struct Pair {
    /// Low body index.
    pub(crate) i: u32,
    /// High body index, or the stencil-side body for cross-cell pairs.
    pub(crate) j: u32,
    /// Mass of body `i`, the radius cubed.
    pub(crate) mi: f32,
    /// Mass of body `j`.
    pub(crate) mj: f32,
}

// Two indices and two masses. Size growth would break the WGSL layout.
const _: () = assert!(size_of::<Pair>() == 16);

impl GpuState {
    /// Records the pair pass: clear the CSR, count, scan, fill. The grid
    /// build must have run this frame, so the buckets hold sorted bodies.
    pub(crate) fn record_pair_pass(&self, encoder: &mut wgpu::CommandEncoder, n: usize) {
        if n == 0 {
            return;
        }
        encoder.clear_buffer(&self.pair_start_buf, 0, None);
        let groups = (n as u32).div_ceil(WORKGROUP);
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("pair count"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.kernels.pair_count);
            pass.set_bind_group(0, &self.bind_grid, &[]);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("pair scan"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.kernels.scan_level1);
            pass.set_bind_group(0, &self.bind_scan_pairs, &[]);
            pass.dispatch_workgroups(super::physics::SCAN_SLICES, 1, 1);
            pass.set_pipeline(&self.kernels.scan_sums);
            pass.dispatch_workgroups(1, 1, 1);
            pass.set_pipeline(&self.kernels.scan_apply);
            pass.dispatch_workgroups(super::physics::SCAN_SLICES, 1, 1);
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("pair fill"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.kernels.pair_fill);
            pass.set_bind_group(0, &self.bind_grid, &[]);
            pass.dispatch_workgroups(groups, 1, 1);
        }
    }

    /// Reads the pair total and the kept pairs back. Test-only.
    pub(crate) fn download_pairs(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        n: usize,
        cap: u32,
    ) -> (u32, Vec<Pair>) {
        if n == 0 {
            return (0, Vec::new());
        }
        let total_stage = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("pair total staging"),
            size: 4,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        // After the exclusive scan, slot n holds the full pair count.
        encoder.copy_buffer_to_buffer(&self.pair_start_buf, n as u64 * 4, &total_stage, 0, 4);
        queue.submit([encoder.finish()]);
        device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("total copy poll fails");
        total_stage.slice(..).map_async(wgpu::MapMode::Read, |_| {});
        device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("total map poll fails");
        let total = {
            let view = total_stage.get_mapped_range(..).expect("total map fails");
            let bytes: Vec<u8> = view.to_vec();
            u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
        };
        total_stage.unmap();

        let kept = (total as u64)
            .min(cap as u64)
            .min(self.pairs_capacity as u64);
        if kept == 0 {
            return (total, Vec::new());
        }
        let pairs_stage = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("pairs staging"),
            size: kept * 16,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        encoder.copy_buffer_to_buffer(&self.pairs_buf, 0, &pairs_stage, 0, kept * 16);
        queue.submit([encoder.finish()]);
        device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("pairs copy poll fails");
        pairs_stage.slice(..).map_async(wgpu::MapMode::Read, |_| {});
        device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("pairs map poll fails");
        // The view owns a mapped slice, so it drops before the unmap.
        let out = {
            let view = pairs_stage.get_mapped_range(..).expect("pairs map fails");
            bytemuck::cast_slice(&view).to_vec()
        };
        pairs_stage.unmap();
        (total, out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::{SimSettings, World};
    use crate::ui::gpu::headless_device;
    use crate::ui::gpu::physics::TABLE_MASK;

    /// Builds the grid and the pairs on the GPU for one frame and returns
    /// the total and the kept pairs.
    fn run_pairs(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        state: &GpuState,
        sim: &SimUniforms,
        n: usize,
    ) -> (u32, Vec<Pair>) {
        state.write_sim(queue, sim);
        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        state.record_grid_build(&mut encoder, n);
        state.record_pair_pass(&mut encoder, n);
        queue.submit([encoder.finish()]);
        device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("pair pass poll fails");
        state.download_pairs(device, queue, n, sim.pair_cap)
    }

    /// The independent oracle: every overlapping pair at i less than j,
    /// found by brute force over all index pairs. Shares no logic with
    /// either implementation.
    fn brute_force(bodies: &[GpuBody]) -> Vec<(u32, u32)> {
        let mut want = Vec::new();
        for i in 0..bodies.len() as u32 {
            for j in (i + 1)..bodies.len() as u32 {
                let (a, b) = (&bodies[i as usize], &bodies[j as usize]);
                let d = [
                    b.pos[0] - a.pos[0],
                    b.pos[1] - a.pos[1],
                    b.pos[2] - a.pos[2],
                ];
                let dist2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
                let min_d = a.radius + b.radius;
                if dist2 < min_d * min_d && dist2 >= 1e-12 {
                    want.push((i, j));
                }
            }
        }
        want.sort_unstable();
        want
    }

    /// GPU pair set vs the brute-force oracle on the same positions.
    fn check_pairs(n: usize) {
        let Some((device, queue)) = headless_device() else {
            eprintln!("no wgpu adapter; skipping");
            return;
        };
        let mut world = World::new();
        world.spawn_wave(n, [0.0, 5.0, 0.0], 4.0);
        let bodies: Vec<GpuBody> = world.bodies.iter().map(GpuBody::from_body).collect();
        let count = bodies.len();
        let sim = SimUniforms::new(
            &SimSettings::default(),
            1.0 / 60.0,
            world.cell_size(),
            TABLE_MASK,
            count,
        );

        let mut state = GpuState::new(&device);
        state.upload_bodies(&device, &queue, &bodies);
        let (total, pairs) = run_pairs(&device, &queue, &state, &sim, count);

        // Cross-cell pairs keep the stencil side as `i`, which may sit
        // above `j`. Normalize for the set compare.
        let mut got: Vec<(u32, u32)> = pairs.iter().map(|p| (p.i.min(p.j), p.i.max(p.j))).collect();
        got.sort_unstable();
        got.dedup();
        let want = brute_force(&bodies);
        assert_eq!(total as usize, want.len(), "pair total differs");
        assert_eq!(got, want, "pair set differs");
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn pairs_match_the_oracle() {
        check_pairs(3000);
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn pairs_replay_bit_identical() {
        let Some((device, queue)) = headless_device() else {
            eprintln!("no wgpu adapter; skipping");
            return;
        };
        let mut world = World::new();
        world.spawn_wave(3000, [0.0, 5.0, 0.0], 4.0);
        let bodies: Vec<GpuBody> = world.bodies.iter().map(GpuBody::from_body).collect();
        let count = bodies.len();
        let sim = SimUniforms::new(
            &SimSettings::default(),
            1.0 / 60.0,
            world.cell_size(),
            TABLE_MASK,
            count,
        );

        let mut a = GpuState::new(&device);
        let mut b = GpuState::new(&device);
        a.upload_bodies(&device, &queue, &bodies);
        b.upload_bodies(&device, &queue, &bodies);
        let (ta, pa) = run_pairs(&device, &queue, &a, &sim, count);
        let (tb, pb) = run_pairs(&device, &queue, &b, &sim, count);
        assert_eq!(ta, tb, "two pair runs diverged on the total");
        assert_eq!(pa, pb, "two pair runs diverged on the pair order");
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn pair_overflow_clips_deterministically() {
        let Some((device, queue)) = headless_device() else {
            eprintln!("no wgpu adapter; skipping");
            return;
        };
        let mut world = World::new();
        // One dense blob guarantees more pairs than the tiny budget.
        world.spawn(
            2000,
            [0.0, 5.0, 0.0],
            0.0,
            crate::engine::Shape::Sphere,
            0.1,
        );
        let bodies: Vec<GpuBody> = world.bodies.iter().map(GpuBody::from_body).collect();
        let count = bodies.len();
        let mut sim = SimUniforms::new(
            &SimSettings::default(),
            1.0 / 60.0,
            world.cell_size(),
            TABLE_MASK,
            count,
        );
        sim.pair_cap = 64;

        let mut state = GpuState::new(&device);
        state.upload_bodies(&device, &queue, &bodies);
        let (total_a, pairs_a) = run_pairs(&device, &queue, &state, &sim, count);
        let (total_b, pairs_b) = run_pairs(&device, &queue, &state, &sim, count);
        assert!(total_a > 64, "the blob made too few pairs: {total_a}");
        assert_eq!(pairs_a.len(), 64, "the budget did not clip");
        assert_eq!((total_a, pairs_a), (total_b, pairs_b), "clipping drifted");
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn pair_pass_times_at_500k() {
        let Some((device, queue)) = headless_device() else {
            eprintln!("no wgpu adapter; skipping");
            return;
        };
        let mut world = World::new();
        world.spawn_wave(500_000, [0.0, 8.0, 0.0], 4.0);
        // Settle the pile, so the pair count matches the dense regime the
        // CPU phase table reports.
        for _ in 0..30 {
            world.step(1.0 / 60.0);
        }
        let bodies: Vec<GpuBody> = world.bodies.iter().map(GpuBody::from_body).collect();
        let count = bodies.len();
        let sim = SimUniforms::new(
            &SimSettings::default(),
            1.0 / 60.0,
            world.cell_size(),
            TABLE_MASK,
            count,
        );
        let mut state = GpuState::new(&device);
        state.upload_bodies(&device, &queue, &bodies);

        let (_, _) = run_pairs(&device, &queue, &state, &sim, count);
        let t = std::time::Instant::now();
        for _ in 0..10 {
            let (total, _) = run_pairs(&device, &queue, &state, &sim, count);
            if total == 0 {
                panic!("the settled pile made no pairs");
            }
            let _ = total;
        }
        let per_frame = t.elapsed().as_secs_f32() * 1000.0 / 10.0;
        println!("pair pass on GPU: {per_frame:.3} ms at {count} bodies");
    }
}
