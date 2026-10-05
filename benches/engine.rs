//! Benchmarks for the physics engine. Run with `cargo bench`.

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;

use particule_simulation_4d::engine::{World, thread_count};

const DT: f32 = 1.0 / 60.0;

/// Builds a settled pile of `n` bodies. Settling happens outside the timed section.
fn settled(n: usize) -> World {
    let mut w = World::new();
    w.spawn_wave(n, [0.0, 8.0, 0.0], 4.0);
    for _ in 0..60 {
        w.step(DT);
    }
    w
}

/// Full `step` cost at growing body counts, on the default pool.
fn bench_step(c: &mut Criterion) {
    let mut group = c.benchmark_group("step");
    for n in [1000, 4000, 16000] {
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, &n| {
            let mut w = settled(n);
            b.iter(|| {
                w.step(DT);
                black_box(w.bodies[0].pos);
            });
        });
    }
    group.finish();
}

/// Same workload on pools of 1, half, and all detected logical cores.
fn bench_thread_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("thread_scaling");
    let counts = [1, thread_count() / 2, thread_count()];
    for &t in &counts {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(t)
            .build()
            .unwrap();
        group.bench_with_input(BenchmarkId::from_parameter(t), &pool, |b, pool| {
            let mut w = settled(4000);
            b.iter(|| {
                pool.install(|| w.step(DT));
                black_box(w.bodies[0].pos);
            });
        });
    }
    group.finish();
}

/// Builds a near-rest cloud of `n` bodies spread over ~40 units, spacing
/// wider than two diameters. Sparse regime: full grid scan, few real pairs.
fn spread(n: usize) -> World {
    let mut w = World::new();
    // Zero gravity. The cloud must stay spread for the whole bench run.
    w.settings.gravity = 0.0;
    w.spawn_wave(n, [0.0, 8.0, 0.0], 4.0);
    let per = (n as f64).cbrt().ceil() as usize;
    let spacing = 40.0 / per as f32;
    for (i, b) in w.bodies.iter_mut().enumerate() {
        let (ix, iy, iz) = (i % per, (i / per) % per, i / (per * per));
        b.pos = [
            ix as f32 * spacing - 20.0,
            0.5 + iy as f32 * spacing,
            iz as f32 * spacing - 20.0,
        ];
        b.vel = [0.0, 0.0, 0.0];
    }
    w
}

/// Sparse regime: bodies spread wide, almost no contacts.
fn bench_spread(c: &mut Criterion) {
    let mut group = c.benchmark_group("spread");
    for n in [1000, 4000, 16000] {
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, &n| {
            let mut w = spread(n);
            b.iter(|| {
                w.step(DT);
                black_box(w.bodies[0].pos);
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_step, bench_spread, bench_thread_scaling);
criterion_main!(benches);
