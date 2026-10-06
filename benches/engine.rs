//! Benchmarks for the physics engine. Run with `cargo bench`.
//!
//! The `step`, `spread`, and `thread_scaling` groups spawn spheres only.
//! The `step_shape` group adds cube and half-and-half piles. Physics
//! reads no shape tag, so the three shape rows must match at each scale.

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;

use particule_simulation_4d::engine::{Shape, World, thread_count};

const DT: f32 = 1.0 / 60.0;

/// Shape distribution of a bench pile.
#[derive(Clone, Copy)]
enum Mix {
    /// Every body is a sphere.
    Sphere,
    /// Every body is a cube.
    Cube,
    /// Half spheres, half cubes, alternating by body index.
    Half,
}

impl Mix {
    /// Row label for this mix.
    fn name(self) -> &'static str {
        match self {
            Mix::Sphere => "sphere",
            Mix::Cube => "cube",
            Mix::Half => "half",
        }
    }

    /// Shape of the body at index `i`.
    fn shape(self, i: usize) -> Shape {
        match self {
            Mix::Sphere => Shape::Sphere,
            Mix::Cube => Shape::Cube,
            Mix::Half if i.is_multiple_of(2) => Shape::Sphere,
            Mix::Half => Shape::Cube,
        }
    }
}

/// Builds a settled pile of `n` bodies. Settling happens outside the timed section.
fn settled(n: usize) -> World {
    settled_mix(n, Mix::Sphere)
}

/// Builds a settled pile of `n` bodies with the shape mix `mix`.
fn settled_mix(n: usize, mix: Mix) -> World {
    let mut w = World::new();
    w.spawn_wave(n, [0.0, 8.0, 0.0], 4.0);
    for (i, b) in w.bodies.iter_mut().enumerate() {
        b.shape = mix.shape(i);
    }
    for _ in 0..60 {
        w.step(DT);
    }
    w
}

/// Full `step` cost at growing body counts, on the default pool.
fn bench_step(c: &mut Criterion) {
    let mut group = c.benchmark_group("step");
    for n in [1000, 4000, 16000, 30000] {
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
    for n in [1000, 4000, 16000, 30000] {
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

/// Shape regimes at the four reference scales. Same count per scale, one
/// settled pile per mix. Physics reads no shape tag, so the three rows of
/// a scale must stay equal. A gap means the engine grew shape-dependent.
fn bench_step_shape(c: &mut Criterion) {
    let mut group = c.benchmark_group("step_shape");
    for n in [10_000, 100_000, 500_000, 1_000_000] {
        group.throughput(Throughput::Elements(n as u64));
        for mix in [Mix::Sphere, Mix::Cube, Mix::Half] {
            group.bench_with_input(BenchmarkId::new(mix.name(), n), &n, |b, &n| {
                let mut w = settled_mix(n, mix);
                b.iter(|| {
                    w.step(DT);
                    black_box(w.bodies[0].pos);
                });
            });
        }
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_step,
    bench_spread,
    bench_thread_scaling,
    bench_step_shape
);
criterion_main!(benches);
