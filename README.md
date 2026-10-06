# Particule_Simulation_4D

Real-time particle simulation in 3D — x, y, z, with time as the fourth dimension.
Written in Rust. Rendered by wgpu. Tuned live from the in-game debug panel.

A remake of my C++ / Raylib project
[Cpp-Raylib-Particule-Simulation](https://github.com/CLVwp/Cpp-Raylib-Particule-Simulation).

## Stack

| | |
|---|---|
| Language | Rust 1.95+ (edition 2024) |
| Rendering | [wgpu](https://wgpu.rs) 30 — one instanced draw call for all bodies |
| Window | [winit](https://docs.rs/winit) 0.30 |
| UI | [egui](https://docs.rs/egui) 0.36 |
| Physics | CPU rayon pool, or GPU compute shaders above a threshold |
| Benchmarks | [criterion](https://github.com/bheisler/criterion.rs) |

## Prerequisites

- [Rust 1.95+](https://rustup.rs) with the `msvc` toolchain on Windows.
- A GPU with Vulkan, DirectX 12, or Metal drivers.
- The app picks the fastest GPU. It logs the adapter at startup.

## Run

```bash
git clone https://github.com/CLVwp/Particule_Simulation_4D.git
cd Particule_Simulation_4D
cargo run --release
```

## Controls

| Input | Action |
|---|---|
| Left drag | Orbit |
| Shift + left drag, or middle drag | Pan |
| Mouse wheel | Zoom (clamped) |
| `Z Q S D` / `W A S D`, `E`, `Q` | Slide, rise, sink — follows the camera |
| `F1` | Show the stats and tuning panel |

The key bindings are preset for QWERTY and AZERTY. Every action is rebindable
in the Settings page. Press `Escape` to cancel a rebind.

## The F1 panel

The panel shows live stats: frame and step times, per-phase physics costs,
contacts, instances painted, memory counters, and the GPU adapter name.

The panel also tunes the optimization systems at runtime. Every knob keeps a
working default:

- **Tile merge (LOD)** — bodies below the pixel radius merge into screen
  tiles. One quad replaces each dense tile. Tune the radius, the tile size,
  and the tile fill.
- **Off-screen cull** — drops quads outside the viewport before the sort.
- **Fluid cutoff** — fluid cells below this density do not draw.
- **Prune dead pairs** — keeps only overlapping pairs in the contact list.
  The dropped pairs add zero in the scan round. One solve round replays
  bit-exact. Later rounds can see overlaps that earlier rounds created,
  so with two rounds long runs drift a little.
- **PAR_MIN** — the rayon pool starts above this body count. Set it to `0`
  to force the parallel path, or raise it to force the inline path.
- **GPU physics** — runs integrate, the broad phase, and the Jacobi solve
  in compute shaders above the body threshold. Below the threshold, on a
  device error, or in fluid mode, the CPU pool takes over. The phase rows
  show the five GPU passes, and the resolve epsilon greys out while the
  GPU steps.

`Reset tuning` restores every default.

## Performance notes

- All bodies draw through one instanced pipeline. The GPU cost stays flat as
  the body count grows.
- The physics step is the frame-time ceiling. The F1 panel shows which phase
  dominates: integrate, grid, contacts, resolve, or floor.
- Above the threshold the step runs in compute shaders. A settled 500k pile
  steps in about 8 ms on the GPU against about 18 ms on the CPU pool.
- Results do not depend on the thread count. A test pins this property.

## Architecture

```
src/
  engine/       physics core. No UI code. Fully unit tested.
    world/      the World container and the five step phases.
    fluid/      the Navier-Stokes fluid mode.
  ui/           window, renderer, input, and the app state.
    widgets.rs  reusable egui components.
    gui/        the pages: menu, settings, HUD, side windows, overlay.
    scene/      the scene build: project, cull, LOD merge, sort.
    gpu/        compute shaders for the physics step, and the body
                buffer the draw reads.
  perf.rs       counting allocator behind the memory stats.
```

The engine holds the laws of motion and the solver. It knows nothing about
the GPU or the UI, so you can test it, bench it, or reuse it anywhere.
`cargo test` runs the whole suite. `cargo bench` measures the step phases.

## Benchmarks

Run one group to keep the wait short:

```bash
cargo bench --bench engine -- step_shape
```

| Group | Measures |
|---|---|
| `step` | full step on settled sphere piles, 1k to 30k bodies |
| `spread` | full step on sparse scenes, almost no contacts |
| `thread_scaling` | step cost at 1, half, and all detected threads |
| `step_shape` | sphere, cube, and half-and-half piles at 10k, 100k, 500k, 1M |
| `scene` | the CPU scene build at the same four scales |

The engine never reads the shape tag. The three `step_shape` rows of one
scale must stay equal. A gap means the physics grew shape-dependent.

A bench case takes longer than one app frame. The bench builds a settled
pile, warms up, then times hundreds of steps for stable statistics. The
app draws one step per frame and shows one sample.

## Roadmap

- [x] 3D particle system and camera
- [x] Newton rigid bodies with contacts
- [x] Navier-Stokes fluid mode
- [x] GPU instanced rendering with tile-merge LOD
- [x] Live tuning panel
- [x] Physics on the GPU with compute shaders

The GPU physics runs integrate, the broad phase, and the Jacobi solve
in compute shaders. Enable it in the F1 panel; below the threshold the
CPU pool takes over. One GPU step holds a settled 500k pile in about
8 ms against about 18 ms on the CPU. Results stay deterministic per
device, and the CPU path remains the reference.

## References

- [wgpu](https://docs.rs/wgpu) and [egui](https://docs.rs/egui) documentation
- Original project: [Cpp-Raylib-Particule-Simulation](https://github.com/CLVwp/Cpp-Raylib-Particule-Simulation)
