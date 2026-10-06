# TODO — CPU frame-time reduction

Baseline at 500 000 bodies (F1 panel, RX 7800 XT): step 31.99 ms.
contacts 24.22 ms, resolve 3.91 ms, grid 3.36 ms. Order the work by
phase weight: contacts first.

After the 2026-10-06 wave (settled pile, `cargo run --release
--example phase_table 500000`): the example settles 60 frames like
the benches, so the pile is tighter than the old 30-frame numbers
and contacts read 865 369. Median step 19.5 ms. contacts 12.0 ms,
grid 3.1 ms, resolve 4.0 ms, scene 7.5 ms. Resolve paid about
0.5 ms for the pair-friction delta: `ContactDelta` grew 20 to 36
bytes to carry the tangent. The example prints the scene line now;
the step loop runs alone, the scene loop follows. Use it before and
after every change.

In-app F1 at 500k, sparse scene (19 893 contacts): frame 19.0 ms,
step 7.747 ms. contacts 3.967 ms, grid 3.127 ms, resolve 0.683 ms.
scene 7.565 ms — about 40 % of the frame. The frame is render-bound
again. The GPU storage buffer item now leads, and grid work beats
body sleep at this contact count.

In-app F1 at 1 000 000 bodies, 2026-10-06 evening. CPU path, vertex
pull on: step 15.7 ms, frame 25.9 ms, 39 FPS, 289k contacts.
grid 6.7 ms (43 %), contacts 9.0 ms (48 %), resolve 1.7 ms. GPU path:
step 4.1 ms, frame 7.4 ms, 140 FPS. The 1M GPU step extrapolation of
14-18 ms was wrong; the pair pass scales better on a sparse scene
than its 500k dense sample. The objective of 60 FPS at 1M is met.

Tags: `[algo]` algorithm, `[par]` parallelism, `[arch]` architecture,
`[visual]` visual shortcut.

Verify every change three ways: `cargo test` (the determinism test must
stay green), `cargo bench`, and the F1 phase table before and after.

## State — 2026-10-06

Read this section before the first change of a new session.

- The F1 panel tunes every optimization live. Render knobs live in
  `Tuning` (`ui/scene/mod.rs`). Engine knobs live in `SimSettings`
  (`par_min`, `prune_dead_pairs`, `max_speed`, `pair_friction`,
  `pair_restitution`, `resolve_rounds`, `resolve_epsilon`). Space
  pauses the sim, and the Physics window scales the step.
- The engine container and its phases live in `engine/world/`. The UI
  pages live in `ui/gui/`, the scene build in `ui/scene/`, and the
  reusable egui components in `ui/widgets.rs`.
- At 500k bodies the step is physics-bound. The render knobs do not
  move the FPS. Work on the engine phases only.
- The scan is a sorted-key CSR with a monotone merge walk per stencil
  offset. `world/grid.rs` holds the key sort. `world/contacts.rs` holds
  the walk, the CSR fill, and the `bc_*` body lists.
- A hash-map scan was measured twice and reverted. It regressed at
  small counts. Do not revisit it without a new design.
- `world/mod.rs::par_each` is the pool gate. It compares the slice length
  with `settings.par_min`. Reuse it for the parallel CSR fill.
- Per-phase timings sit in `World::phase_ms`. The F1 table reads them.
- Every number in this file comes from sphere-only piles. The `step_shape`
  bench group adds cube and half-and-half piles at 10k, 100k, 500k, and
  1M, one settled pile per mix, same count per scale. Physics reads no
  shape tag, so the three rows of a scale must stay equal. A gap means
  the engine grew shape-dependent.
- The GPU side lives in `ui/gpu/`. The render path draws bodies from a
  GPU storage buffer (`Tuning.gpu_render`); the CPU scene build keeps
  only the lines and axis labels. The full physics step (integrate,
  grid, pairs, solve, floor) runs on device behind the F1 toggle
  `Tuning.gpu_physics` and its body threshold. Tests:
  `cargo test --release --lib ui::gpu -- --nocapture
  --test-threads=1`. Release, settled 500k pile: full GPU step
  7.99 ms against 18.28 ms on the CPU. In app at 1M: 4.1 ms and
  140 FPS. The residency ends past the fixed CSR region, so the
  wrap cannot happen. Determinism is per device; the CPU path
  stays the reference. See the GPU compute protocol in CLAUDE.md.

## Contacts — 24.2 ms (75 % of the step)

- [x] `[par]` **Parallel CSR fill.** Atomic fetch-add ranks, then an
  insertion sort restores the ascending contact order per body. One code
  path serves both drivers, so the sums stay deterministic. Result: the
  fill alone moved 21.78 → 20.16 ms. The scan, not the fill, was the
  real cost.
- [x] `[algo]` **Merge scan.** New item, found by profiling the scan.
  A translated cell key is monotone in the own key, so each of the 13
  stencil offsets owns one merge pointer that only moves forward. One
  binary search per offset and chunk replaces one search per cell and
  offset. Result: contacts 20.16 → 10.12 ms at 500k. Pair count
  unchanged (568 426). Determinism tests stay green.
- [ ] `[algo]` **Body sleep.** A body at rest under a small speed leaves
  the scan until an impulse wakes it. A settled pile then costs almost
  nothing. Effort: medium-high. Touches spawn, solver, and the floor.
- [ ] `[arch]` **SoA body arrays.** pos, vel, radius in separate vectors
  improve cache locality in the scan, the delta pass, and the key fill.
  Est: 10-20 % on contacts and grid. Effort: medium. Touches the engine.
- [ ] `[algo]` **Precomputed shares.** Store the two share vectors in
  `ContactDelta` at delta time. The apply loop becomes pure adds.
  `ContactDelta` grows from 20 to 44 bytes. Trades memory traffic for ALU.
  Est: small on contacts and resolve. Effort: small.
- [x] `[arch]` **GPU broad phase.** Hash grid and pair list in compute
  shaders. Landed 2026-10-06 with the full GPU step: integrate, grid,
  pairs, solve, and floor run on device. 7.99 ms at 500k settled
  against 18.28 ms CPU. The pair pass reads a per-body cell cache the
  scatter writes. Remaining headroom: the fixed 2^22 hash region and
  the extra scans.

## Resolve — 3.9 ms (12 % of the step)

- [x] `[algo]` **Expose the round count.** `SimSettings.resolve_rounds`,
  live slider in the F1 panel. Default stays 2.
- [x] `[algo]` **Early exit.** `SimSettings.resolve_epsilon`. Stops the
  rounds when the mean delta motion drops below it. Zero disables.
  Default zero, so the default physics is unchanged.
- [ ] `[par]` **SIMD delta math.** `wide` or portable SIMD on the delta
  pass. Est: 20-40 % of the phase. Effort: small-medium.

## Grid — 3.4 ms (10 % of the step)

- [ ] `[algo]` **Counting-sort build.** REJECTED by arithmetic, not by
  measurement. The key space is 63 bits, so a counting sort needs radix
  passes first: 4 passes by 16-bit digits cost more in histogram and
  scatter traffic than the rayon comparison sort at 500k. Revisit only
  with a dense cell id source.
- [ ] `[par]` **Parallel run counting.** Build `cell_keys` and
  `cell_start` in chunks with one boundary fixup. Today this pass is
  sequential. Est: about -0.5 ms at 500k. Deprioritized: about 3 % of
  the step now.
- [ ] `[algo]` **Radix sort on the 21-bit fields.** Three 7-bit passes
  beat the comparison sort at this size. The counting-sort build above
  may replace this item.

## Cross-cutting

- [x] `[arch]` **Render from a GPU storage buffer.** The vertex shader
  reads raw bodies and projects on device. Landed 2026-10-06
  (`Tuning.gpu_render`): the CPU scene build drops to lines and axis
  labels, and the 500k sparse frame went 19.0 to about 12 ms. Tile
  merge and off-screen cull only shape the CPU path; the F1 panel
  greys them out while the GPU draws.
- [ ] `[visual]` **Adaptive LOD.** Raise `merge px` automatically so
  instances stay under a budget. Bounds the scene phase at any count.
  Effort: small.
- [ ] `[visual]` **Instance budget.** Cap the drawn instances, nearest
  first. A hard bound for extreme scenes. Effort: small.
- [ ] `[arch]` **mimalloc.** Swap the global allocator. Small wins on
  allocation-heavy frames. Effort: trivial.

## CI — solved 2026-10-06

`checks` runs on Linux and Windows msvc. The miri fixes are committed:
`CARGO_NET_RETRY` for the nightly cache, and `MIRIFLAGS` with Tree
Borrows for the `crossbeam-epoch` false positive. The job now also
sets `timeout-minutes` and a `concurrency` group, so a stale run
cannot stack for six hours. The multi-hundred-step test loops carry
`#[cfg_attr(miri, ignore)]`. One parallel step still runs under miri,
so the pool and the atomic fill stay checked.
