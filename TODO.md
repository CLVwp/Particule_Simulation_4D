# TODO — CPU frame-time reduction

Baseline at 500 000 bodies (F1 panel, RX 7800 XT): step 31.99 ms.
contacts 24.22 ms, resolve 3.91 ms, grid 3.36 ms. Order the work by
phase weight: contacts first.

After the 2026-10-06 wave (settled pile, `cargo run --release
--example phase_table 500000`): step 15.53 ms. contacts 10.12 ms,
grid 2.97 ms, resolve 2.18 ms. The phase table example reproduces the
F1 table headless. Use it before and after every change.

In-app F1 at 500k, sparse scene (19 893 contacts): frame 19.0 ms,
step 7.747 ms. contacts 3.967 ms, grid 3.127 ms, resolve 0.683 ms.
scene 7.565 ms — about 40 % of the frame. The frame is render-bound
again. The GPU storage buffer item now leads, and grid work beats
body sleep at this contact count.

In-app F1 at 1 000 000 bodies: contacts about 12 ms (55-69 %),
grid about 6 ms (25-30 %), resolve about 2.5 ms (10 %). Grid grows
linearly with the body count, so its share rises with every sparse
scene. Parallel run counting pays from 500k already.

Tags: `[algo]` algorithm, `[par]` parallelism, `[arch]` architecture,
`[visual]` visual shortcut.

Verify every change three ways: `cargo test` (the determinism test must
stay green), `cargo bench`, and the F1 phase table before and after.

## State — 2026-10-06

Read this section before the first change of a new session.

- The F1 panel tunes every optimization live. Render knobs live in
  `Tuning` (`ui/scene/mod.rs`). Engine knobs live in `SimSettings`
  (`par_min`, `prune_dead_pairs`).
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
- [ ] `[arch]` **GPU broad phase.** Hash grid and pair list in compute
  shaders. The CPU reads counts only. Est: contacts under 2 ms.
  Effort: high. See the roadmap item in the README.

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

- [ ] `[arch]` **Render from a GPU storage buffer.** Give the vertex
  shader the raw body positions; the GPU projects and culls. The scene
  phase drops to one buffer write. Tile merge goes away, and the GPU
  absorbs the overdraw. Effort: medium-high.
- [ ] `[visual]` **Adaptive LOD.** Raise `merge px` automatically so
  instances stay under a budget. Bounds the scene phase at any count.
  Effort: small.
- [ ] `[visual]` **Instance budget.** Cap the drawn instances, nearest
  first. A hard bound for extreme scenes. Effort: small.
- [ ] `[arch]` **mimalloc.** Swap the global allocator. Small wins on
  allocation-heavy frames. Effort: trivial.

## CI — for a dedicated session

`checks` is green since the wgpu migration. `miri` failed for three
known causes:

1. gpui era: `yeslogic-fontconfig-sys` failed on Linux. The migration
   removed the dependency. Solved.
2. Network: the nightly cache expires daily. Downloads hit broken
   pipes. Drafted fix: `CARGO_NET_RETRY: "10"`.
3. Miri: `crossbeam-epoch` trips a Stacked Borrows false positive in
   the rayon pool (`internal.rs:567`). Known upstream issue. Not a
   bug here. Drafted fix: `MIRIFLAGS: -Zmiri-tree-borrows`.

The two fixes sit uncommitted in `.github/workflows/ci.yml`. Verify
them, then commit.
