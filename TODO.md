# TODO — CPU frame-time reduction

Baseline at 500 000 bodies (F1 panel, RX 7800 XT): step 31.99 ms.
contacts 24.22 ms, resolve 3.91 ms, grid 3.36 ms. Order the work by
phase weight: contacts first.

Tags: `[algo]` algorithm, `[par]` parallelism, `[arch]` architecture,
`[visual]` visual shortcut.

Verify every change three ways: `cargo test` (the determinism test must
stay green), `cargo bench`, and the F1 phase table before and after.

## Contacts — 24.2 ms (75 % of the step)

- [ ] `[par]` **Parallel CSR fill.** Count contacts per body in chunks,
  prefix-sum, then scatter with per-contact ranks. The order inside each
  body list stays exact, so the Jacobi sum stays deterministic. The wave-2
  stop rule blocked this under ~100k contacts; 959k contacts clear it.
  Est: contacts 24 → 12-16 ms. Effort: medium.
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

- [ ] `[algo]` **Expose the round count.** Put `RESOLVE_ROUNDS` in the
  tuning panel. One round halves the phase; piles settle softer. A direct
  stiffness-for-time trade. Effort: trivial.
- [ ] `[algo]` **Early exit.** Stop the rounds when the summed delta drops
  below a small epsilon. Resting piles then skip most rounds.
  Effort: small.
- [ ] `[par]` **SIMD delta math.** `wide` or portable SIMD on the delta
  pass. Est: 20-40 % of the phase. Effort: small-medium.

## Grid — 3.4 ms (10 % of the step)

- [ ] `[algo]` **Counting-sort build.** Count bodies per cell, prefix the
  runs, then scatter with ranks. Removes the comparison sort entirely.
  Body order inside a cell stays ascending, so emission order holds.
  Est: grid → about 1 ms. Effort: medium.
- [ ] `[par]` **Parallel run counting.** Build `cell_keys` and
  `cell_start` in chunks with one boundary fixup. Today this pass is
  sequential. Est: about -1 ms. Effort: small.
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
