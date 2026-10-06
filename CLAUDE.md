# CLAUDE.md

## Rust rules — rust-skills (absolute)

This repo uses the rust-skills skill (source: leonardomso/rust-skills, 292 rules, 28 categories,
consolidated with the Microsoft Pragmatic Rust Guidelines).
The skill is linked at `.claude/skills/rust-skills`. The files live at `.agents/skills/rust-skills`.

- Apply rust-skills to every Rust task. Write, review, refactor, or fix — no exception.
- Read `.agents/skills/rust-skills/SKILL.md` first. It holds the priority table.
- Rules are deterministic: one file per rule, in `.agents/skills/rust-skills/rules/`.
  Find a rule by its prefix. Example: `rules/own-*.md` holds the ownership rules.
- CRITICAL priority prefixes, always obey: `own-` (ownership), `err-` (errors),
  `mem-` (memory), `unsafe-`.
- HIGH priority prefixes, apply when relevant: `api-`, `opt-`, `conc-`, `num-`.
- After Rust changes, run `cargo test`. Do not claim success without test output.

## Performance protocol — engine work

- Measure before and after every engine change. Run
  `cargo run --release --example phase_table 500000`. Compare the
  phase table. State the numbers in the report.
- The per-body contact order is a hard invariant. Each body's contact
  list must stay in ascending contact order. Any other order breaks
  the determinism of the Jacobi sum. Keep the order when you touch the
  scan, the CSR fill, or the solver. The test
  `result_does_not_depend_on_thread_count` guards it.
- Profile before you pick a target. The heaviest phase moves with the
  scene: a dense pile stresses contacts, a sparse scene stresses grid
  and scene build. Read the F1 table or the phase table first.

## GPU compute protocol — `ui/gpu` work

- Verify GPU physics with
  `cargo test --release --lib ui::gpu -- --nocapture --test-threads=1`.
  Run the GPU suite serially. Parallel GPU tests can wedge a driver.
  The parity gaps must stay at 0.0. The phase table still guards the
  CPU path.
- Two determinism tiers. CPU: bit-exact across thread counts, pinned
  by the thread-count test. GPU: bit-exact per device across runs.
  GPU against CPU compares sets with tolerance, never bits.
- One kernel per compute pass. The pass boundary is the memory barrier.
  Two kernels in one pass have no order guarantee on the writes.
- Storage structs use flat `[f32; 3]` arrays. A `vec3` forces align 16
  and grows the stride to 48 bytes. Uniform structs stay flat scalars,
  sixteen floats at most per 64-byte block. Pin every GPU struct with a
  static size assert, on both sides.
- The `Sim` struct repeats in every shader module. One field rename or
  one new field lands in all copies and in the Rust `SimUniforms` in
  the same commit.
- The GPU grid hashes the cell triple, not the packed 63-bit key. WGSL
  has no 64-bit integers. Exact cell checks reject hash collisions.
- The pair CSR scans a fixed 2^22-slot region. The GPU pair pass caps
  near 4.2 million bodies.
- `select` evaluates both arms. An arm with a buffer index must stay in
  bounds on both sides, so branch instead when one arm indexes past the
  end.
- wgpu 30 traps: `get_mapped_range` returns a `Result`, and its view
  owns the map. Drop the view inside a block before `unmap`.
  `NonZeroU64` comes from `std::num`, not from `wgpu`.

## Documentation rule — ASD-STE100 (absolute)

All documentation in this repo obeys ASD-STE100 (Simplified Technical English):
README, markdown docs, code comments, and UI help text.

- Write short sentences. Use 20 words maximum per sentence.
- Write one idea per sentence. In procedures, write one action per step.
- Use the imperative for instructions. Example: "Run `cargo build`."
- Use the active voice.
- Use simple, everyday words. No slang. No idioms. No contractions.
- Use one word for one meaning. Do not switch between synonyms.
- Technical terms from the code are approved: particle, body, grid, spawn, wave, camera.

Comments state what the code does. They do not tell history. They do not list alternatives.
