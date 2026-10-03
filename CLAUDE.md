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
