# proj-ai-friendly

> Design for AI use: one path per item, no meta design docs, Rust-shaped solutions

Source: Microsoft Pragmatic Rust Guidelines (`M-DESIGN-FOR-AI`, `M-SINGLE-ITEM-PATH`, `M-NO-META-DESIGN-DOCUMENTATION`, `M-RUST-SHAPED`).

## Why It Matters

AI agents and humans both navigate code by names and paths. An item reachable through several re-exports gives inconsistent context and duplicated docs. Meta documents about design drift from the code and mislead. Problems in Rust get solved with Rust idioms; a port of another language's pattern reads as noise.

## Bad

```rust
pub mod prelude { pub use crate::client::{Client, ClientBuilder}; }
pub use crate::client::Client;      // second path to the same item
// DESIGN_DECISIONS.md: why we wrapped the client in a manager factory...
```

## Good

```rust
pub use crate::client::Client;      // one canonical path
// docs live in rustdoc, next to the code they describe
```

## See Also

- [proj-pub-use-reexport](proj-pub-use-reexport.md) - re-export deliberately
- [doc-all-public](doc-all-public.md) - document all public items
