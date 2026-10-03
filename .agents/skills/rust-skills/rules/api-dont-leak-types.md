# api-dont-leak-types

> Do not expose external crate types in your public API

Source: Microsoft Pragmatic Rust Guidelines (`M-DONT-LEAK-TYPES`).

## Why It Matters

A leaked foreign type in the public surface forces users to depend on that crate. A minor-version bump of the foreign crate then becomes a breaking change for your users. Keep foreign types at the boundary; convert to your own types or stdlib types.

## Bad

```rust
// leaks serde_json::Value into the public API
pub fn parse_config(raw: &str) -> serde_json::Value;
```

## Good

```rust
// the public API speaks your own types
pub struct Config { /* ... */ }

pub fn parse_config(raw: &str) -> anyhow::Result<Config>;
```

## See Also

- [proj-pub-use-reexport](proj-pub-use-reexport.md) - re-export deliberately
- [api-escape-hatches](api-escape-hatches.md) - wrapped types keep an escape hatch
