# api-escape-hatches

> Wrap external types, and expose a native escape hatch to get them back

Source: Microsoft Pragmatic Rust Guidelines (`M-ESCAPE-HATCHES`).

## Why It Matters

A wrapper keeps the public API clean, but users sometimes need the inner type for a case you did not cover. Without an escape hatch they drop your wrapper entirely. Provide `into_inner()`, an `as_inner()`, or a raw constructor.

## Bad

```rust
pub struct Timeout(tokio::time::Duration);
// no way back to tokio::time::Duration
```

## Good

```rust
pub struct Timeout(Duration);

impl Timeout {
    pub fn from_secs(secs: u64) -> Self;
    pub fn into_std(self) -> std::time::Duration; // escape hatch
}
```

## See Also

- [api-dont-leak-types](api-dont-leak-types.md) - do not leak external types
