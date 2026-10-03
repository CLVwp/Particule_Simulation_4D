# const-documented-magic

> Every magic value becomes a named, documented constant

Source: Microsoft Pragmatic Rust Guidelines (`M-DOCUMENTED-MAGIC`).

## Why It Matters

A bare number in code forces the reader to guess its origin and its unit. A named constant answers "what is this?" in one place and makes the value greppable. Document the unit and the origin on the constant, not at each use site.

## Bad

```rust
if elapsed > 300 { timeout_occurred(); } // 300 what? why?
```

## Good

```rust
/// Session timeout in seconds. Matches the backend session TTL.
const SESSION_TIMEOUT_SECS: u64 = 300;

if elapsed > SESSION_TIMEOUT_SECS {
    timeout_occurred();
}
```

## See Also

- [const-vs-static](const-vs-static.md) - `const` vs `static`
- [name-consts-screaming](name-consts-screaming.md) - constant naming
