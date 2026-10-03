# err-build-result

> Builders validate in the final `.build()`, not in every setter

Source: Microsoft Pragmatic Rust Guidelines (`M-BUILD-RESULT`).

## Why It Matters

Validation in every setter forces setters to return `Result` and makes chaining painful. Collect checks in one place: `.build()` returns `Result` and reports all missing or conflicting state. Setters stay infallible and chainable.

## Bad

```rust
let b = ClientBuilder::new().timeout(secs)?.retries(n)?; // fallible setters
```

## Good

```rust
let client = ClientBuilder::new()
    .timeout(secs)
    .retries(n)
    .build()?; // single validation point
```

## See Also

- [api-builder-pattern](api-builder-pattern.md) - builders for complex construction
