# lint-expect-over-allow

> Silence a lint with `#[expect]`, not `#[allow]`

Source: Microsoft Pragmatic Rust Guidelines (`M-LINT-OVERRIDE-EXPECT`).

## Why It Matters

`#[allow]` stays true forever, even after the code changes and the lint no longer fires. `#[expect]` warns when the expectation stops holding. The override removes itself from the code path once it is obsolete.

## Bad

```rust
#[allow(dead_code)] // stays silent forever, even when wrong
fn legacy_path() {}
```

## Good

```rust
#[expect(dead_code)] // warns once the function is used or removed
fn legacy_path() {}
```

## See Also

- [lint-warn-style](lint-warn-style.md) - enable the style lint group
