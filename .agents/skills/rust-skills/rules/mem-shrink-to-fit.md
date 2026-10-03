# mem-shrink-to-fit

> Shrink collections to fit after building them

Source: Microsoft Pragmatic Rust Guidelines (`M-SHRINK-TO-FIT`).

## Why It Matters

`with_capacity` followed by growth or filtering often leaves large spare capacity. A long-lived collection keeps that memory forever. Call `shrink_to_fit()` (or build with `Box<[T]>`) once construction is done and the collection is long-lived.

## Bad

```rust
let mut names = Vec::with_capacity(10_000); // parsed from a 200-item file
// ... names.len() == 200; capacity stays 10_000
```

## Good

```rust
let mut names = Vec::with_capacity(10_000);
// ...
names.shrink_to_fit(); // long-lived; free the spare capacity
```

## See Also

- [mem-with-capacity](mem-with-capacity.md) - reserve up front
- [mem-boxed-slice](mem-boxed-slice.md) - `Box<[T]>` for fixed owned data
