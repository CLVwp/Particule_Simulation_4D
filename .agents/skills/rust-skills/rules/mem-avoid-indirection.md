# mem-avoid-indirection

> Do not add pointer layers to nested types without a reason

Source: Microsoft Pragmatic Rust Guidelines (`M-AVOID-INDIRECTION`).

## Why It Matters

`Box`, `Rc`, and `Option<Box<...>>` inside a struct scatter the data across the heap. Every layer costs an allocation and a pointer dereference on access. Only add indirection for a concrete need: breaking a size cycle, sharing, or shrinking a large enum.

## Bad

```rust
struct Node {
    // double indirection; one Box is enough
    next: Option<Box<Rc<Node>>>,
}
```

## Good

```rust
struct Node {
    next: Option<Box<Node>>, // one heap allocation per link
}
```

## See Also

- [mem-box-large-variant](mem-box-large-variant.md) - `Box` large enum variants
- [opt-cache-friendly](opt-cache-friendly.md) - keep data contiguous
