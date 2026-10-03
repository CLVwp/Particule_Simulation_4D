# api-types-send

> Design public types to be `Send` and `Sync` unless the type is local by design

Source: Microsoft Pragmatic Rust Guidelines (`M-TYPES-SEND`).

## Why It Matters

A non-`Send` public type spreads its restriction to every caller that stores it. Users discover the restriction late, inside spawned tasks or threads. Design the common case as `Send`; use `Rc`, `RefCell`, or `LocalSet` only for types that are local by design.

## Bad

```rust
pub struct Session {
    // Rc makes the whole type non-Send
    state: Rc<RefCell<Inner>>,
}
```

## Good

```rust
pub struct Session {
    // Arc works across threads; the type stays Send + Sync
    state: Arc<Mutex<Inner>>,
}
```

## See Also

- [own-arc-shared](own-arc-shared.md) - use `Arc<T>` for shared ownership
- [own-rc-single-thread](own-rc-single-thread.md) - keep `Rc<T>` for local types only
