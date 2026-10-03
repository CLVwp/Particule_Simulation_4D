# api-services-clone

> Make service types cheap to clone, and derive `Clone`

Source: Microsoft Pragmatic Rust Guidelines (`M-SERVICES-CLONE`).

## Why It Matters

Service types (connections, clients, handlers) get passed into tasks, handlers, and layers. If cloning is impossible, users reach for `Arc` wrappers themselves and the API diverges. Put cheap fields behind `Arc` so a clone is a handle, not a deep copy.

## Bad

```rust
pub struct Db {
    pool: Pool, // deep clone semantics unclear; no Clone
}
```

## Good

```rust
#[derive(Clone)]
pub struct Db {
    // Pool is internally an Arc; clone is cheap
    pool: Pool,
}
```

## See Also

- [own-arc-shared](own-arc-shared.md) - shared ownership across threads
- [own-clone-explicit](own-clone-explicit.md) - clone where the cost is meaningful
