# proj-no-glob-reexports

> Do not glob re-export items. Name every re-exported item.

Source: Microsoft Pragmatic Rust Guidelines (`M-NO-GLOB-REEXPORTS`).

## Why It Matters

`pub use dep::*;` re-exports whatever the dependency exports today, and something else tomorrow. Names collide silently, and the public surface becomes unpredictable. Enumerate the items.

## Bad

```rust
pub use serde::*; // public surface follows an external crate
```

## Good

```rust
pub use serde::{Deserialize, Serialize}; // explicit, stable
```

## See Also

- [proj-pub-use-reexport](proj-pub-use-reexport.md) - re-export deliberately
