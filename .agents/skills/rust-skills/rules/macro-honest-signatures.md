# macro-honest-signatures

> Macros must not hide signatures. Prefer declarative macros over proc macros.

Source: Microsoft Pragmatic Rust Guidelines (`M-MACROS-DONT-LIE`, `M-EXAMPLE-OVER-PROC`).

## Why It Matters

A macro that invents fields, impls, or visibility surprises both users and tools: rustdoc, IDEs, and reviewers see the expansion, not the intent. Users cannot reason about a function whose real signature differs from what the macro shows. A declarative `macro_rules!` covers most needs; a proc macro is a heavyweight dependency for author and user.

## Bad

```rust
// generates a hidden getter and a hidden impl behind the call site
make_entity!(User { id: u64, name: String }); // + hidden fn as_row(), hidden From impl
```

## Good

```rust
// visible surface: the struct says everything
struct User { id: u64, name: String }

impl User {
    fn as_row(&self) -> Row { /* plain, greppable, documented */ }
}
```

## See Also

- [macro-prefer-functions](macro-prefer-functions.md) - reach for a macro last
- [macro-proc-two-crate](macro-proc-two-crate.md) - proc macros in a dedicated crate
