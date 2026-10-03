# doc-inline-reexports

> Mark re-exported types with `#[doc(inline)]`

Source: Microsoft Pragmatic Rust Guidelines (`M-DOC-INLINE`).

## Why It Matters

A re-exported type documents at its home module by default. Users of your crate jump to the docs and land in a foreign crate. `#[doc(inline)]` brings the documentation into your docs, so your crate reads as the reference.

## Bad

```rust
pub use foreign_crate::Client; // docs point away from your crate
```

## Good

```rust
#[doc(inline)]
pub use foreign_crate::Client; // docs render inside your crate
```

## See Also

- [proj-pub-use-reexport](proj-pub-use-reexport.md) - re-export deliberately
- [doc-intra-links](doc-intra-links.md) - link items with intra-doc links
