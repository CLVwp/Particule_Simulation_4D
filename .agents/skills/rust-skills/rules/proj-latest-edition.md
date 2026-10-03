# proj-latest-edition

> New crates target the latest Rust edition

Source: Microsoft Pragmatic Rust Guidelines (`M-LATEST-EDITION`).

## Why It Matters

Each edition removes old hazards and unlocks new semantics (2024: `unsafe extern`, disjoint closure captures, `gen` reserved). Starting on the latest edition keeps the codebase current without a migration later. Existing crates migrate deliberately, with `cargo fix --edition`.

## Bad

```toml
[package]
edition = "2018" # new crate on an old edition
```

## Good

```toml
[package]
edition = "2024"
```

## See Also

- [proj-msrv-declare](proj-msrv-declare.md) - declare the minimum supported Rust version
