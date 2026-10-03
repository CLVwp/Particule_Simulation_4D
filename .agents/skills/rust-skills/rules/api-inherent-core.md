# api-inherent-core

> Put core functionality in inherent methods; prefer free functions when there is no receiver

Source: Microsoft Pragmatic Rust Guidelines (`M-ESSENTIAL-FN-INHERENT`, `M-REGULAR-FN`).

## Why It Matters

Users look for the main operation on the type itself. An essential operation hidden in a trait or a helper module is hard to find and hard to document. Inherent impls also avoid trait-resolution surprises. When the function has no natural receiver, write a plain free function instead of forcing an associated function on an unrelated type.

## Bad

```rust
pub trait ZipExt { fn write_zip(&self, out: &mut Vec<u8>); } // essential op in a trait
impl Archive { fn default_name() -> String } // no receiver; forced associated fn
```

## Good

```rust
impl Archive {
    pub fn write_zip(&self, out: &mut Vec<u8>); // inherent: core operation
}

pub fn default_name() -> String; // free function: no receiver involved
```

## See Also

- [api-common-traits](api-common-traits.md) - implement standard traits for public types
