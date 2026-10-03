# app-mimalloc

> Applications use a fast general allocator, such as mimalloc

Source: Microsoft Pragmatic Rust Guidelines (`M-MIMALLOC-APPS`).

## Why It Matters

The system allocator is conservative. A drop-in allocator like mimalloc lifts allocation-heavy workloads for one line and one dependency. Applications own the whole process, so they can choose freely; libraries must not impose an allocator on their users.

## Bad

```rust
// library forces an allocator on every downstream user
#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;
```

## Good

```rust
// application main.rs: owns the process, picks the allocator
#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

fn main() {}
```

## See Also

- [mem-arena-allocator](mem-arena-allocator.md) - arena for batch allocation
