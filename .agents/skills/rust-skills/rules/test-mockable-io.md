# test-mockable-io

> I/O and system calls sit behind traits, and test utilities are feature-gated

Source: Microsoft Pragmatic Rust Guidelines (`M-MOCKABLE-SYSCALLS`, `M-TEST-UTIL`).

## Why It Matters

Code that calls the filesystem or the clock directly cannot be tested without a real environment. A narrow trait (or injected dependency) makes the seam explicit. Test helpers shipped to real users widen the API and the dependency tree; gate them behind a `test-util` feature.

## Bad

```rust
pub fn load_user(id: u32) -> User {
    let raw = std::fs::read_to_string(format!("/data/{id}.json")).unwrap(); // untestable
    parse(&raw)
}

pub fn fixture_user() -> User { /* shipped to production users */ }
```

## Good

```rust
pub trait UserSource { fn load(&self, id: u32) -> anyhow::Result<User>; }

pub struct FsSource;                       // production
pub struct MemorySource(pub Vec<User>);    // tests

#[cfg(feature = "test-util")]
pub fn fixture_user() -> User;
```

## See Also

- [test-mock-traits](test-mock-traits.md) - traits enable mocking
- [test-integration-dir](test-integration-dir.md) - integration tests under `tests/`
