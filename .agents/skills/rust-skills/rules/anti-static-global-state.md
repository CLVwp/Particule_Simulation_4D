# anti-static-global-state

> Avoid `static` items. Pass state explicitly, or use `OnceLock`/`LazyLock`.

Source: Microsoft Pragmatic Rust Guidelines (`M-AVOID-STATICS`).

## Why It Matters

Global statics hide dependencies, break test isolation, and force `unsafe` when mutable. State passed as parameters stays visible and testable. When a value is truly global by nature (a registry, a timezone database), use `LazyLock` for safe one-time initialization.

## Bad

```rust
static CONFIG: Mutex<Config> = Mutex::new(Config::new()); // hidden global dependency
```

## Good

```rust
pub struct App { config: Config } // state travels with the owner

static REGISTRY: LazyLock<Registry> = LazyLock::new(Registry::load); // true global, safe init
```

## See Also

- [const-vs-static](const-vs-static.md) - `const` inlines, `static` addresses
- [conc-thread-local](conc-thread-local.md) - thread-local over `static mut`
