# ffi-core-translates

> Business logic lives in a core crate; the FFI layer only translates. Isolate state. Follow FFI naming.

Source: Microsoft Pragmatic Rust Guidelines (`M-FFI-TRANSLATES`, `M-ISOLATE-DLL-STATE`, `M-FFI-NAMING`).

## Why It Matters

A thin FFI layer keeps the logic testable in pure Rust and the surface auditable. Global state shared across DLL boundaries creates order-of-init and unload bugs; each FFI library must own its state. Naming must match what host languages expect.

## Bad

```rust
// logic embedded in the exported function; global state behind the boundary
#[no_mangle]
pub extern "C" fn compute(a: f64, b: f64) -> f64 {
    unsafe { GLOBAL_CONFIG.risk_limit * (a + b) } // logic + hidden state
}
```

## Good

```rust
// core crate: pure, testable
pub fn compute(cfg: &RiskConfig, a: f64, b: f64) -> f64 { cfg.risk_limit * (a + b) }

// ffi crate: translation only, per-library state handle
#[unsafe(no_mangle)]
pub extern "C" fn risk_compute(h: *mut RiskHandle, a: f64, b: f64) -> f64 { /* translate */ }
```

## See Also

- [unsafe-safety-comment](unsafe-safety-comment.md) - `// SAFETY:` on every unsafe block
- [unsafe-minimize-scope](unsafe-minimize-scope.md) - keep unsafe blocks small
