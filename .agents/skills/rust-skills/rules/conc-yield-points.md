# conc-yield-points

> Long-running loops need yield points

Source: Microsoft Pragmatic Rust Guidelines (`M-YIELD-POINTS`).

## Why It Matters

A loop that never yields starves the runtime or the OS scheduler. On async, a loop without `.await` blocks the executor worker. On threads, a CPU-bound loop burns a core and delays shutdown. Check a cancel flag, sleep briefly, or `yield_now()` at a predictable interval.

## Bad

```rust
// async worker without an await point
loop {
    poll_source(); // executor worker is stuck here
}
```

## Good

```rust
loop {
    if cancel.is_cancelled() { break; }
    poll_source();
    tokio::task::yield_now().await; // executor stays responsive
}
```

## See Also

- [async-cancellation-token](async-cancellation-token.md) - graceful shutdown
- [conc-rayon-par-iter](conc-rayon-par-iter.md) - CPU-bound work on rayon
