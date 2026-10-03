# async-stack-size

> Keep hot async functions small on the stack

Source: Microsoft Pragmatic Rust Guidelines (`M-ASYNC-STACK-SIZE`).

## Why It Matters

An async fn stores all its locals inside its future, so a big buffer or a deep call inlined into a task becomes part of every task's footprint. Thousands of spawned tasks multiply that cost. Move large buffers out of the future (allocate on the heap, or extract into a non-async fn).

## Bad

```rust
async fn process() {
    let buf = [0u8; 64 * 1024]; // 64 KiB inside every future
    // ...
}
```

## Good

```rust
async fn process() {
    let buf = vec![0u8; 64 * 1024]; // one heap allocation, small future
    // ...
}
```

## See Also

- [async-spawn-blocking](async-spawn-blocking.md) - heavy work off the async runtime
