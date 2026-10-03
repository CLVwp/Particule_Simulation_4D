# opt-log-overhead

> Telemetry must stay cheap on disabled paths

Source: Microsoft Pragmatic Rust Guidelines (`M-LOG-OVERHEAD`).

## Why It Matters

Formatting arguments for a log line costs cycles even when nothing consumes the message. Hot-path logging with `format!`-style eager arguments shows up in profiles. Use lazy, structured fields; guard expensive context behind a level check or a gated span.

## Bad

```rust
info!("processed {} items in {:?}", expensive_summary(&items), elapsed); // always formats
```

## Good

```rust
if tracing::enabled!(tracing::Level::INFO) {
    info!(items = items.len(), elapsed_ms = elapsed.as_millis(), "processed");
}
```

## See Also

- [obs-structured-fields](obs-structured-fields.md) - structured fields over interpolated strings
- [anti-format-hot-path](anti-format-hot-path.md) - no `format!` in hot paths
