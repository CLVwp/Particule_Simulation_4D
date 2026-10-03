# err-panic-policy

> Panic means "stop the program". Report bugs as panics; report expected failure as `Result`.

Source: Microsoft Pragmatic Rust Guidelines (`M-PANIC-IS-STOP`, `M-PANIC-ON-BUG`, `M-PANIC-CONTINUATION`, `M-PANIC-MESSAGE`).

## Why It Matters

`panic!` and `Result` answer different questions. A broken invariant is a bug: panic, with a message that states the invariant. An expected failure (bad input, I/O) is an error: return `Result`. Catching a panic to continue running is the last resort; default to aborting. A panic without a helpful message turns debugging into guesswork.

## Bad

```rust
// expected failure, but panics
pub fn parse_port(s: &str) -> u16 {
    s.parse().unwrap()
}

// bug hidden behind an error type
if ptr.is_null() { return Err(Error::NullPointer); } // unreachable in a correct caller
```

## Good

```rust
// expected failure: Result
pub fn parse_port(s: &str) -> Result<u16, ParsePortError> {
    s.parse().map_err(ParsePortError)
}

// broken invariant: panic with a clear message
assert!(!ptr.is_null(), "caller passed a null pointer; see safety contract");
```

## See Also

- [err-result-over-panic](err-result-over-panic.md) - return `Result` for recoverable errors
- [err-expect-bugs-only](err-expect-bugs-only.md) - `expect` for invariants only
