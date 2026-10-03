# api-parameter-order

> Keep parameter order consistent: receiver, context/config, data, callback last

Source: Microsoft Pragmatic Rust Guidelines (`M-PARAMETER-CONSISTENCY`).

## Why It Matters

Consistent order turns into muscle memory. Users guess a call site correctly the first time. Put the receiver first, then configuration, then the data, and put callbacks or output parameters last.

## Bad

```rust
fn send(retry: usize, client: &Client, payload: Bytes, on_error: impl Fn(Error));
```

## Good

```rust
fn send(client: &Client, payload: Bytes, retry: usize, on_error: impl Fn(Error));
```

## See Also

- [api-builder-pattern](api-builder-pattern.md) - many options belong in a builder
