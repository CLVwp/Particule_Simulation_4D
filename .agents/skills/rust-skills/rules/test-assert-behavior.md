# test-assert-behavior

> Tests assert behavior, not a copy of the input

Source: Microsoft Pragmatic Rust Guidelines (`M-TAUTOLOGICAL-TESTS`).

## Why It Matters

A test that re-derives the expected value with the same logic as the code passes always and proves nothing. Assert the ground truth by hand, or assert an observable effect.

## Bad

```rust
#[test]
fn total_price() {
    let cart = Cart::with_items(vec![item(100), item(200)]);
    assert_eq!(cart.total(), cart.items().map(|i| i.price).sum::<u64>()); // same math twice
}
```

## Good

```rust
#[test]
fn total_price() {
    let cart = Cart::with_items(vec![item(100), item(200)]);
    assert_eq!(cart.total(), 300); // hand-written ground truth
}
```

## See Also

- [test-arrange-act-assert](test-arrange-act-assert.md) - structure tests in three parts
- [test-descriptive-names](test-descriptive-names.md) - names explain the behavior
