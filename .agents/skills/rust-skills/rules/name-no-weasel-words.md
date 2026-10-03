# name-no-weasel-words

> Names carry meaning. No weasel words, no filler.

Source: Microsoft Pragmatic Rust Guidelines (`M-WEASEL-WORDS`, `M-SHORT-NAMES`).

## Why It Matters

Names like `Manager`, `Helper`, `Utils`, `Data`, or `Info` hide what the type does. A reader must open the file to learn anything. Short, specific names also make errors and docs readable.

## Bad

```rust
pub struct DataManager;      // manages what?
pub fn process_util(data: &Info) -> Result<Output>;
```

## Good

```rust
pub struct ConnectionPool;
pub fn parse_record(raw: &[u8]) -> Result<Record>;
```

## See Also

- [name-types-camel](name-types-camel.md) - naming conventions
- [name-is-has-bool](name-is-has-bool.md) - boolean method prefixes
