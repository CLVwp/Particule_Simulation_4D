# doc-first-sentence

> The first doc sentence is one short line, and docs use canonical sections

Source: Microsoft Pragmatic Rust Guidelines (`M-FIRST-DOC-SENTENCE`, `M-CANONICAL-DOCS`).

## Why It Matters

The first sentence becomes the summary line in rustdoc lists, IDE hovers, and search results. A paragraph there renders as a wall of text. Keep it to one line of about 15 words, then explain, then use the canonical sections (`# Examples`, `# Errors`, `# Panics`, `# Safety`).

## Bad

```rust
/// Parses the config file which can be in several formats and also handles
/// environment overrides and a long list of historical quirks, returning the
/// final configuration object.
pub fn load_config() -> Config;
```

## Good

```rust
/// Loads the effective configuration from file and environment.
///
/// Reads `app.toml`, then applies environment overrides.
///
/// # Errors
///
/// Returns an error when the file is missing or invalid.
pub fn load_config() -> anyhow::Result<Config>;
```

## See Also

- [doc-errors-section](doc-errors-section.md) - `# Errors` section
- [doc-examples-section](doc-examples-section.md) - `# Examples` section
