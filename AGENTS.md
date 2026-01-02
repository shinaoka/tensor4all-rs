# Agent Guidelines for tensor4all-rs

Before starting work, read the repository root `README.md` as well as this `AGENTS.md`.

## General Guidelines

- Use the same language as in past conversations with the user (if it has been Japanese, use Japanese)
- All source code and documentation must be in English
- Each crate in `crates/` is an independent Rust package with its own `Cargo.toml`, `src/`, and `tests/` directories. Understand the package structure before making changes
- When working on a crate, navigate into its directory and work as if it were a standalone package. Be aware of dependencies between crates

## Error Handling

### Prefer `anyhow` for Generic Error Handling

For generic error handling and error context propagation, **prefer using `anyhow`** over manual error construction. This provides better error messages and error chaining.

**Recommended pattern:**
```rust
use anyhow::{Context, Result};

fn some_function() -> Result<()> {
    let value = some_operation()
        .context("Failed to perform operation")?;
    Ok(())
}
```

**When converting errors to custom error types:**
```rust
use anyhow;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum MyError {
    #[error("Operation failed: {0}")]
    OperationError(#[from] anyhow::Error),
}

// Usage:
let result = some_operation()
    .map_err(|e| anyhow::anyhow!("Context: {:?}", e))
    .map_err(MyError::OperationError)?;
```

**Benefits:**
- Better error messages with context
- Automatic error chaining
- Consistent error handling across the codebase
- Easy to add context at any point in the call stack

### When to Use `thiserror`

Use `thiserror` for:
- Public API error types that need to be matched by callers
- Error types that need to be serialized or sent across boundaries
- Error types that are part of the public interface

Use `anyhow` for:
- Internal error handling and propagation
- Adding context to errors
- Generic error handling where the specific error type doesn't matter to callers

## Running Tests

**Full test suite:**
```bash
cd /path/to/crate
cargo test
```

**Running specific tests:**
```bash
cargo test --test test_name
```

**Running with output:**
```bash
cargo test -- --nocapture
```

**Running tests for all crates:**
```bash
cargo test --workspace
```

## Code Style

- Follow standard Rust style guidelines
- Use `rustfmt` for formatting (run `cargo fmt`)
- Use `clippy` for linting (run `cargo clippy`)
- Prefer explicit error handling over `unwrap()` or `expect()` in library code

## Dependencies

- When adding dependencies, consider:
  - Is it already used in another crate? (check `Cargo.toml` files)
  - Is it a workspace dependency? (check root `Cargo.toml`)
  - Does it fit the crate's purpose and scope?

## Git Workflow

- Each crate is typically developed independently
- Follow standard git workflow: create branches, commit changes, create PRs
- Test changes before committing

