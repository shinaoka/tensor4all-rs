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

## API Design

- **Do not make functions `pub` unnecessarily**: Only expose functions that are part of the public API. Internal helper functions should remain private.
- Functions should be `pub` only when they need to be used by other crates or are part of the documented public interface.

## Testing Guidelines

- **Private function testing**: For private functions and internal helpers, add tests at the end of the source file using `#[cfg(test)]` modules.
- **Integration test file naming**: When creating test files in the `tests/` directory, follow the naming convention:
  - Use the same name as the source file: `src/index.rs` → `tests/index.rs`
  - For testing specific features: `src/index.rs` → `tests/index_ops.rs` (where `_ops` indicates operations)
  - The test file name should clearly indicate what is being tested

**Example:**
```rust
// src/index.rs
fn internal_helper() -> usize {
    42
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_internal_helper() {
        assert_eq!(internal_helper(), 42);
    }
}
```

For public API testing, use `tests/index.rs` or `tests/index_ops.rs` as appropriate.

- **Avoid code duplication in tests**: When writing tests, avoid unnecessary code duplication. If you have nearly identical tests for different types (e.g., `f64` and `Complex64`), consider using generic test functions or macros to reduce duplication.

**Example:**
```rust
// Instead of duplicating tests for f64 and Complex64:
#[test]
fn test_operation_f64() {
    let x: f64 = 1.0;
    let y: f64 = 2.0;
    assert_eq!(operation(x, y), 3.0);
}

#[test]
fn test_operation_c64() {
    let x = Complex64::new(1.0, 0.0);
    let y = Complex64::new(2.0, 0.0);
    assert_eq!(operation(x, y), Complex64::new(3.0, 0.0));
}

// Consider using a generic test helper:
fn test_operation_generic<T: Scalar + PartialEq + std::fmt::Debug>()
where
    T: From<f64>,
{
    let x = T::from(1.0);
    let y = T::from(2.0);
    assert_eq!(operation(x, y), T::from(3.0));
}

#[test]
fn test_operation_f64() {
    test_operation_generic::<f64>();
}

#[test]
fn test_operation_c64() {
    test_operation_generic::<Complex64>();
}
```

This approach reduces maintenance burden and ensures consistency across similar test cases.

## Dependencies

- When adding dependencies, consider:
  - Is it already used in another crate? (check `Cargo.toml` files)
  - Is it a workspace dependency? (check root `Cargo.toml`)
  - Does it fit the crate's purpose and scope?

## Git Workflow

- Each crate is typically developed independently
- Follow standard git workflow: create branches, commit changes, create PRs
- Test changes before committing

