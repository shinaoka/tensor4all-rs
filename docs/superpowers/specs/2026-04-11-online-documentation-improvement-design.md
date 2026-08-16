> [!NOTE] Superseded (2026-08, #639)
> `tensor4all-tcicore` was dissolved into `tensor4all-core`; the crate no
> longer exists. References below describe the pre-dissolution state.

# Online Documentation Improvement Design

**Date:** 2026-04-11
**Goal:** Improve online documentation so that humans and AI agents can understand usage without reading the codebase.

## Target Audiences

1. **Rust library users** — directly importing tensor4all crates
2. **AI agents** — writing correct code from API documentation alone

## Approach: Bottom-Up (Dependency Order)

Improve each crate from the top of the dependency tree downward, excluding C-API (`tensor4all-capi`). For each crate, improve both rustdoc comments and mdBook guides simultaneously.

### Work Order

```
0. AGENTS.md update (codify documentation rules)
1. CI hardening (add mdbook test, verify cargo test --doc)
2. tensor4all-core
3. tensor4all-simplett
4. tensor4all-tcicore
5. tensor4all-tensorci
6. tensor4all-treetn
7. tensor4all-quanticstransform
8. tensor4all-quanticstci
9. tensor4all-itensorlike
10. tensor4all-hdf5
11. tensor4all-treetci
12. remaining crates (partitionedtt, etc.)
```

## Rustdoc Improvement Standards

### Types (struct/enum/trait)
- **Summary**: What it represents, when to use it (1-2 sentences)
- **Related types**: Relationship to similar types (e.g., "SimpleTT is the simple version; TreeTN is the general version")
- **Examples**: Creation -> basic operation -> result verification with assertions

### Functions/Methods
- **Summary**: What it does (1 sentence)
- **Arguments**: Meaning, constraints, typical values for each parameter
- **Returns**: What is returned, how to use it
- **Panics/Errors**: Under what conditions it fails
- **Examples**: Minimal runnable example with assertions

### Options/Config Types (Critical)
- **Each field**: Meaning and recommended values
- **Field relationships**: Trade-offs (e.g., `rtol` vs `max_bond_dim`)
- **"When in doubt" defaults**: Recommended starting values

### What NOT to Write
- Internal implementation details (algorithm derivations belong in mdBook guides)
- Developer rules already in AGENTS.md

## mdBook Guide Improvement Standards

### Existing Guide Fixes
- **Convert all `rust,ignore` blocks to runnable** — currently 34 occurrences across 8 guides
- **Add assertions** to every code example to verify correctness
- **Add parameter selection guidance** (e.g., "`rtol=1e-8` is general-purpose; `1e-12` for high precision")

### Content Gaps to Fill

| Guide | Missing Content |
|-------|----------------|
| tensor-basics | Index ID matching semantics, Storage type selection |
| tensor-train | End-to-end workflow (create -> compress -> evaluate -> verify accuracy) |
| tci | Parameter selection guidance, convergence diagnostics, continuous vs discrete selection criteria |
| tci-advanced | Initial pivot strategies, batch evaluation |
| compress | Examples of compressible vs incompressible functions, tolerance selection |
| quantics | Apply method (naive/zipup/fit) selection criteria, Steiner tree explanation |
| tree-tn | Actual tree (non-chain) topology example, SimpleTT/TreeTN selection guidance |
| qft | Simpler introduction, output interpretation |

### No New Guides
Prioritize filling gaps in existing guides. Minimize structural changes.

## CI Hardening

### Current State
- `deploy-docs.yml`: `mdbook build` only (no code execution)
- `cargo nextest`: does NOT run doctests
- `cargo test --doc`: status in CI unknown

### Additions

**1. `cargo test --doc --release --workspace`**
- Executes all rustdoc code examples
- Add to test CI workflow (runs on PRs)

**2. `mdbook test docs/book`**
- Executes all mdBook Rust code blocks
- Add to `deploy-docs.yml` before `mdbook build`
- Configure `book.toml` with `[rust]` section for dependency resolution

**3. `book.toml` Changes**
```toml
[rust]
edition = "2021"
```

Code blocks use hidden lines (`# ` prefix) for `use` statements and `fn main()` wrappers (standard mdBook pattern).

## AGENTS.md Update

Expand the existing "Documentation Requirements" section (currently 3 lines) to codify all rules from this design:

- Rustdoc standards for types, functions, Options types
- Code example rules: `ignore`/`no_run` prohibited, all examples must have assertions
- mdBook guide standards: all code blocks runnable with assertions, use hidden lines
- CI verification: `cargo test --doc`, `mdbook test` must pass

This is done as step 0, before any per-crate work.

## Per-Crate Work Procedure

For each crate:
1. Read `docs/api/<crate>.md` and source to understand full public API
2. Improve rustdoc comments (per Rustdoc standards above)
3. Improve corresponding mdBook guide if one exists (per mdBook standards above)
4. Verify: `cargo test --doc -p <crate> --release`
5. Verify: `cargo fmt --all && cargo clippy --workspace`

## Completion Criteria

### Per Crate
- All public types/functions/traits have standard-compliant doc comments
- All doctest examples are runnable (`ignore`/`no_run` prohibited)
- All doctest examples include assertions verifying correctness
- `cargo test --doc -p <crate> --release` passes

### Overall
- mdBook guides have 0 `rust,ignore` blocks
- All mdBook code examples include assertions
- `mdbook test docs/book` passes in CI
- `cargo test --doc --release --workspace` passes in CI

## Out of Scope
- New crate creation
- API design changes (discovered issues filed as GitHub issues)
- C-API crate (`tensor4all-capi`)
- mdBook structural changes (new guides, navigation changes)
