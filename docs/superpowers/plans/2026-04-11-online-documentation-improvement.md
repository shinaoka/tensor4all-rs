> [!NOTE] Superseded (2026-08, #639)
> `tensor4all-tcicore` was dissolved into `tensor4all-core`; the crate no
> longer exists. References below describe the pre-dissolution state.

# Online Documentation Improvement Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Improve rustdoc and mdBook documentation so humans and AI agents can understand usage without reading source code.

**Architecture:** Bottom-up approach starting from foundational crates. Each crate gets rustdoc comment improvements + corresponding mdBook guide fixes. CI enforces all code examples are runnable with assertions.

**Tech Stack:** Rust doc comments, mdBook, GitHub Actions CI

**Spec:** `docs/superpowers/specs/2026-04-11-online-documentation-improvement-design.md`

---

## Dependency Order (regular deps only, excluding C-API)

```
tensor4all-tensorbackend  (0 internal deps)
tensor4all-tcicore        (0 internal deps)
tensor4all-core           (depends on: tensorbackend, tcicore)
tensor4all-simplett       (depends on: core, tcicore, tensorbackend)
tensor4all-tensorci       (depends on: simplett, tcicore)
tensor4all-treetn         (depends on: core, simplett)
tensor4all-quanticstransform (depends on: core, simplett, treetn)
tensor4all-itensorlike    (depends on: core, treetn)
tensor4all-quanticstci    (depends on: core, simplett, tcicore, treetci, treetn)
tensor4all-treetci        (depends on: core, tcicore, treetn)
tensor4all-hdf5           (depends on: core, itensorlike)
tensor4all-partitionedtt  (depends on: core, itensorlike)
```

## Crate-to-Guide Mapping

| Crate | mdBook Guide |
|-------|-------------|
| tensor4all-core | `guides/tensor-basics.md` |
| tensor4all-simplett | `guides/tensor-train.md`, `guides/compress.md` |
| tensor4all-tcicore | `guides/tci-advanced.md` (CachedFunction) |
| tensor4all-tensorci | `guides/tci.md` (crossinterpolate2 section) |
| tensor4all-treetn | `guides/tree-tn.md` |
| tensor4all-quanticstransform | `guides/quantics.md`, `guides/qft.md` |
| tensor4all-quanticstci | `guides/tci.md` (quanticscrossinterpolate section) |
| tensor4all-itensorlike | `guides/tensor-train.md` (ITensorLike section) |
| tensor4all-tensorbackend | (no guide - internal crate) |
| tensor4all-treetci | (no guide yet) |
| tensor4all-hdf5 | (no guide yet) |
| tensor4all-partitionedtt | (no guide yet) |

---

### Task 1: Update AGENTS.md Documentation Requirements

**Files:**
- Modify: `AGENTS.md:37-42`

- [ ] **Step 1: Replace the Documentation Requirements section**

Replace lines 37-42 in `AGENTS.md` with the expanded documentation rules:

```markdown
## Documentation Requirements

### Rustdoc Standards

Every public type, trait, and function **must** have doc comments with the following:

**Types (struct/enum/trait):**
- Summary: what it represents, when to use it (1-2 sentences)
- Related types: relationship to similar types (e.g., "`TensorTrain` is the simple chain version; `TreeTN` is the general tree version")
- `# Examples` section with runnable code and assertions

**Functions/methods:**
- Summary: what it does (1 sentence)
- Arguments: meaning, constraints, typical values for each parameter (especially for `Options` types)
- Returns: what is returned, how to use it
- `# Panics` or `# Errors`: under what conditions it fails
- `# Examples` section with runnable code and assertions

**Options/Config types (critical for usability):**
- Each field: meaning, recommended values, and default behavior
- Field relationships and trade-offs (e.g., `rtol` vs `max_bond_dim`)
- "When in doubt" defaults

### Code Example Rules

- All doc examples **must** be runnable (`ignore` and `no_run` attributes are **prohibited**)
- All doc examples **must** include assertions verifying correctness (not just compilation/execution)
  - Use `assert!`, `assert_eq!`, `approx::assert_abs_diff_eq!`, etc.
- mdBook guide code blocks follow the same rules: runnable with assertions
- mdBook code blocks use hidden lines (`# ` prefix) for `use` statements and `fn main()` wrappers

### CI Verification

- `cargo test --doc --release --workspace` must pass (rustdoc examples)
- `mdbook test docs/book` must pass (mdBook guide examples)
```

- [ ] **Step 2: Verify AGENTS.md is well-formed**

Run: `cat AGENTS.md | head -60`
Expected: The new Documentation Requirements section is properly formatted

- [ ] **Step 3: Commit**

```bash
git add AGENTS.md
git commit -m "docs: expand Documentation Requirements in AGENTS.md

Codify rustdoc standards, code example rules (no ignore/no_run,
assertions required), and CI verification requirements."
```

---

### Task 2: Add mdbook test to CI

**Files:**
- Modify: `docs/book/book.toml`
- Modify: `.github/workflows/deploy-docs.yml`
- Modify: `.github/workflows/CI_rs.yml`

**Context:** Currently `mdbook test` fails because code blocks can't resolve crate imports.
mdBook's `mdbook test` passes `--library-path` and `--edition` from `book.toml`'s `[rust]` section,
but it uses `rustdoc --test` under the hood, which needs `--extern` flags and library paths.
The practical approach is to run `mdbook test` with the library path pointing to the release build output.

- [ ] **Step 1: Update book.toml**

Add the `[rust]` section to `docs/book/book.toml`:

```toml
[rust]
edition = "2021"
```

- [ ] **Step 2: Add mdbook test to deploy-docs.yml**

In `.github/workflows/deploy-docs.yml`, add a test step after the Rust setup and before the mdBook build.
The key is to build the workspace first so libraries are available, then run `mdbook test` with the library path:

```yaml
      - name: Build workspace (for mdbook test)
        run: cargo build --release --workspace

      - name: Test mdBook examples
        run: mdbook test docs/book -L target/release/deps
```

Insert these steps after the "Setup Rust" step and before the "Build mdBook" step.

- [ ] **Step 3: Add doctest and mdbook test to CI_rs.yml**

Add a new job to `.github/workflows/CI_rs.yml` that runs both `cargo test --doc` and `mdbook test`:

```yaml
  doctest:
    name: Doctests
    runs-on: ubuntu-22.04
    steps:
      - uses: actions/checkout@v4

      - name: Install HDF5
        run: sudo apt-get update && sudo apt-get install -y libhdf5-dev

      - name: Install Rust
        uses: dtolnay/rust-toolchain@stable

      - uses: Swatinem/rust-cache@v2

      - name: Install mdBook
        run: cargo install mdbook --version 0.5.2 --locked

      - name: Build workspace
        run: cargo build --release --workspace

      - name: Run rustdoc tests
        run: cargo test --doc --release --workspace

      - name: Run mdBook tests
        run: mdbook test docs/book -L target/release/deps
```

Also add `doctest` to the `needs` list in the `rollup-rs` job.

- [ ] **Step 4: Test locally**

Run:
```bash
cargo build --release --workspace
mdbook test docs/book -L target/release/deps
```

Expected: Currently fails on `rust,ignore` blocks (they are skipped) and non-ignore blocks that exist in `getting-started.md` and `tensor-basics.md` should pass.

Note: `rust,ignore` blocks are skipped by `mdbook test`, so existing guides won't break CI.
As we convert `ignore` blocks to runnable in later tasks, they will start being tested.

- [ ] **Step 5: Commit**

```bash
git add docs/book/book.toml .github/workflows/deploy-docs.yml .github/workflows/CI_rs.yml
git commit -m "ci: add doctest and mdbook test to CI

- Add cargo test --doc to verify rustdoc examples
- Add mdbook test to verify mdBook guide examples
- Build workspace first so mdbook test can resolve crate imports"
```

---

### Task 3-N: Per-Crate Documentation Improvement

Each crate follows the same procedure. The work order is:

1. `tensor4all-tensorbackend` (internal, minimal public API - likely quick)
2. `tensor4all-tcicore`
3. `tensor4all-core`
4. `tensor4all-simplett`
5. `tensor4all-tensorci`
6. `tensor4all-treetn`
7. `tensor4all-quanticstransform`
8. `tensor4all-itensorlike`
9. `tensor4all-quanticstci`
10. `tensor4all-treetci`
11. `tensor4all-hdf5`
12. `tensor4all-partitionedtt`

#### Per-Crate Procedure

For each crate `<crate>`:

- [ ] **Step A: Audit public API**

Run:
```bash
cargo run -p api-dump --release -- . -o docs/api
```

Read `docs/api/<crate>.md` to identify all public types, functions, and traits.
For each item, classify documentation status:
- OK: has summary + params + errors + runnable example with assertions
- Needs improvement: has some docs but missing elements
- Missing: no doc comment or only a one-liner

- [ ] **Step B: Improve rustdoc comments**

For each item needing improvement, add/expand doc comments per AGENTS.md standards:
- Summary, arguments, returns, panics/errors
- Runnable `# Examples` with assertions
- For Options types: field meanings, recommended values, trade-offs

- [ ] **Step C: Verify rustdoc examples**

Run:
```bash
cargo test --doc --release -p <crate>
```

Expected: All doctests pass, 0 ignored.

- [ ] **Step D: Improve mdBook guide (if applicable)**

For the corresponding mdBook guide (see Crate-to-Guide Mapping above):
1. Convert all `rust,ignore` blocks to runnable code
2. Add hidden lines (`# ` prefix) for `use` statements and `fn main()` wrapper
3. Add assertions to verify results
4. Add missing content per the spec (parameter guidance, workflow examples, selection criteria)

- [ ] **Step E: Verify mdBook examples**

Run:
```bash
cargo build --release --workspace
mdbook test docs/book -L target/release/deps
```

Expected: All code blocks in the modified guide pass.

- [ ] **Step F: Lint and format**

Run:
```bash
cargo fmt --all
cargo clippy --workspace
```

Expected: No warnings or errors.

- [ ] **Step G: Commit**

```bash
git add -A
git commit -m "docs(<crate>): improve rustdoc and mdBook documentation

- Expand doc comments for all public API items
- Add runnable examples with assertions
- Convert mdBook guide code blocks from ignore to runnable"
```

#### Guide-Specific Content Additions

When working on each crate's mdBook guide, add the following missing content:

**tensor-basics.md** (tensor4all-core):
- Index ID matching semantics: explain that contraction matches axes by Index identity (UUID-based), not by dimension or position
- Storage type selection: when to use dense vs diagonal tensors
- Factorization selection guide: SVD (truncation, singular values) vs QR (fast, no truncation) vs LU vs CI

**tensor-train.md** (tensor4all-simplett, tensor4all-itensorlike):
- End-to-end workflow: create TT -> compress -> evaluate -> verify accuracy
- Bond dimension guidance: how tolerance affects bond dim, typical values
- SimpleTT vs ITensorLike selection criteria
- ITensorLike column-major convention explanation

**tci.md** (tensor4all-tensorci, tensor4all-quanticstci):
- Parameter selection: `tolerance`, `max_bond_dim`, `nrandominitpivot` guidance
- Convergence diagnostics: how to read the errors vector, detect stalled convergence
- `quanticscrossinterpolate` vs `quanticscrossinterpolate_discrete` selection criteria
- Index convention highlight: 0-indexed (low-level) vs 1-indexed (quantics grid)

**tci-advanced.md** (tensor4all-tcicore):
- Initial pivot strategies with concrete examples
- CachedFunction usage and performance implications
- Batch evaluation patterns

**compress.md** (tensor4all-simplett):
- Compressible vs incompressible function examples
- Tolerance selection guidance with accuracy/cost trade-off
- Bond dimension interpretation

**quantics.md** (tensor4all-quanticstransform):
- Apply method selection: naive (simple), zipup (fast), fit (best compression) - when to use each
- Steiner tree explanation with diagram
- Multi-variable encoding motivation

**tree-tn.md** (tensor4all-treetn):
- Non-chain topology example (e.g., star or Y-shape tree)
- SimpleTT vs TreeTN selection guide
- When tree structure is needed vs chain

**qft.md** (tensor4all-quanticstransform):
- Simpler 1D introduction before the advanced partial apply example
- Output interpretation: what the QFT result represents, how to read frequencies
- Bit-reversal behavior explanation

---

### Final Task: Verify Full CI Pipeline

- [ ] **Step 1: Run full doctest suite**

```bash
cargo test --doc --release --workspace
```

Expected: All pass, 0 ignored.

- [ ] **Step 2: Run full mdBook test**

```bash
cargo build --release --workspace
mdbook test docs/book -L target/release/deps
```

Expected: All pass, 0 `rust,ignore` blocks remain.

- [ ] **Step 3: Verify no rust,ignore remains in guides**

```bash
grep -r "rust,ignore\|rust,no_run" docs/book/src/
```

Expected: No matches.

- [ ] **Step 4: Run full lint suite**

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo doc --workspace --no-deps
```

Expected: All pass.
