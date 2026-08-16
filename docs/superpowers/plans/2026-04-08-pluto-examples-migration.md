> [!NOTE] Superseded (2026-08, #639)
> `tensor4all-tcicore` was dissolved into `tensor4all-core`; the crate no
> longer exists. References below describe the pre-dissolution state.

# Pluto Examples Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port T4APlutoExamples content to tensor4all-rs mdBook guides with Rust code examples that include assertions verifying computational correctness.

**Architecture:** Enrich existing `guides/tci.md` with practical examples, add 3 new guide pages (`tci-advanced.md`, `compress.md`, `qft.md`), and update `SUMMARY.md`. All code examples use `rust,ignore` blocks with assertions that verify interpolation accuracy, integral correctness, and transform results.

**Tech Stack:** mdBook, tensor4all-quanticstci, tensor4all-tensorci, tensor4all-tcicore, tensor4all-quanticstransform, tensor4all-treetn, tensor4all-simplett

**Spec:** `docs/superpowers/specs/2026-04-08-pluto-examples-migration-design.md`

**Worktree:** `../tensor4all-rs-pluto-examples` (branch: `feat/pluto-examples-migration`)

---

## Task 1: Update SUMMARY.md with New Pages

**Files:**
- Modify: `docs/book/src/SUMMARY.md`

- [ ] **Step 1: Add new pages to SUMMARY**

Replace the Guides section in `docs/book/src/SUMMARY.md` with:

```markdown
- [Guides]()
  - [Tensor Basics](guides/tensor-basics.md)
  - [Tensor Train](guides/tensor-train.md)
  - [Tensor Cross Interpolation](guides/tci.md)
  - [TCI Advanced Topics](guides/tci-advanced.md)
  - [Compressing Existing Data](guides/compress.md)
  - [Quantics Transform](guides/quantics.md)
  - [Quantum Fourier Transform](guides/qft.md)
  - [Tree Tensor Networks](guides/tree-tn.md)
```

- [ ] **Step 2: Create stub files**

```bash
touch docs/book/src/guides/tci-advanced.md
touch docs/book/src/guides/compress.md
touch docs/book/src/guides/qft.md
```

Write a single heading in each stub:
- `# TCI Advanced Topics`
- `# Compressing Existing Data`
- `# Quantum Fourier Transform`

- [ ] **Step 3: Verify mdbook builds**

```bash
mdbook build docs/book
```

- [ ] **Step 4: Commit**

```bash
git add docs/book/
git commit -m "docs(book): add stub pages for TCI advanced, compress, QFT"
```

---

## Task 2: Enrich guides/tci.md with Practical Examples

**Files:**
- Modify: `docs/book/src/guides/tci.md`

This task enriches the existing TCI guide with practical examples corresponding to quantics1d.jl and quantics2d.jl. Each section starts with a link to the corresponding Pluto example.

- [ ] **Step 1: Rewrite the Low-Level TCI section**

Replace the existing Low-Level TCI example in `docs/book/src/guides/tci.md` with a more practical example. The section should include:

1. A link: `> This section corresponds to the Julia [Quantics TCI of univariate function](https://tensor4all.org/T4APlutoExamples/quantics1d.html) notebook.`

2. The existing example with improved assertions (already done in prior commit — keep as is).

3. After the existing example, add a new subsection **"Practical Example: Multi-scale Function"** with this code:

```rust,ignore
use tensor4all_quanticstci::{
    quanticscrossinterpolate, DiscretizedGrid, QtciOptions,
};

// Multi-scale function: has structure at scale ~2^(-30)
let b: f64 = 2.0_f64.powi(-30);
let f = |x: &[f64]| -> f64 {
    let xv = x[0];
    (xv / b).cos() * (xv / (4.0 * 5.0_f64.sqrt() * b)).cos() * (-xv * xv).exp()
        + 2.0 * (-xv).exp()
};

// 2^40 grid points on [0, ln(20)] — captures the fine structure
let r = 40;
let grid = DiscretizedGrid::builder(&[r])
    .with_lower_bound(&[0.0])
    .with_upper_bound(&[20.0_f64.ln()])
    .build()?;

let (qtci, _ranks, errors) = quanticscrossinterpolate(
    &grid,
    f,
    None,
    QtciOptions::default()
        .with_tolerance(1e-8)
        .with_max_bond_dim(20),
)?;

// Verify convergence
assert!(*errors.last().unwrap() < 1e-8);

// Verify interpolation accuracy at several points (1-indexed grid)
for grid_idx in [1, 100, 10000, 1_000_000] {
    let val = qtci.evaluate(&[grid_idx])?;
    // Convert grid index to coordinate for reference
    let x = (grid_idx as f64 - 1.0) / (2.0_f64.powi(r as i32)) * 20.0_f64.ln();
    let exact = f(&[x]);
    assert!((val - exact).abs() < 1e-6, "mismatch at grid_idx={grid_idx}");
}

// Compute integral: ∫₀^{ln20} f(x) dx ≈ 19/10
let integral = qtci.integral()?;
assert!((integral - 1.9).abs() < 1e-4, "integral={integral}, expected≈1.9");
```

- [ ] **Step 2: Add a 2D multivariate example section**

After the 1D practical example, add a new section **"Multivariate (2D) Example"** with a link:
`> This section corresponds to the Julia [Quantics TCI of multivariate function](https://tensor4all.org/T4APlutoExamples/quantics2d.html) notebook.`

```rust,ignore
use tensor4all_quanticstci::{
    quanticscrossinterpolate_discrete, QtciOptions,
};

// 2D function on a 256 × 256 grid (2^8 per dimension)
let f = |idx: &[i64]| -> f64 {
    let x = idx[0] as f64;
    let y = idx[1] as f64;
    (-0.01 * (x * x + y * y)).exp() + (0.1 * x * y).sin()
};

let sizes = vec![256, 256];
let (qtci, _ranks, errors) = quanticscrossinterpolate_discrete(
    &sizes,
    f,
    None,
    QtciOptions::default().with_tolerance(1e-8),
)?;

// Verify convergence
assert!(*errors.last().unwrap() < 1e-8);

// Verify accuracy at multiple points (1-indexed)
for (ix, iy) in [(1, 1), (50, 100), (128, 128), (200, 256)] {
    let val = qtci.evaluate(&[ix, iy])?;
    let exact = f(&[ix as i64, iy as i64]);
    assert!((val - exact).abs() < 1e-6, "mismatch at ({ix},{iy})");
}
```

- [ ] **Step 3: Verify mdbook builds**

```bash
mdbook build docs/book
```

- [ ] **Step 4: Commit**

```bash
git add docs/book/src/guides/tci.md
git commit -m "docs(book): enrich TCI guide with 1D/2D practical examples"
```

---

## Task 3: Write guides/tci-advanced.md

**Files:**
- Modify: `docs/book/src/guides/tci-advanced.md`

- [ ] **Step 1: Write the full page**

Replace the stub with content covering these sections:

**Header:**
```markdown
# TCI Advanced Topics

> This guide corresponds to the Julia
> [Quantics TCI of univariate function (advanced topics)](https://tensor4all.org/T4APlutoExamples/quantics1d_advanced.html)
> notebook.

This guide covers low-level TCI usage: direct `crossinterpolate2` calls,
initial pivot selection, cached function evaluation, and manual integral
computation.
```

**Section 1: Direct crossinterpolate2 Usage**

Show how to bypass the high-level `quanticscrossinterpolate` wrapper and use
`crossinterpolate2` directly. The worker must:

1. Read the `crossinterpolate2` signature (file: `crates/tensor4all-tensorci/src/tensorci2.rs:837`)
2. Write an example that:
   - Defines a function `f: |idx: &Vec<usize>| -> f64`
   - Creates local_dims for a quantics grid (e.g., `vec![2; 40]` for R=40 binary sites)
   - Chooses initial pivots where `|f|` is large
   - Calls `crossinterpolate2` with `TCI2Options`
   - Converts to tensor train via `.to_tensor_train()?`
   - Evaluates and asserts accuracy
   - Computes integral as `tt.sum() * dx` and asserts accuracy

**Section 2: Initial Pivot Selection**

Explain why choosing points with large |f(x)| improves convergence.
Show computing the function at a few candidate points and selecting the maximum.

**Section 3: CachedFunction**

Show how to use `CachedFunction` from `tensor4all-tcicore`:

```rust,ignore
use tensor4all_tcicore::CachedFunction;

let local_dims = vec![2usize; 40];
let cf = CachedFunction::new(
    |idx: &Vec<usize>| { /* compute f at quantics index */ },
    &local_dims,
)?;

// After TCI, check how many evaluations were needed
println!("Cache size: {}", cf.cache_size());
```

**Section 4: Manual Integral Computation**

Show: `integral = tt.sum() * (x_max - x_min) / 2^R`
Assert against known analytic value.

All code blocks use `rust,ignore`. All examples include assertions.

- [ ] **Step 2: Verify mdbook builds**

```bash
mdbook build docs/book
```

- [ ] **Step 3: Commit**

```bash
git add docs/book/src/guides/tci-advanced.md
git commit -m "docs(book): write TCI advanced topics guide"
```

---

## Task 4: Write guides/compress.md

**Files:**
- Modify: `docs/book/src/guides/compress.md`

- [ ] **Step 1: Write the full page**

Replace the stub with content covering:

**Header:**
```markdown
# Compressing Existing Data

> This guide corresponds to the Julia
> [Compressing existing data](https://tensor4all.org/T4APlutoExamples/compress.html)
> notebook.

TCI can compress existing multi-dimensional data by treating element access as a
function evaluation. This avoids materializing the full tensor while achieving
controlled approximation error.
```

**Section 1: Problem Setup**

Explain: you have a 3D array `data[i][j][k]` and want to compress it.
Create a synthetic dataset:

```rust,ignore
// Synthetic 3D dataset: cos(x) + cos(y) + cos(z)
// Grid: 128 × 128 × 128 (2^7 per dimension)
let n = 128usize;
let data_fn = |idx: &Vec<usize>| -> f64 {
    let x = idx[0] as f64 * std::f64::consts::PI / n as f64;
    let y = idx[1] as f64 * std::f64::consts::PI / n as f64;
    let z = idx[2] as f64 * std::f64::consts::PI / n as f64;
    x.cos() + y.cos() + z.cos()
};
```

**Section 2: TCI Compression**

```rust,ignore
use tensor4all_tensorci::{crossinterpolate2, TCI2Options};
use tensor4all_simplett::AbstractTensorTrain;

let local_dims = vec![128, 128, 128];
let initial_pivots = vec![vec![0, 0, 0]];

let (tci, _ranks, errors) = crossinterpolate2::<f64, _, fn(&[Vec<usize>]) -> Vec<f64>>(
    data_fn,
    None,
    local_dims,
    initial_pivots,
    TCI2Options { tolerance: 1e-10, ..Default::default() },
)?;

// Verify convergence
assert!(*errors.last().unwrap() < 1e-10);

// Convert to tensor train
let tt = tci.to_tensor_train()?;

// Verify accuracy at multiple random points
for (i, j, k) in [(0,0,0), (64,64,64), (127,0,127), (33,99,7)] {
    let compressed = tt.evaluate(&[i, j, k])?;
    let exact = data_fn(&vec![i, j, k]);
    assert!((compressed - exact).abs() < 1e-8,
        "mismatch at ({i},{j},{k}): got {compressed}, expected {exact}");
}
```

**Section 3: Inspecting Compression Quality**

Show bond dimensions and compression ratio.

- [ ] **Step 2: Verify mdbook builds**

```bash
mdbook build docs/book
```

- [ ] **Step 3: Commit**

```bash
git add docs/book/src/guides/compress.md
git commit -m "docs(book): write data compression guide"
```

---

## Task 5: Write guides/qft.md

**Files:**
- Modify: `docs/book/src/guides/qft.md`

This is the most complex task. The worker must read actual source code to understand
the API for Fourier operators, TreeTN construction, and partial apply.

- [ ] **Step 1: Write the full page**

Replace the stub with content covering:

**Header:**
```markdown
# Quantum Fourier Transform

> This guide corresponds to the Julia
> [Quantum Fourier Transform](https://tensor4all.org/T4APlutoExamples/qft.html)
> notebook.

The Quantics Fourier Transform (QFT) operates on functions represented as
Quantics Tensor Trains (QTT). tensor4all-rs provides `quantics_fourier_operator`
to construct the Fourier operator as a `LinearOperator`, which can be applied to
a `TreeTN` via `apply_linear_operator`.
```

**Section 1: 1D QFT**

The worker must:

1. Read `quantics_fourier_operator` (file: `crates/tensor4all-quanticstransform/src/fourier.rs:143`)
2. Read `apply_linear_operator` (file: `crates/tensor4all-treetn/src/operator/apply.rs:139`)
3. Read how to convert `QuanticsTensorCI2` → `TreeTN`:
   - `qtci.tci()` returns `&TreeTCI2<V>`
   - `tensor4all_treetci::materialize::to_treetn(tci, batch_eval, center)` returns `TreeTN<IdxTensor, usize>`
4. Read `FourierOptions` (file: `crates/tensor4all-quanticstransform/src/fourier.rs:22`)

Write an example that:
- Defines a simple 1D function (e.g., `f(x) = exp(-a*x)` on [0, 1))
- Constructs QTT via `quanticscrossinterpolate`
- Converts to `TreeTN` via the `tci()` → `to_treetn()` path
- Builds Fourier operator via `quantics_fourier_operator(r, FourierOptions::default())`
- Applies operator via `apply_linear_operator(&fourier_op, &state_treetn, ApplyOptions::default())`
- Normalizes result by `1/sqrt(2^R)` if `normalize: true`
- Verifies specific Fourier coefficients against the analytic DFT:
  For `f(x) = exp(-a*x)`, the DFT coefficient at frequency k is:
  `F[k] = sum_{n=0}^{N-1} f(n/N) * exp(-2πi*k*n/N)` (but we work with real functions so check magnitude)

**Assertions must verify that the Fourier transform produces correct results.**

**Section 2: 2D QFT via Partial Apply**

Explain the approach:
- 2D function in interleaved quantics encoding: sites [x₁,y₁,x₂,y₂,...,xᵣ,yᵣ]
- 1D Fourier operator acts on subset of sites
- `apply_linear_operator` supports partial apply (auto-extends with identity)

The worker must:
1. Read `test_apply_linear_operator_partial` (file: `crates/tensor4all-treetn/src/operator/apply/tests/mod.rs:186`) to understand partial apply pattern
2. Understand how `LinearOperator` nodes map to `TreeTN` nodes

Write an example or explain the approach with code outline. Include a note:
```markdown
> **Note:** A dedicated multivar Fourier API is not yet available. This example
> uses partial apply to achieve 2D transforms manually.
```

**Assert correctness of 2D Fourier result at specific frequency pairs.**

- [ ] **Step 2: Verify mdbook builds**

```bash
mdbook build docs/book
```

- [ ] **Step 3: Commit**

```bash
git add docs/book/src/guides/qft.md
git commit -m "docs(book): write QFT guide with 1D and manual 2D examples"
```

---

## Task 6: Final Verification

- [ ] **Step 1: Build and check everything**

```bash
mdbook build docs/book
cargo doc --workspace --no-deps
```

Both must succeed.

- [ ] **Step 2: Verify all new pages are linked**

Open `docs/book/book/index.html` and verify:
- TCI Advanced Topics appears in sidebar
- Compressing Existing Data appears in sidebar
- Quantum Fourier Transform appears in sidebar
- All links work

- [ ] **Step 3: Check Pluto example links**

Each guide should have a link to the corresponding Pluto notebook at the top.
Verify these URLs are correct:
- `https://tensor4all.org/T4APlutoExamples/quantics1d.html`
- `https://tensor4all.org/T4APlutoExamples/quantics1d_advanced.html`
- `https://tensor4all.org/T4APlutoExamples/compress.html`
- `https://tensor4all.org/T4APlutoExamples/qft.html`
- `https://tensor4all.org/T4APlutoExamples/quantics2d.html`

- [ ] **Step 4: Commit if any fixes needed**

```bash
git add -A
git commit -m "docs(book): final verification and fixes"
```
