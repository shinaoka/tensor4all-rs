// tensor4all-rs Presentation Slides
// Author: Hiroshi Shinaoka

#set page(
  paper: "presentation-16-9",
  margin: (x: 2cm, y: 1.5cm),
  numbering: "1",
  footer: context [
    #h(1fr)
    #text(size: 14pt, fill: gray)[#counter(page).display()]
  ],
)

#set text(
  font: "Hiragino Sans",
  size: 18pt,
)

#let title-slide(title, subtitle: none, author: none, date: none, url: none) = {
  set align(center + horizon)
  block[
    #text(size: 48pt, weight: "bold")[#title]
    #if subtitle != none {
      v(0.5em)
      text(size: 28pt, fill: gray.darken(20%))[#subtitle]
    }
    #if author != none {
      v(1.5em)
      text(size: 24pt)[#author]
    }
    #if date != none {
      v(0.5em)
      text(size: 20pt, fill: gray)[#date]
    }
    #if url != none {
      v(1em)
      text(size: 16pt, fill: rgb("#2563eb"))[#link(url)[#url]]
    }
  ]
}

#let slide(title, body) = {
  pagebreak()
  block(width: 100%, height: 100%)[
    #text(size: 36pt, weight: "bold", fill: rgb("#2563eb"))[#title]
    #v(0.8em)
    #line(length: 100%, stroke: 0.5pt + rgb("#2563eb"))
    #v(1em)
    #body
  ]
}

#let code-block(code) = {
  block(
    fill: rgb("#f1f5f9"),
    inset: 16pt,
    radius: 6pt,
    width: 100%,
  )[
    #text(font: "Menlo", size: 14pt)[#code]
  ]
}

// Slide 1: Title
#title-slide(
  "tensor4all-rs",
  subtitle: "Tensor Network Computing with Vibe Coding",
  author: "Hiroshi Shinaoka",
  date: "January, 2026",
  url: "https://github.com/tensor4all/tensor4all-rs",
)

// Slide 2: What is Vibe Coding?
#slide("What is Vibe Coding?")[
  #text(size: 22pt)[
    A development style where you code through *intuitive, iterative dialogue with AI*
  ]

  #v(0.5em)

  - Express intent in natural language
  - AI generates and refines code
  - Rapid trial-and-error cycles
  - *Type system catches AI mistakes*

  Rust is ideal: compile-time type checking instantly catches bugs in AI-generated code.
]

// Slide 3: AI Agents I Used
#slide("AI Agents I Used")[
  *Claude Opus 4.5* — One of the best AI coding models

  #v(1em)

  - *Claude Code* (CLI tool by Anthropic) \
    → Focuses on autonomous development

  - *Cursor* (AI-powered IDE) \
    → Better for collaborative editing with human

  #v(1em)

  Both provide agentic coding capabilities: \
  code generation, error fixing, and refactoring
]

// Slide 4: Workflow
#slide("Typical Workflow")[
  + *Analyze*: Ask agent to analyze the original Julia library

  + *Plan*: Agent asks questions and presents a plan \
    → Human approves the plan

  + *Generate*: Agent generates source code and tests \
    → Autonomously fixes compilation and test errors

  + *Review*: Human reviews the code \
    - Focus on global structure and logic. The compiler handles memory safety, lifetimes, etc.

  + *Refactor*: If better ideas emerge, ask agent for large-scale changes \
    → Agent can refactor entire architecture autonomously
]

// Slide 5: Workflow (2) - Issue-based
#slide("Typical Workflow (2): Issue-based")[
  *For larger projects with multiple tasks:*

  #v(0.5em)

  + *Investigate*: Ask agent to analyze codebase and identify issues \
    → Agent creates GitHub issues via `gh` command

  + *Accumulate*: Build up a backlog of well-defined issues

  + *Resolve*: Agent picks and solves issues one by one \
    → Each issue becomes a focused PR

  #v(0.5em)

  *Advantage*: Clear task boundaries, easy to track progress, \
  parallelizable with multiple agents
]

// Slide: Design Philosophy
#slide("Design Philosophy of tensor4all-rs")[
  #v(0.5em)

  - *Modular architecture* \
    Independent crates enable fast compilation and isolated testing

  - *ITensors.jl-like dynamic structure* \
    Flexible Index system preserves intuitive API

  - *Static error detection* \
    Rust's type system catches errors at compile time

  - *Multi-language support* \
    C-API exposes full functionality to Julia and Python
]

// Slide 4: Project Structure
#slide("Project Structure")[
  #text(size: 16pt)[
  #table(
    columns: (auto, auto),
    stroke: 0.5pt,
    [*Crate*], [*Description*],
    [tensor4all-core], [Index, Tensor, SVD, QR, LU],
    [tensor4all-simplett], [Simple TT/MPS with canonical forms],
    [tensor4all-tensorci], [TCI algorithms],
    [tensor4all-quanticstci], [High-level Quantics TCI],
    [tensor4all-capi], [C FFI for language bindings],
    [tensor4all-treetn], [Tree tensor networks (WIP)],
    [tensor4all-itensorlike], [ITensors.jl-like API (WIP)],
    [tensor4all-quanticstransform], [Quantics transformation operators (WIP)],
    [quanticsgrids], [Quantics grid structures],
  )
  ]
  Full functionality implemented in Rust and exposed to other languages via C-API.
]

// Slide 5: Type Correspondence
#slide("Type Correspondence with ITensors.jl")[
  #text(size: 16pt)[
  #table(
    columns: (auto, auto),
    stroke: 0.5pt,
    [*ITensors.jl*], [*tensor4all-rs*],
    [`Index{Int}`], [`Index<Id, NoSymmSpace>`],
    [`ITensor`], [`TensorDynLen<Id, Symm>`],
    [`Dense` / `Diag`], [`Storage::DenseF64` / `DiagF64`],
    [`A * B`], [`a.contract(&b)` or `T::contract(&[a, b])`],
    [`cutoff`], [`rtol` (= √cutoff)],
  )
  ]
  Truncation: $ norm(A - A_"approx")_F / norm(A)_F <= "rtol" $
]

// Slide 6: Index System Comparison
#slide("Index System: ITensors.jl vs QSpace")[
  #text(size: 20pt)[
  #table(
    columns: (auto, auto, auto),
    stroke: 0.5pt,
    inset: 10pt,
    [*Aspect*], [*ITensors.jl*], [*QSpace*],
    [Central entity], [Index], [Tensor],
    [Index identity], [UUID (auto)], [itag name (string)],
    [Connection], [Share same Index], [Same itag + opposite direction],
    [Direction], [Undirected], [Directed (Ket/Bra)],
  )
  ]

  #v(1em)
  tensor4all-rs supports *both* via `IndexLike` trait with `ConjState`
]

// Slide 7: Design Challenge
#slide("Design Challenge: Extensibility")[
  *Initial focus*: Dense tensors without symmetry

  #v(0.5em)

  *Challenge*: Maintain compatibility with different index systems \
  (ITensors.jl, QSpace, etc.) and keep code generic for future extensions

  #v(0.5em)

  *Problem*: A single Index type cannot support all use cases

  #v(0.5em)

  *Solution*: Define minimal trait requirements for Index and Tensor \
  → Tree TN algorithms are generic over any type implementing the traits

  #v(0.5em)

  *Trait*: A set of requirements a type must satisfy (similar to C++ concepts)
]

// Slide 8: IndexLike Trait
#slide("IndexLike Trait")[
  #code-block[
    ```rust
    pub trait IndexLike: Clone + Eq + Hash {
        type Id: Clone + Eq + Hash;

        fn id(&self) -> &Self::Id;
        fn dim(&self) -> usize;
        fn conj_state(&self) -> ConjState;  // Undirected, Ket, Bra
        fn conj(&self) -> Self;
        fn is_contractable(&self, other: &Self) -> bool;
    }
    ```
  ]

  *Important*: `is_contractable()` checks ID, dim, *and* ConjState \
  → Same ID alone does NOT guarantee contractability!
]

// Slide 9: Contractability Rules
#slide("Contractability Rules (`is_contractable`)")[
  `is_contractable(&self, other)` returns `true` iff *all* conditions hold:

  #v(0.5em)

  + Same `id()` — must reference the same logical index
  + Same `dim()` — dimensions must match
  + Compatible `ConjState`:
    - `(Ket, Bra)` or `(Bra, Ket)` → *contractable*
    - `(Undirected, Undirected)` → *contractable*
    - Mixed directed/undirected → *forbidden*
]

// Slide 9: TensorLike Trait
#slide("TensorLike Trait")[
  #text(size: 16pt)[
  #code-block[
    ```rust
    pub trait TensorIndex: Sized + Clone {
        type Index: IndexLike;
        fn external_indices(&self) -> Vec<Self::Index>;
        fn replaceind(&self, old: &Self::Index, new: &Self::Index) -> Result<Self>;
    }

    pub trait TensorLike: TensorIndex {
        fn contract(tensors: &[Self]) -> Result<Self>;  // Auto-contracts via is_contractable
        fn factorize(&self, left_inds: &[Self::Index], options: &FactorizeOptions)
            -> Result<FactorizeResult<Self>>;
        fn conj(&self) -> Self;
        fn norm_squared(&self) -> f64;
        // ... other operations
    }
    ```
  ]
  ]
  Tree TN algorithms are generic over any `TensorLike` implementor.
]

// Slide 10: Contraction Methods
#slide("Contraction Methods")[
  #text(size: 18pt)[
  *Automatic contraction* — uses `is_contractable()` to find pairs:
  #code-block[
    ```rust
    // Contractable index pairs are automatically found and contracted
    let c = T::contract(&[a, b, c])?;
    // Or using * operator for TensorDynLen
    let c = &a * &b;
    ```
  ]

  #v(0.5em)

  *Explicit contraction* — on concrete type (TensorDynLen):
  #code-block[
    ```rust
    // Contract specific indices i from A with j from B
    let c = a.tensordot(&b, &[(i, j)])?;
    ```
  ]
  ]

  #v(0.5em)
  `contract`: generic, auto-matching / `tensordot`: explicit control (concrete type only)
]

// Slide 11: Type Hierarchy Overview
#slide("Type Hierarchy Overview")[
  #text(size: 20pt)[
  ```
  TensorDynLen (implements TensorLike)
      │
      ├── indices: Vec<DynIndex>     ← Index information
      │
      └── data: TensorData           ← Actual tensor data
              │
              └── components: Vec<TensorComponent>
                      │
                      └── storage: Storage   ← Dense/Diag × F64/C64
  ```

  #v(0.8em)

  - *TensorDynLen*: User-facing tensor type (implements `TensorLike`)
  - *TensorData*: Lazy outer product of tensor components
  - *Storage*: Low-level data storage (`mdarray` backend)
  ]
]

// Slide 12: DynIndex
#slide("DynIndex: Default Index Type")[
  #text(size: 18pt)[
  Type alias: `DynIndex = Index<DynId, TagSet>`

  #v(0.5em)

  #code-block[
    ```rust
    pub struct Index<Id, Tags> {
        pub id: Id,      // Unique identifier
        pub dim: usize,  // Dimension
        pub tags: Tags,  // String tags for labeling
    }
    ```
  ]

  #v(0.5em)

  - *DynId*: UUID-based unique identifier (like ITensors.jl)
  - *TagSet*: ITensor-compatible string tags for labeling (e.g., `"Site,n=1"`)
  - Implements `IndexLike` trait
  ]
]

// Slide 13: TensorData
#slide("TensorData: Lazy Outer Products")[
  #text(size: 18pt)[
  Stores tensors as *lazy outer products* of components:
  #code-block[
    ```rust
    pub struct TensorData {
        pub components: Vec<TensorComponent>,  // Storage + indices
        pub external_ids: Vec<DynId>,          // User-facing order
    }
    ```
  ]

  #v(0.5em)

  *Advantages*:
  - Diagonal × Dense = lazy (no memory explosion)
  - Actual expansion only when needed
  - Permutations are tracked, not executed
  ]
]

// Slide 14: Storage
#slide("Storage: Backend Layer")[
  #text(size: 18pt)[
  Defined in `tensor4all-tensorbackend` crate:

  #v(0.5em)

  #code-block[
    ```rust
    pub enum Storage {
        DenseF64(DenseStorageF64),  // Dense real
        DenseC64(DenseStorageC64),  // Dense complex
        DiagF64(DiagStorageF64),    // Diagonal real
        DiagC64(DiagStorageC64),    // Diagonal complex
    }
    ```
  ]

  #v(0.5em)

  *Backend libraries*:
  - `mdarray`: Multi-dimensional array storage
  - `mdarray-linalg`: Linear algebra operations (SVD, QR, LU)
  ]
]

// Slide 15: Contraction Path Optimization
#slide("Contraction Path Optimization")[
  #text(size: 18pt)[
  *Problem*: `TensorData` can contain mixed Dense + Diag components \
  → Need optimal contraction order for the component list

  #v(0.3em)

  *Example* (SVD-like): `U(i,j)` × `s(j)` × `V(j,k)` where `s` is diagonal
  - Index `j` is a *hyperedge* (shared by 3 tensors)
  - Naive order may expand diagonal unnecessarily

  #v(0.3em)

  *Solution*: #link("https://github.com/GiggleLiu/omeco")[omeco] (Rust port of OMEinsumContractionOrders.jl)
  - GreedyMethod: O(n² log n) near-optimal ordering
  - Also supports TreeSA (simulated annealing)

  #v(0.3em)

  *Workflow*: `TensorData.components` → omeco → contraction tree → execute
  ]
]

// Rust Examples (3 slides): Index & Tensor → TensorTrain → TCI
#slide("Rust: Index & Tensor")[
  #code-block[
    ```rust
    use tensor4all_core::{Index, Tensor};

    // Create indices with tags
    let i = Index::new_dyn_with_tag(2, "Site,n=1")?;
    let j = Index::new_dyn_with_tag(3, "Link")?;

    // Create tensors and contract
    let a = Tensor::random(&[i.clone(), j.clone()]);
    let b = Tensor::random(&[j.clone(), k.clone()]);
    let c = &a * &b;  // Contract on shared index j (uses is_contractable)
    ```
  ]
]

#slide("Rust: TensorTrain")[
  #code-block[
    ```rust
    use tensor4all_simplett::{TensorTrain, AbstractTensorTrain};

    // Create a constant tensor train with local dimensions [2, 3, 4]
    let tt = TensorTrain::<f64>::constant(&[2, 3, 4], 1.0);

    // Evaluate at a specific multi-index
    let value = tt.evaluate(&[0, 1, 2])?;

    // Compress with tolerance (rtol=1e-10, maxrank=20)
    let compressed = tt.compressed(1e-10, Some(20))?;
    ```
  ]
]

#slide("Rust: TCI")[
  #code-block[
    ```rust
    use tensor4all_tensorci::{crossinterpolate2, TCI2Options};

    let f = |idx: &Vec<usize>| -> f64 {
        ((1 + idx[0]) * (1 + idx[1]) * (1 + idx[2])) as f64
    };

    let options = TCI2Options { tolerance: 1e-10, ..Default::default() };
    let tensor4all_tensorci::TCI2OptimizationResult { tci, ranks, errors, .. } = crossinterpolate2(
        f, None, vec![4, 4, 4], vec![vec![0, 0, 0]], options)?;
    ```
  ]
]

// Julia Examples (3 slides): Index & Tensor → TensorTrain → TCI
#slide("Julia: Index & Tensor")[
  #code-block[
    ```julia
    using Tensor4all.ITensorLike

    i = Index(2, tags="Site,n=1")
    j = Index(3, tags="Link")
    k = Index(2, tags="Site,n=2")

    t1 = Tensor([i, j], randn(2, 3))
    t2 = Tensor([j, k], randn(3, 2))
    result = contract(t1, t2)  # Contract on shared index j
    ```
  ]
]

#slide("Julia: TensorTrain")[
  #code-block[
    ```julia
    using Tensor4all.SimpleTT

    # Create and manipulate tensor trains
    tt = TensorTrain(tensors)

    # Orthogonalize and truncate
    orthogonalize!(tt, 2)
    truncate!(tt; maxdim=3, rtol=1e-10)

    # Evaluate and sum
    println("Sum: ", sum(tt))
    ```
  ]
]

#slide("Julia: TCI")[
  #code-block[
    ```julia
    using Tensor4all.TensorCI

    f(i, j, k) = Float64((1 + i) * (1 + j) * (1 + k))
    tt, err = crossinterpolate2(f, [4, 4, 4]; tolerance=1e-10)

    println(tt(0, 0, 0))  # 1.0
    println(tt(3, 3, 3))  # 64.0
    ```
  ]
]

// Python Examples (3 slides): Index & Tensor → TensorTrain → TCI
#slide("Python: Index & Tensor")[
  #code-block[
    ```python
    from tensor4all import Index, Tensor
    import numpy as np

    i = Index(2, tags="Site")
    j = Index(3, tags="Link")

    data = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
    t = Tensor([i, j], data)

    arr = t.to_numpy()  # Convert back to NumPy
    ```
  ]
]

#slide("Python: TensorTrain")[
  #code-block[
    ```python
    from tensor4all import TensorTrain

    # Create tensor train
    tt = TensorTrain.constant([2, 3, 4], 1.0)

    # Evaluate and compress
    value = tt.evaluate([0, 1, 2])
    compressed = tt.compressed(rtol=1e-10, maxrank=20)

    print("Sum:", tt.sum())
    ```
  ]
]

#slide("Python: TCI")[
  #code-block[
    ```python
    from tensor4all import crossinterpolate2

    def f(i, j, k):
        return float((1 + i) * (1 + j) * (1 + k))

    tt, err = crossinterpolate2(f, [4, 4, 4], tolerance=1e-10)

    print(tt(0, 0, 0))  # 1.0
    print(tt(3, 3, 3))  # 64.0
    ```
  ]
]

// Slide: Future Extensions
#slide("Future Extensions")[
  - Tree TCI
  - GPU acceleration, Automatic differentiation
  - Quantum number symmetries: U(1), Z_n, SU(2), SU(N)
]
