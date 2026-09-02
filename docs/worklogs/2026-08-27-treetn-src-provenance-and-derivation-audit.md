# TreeTN SRC provenance and derivation audit

Date: 2026-08-27
Worktree: `.worktrees/treetn-src`
Branch: `feature/treetn-src`
Base implementation commit before this audit: `4167f3c4dde7552d95f1dbfa2a143cbdf841a132`

## Scope and audit rule

This audit covers the SRC implementation and all support code added for its
execution path. The question is not whether the code produces plausible
answers, but whether each non-trivial code fragment has a traceable basis.
Every row below is classified as one of:

- `[Paper]`: directly specified by the cited equation, section, theorem, or
  Algorithm 1 in the paper.
- `[Author]`: directly cross-checked against the named function or code block in
  the author's repository.
- `[Derived]`: manually derived from the paper's tensor identities or linear
  algebra, with the derivation written out in this document.
- `[Repo]`: an existing tensor4all-rs abstraction or invariant used as the
  implementation seam.
- `[AI-Supplied]`: no source-level basis was found. This is an engineering
  choice or a tree generalization and must not be described as a literal port.

The primary sources are:

1. Camaño, Epperly, and Tropp, “Successive randomized compression: A
   randomized algorithm for the compressed MPO-MPS product,” arXiv:2504.06475.
   The exact source snapshot was inspected. Relevant locations are Section 2.2
   and the Khatri--Rao definition, Algorithm 1, the exact-recovery theorem and
   complexity paragraph, and Appendices C--D.
   <https://arxiv.org/abs/2504.06475>
2. `chriscamano/RandomMPOMPS`, commit
   `fe6ad494fc6f3605fc3963360f626d83f47bc2ce`:
   `code/tensornetwork/contraction.py` (`random_contraction` and
   `random_contraction_inc`), `incrementalqr.py` (`IncrementalQR`), and
   `incrementalqr.cpp` (`setup`, `add_cols`, `get_error_estimate`).
   <https://github.com/chriscamano/RandomMPOMPS>
3. tensor4all-rs issue #563 and maintainer clarification comment
   `5396107820`. The clarification requires factorized MPO--MPO probes: do not
   first form a fused local physical dimension `d^2`.
   <https://github.com/tensor4all/tensor4all-rs/issues/563>

No source text was copied into the Rust implementation. The author repository
was used for exact control-flow and storage-layout comparison. Scientific
citations in this file and the module-level provenance comments are references,
not a claim that the tree code exists in the author repository.

For reproducibility, the inspected paper source snapshot has these concrete
locations in `report.tex`: Khatri--Rao construction, lines 283--314; chain
derivation, lines 351--530; Algorithm 1, lines 638--670; exact-recovery theorem
and complexity, lines 674--700; exact-recovery proof, lines 1102--1143;
adaptive estimator and adaptive algorithm, lines 1146--1282; and incremental QR
update, lines 1288--1355. In the author checkout, the corresponding concrete
locations are `contraction.py:82-353` (`random_contraction`),
`contraction.py:357-593` (`random_contraction_inc`),
`incrementalqr.py:50-175` (`IncrementalQR`), and
`incrementalqr.cpp:21-119` (`setup`, `add_cols`, and
`get_error_estimate`).

## Paper algorithm, reconstructed by hand

### Factorized MPO--MPO probe

At a site, let the local product be

\[
C^{s,t}=\sum_u A^{s,u}B^{u,t}.
\]

For independent probe vectors `x` and `y`, define the factorized probe
`z_{s,t,k}=x_{s,k}y_{t,k}`. Then, for one column `k`,

\[
\begin{aligned}
\langle C,z_k\rangle
 &=\sum_{s,t} C^{s,t}x_{s,k}y_{t,k}\\
 &=\sum_{s,t,u} A^{s,u}B^{u,t}x_{s,k}y_{t,k}\\
 &=\sum_u\left(\sum_s A^{s,u}x_{s,k}\right)
             \left(\sum_t B^{u,t}y_{t,k}\right).
\end{aligned}
\]

Thus each operand may be contracted with its own one-index probe before the
shared physical/virtual legs are contracted. This is exactly the issue #563
requirement and is the reason the implementation does not materialize a local
`d^2` tensor. For a batch, the batch index `k` is retained, so the expression is
evaluated independently for each `k`; summing the batch axis would incorrectly
mix independent probes.

The paper's sketch is `A Omega`, not `A conj(Omega)`. Conjugation appears in
the projection `Q^\dagger A` and in the estimator. This distinction motivated
the removal of the unused helpers that conjugated the probe vectors: they had
no production caller and did not implement the paper's sketch convention.

### Chain recurrence

Let `C_j` be the local MPO--MPO product and let `ω_j` be the factorized probe
over the surviving physical indices at site `j`. Define the probed prefix

\[
P_1=C_1\,\omega_1,\qquad
P_j=P_{j-1}\,C_j\,\omega_j.
\]

For the last site, the paper forms

\[
Y_n=P_{n-1}C_n,\qquad Y_n=Q_nR_n,
\]

and projects the carry into the next contraction as `S_n=Q_n^\dagger C_n`.
For an internal site, the right environment is already available and the
corresponding equations are

\[
Y_j=P_{j-1}C_jS_{j+1},\qquad
Y_j=Q_jR_j,\qquad
S_j=Q_j^\dagger C_jS_{j+1}.
\]

The first factor is `C_1 S_2`. To verify the recurrence, insert the identity
`Q_jQ_j^\dagger` at each cut:

\[
Y_j \approx Q_jQ_j^\dagger Y_j.
\]

If `Q_j` has all columns, this is equality because `Y_j=Q_jR_j` and
`Q_j^\dagger Q_j=I`. Replacing `Y_j` by the selected QR columns is the only
compression step; the subsequent projected carry is exactly the contraction of
the original local factor with `Q_j^\dagger`. This is the manual reason that
the right-to-left schedule in the Rust chain code matches Algorithm 1 rather
than being a generic fit or zip-up implementation.

### Rooted-tree recurrence

The paper only specifies the chain/MPO--MPS case. The following is the manual
extension used for TreeTN and is therefore explicitly `[AI-Supplied]`.

Root the input tree at the requested center. For a directed edge `v -> p`, let
`U_v` be the contraction of the local product at `v` with all already projected
child messages. Let `E_{v->p}` be the contraction of everything outside the
subtree rooted at `v`, with all outside physical legs probed. The edge sketch is

\[
Y_{v\to p}=U_vE_{v\to p},\qquad
Y_{v\to p}=Q_{v\to p}R_{v\to p},
\]

and the projected message sent to the parent is

\[
S_{v\to p}=Q_{v\to p}^{\dagger}U_v.
\]

The upward postorder is valid because every child message is available before
assembling `U_v`. The reverse postorder is valid for complement environments:
at `p`, the message from the parent side and all sibling messages are available
before constructing `E_{v->p}`. At the root, all projected child messages are
contracted with the root local product. This is just associativity of tensor
contraction plus the same `Q Q^\dagger` insertion used in the chain proof.

The repository's `edges_to_canonicalize_by_names(center)` supplies the
child-to-parent postorder; the source for that traversal is
`crates/tensor4all-treetn/src/node_name_network.rs`, not the paper or the
author repository.

### Incremental QR update

Suppose the current sketch is `A=QR` and a block `B` is appended. Applying the
old `Q^\dagger` gives

\[
Q^\dagger B=\begin{bmatrix}C\\D\end{bmatrix}.
\]

Factor the residual with `D=Q_2R_2`. In the enlarged orthogonal basis, the
factorization is

\[
[A\ B]
 = [Q\ Q_\perp Q_2]
   \begin{bmatrix}R&C\\0&R_2\end{bmatrix}.
\]

Therefore only the old reflectors must be applied to `B`, and only `D` needs a
new QR. The leading cost is `O(m r k)` for the old-reflector application plus
`O((m-r)k^2)` for the residual block, instead of refactorizing all `r+k`
columns. This is the performance mechanism required by Appendix C and by the
author's `IncrementalQR.append`; it is not merely a cached full QR.

The inverse-adjoint block update in Appendix C is equivalent to retaining the
actual `R` and recomputing the small `R^\dagger` solve for the estimator. If
`G=R^{-\dagger}`, the author stores the corresponding inverse triangular data;
the Rust backend stores actual `R` and solves `R^\dagger X=I`, yielding
`X=G`. This uses the same small-matrix asymptotic work and avoids exposing an
inverse as a factorization result.

### Adaptive estimator

Appendix C defines, for `p` probe columns,

\[
\widehat{\mathrm{Err}}
 =\left(\frac1p\sum_{i=1}^p\|g_i\|^{-2}\right)^{1/2},
\qquad G=R^{-\dagger},
\]

where `g_i` are the columns of `G` in the paper's notation. For the norm,
`Y=QR` and `Q^\dagger Q=I` imply

\[
\|Y\|_F^2
 =\operatorname{tr}(Y^\dagger Y)
 =\operatorname{tr}(R^\dagger Q^\dagger QR)
 =\operatorname{tr}(R^\dagger R)
 =\|R\|_F^2,
\]

so `NormHat = ||R||_F / sqrt(p)`. The stopping test is therefore
`ErrHat <= atol + rtol * NormHat`. The final SVD safety factor uses the paper's
experimental choice of checking the sketch at `0.1 * requested tolerance`.

## Function-by-function traceability

Line numbers refer to the audited worktree at the date above. A row may group
straight-line validation and error wrapping with its immediately preceding
algorithmic block; no grouped row hides an unclassified algorithmic decision.

### Public dispatch and SRC options

| Location | Code fragment | Basis and classification |
|---|---|---|
| `contraction.rs:1353-1385` | `ContractionMethod::Src` and its public documentation | The issue's requested sibling method and the existing `Zipup`, `Fit`, and `Naive` API layout: `[Repo]`. The paper supplies the algorithm, not this enum. |
| `contraction.rs:1388-1431` | `SrcOptions` fields/defaults | `rtol`, rank, sketch, and final truncation concepts come from Algorithm 1, Appendix C, and the issue comments: `[Paper]`. Exact defaults `min_rank=2`, `rank_increment=3`, `final_svd=false`, and `seed=0` are policy choices: `[AI-Supplied]`; `false` is selected because the paper and author API make the final round opt-in. |
| `contraction.rs:1423-1431` | `Default` values | The paper only requires independent Gaussian probes and an optional final SVD; exact defaults are `[AI-Supplied]`. |
| `contraction.rs:1425-1434` | `sketch_options` and `0.1 * rtol` | The safety factor is the paper's adaptive experiment convention (`report.tex:1286`): `[Paper]`. Applying it only when `final_svd` is enabled is an API policy: `[AI-Supplied]`. |
| `contraction.rs:1466-1590` | option builders | Field assignment is `[Repo]` API plumbing. The constraints in `validate` (non-negative tolerances, nonzero rank controls, fixed/adaptive mutual exclusion) are required to keep the stated algorithm defined, but their exact error policy is `[AI-Supplied]`. |
| `contraction.rs:1652-1838` | `ContractionOptions` SRC dispatch fields | Reuses the existing TreeTN contraction API: `[Repo]`. The paper does not specify this Rust option object: `[AI-Supplied]` where it chooses defaults or validation behavior. |
| `contraction.rs:1922-2012` | top-level `contract` dispatch to `src_tree::contract` | Selecting the SRC implementation from `ContractionMethod::Src` is `[Repo]`; rejecting invalid options and preserving existing dense/fit/zip-up paths is `[Repo]`. |

### Probe construction and local MPO--MPO contraction

| Location | Code fragment | Basis and classification |
|---|---|---|
| `src_probe.rs:24-35` | `ProbeBank` storage | Independent Gaussian probe columns and prefix extension are `[Paper]`/`Algorithm 1`; storing a reusable `HashMap` bank is `[AI-Supplied]`. |
| `src_probe.rs:37-69` | `ProbeBank::new` validation and initial extension | Gaussian bank initialization follows Algorithm 1: `[Paper]`. Duplicate-index, zero-dimension, overflow, and allocation checks are `[AI-Supplied]` safety policy. |
| `src_probe.rs:72-104` | `width`, `coefficients`, `column` | Column extraction and bounds checks are `[Repo]`/`[AI-Supplied]` storage plumbing. |
| `src_probe.rs:106-123` | `extend_to` | Appending new columns while preserving the old prefix is needed by the adaptive reuse schedule: `[Derived]`. Persistent `StdRng` ordering and map-backed storage are `[AI-Supplied]`; the author instead calls `np.random.randn` at each sketch. |
| `src_probe.rs:126-130` | Box--Muller `standard_normal` | The paper requires standard normal entries, and the author uses NumPy standard normals, so the distribution is `[Paper]`/`[Author]`. This exact RNG transform and deterministic seeding are `[AI-Supplied]`. |
| `src_probe.rs:132-177` | `single_probe`, `single_probe_batch` | Turning coefficients into one-index/batched tensors is `[Repo]` bridge. The no-conjugation coefficient copy is `[Paper]`; range and payload checks are `[AI-Supplied]`. |
| `src_probe.rs:179-206` | `site_probe` | Tensor-product probe construction is the Khatri--Rao definition in Section 2.2 and Algorithm 1: `[Paper]`. The mixed-radix loop is a manual derivation of column-major indexing: `[Derived]`; construction/error wrapping is `[Repo]`. |
| `src_probe.rs:208-270` | `site_probe_batch` and `site_probe_batch_range` | Retaining a batch axis for independent columns is `[Derived]` from Algorithm 1. The scalar empty-output bridge, checked products, and batch-axis placement are `[AI-Supplied]`. |
| `src_probe.rs:273-279` | `product_dim` | Checked multiplication of physical dimensions is required by the matrix row-space count: `[Derived]`; overflow reporting is `[AI-Supplied]` safety behavior. |
| `src_probe.rs:281-307` | `probed_site_pair` | The sequence “probe A, probe B, then contract their shared leg” follows issue #563 comment 5396107820 and the hand derivation above: `[Paper]`/`[Derived]`. Tensor API assembly is `[Repo]`. |
| `src_probe.rs:309-349` | `probed_site_pair_batch_range` | Same factorized MPO--MPO identity with the probe column retained: `[Derived]`. The scalar broadcast branch and error handling are `[AI-Supplied]`. |
| `src_probe.rs:351-387` | `contract_prefix_with_probed_site_pair_batch_range` | Prefix recurrence is Algorithm 1 / chain derivation: `[Paper]`/`[Derived]`. Explicit argument grouping and retained-index plumbing are `[AI-Supplied]`. |
| `src_probe.rs:389-433` | `partition_probes` | Splitting A-side and B-side external probes is required by issue #563: `[Derived]`. The exact error policy for absent/shared output indices is `[AI-Supplied]`. |
| `src_probe.rs:435-463` | `contract_operand_with_probes` | Sequential retained contraction preserves probe column independence: `[Derived]`. Clone/error behavior is `[Repo]`/`[AI-Supplied]`. |
| `src_probe.rs:465-477` | `contract_site_pair` | Generic local product contraction is existing tensor-network semantics: `[Repo]`. |
| `src_probe.rs:479-500` | `contract_retaining` | “Shared batch is elementwise, not summed” is the required batched interpretation: `[Derived]`. Trailing-axis normalization is `[AI-Supplied]` tensor plumbing. |
| `src_probe.rs:501-539` | `site_operands`, `local_site_pairs` | Match corresponding named sites and fetch local tensors: `[Repo]`; the paper has no named-TreeTN API. |
| `src_probe.rs:541-591` | `local_output_indices` | Remove internal bond indices and retain the symmetric difference of local external indices: `[Derived]` from the MPO--MPO index equation. Deterministic sorting and missing-edge diagnostics are `[Repo]`/`[AI-Supplied]`. |
| `src_probe.rs:593-627` | `fixed_probe_width`, `maximum_site_width`, `initial_width` | `fixed_probe_width` implements the paper's recommended `max(ceil(1.5*chibar), chibar+10)` from `report.tex:627-636`: `[Paper]`. The `min` with row space is a matrix-shape consequence `[Derived]`; the `min` with the A/B cut dimension, adaptive initial width, and exact policy choices are `[AI-Supplied]` tree/runtime policy. |
| `src_probe.rs:629-690` | `factorize_probe_columns` loop | QR-only core and adaptive expansion follow Algorithm 1, Appendix C, and author `random_contraction_inc`: `[Paper]`/`[Author]`. Reuse of `previous`, rank-saturation stopping, rank increment, and propagating estimator errors are `[Derived]`/`[AI-Supplied]` numerical policy. In particular, estimator failure is no longer silently converted into success. |
| `src_probe.rs:692-724` | `connect_result_edge` | Connecting the newly created cap indices into TreeTN edges is `[Repo]`; the paper uses chain tensor notation, not this graph API. |
| `src_probe.rs:726-756` | `mark_result_canonical` | `Q^\dagger Q=I` makes each QR factor left-isometric; setting the existing TreeTN orientation metadata is `[Derived]` and `[Repo]`. Avoiding a second QR is an optimization and is `[AI-Supplied]`, validated by `validate_ortho_consistency`. |

### Chain implementation

| Location | Code fragment | Basis and classification |
|---|---|---|
| `src_chain.rs:29-106` | `contract` validation, site collection, probe-index collection, and mode selection | Chain order and site-local product are `[Repo]`; choosing the last site as the sweep endpoint follows Algorithm 1: `[Paper]`. Sim-internal-index hygiene, deterministic probe-index ordering, and empty-chain diagnostics are `[AI-Supplied]`/`[Repo]`. |
| `src_chain.rs:108-150` | last-site adaptive sketch and scalar case | `Y_n=P_{n-1}C_n`, QR, and projected carry are Algorithm 1: `[Paper]`/`[Derived]`. Dimension-one scalar bridge is `[AI-Supplied]`. |
| `src_chain.rs:152-200` | internal right-to-left adaptive sweep and first site | `Y_j=P_{j-1}C_jS_{j+1}`, QR, `S_j=Q_j^\dagger C_jS_{j+1}`, and `C_1S_2` are Algorithm 1: `[Paper]`/`[Derived]`. Exact TreeTN factor assembly is `[Repo]`; dimension caps and diagnostics are `[AI-Supplied]`. |
| `src_chain.rs:202-227` | result assembly and optional final truncate | Existing TreeTN topology construction is `[Repo]`; optional final SVD is permitted by issue #563's core-loop clarification and the paper's adaptive experiments: `[Paper]`/`[Issue]`. It is now opt-in, matching the author function's `finalround=None` default. |
| `src_chain.rs:230-395` | `FixedContractionRequest` and `contract_fixed` | Fixed-width QR sweep is Algorithm 1: `[Paper]`/`[Author]`. The scalar-site branch, exact cut-dimension cap, batched prefix cache, and result metadata are `[Derived]`/`[AI-Supplied]`. |
| `src_chain.rs:397-431` | `chain_cut_dimensions` | Product of corresponding A/B bond dimensions is the MPO--MPO cut dimension from the hand index count: `[Derived]`; checked arithmetic and graph lookup are `[Repo]`/`[AI-Supplied]`. |
| `src_chain.rs:433-452` | `factorize_fixed_batch` | Full-rank left QR is Algorithm 1 and the author code's `np.linalg.qr`/QR path: `[Paper]`/`[Author]`. Calling the tensor4all factorization seam is `[Repo]`. |
| `src_chain.rs:454-586` | `PrefixCache`, `BatchedPrefixCache`, segment growth and concatenation | Prefix recurrence is `[Paper]`/`[Derived]`. Segmenting newly generated columns and concatenating them is a Rust performance optimization not present in the author source: `[AI-Supplied]`. The exact-prefix reset on a width decrease is defensive `[AI-Supplied]` behavior. |
| `src_chain.rs:589-646` | scalar `PrefixCache::ensure_width`/`column` | Reusing previously computed `P_j` columns follows the adaptive prefix idea in the author code: `[Author]`/`[Derived]`; vector storage and cache shape are `[AI-Supplied]`. |
| `src_chain.rs:648-700` | `FactorizeSiteRequest` and `factorize_site_adaptive` | Building the left row indices, factoring the sketch, conjugating the factor for the environment, and propagating the right environment follow Algorithm 1: `[Paper]`/`[Derived]`. Request-struct abstraction and scalar handling are `[AI-Supplied]`. |

### Tree implementation

| Location | Code fragment | Basis and classification |
|---|---|---|
| `src_tree.rs:31-64` | chain specialization and tree validation | Chain endpoint specialization is `[Paper]`/`[Author]`; topology/center validation is `[Repo]`. |
| `src_tree.rs:67-121` | deterministic node/root setup, local products, output lists, probe bank, environment cache | Named-tree data access is `[Repo]`; root and probe-bank organization are `[AI-Supplied]` tree plumbing. |
| `src_tree.rs:123-236` | upward edge loop, source assembly, edge sketch, QR, projection, child message storage | The local `Y=UE`, QR, and `S=Q^\dagger U` operations are `[Derived]` from the hand tree recurrence. The entire rooted-tree schedule has no author-code counterpart and is `[AI-Supplied]` as a tree generalization. Scalar subtrees and cut-dimension caps are `[AI-Supplied]`. |
| `src_tree.rs:238-270` | root merge, graph assembly, final truncation/canonical metadata | Root merge follows associativity and the hand recurrence: `[Derived]`; graph construction and canonical metadata are `[Repo]`. Final SVD policy is `[Issue]`/`[AI-Supplied]`. |
| `src_tree.rs:273-352` | `EnvironmentCache::new`, scalar `ensure_width` | A complement environment is the contraction of the probed complement: `[Derived]`. Caching one environment per column is `[AI-Supplied]`; tensor lookups/errors are `[Repo]`. |
| `src_tree.rs:354-428` | batched environment cache and `column` | Retained batch independence is `[Derived]`; batched directed-message reuse and cache keys are `[AI-Supplied]`. |
| `src_tree.rs:431-455` | `merge_projected`, `site_factors` | Associative local merge of projected child messages is `[Derived]`; factor-vector assembly is `[Repo]`/`[AI-Supplied]`. |
| `src_tree.rs:457-475` | `uncontracted_indices` | Count-one indices are exactly the open legs of a contracted subtree; this is a direct tensor-network index invariant: `[Derived]`. Stable ordering is `[Repo]`. |
| `src_tree.rs:477-486` | `contract_factors` | Empty/singleton/multiple factor handling is `[Repo]` API plumbing; exact error wording is `[AI-Supplied]`. |
| `src_tree.rs:488-514` | `edge_bonds` | Obtain the two corresponding A/B bonds and count their product dimension: `[Derived]`/`[Repo]`. |
| `src_tree.rs:516-578` | scalar upward/downward directed messages | Two-pass message equations are the hand derivation above: `[Derived]`; traversal order and choosing the available side message are `[AI-Supplied]` because no tree reference exists in the paper or author code. |
| `src_tree.rs:580-641` | batched upward/downward directed messages | Same two-pass derivation with the batch index retained elementwise: `[Derived]`; batch-specific implementation is `[AI-Supplied]`. |

### Incremental QR backend

| Location | Code fragment | Basis and classification |
|---|---|---|
| `incremental_qr.rs:1-31` | module contract and state description | Reflector append layout is `[Author]`/`Appendix C`; safe Rust and actual-R representation are explicitly not literal author storage: `[Derived]`/`[AI-Supplied]`. |
| `incremental_qr.rs:34-82` | `IncrementalQrScalar` and scalar implementations | Conjugation is required by complex `Q^\dagger`: `[Paper]`/`[Derived]`. `from_real` is scalar plumbing `[Repo]`/`[AI-Supplied]`. |
| `incremental_qr.rs:113-178` | state fields and `from_factors` | Existing `Q,R` resume is required by Appendix C's update contract: `[Author]`/`[Derived]`. Factoring a supplied Q into `Q_qR_q` and storing `R_qR` is manually justified by `QR=(Q_q)(R_qR)`: `[Derived]`; validation is `[AI-Supplied]`. |
| `incremental_qr.rs:181-223` | `new` and initial factorization | Thin QR is Algorithm 1/Appendix C and author LAPACK QR: `[Paper]`/`[Author]`. Custom storage construction is `[AI-Supplied]`. |
| `incremental_qr.rs:225-363` | `append` | Apply old reflectors, factor residual columns, form block-triangular R, and append new reflectors: `[Author]`/`Appendix C` and the hand derivation above. Copying reflector storage and column-major loops are `[AI-Supplied]` implementation details. Residual tolerance, dependent-column skipping, and rank decrease behavior (`285-317`) are not in the paper/author code: `[AI-Supplied]` numerical-rank policy. |
| `incremental_qr.rs:365-403` | `q` and `rank` | Forming Q from stored reflectors and reporting its width are standard Householder QR: `[Derived]`; public API shape is `[Repo]`. |
| `incremental_qr.rs:405-464` | `q_columns` | Applying the complete reflector product to selected basis columns is `[Derived]`. Reusing old Q columns and materializing only appended columns is `[AI-Supplied]` optimization. |
| `incremental_qr.rs:466-514` | `r` and `error_estimate` | R exposure is `[Repo]`; Appendix C estimator delegation is `[Paper]`/`[Author]`. |
| `incremental_qr.rs:517-563` | `householder_factor` | Factor all columns with stored Householder vectors and extract upper R: `[Derived]` from standard QR and author LAPACK's packed reflector representation. The safe-Rust loop is `[AI-Supplied]`. |
| `incremental_qr.rs:565-611` | complex-safe `householder_vector` | The formula and verification below are `[Derived]`; author code delegates this arithmetic to LAPACK, so this exact implementation is not a source port. |
| `incremental_qr.rs:613-648` | `apply_reflector`, `apply_q_adjoint` | `H=I-\tau vv^\dagger` and ordered reflector application are `[Derived]`; loop/storage details are `[AI-Supplied]`. |
| `incremental_qr.rs:650-667` | `form_q` | Forming the thin Q from the reflector product is `[Derived]`; output allocation is `[Repo]`/`[AI-Supplied]`. |

### Core tensor seams and dense layout bridge

| Location | Code fragment | Basis and classification |
|---|---|---|
| `tensor_like.rs:470-542` | private incremental state in `FactorizeResult`, `new`, state attach/accessors | The factor result must carry an opaque update state across adaptive iterations: `[Derived]`. Making it private, exposing `new`, and preserving existing public fields are `[Repo]` API plumbing. |
| `tensor_like.rs:842-899` | `contract_retaining_indices` trait seam/default | Batch-preserving contraction is required by the independent-column derivation: `[Derived]`. The generic fallback and error shape are `[AI-Supplied]`. |
| `tensor_like.rs:1005-1072` | incremental factorization trait/default | The previous/all/appended contract follows Appendix C and author `append`: `[Author]`/`[Derived]`. The generic full-refactor fallback is `[AI-Supplied]` and is deliberately not used by the native `IdxTensor` path. |
| `tensor_like.rs:1174-1242` | `from_dense_any` default | The probe and reflector bridges need dense construction; mixed-radix column-major decoding is `[Derived]`/`[Repo]`. The one-hot accumulation fallback is `[AI-Supplied]` compatibility plumbing. |
| `tensor_like.rs:1244-1342` | `stack_along_new_index` default | Stacking independent probe columns is `[Derived]`; one-hot/outer-product fallback, axis policy, and validation are `[AI-Supplied]`. |
| `tensor_like.rs:1344-1457` | `concatenate_along_new_index` default | Concatenating old and new Q blocks is `[Derived]` from the incremental Q construction. Slice-and-stack fallback is `[AI-Supplied]`. |
| `idx_tensor.rs:3648-3785` | pairwise retained contraction | Batched GEMM with retained axes is the required elementwise batch semantics: `[Derived]`. `dot_general_with_conj`, fallback conditions, output permutation, and materialized dense bridge are `[Repo]`/`[AI-Supplied]`. |
| `idx_tensor.rs:5647-5906` | native `IdxTensor` incremental factorization | Matrix reshape from left indices and the Q/R tensor reconstruction follow the hand QR derivation: `[Derived]`. f64/Complex64 dispatch, private state enum, prefix concatenation, and scalar storage are `[AI-Supplied]` implementation policy. |
| `idx_tensor.rs:5981-5992`, `6041-6115` | trait overrides for retained contraction and dense/stack/concatenate constructors | These connect the generic SRC seams to existing dense tensor operations: `[Repo]`/`[AI-Supplied]`; no paper or author function has this Rust API. |
| `factorize.rs:305-827`, `fit.rs:720-730`, `swap.rs:56-85` | replacement of public `FactorizeResult` literals with `FactorizeResult::new` | Required because the algorithm-private QR state made direct struct literals invalid. This is `[Repo]` compatibility plumbing, not SRC mathematics. |

## Manual check of the complex Householder formula

For a column segment `x=[alpha; y]`, define `mu=||x||_2`, choose
`phi=alpha/|alpha|` when `alpha != 0` and `phi=1` otherwise, and set

\[
\beta=-\phi\mu,\qquad
\delta=\alpha-\beta,\qquad
v=[1;y/\delta],\qquad
\tau=(\beta-\alpha)/\beta.
\]

Because `beta` has the phase of `-alpha`,

\[
v^\dagger x
 =\alpha + \frac{y^\dagger y}{\bar\delta}
 =-\beta.
\]

The final equality follows from
`y^\dagger y=mu^2-|alpha|^2` and the definitions of `beta`, `delta`, and
`phi`; it can also be checked by multiplying both sides by `\bar\delta` and
using `alpha=phi|alpha|` and `beta=-phi mu`. Therefore

\[
Hx=x-\tau v(v^\dagger x)
   =x+\tau\beta v,
\]

whose first entry is `alpha+(beta-alpha)=beta`, while every tail entry is
`y+(beta-alpha)y=0`. Hence `Hx=beta e_1`. Further, `tau` is real for this
phase choice, so `H=I-\tau vv^\dagger` is Hermitian and applying the same
operation implements the needed adjoint reflector. The Rust code at
`incremental_qr.rs:565-636` follows this derivation and uses conjugated inner
products; the complex reconstruction tests are a numerical check of the
identity, not an external provenance source.

## Test-by-test oracle map

Tests are executable checks of the claims above; they are not independent
sources for the algorithm. The following map records what each SRC-related
test is intended to prove, so a passing test is not mistaken for provenance.

| Location | Test group | Oracle or derivation |
|---|---|---|
| \`src_probe.rs:768-1000\` | probe-bank prefix, zero-width/dimension guards, column-major Khatri--Rao order | Prefix equality is \`[Derived]\` from adaptive probe reuse; invalid dimensions and overflow are \`[AI-Supplied]\` safety policy; tensor-product values follow the mixed-radix derivation below \`[Derived]\`. |
| \`src_probe.rs:827-923\` | single-site and batched MPO--MPO probes | Expected values are hand-evaluated from \`C^{s,t}=sum_u A^{s,u}B^{u,t}\` and \`z_{s,t,k}=x_{s,k}y_{t,k}\` \`[Derived]\`, with the factorized requirement from issue comment \`5396107820\` \`[Issue]\`. |
| \`src_probe.rs:946-1000\` | adaptive request count, cut cap, and final-SVD safety factor | Request sequencing follows Algorithm 1/Appendix C \`[Paper]\`; rank saturation and dimension-one behavior are \`[AI-Supplied]\` policy; \`0.1*rtol\` is the paper's adaptive experiment convention \`[Paper]\`. |
| \`incremental_qr.rs:675-843\` | reconstruction, append, dependent columns, \`from_factors\`, shape errors, complex projection | Reconstruction and append use the block-QR derivation and Appendix C \`[Paper]/[Derived]\`; supplied-factor reconstruction follows \`QR=(Q_q)(R_qR)\` \`[Derived]\`; dependent-column handling and shape diagnostics are \`[AI-Supplied]\`; complex projection checks the Hermitian Householder derivation above \`[Derived]\`. |
| \`backend/tests/mod.rs:7-73\` | real/complex/singular/single-precision SRC estimator | Expected values are direct evaluations of \`G=R^{-dagger}\` and \`(p^{-1}sum ||g_i||^{-2})^{1/2}\` from Appendix C \`[Paper]/[Derived]\`; rejection and scalar coverage are backend safety/portability checks \`[Repo]/[AI-Supplied]\`. |
| \`tensor_like/tests/mod.rs:117-284\` | dense construction, batch stacking/concatenation, incremental prefixes and multi-axis rows | Column-major payloads follow the repository dense-layout rule \`[Repo]\`; batch assembly and prefix reconstruction follow the retained-column derivation \`[Derived]\`; generic fallback coverage is \`[AI-Supplied]\` plumbing. |
| \`defaults/contract/tests/mod.rs:236-315\` | retained shared batch/index semantics | The expected result keeps the shared batch label as an elementwise axis rather than contracting it, exactly as required by independent probe columns \`[Derived]\`; the remaining retained-index cases exercise the existing contraction contract \`[Repo]\`. |
| \`contraction/tests/mod.rs:519-835,890-997\` | chain/tree exactness, adaptive caps, scalar bridges, and complex case | Chain exactness is Algorithm 1 with a full probe cap \`[Paper]/[Derived]\`; tree exactness is the hand rooted-tree recurrence \`[Derived]\` and is explicitly not evidence that a published tree implementation exists \`[AI-Supplied]\`; topology and canonical-center assertions use existing TreeTN invariants \`[Repo]\`; fixture values are test data \`[AI-Supplied]\`. |

The test fixtures, deterministic seeds, tolerance values, and assertion
thresholds are not statements from the paper unless a row explicitly says so.
They are \`[AI-Supplied]\` test-engineering choices. In particular, these tests
establish the audited identities for the covered small cases; they do not
prove the unestablished general tree theorem or the asymptotic benchmark claim.

## Mixed-radix and column-major check

For logical indices with dimensions `d_0,...,d_{q-1}`, column-major linear
index `l` has coordinate

\[
p_i=\left\lfloor\frac{l}{\prod_{h<i}d_h}\right\rfloor\bmod d_i.
\]

The loop in `site_probe` implements this by repeatedly taking `remainder %
d_i` and then dividing `remainder /= d_i`. Consequently the tensor-product
probe value is `\prod_i Omega^{(i)}[p_i,k]`, which is the Khatri--Rao column.
The index order and flat-buffer convention are repository column-major
semantics; the paper specifies the abstract Kronecker/Khatri--Rao product but
not this Rust storage bridge. The bridge is therefore `[Derived]` plus
`[Repo]`, not a claimed author-code detail.

## Estimator representation check

The author stores inverse-triangular data and computes inverse-row norms in
`incrementalqr.cpp::get_error_estimate` (lines 106--119). The Rust backend at
`backend.rs:409-489` stores actual `R`, forms `R^\dagger`, and solves
`R^\dagger X=I`. If the paper/author variable is `G=R^{-\dagger}`, then
`X=G`. If an implementation instead describes the same data as `R^{-1}`, the
columns of `X` are the conjugate-transposes of the rows of that inverse, so
the corresponding norms are equal. The estimator sum is therefore invariant
under this representation change. The norm identity is the Frobenius derivation
above. This is `[Paper]` plus `[Derived]`, while choosing actual-R storage is
`[AI-Supplied]`.

## Mismatch ledger

These are deliberate differences from the sources and must remain visible in
future reviews:

| Topic | Paper/author | Current implementation | Classification |
|---|---|---|---|
| Product type | Paper algorithm is MPO--MPS; issue #563 requires factorized MPO--MPO | Separate A/B probes and local shared-leg contraction | `[Derived]` from issue clarification |
| Topology | Chain recurrence only | Rooted tree with upward projected messages and complement environments | `[AI-Supplied]` tree extension |
| QR storage | Author inverse-R state plus LAPACK routines | Safe-Rust Householder reflectors plus actual R | `[Derived]` mathematics, `[AI-Supplied]` representation |
| Q materialization | Author extracts Q when needed | `q_columns` materializes only requested appended columns | `[AI-Supplied]` optimization |
| Probe generation | Author uses NumPy random draws per sketch | Persistent seeded `StdRng` bank preserving prefixes | `[AI-Supplied]` engineering |
| Batch execution | Author loops/updates sketch blocks | Retained-index contraction and dense batched GEMM | `[AI-Supplied]` tensor4all bridge |
| Rank deficiency | Author path assumes the QR block is usable | Epsilon-scaled residual threshold skips dependent appended columns | `[AI-Supplied]` numerical policy |
| Final compression | Optional SVD is allowed | `final_svd` defaults to off; explicit final tolerance policy enables the paper's `0.1` sketch safety factor | `[Paper]`/`[Issue]` plus `[AI-Supplied]` API policy |
| Scalar subtrees | Not specified in source | Dimension-one structural bridges | `[AI-Supplied]` topology-preserving bridge |

## Findings and changes made during this audit

1. The unused `probed_site`, `probed_site_batch`, and
   `probed_site_batch_range` helpers that conjugated probes were removed. The
   production implementation now has only the no-conjugation `A Omega` path.
2. The empty `src_qr.rs` module and declaration were removed; it provided no
   implementation or provenance.
3. The scalar batch test now exercises the actual scalar branch of
   `probed_site_pair_batch_range` and checks broadcast values.
4. Probe range arithmetic now checks `first_column + width` and
   `first_column + column` for overflow.
5. Adaptive estimator failures are propagated. A failed Appendix C estimator
   cannot silently be interpreted as having met the requested tolerance.
6. Module-level provenance comments identify the exact paper sections and
   author functions. Unsupported tree and tensor-bridge sections are marked
   `[AI-Supplied]` here instead of being presented as source ports.

## Re-cloned reference and second implementation pass (2026-08-27)

The reference was cloned afresh and pinned to author commit
`fe6ad494fc6f3605fc3963360f626d83f47bc2ce`. Within that checkout, the local
paper sources are `arxiv-source/report.tex` and
`CET26-Successive-Randomized-quantum.pdf`. The relevant source locations are
`code/tensornetwork/contraction.py:82-353` (`random_contraction`),
`code/tensornetwork/contraction.py:357-593` (incremental variant),
`code/tensornetwork/incrementalqr.py:50-175`,
`code/tensornetwork/incrementalqr.cpp:21-119`, and
`code/tensornetwork/util/benchmarking.py:52-85,398-409`.

The decisive comparison was:

| Source evidence | Consequence for Rust |
|---|---|
| Algorithm 1, `report.tex:638-670`, sketches an MPO--MPS product and performs QR on the projected columns | The MPO--MPO extension must make separate `A Omega` and `B Omega` probes, then contract the shared physical leg; it must not first materialize the full local MPO--MPO product. |
| Author `random_contraction` contracts the left environment, applies the current site to the probe, then projects against the right environment | Fixed and adaptive chain paths now use environment-first prefix/current-site contractions. |
| Author incremental code adds a block using the Appendix C inverse-triangular update and estimates from inverse-triangular rows | Rust `IncrementalQr` now maintains the equivalent inverse-adjoint state and uses the block update; the estimator is no longer recomputed through an unrelated factorization path. |
| `report.tex:627-636,670` and author `finalround=None` make final MPS SVD compression optional | `SrcOptions::final_svd` now defaults to `false`; an explicit final truncation policy is required to enable it. |
| Author benchmark `util/benchmarking.py:52-85` separates as-is SRC from oversampled plus final-round SRC | The benchmark no longer silently combines a fixed oversized sketch with an unreported final SVD; fixed SRC requests the same target rank as the global fit comparison. |

The performance bug was therefore not merely a missing optimization. The old
adaptive path formed a large local MPO--MPO product before applying probes,
which changed the effective contraction order and violated the mechanism used
by the paper and author implementation. The adaptive prefix cache also
constructed new columns one at a time through that path. Both were replaced by
batched, environment-first prefix construction. The tree path uses the same
local projected-message recurrence; its tree-specific recurrence remains
`[AI-Supplied]` because neither the paper nor the author repository publishes
that extension.

The focused regression test
`prefix_probe_contraction_matches_local_product_but_uses_environment_first_order`
checks the optimized path against the mathematically equivalent local product
on both probed and unprobed site pairs. The default-final-SVD regression test
also checks that adaptive SRC stops at the requested tolerance without an
implicit final SVD.

## Verification status

The focused probe test was rerun after the cleanup:

```text
cargo test -p tensor4all-treetn --lib --release src_probe::tests -- --nocapture
12 passed; 0 failed
```

The broader release checks completed as follows:

```text
cargo test -p tensor4all-tensorbackend --lib --release    205 passed
cargo test -p tensor4all-core --lib --release              429 passed, 1 ignored
cargo test -p tensor4all-treetn --lib --release            482 passed, 1 ignored
cargo test --doc --release -p tensor4all-tensorbackend -p tensor4all-treetn
  148 + 140 passed
cargo clippy --workspace --all-targets -- -D warnings ...  passed
cargo fmt --all -- --check                               passed
```

The full single-process benchmark was rerun after this second pass. Its
generated profile `full-corrected-ordering-20260827` contains 60 records for
the three benchmark cases and `N=2,8,32,128`; all correctness gates passed. In
Case 3 at `N=128`, the measured times were fit
`0.362040 s`, zip-up `0.248990 s`, fixed SRC `0.104899 s`, and adaptive SRC
`0.181698 s`, with errors below `1e-4`. The run used one process, one Rayon
thread, one BLAS thread, and CPU affinity to one core. These measurements are
validation evidence for the current dirty worktree, not a reproducibility
claim for a committed revision.

## Remaining review risks

- The rooted-tree recurrence is mathematically motivated by contraction
  associativity and the chain `Q Q^\dagger` argument, but it has no external
  author implementation or published tree theorem. It needs independent
  numerical tests against dense contraction on representative branching
  patterns before being described as validated generally.
- The epsilon-scaled dependent-column policy changes the effective sketch rank
  when appended columns are numerically dependent. It is useful defensive
  behavior, but it is not part of SRC's stated randomized model and should be
  reviewed separately from the algorithmic proof.
- The generic trait fallbacks are correctness bridges, not performance paths.
  A non-`IdxTensor` implementation that does not override them may satisfy the
  API while losing the intended complexity.
- Only the TreeTN engine is in scope in this worktree, following Hiroshi's
  direction. No SimpleTT SRC implementation is claimed by this audit.
