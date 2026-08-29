# `src_chain.rs` endpoint-center canonical-direction bug (found, not fixed)

## Summary

While implementing Task 6 of the
`docs/superpowers/plans/2026-08-29-src-audit-remediation.md` remediation
plan (a numerical isometry check for SRC contraction results), the
implementer found — and an independent reviewer separately reproduced — a
real production bug in `crates/tensor4all-treetn/src/treetn/contraction/src_chain.rs`:
the chain-specialized SRC contraction path's declared canonical-center
metadata (`canonical_region`, `ortho_towards`) is systematically backwards
relative to the actual numerical structure of the result tensors, whenever
the requested contraction center is a chain endpoint.

**Practical severity:** the *contracted value* is correct — every existing
dense-oracle (`contract_naive`) comparison passes, and this is not a
numerical-accuracy bug. Only the canonical-center *metadata* is wrong. No
test in the existing suite catches it, because the two ways tests validate
SRC results (dense-oracle value comparison, and
`validate_ortho_consistency()`) are both blind to this specific
discrepancy — see "Why existing tests don't catch it" below.

This bug was **not fixed** by the 2026-08-29 remediation plan — it was out
of scope for Task 6 (a test-authoring task scoped to the test file only)
and deserves its own reviewed change. This worklog is the tracked record
of the finding, per that plan's Task 6 report. It is also documented
in-code at
`crates/tensor4all-treetn/src/treetn/contraction/tests/mod.rs:570-596`
(the comment above `src_result_tensor_is_numerically_isometric`), which
this worklog expands on with full reproduction detail that would otherwise
only exist in transient session-local planning artifacts.

## Root cause

`src_chain.rs::contract` dispatches to `contract_fixed` (or the adaptive
loop that wraps it) only when the requested contraction center is a chain
**endpoint** (`src_tree.rs`'s dispatcher checks
`chain.last() == Some(center)` and routes to `src_chain::contract` in that
case; any other requested center goes to the general rooted-tree path in
`src_tree.rs` instead).

`contract_fixed`'s sweep runs from the requested endpoint *backward*
toward the opposite endpoint, building genuine left-canonical QR factors
at each step (`factorize_full_rank(..., FactorizeCanonical::Left)`,
`src_chain.rs:453`, called once per step of the sweep). Under this crate's
own `Canonical::Left` convention (cross-checked against
`crates/tensor4all-core/tests/linalg_factorize.rs::assert_canonical`: for
a left-canonical factor of shape `(rows, rank)`,
`sum_row conj(M[row,a]) * M[row,b] = delta(a,b)` — i.e. the factor is
isometric *towards its kept/new bond*, which in this sweep is the bond
pointing toward the *opposite* endpoint from the one the sweep started
at), each factor produced by the sweep ends up isometric toward the node
closer to the **opposite** endpoint, not toward the requested center. The
opposite endpoint absorbs the final, unfactorized "environment" tensor —
i.e. the opposite endpoint is where the actual data/norm lives, exactly as
if the true orthogonality center were the *other* endpoint.

`mark_result_canonical` (`src_probe.rs:689`, called at
`src_chain.rs:235` and `src_chain.rs:402`) then unconditionally labels
`canonical_region = [center]` with every edge's `ortho_towards` pointing
toward the *requested* center — which is backwards relative to what the
sweep actually built. The function has no logic that inspects which
direction the sweep's factors actually ended up isometric toward; it
trusts the caller's `center` argument unconditionally.

Confirmed isolated to this endpoint-center chain specialization only:
requesting an **interior** center on the same chain topology (which
dispatches to the general `src_tree.rs` rooted-tree path instead, since
`chain.last() != center`) produces declared and actual directions that
match exactly. The existing branched-tree fixture (hub center, also
routed through `src_tree.rs`) likewise matches exactly. So the general
tree path is unaffected; the bug is specific to `src_chain.rs`'s
endpoint-center sweep-direction / labeling mismatch.

## Reproduction methodology and numbers

### Original discovery (Task 6 implementer)

Before trusting any measurement against SRC output, the implementer first
sanity-checked the Gram-matrix isometry-checking method itself against
`contract_zipup` (a mature, independently-tested contraction path) on the
same 3-node chain fixture: both declared ortho directions
(edge(A,B)&rarr;B and edge(B,C)&rarr;C) gave residuals of order `1e-16`,
confirming the Gram-matrix construction (matricize via `fuse_indices`,
form `M^\dagger M`, compare to `IdxTensor::diagonal`) is itself correct
before using it to check SRC.

Running the same check against
`contract(&tn_a, &tn_b, &"C", ContractionOptions::src()...)` (fixed-rank,
full probe cap, same 3-node chain `"A"-"B"-"C"`, requested center `"C"`,
an endpoint) gave:

| node | bond checked | residual |
|---|---|---|
| A | towards B (declared direction) | ~4.4e10 |
| B | towards A (reverse of declared) | ~2.2e-16 |
| B | towards C (declared direction) | ~2.98 |

I.e. the *declared* canonical direction (`canonical_region = {"C"}`,
`ortho_towards` pointing every edge toward `"C"`) is numerically wrong:
node B is actually isometric in the *opposite* direction (toward A, not
toward C), and node A — which the metadata implies should carry the
isometric factor toward B — instead holds the "data"/norm tensor, exactly
as if the true orthogonality center were `"A"` rather than `"C"`.

Repeating with the request reversed (center `"A"` instead of `"C"`)
produced the same pattern, mirrored: node C became the wildly-non-isometric
"data" node, and node B was isometric toward C instead of the
declared-toward-A direction. This confirms the discrepancy is systematic
— a structural mislabeling, not a fluke of one seed or one direction
choice.

### Independent re-verification (Task 6 review)

A separate reviewer, tasked with independently reproducing the claim
rather than accepting the report at face value, first derived the same
root-cause mechanism purely by reading `src_chain.rs` (the Left-canonical
factorization direction vs. the sweep direction vs.
`mark_result_canonical`'s unconditional labeling) *before* running
anything, then independently reproduced the numerical discrepancy with
their own separate throwaway test (added, run, and cleanly reverted via
`git checkout` afterward — not part of any committed diff). The
reviewer's own measured residuals matched the implementer's almost
exactly: `~4.4e10`, `~2.98`, and `~1e-16` across multiple configurations
and both center-reversal directions. The reviewer likewise confirmed the
discrepancy is isolated to `src_chain.rs`'s endpoint-center path, with
interior centers and the general `src_tree.rs` path unaffected in both
investigations.

## Which existing tests are affected, and why they don't catch it

Every existing test that requests an **endpoint** center on a chain
topology through the SRC path is affected — i.e. every test using
`make_three_node_chain_pair()` (always called with center `"C"`, an
endpoint in that fixture) or an equivalent endpoint-center chain fixture.
Named specifically:

- `src_fixed_matches_exact_contraction_when_probe_cap_is_full`
- `src_dispatch_preserves_public_contract`
- `src_adaptive_contracts_and_honors_rank_cap`
- `src_adaptive_uses_the_minimum_rank_when_the_estimate_is_already_small`
- `src_adaptive_matches_exact_contraction_on_a_small_chain` (added by this
  same remediation plan's Task 6, using the brief-specified endpoint
  center `"C"`)

These tests keep passing despite the bug for two independent reasons,
neither of which is sensitive to which internal node is (mis)labeled as
the canonical center:

1. **Dense-oracle comparisons** (`to_dense()` + `.sub(&expected).maxabs()`
   against `contract_naive`) check only the overall contracted tensor
   value. The contracted value is correct regardless of which internal
   node the network is labeled as canonical around — canonicalization is
   an internal gauge choice that does not change the contracted result.
2. **`validate_ortho_consistency()`** (used by tests that check structural
   canonical-form properties) only checks that the declared metadata is
   *internally self-consistent* — that `ortho_towards` directions agree
   with `canonical_region` and the tree's connectivity — never that the
   metadata matches the actual tensor *values*. A network that is
   internally consistent about the wrong center passes this check just as
   cleanly as one that is consistent about the right one.

This is precisely the class of gap flagged (as a theoretical concern) by
the original SRC provenance audit's WS-tests §5c finding ("no test in the
suite proves a result tensor is numerically unitary/isometric — only
metadata is checked"). Task 6 of this remediation plan set out to close
that theoretical gap with a genuine Gram-matrix isometry check, and in
doing so turned it from a theoretical concern into a confirmed, reproduced
production bug.

## Disposition: not fixed, recommended as a follow-up task

This bug was **not fixed** as part of the 2026-08-29 remediation plan.
Task 6's brief scoped changes to the test file only
(`crates/tensor4all-treetn/src/treetn/contraction/tests/mod.rs`); a fix to
`contract_fixed`'s sweep-direction/labeling mismatch (and the shared
`mark_result_canonical` call sites at `src_chain.rs:235` and `:402`) is a
nontrivial, independent behavioral change to production contraction code
that deserves its own dedicated review, not a drive-by fix bundled into a
test-authoring task.

The delivered isometry test
(`src_result_tensor_is_numerically_isometric`) deliberately uses a
4-node-chain fixture with an **interior** center (`"S1"`), which dispatches
through the unaffected general `src_tree.rs` path, so that it validates
the intended property — a *correct* SRC result is numerically isometric —
without either failing on this known, separately-tracked issue or
laundering the wrong direction into a passing assertion by empirically
fitting the test to whatever the buggy path happens to produce.

**Recommended follow-up task:** fix `src_chain.rs`'s canonical-direction
bookkeeping. The likely fix shape is either (a) reversing which endpoint
`contract_fixed`'s sweep treats as the "start" so the Left-canonical
factors' isometric direction matches the requested center, or (b) having
`mark_result_canonical`'s call sites in `src_chain.rs` pass the sweep's
*actual* resulting orthogonality-center direction rather than trusting the
requested `center` argument unconditionally — whichever is correct should
be determined by reading `contract_fixed`'s full sweep loop, not guessed
from this worklog alone. After the fix, broaden
`src_result_tensor_is_numerically_isometric` (or add a sibling test) to
also cover the endpoint-center case, since an endpoint request is the more
commonly used center choice in this test file's existing fixtures, and a
regression test that would have caught this bug should exist going
forward.
