# AI-ready issue batch implementation design

## Objective

Resolve every currently open `ai-ready` issue in a small number of coherent changes without compatibility shims. The work stays on one integration branch so the final changes can be submitted as the minimum number of reviewable PRs; commits remain grouped by owning abstraction.

## Workstreams

1. **Public-contract cleanup** — #633, #548, #547, #634.
   Remove inert public options, enforce full-index identity and validated mutation, reject ambiguous topology/index inference, validate linear-chain conversion, and make TreeTN/PartitionedTT mutations transactional.
2. **Boundary and numerical error hardening** — #543, #544, #546, #550.
   Centralize checked shape/length/label conversions at public boundaries, propagate evaluation/callback failures, and add focused regressions before allocation, indexing, or backend execution.
3. **AD and repository tooling** — #545, #637.
   Preserve or explicitly reject tracked AD operations, and make the generated API inventory complete, ignored, reproducible, and CI-checked.

## Implementation rules

- Fix each contract at its owning crate; do not add compatibility aliases for removed inert APIs.
- Reuse existing typed error and checked-arithmetic helpers before introducing new ones.
- Validate all caller-controlled metadata before allocation, pointer arithmetic, cache lookup, mutation, or backend calls.
- Keep public docs, runnable examples, tests, C API headers, and contributor instructions synchronized.
- Add one regression per distinct error/control-flow path and use release-mode tests for the final checks.
- Keep changes limited to the issue scopes; stop and record a blocker if a new API/product decision is required.

## Verification plan

During development, run focused `cargo test` checks for changed crates and the relevant Python/script tests. Before completion, run `cargo fmt --all`, the formatting check, workspace clippy with `-D warnings`, release tests for changed crates, `cargo nextest run --release --workspace`, `cargo doc --workspace --no-deps`, `./scripts/test-mdbook.sh` when docs/examples change, and the repository-rules review/test scripts. Regenerate/check the API inventory using the new documented command and run coverage or an explicit coverage attestation for all changed paths.

## Review state

This document is the pre-implementation design record for the batch. Any delegated implementation must receive an independent read-only cross-model review of this design before it starts and of its complete diff afterward. No delegated implementation has started yet.
