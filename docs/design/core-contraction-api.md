# tensor4all-core contraction API direction

Source of the still-live API decision from the deleted
`plan/tensor4all-core-api-cleanup.md` (tensor4all-rs#566 Task 11 debris
removal).

## Direction (current and pending)

- Prefer one public N-ary contraction entry point over separate binary and
  N-ary APIs. Binary contraction is expressed as N-ary contraction with two
  operands.
- The default API takes borrowed operands (`contract(&[&a, &b, &c])`); an
  owned variant (`contract_owned(vec![a, b, c])`) exists as an explicit
  optimization path for callers that can transfer ownership.
- The public API accepts structural labels directly instead of building a
  string equation and parsing it again downstream.
- `contract_pair` remains exported as a temporary convenience; the direction
  is to converge on N-ary `contract`/`contract_owned` plus explicit
  `outer_product`, then remove `contract_pair`.

## Implemented

- `contract` / `contract_owned` and their `contract_with_options` /
  `contract_owned_with_options` variants:
  `crates/tensor4all-core/src/defaults/contract.rs`
  (re-exported from `crates/tensor4all-core/src/lib.rs`).
- Semantics tests: `crates/tensor4all-core/src/defaults/contract/tests/mod.rs`.
- `outer_product` for disconnected inputs:
  `crates/tensor4all-core/src/defaults/contract.rs`.

## Follow-up

- Remove the `contract_pair` convenience export once consumers migrate to
  N-ary `contract` (PR 3 API-vocabulary work in tensor4all-rs#566).
