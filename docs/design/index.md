# Design Documents

## Architecture & Backend

| Document | Description |
|----------|-------------|
| [t4a_unified_tensor_backend.md](./t4a_unified_tensor_backend.md) | Unified tensor backend design (tenferro-rs integration) |
| [tenferro-main-session-migration.md](./tenferro-main-session-migration.md) | First issue #623 slice: pin current tenferro main and migrate internal CPU calls to its canonical session API |
| [torch_backend.md](./torch_backend.md) | PyTorch backend design exploration |
| [tenferro_ad_scalar_operator_extension_note.md](./tenferro_ad_scalar_operator_extension_note.md) | Tenferro AD scalar operator extension notes |
| [build-profiles.md](./build-profiles.md) | Debug-free ordinary Cargo profiles and the opt-in full-debug release profile |

## Tensor Networks

| Document | Description |
|----------|-------------|
| [adaptive-tci-interpolation.md](./adaptive-tci-interpolation.md) | Adaptive TCI patching, convergence, pivot recycling, and structured embedding |
| [partitionedtt-projector-invariants.md](./partitionedtt-projector-invariants.md) | Issue #634 design for coherent projector identity, validation, and transactional PartitionedTT mutation |
| [partitioned-treetn.md](./partitioned-treetn.md) | Issue #648 migration design for TreeTN-native eager partitioning and adaptive patching |
| [gse-chain-mps-algorithm.md](./gse-chain-mps-algorithm.md) | Chain MPS global subspace expansion analysis for TreeTN GSE-TDVP planning |
| [itensormps-compatible-zipup.md](./itensormps-compatible-zipup.md) | ITensorMPS-compatible chain zip-up contraction schedule and policy-aware decomposition follow-up |
| [fit-sum.md](./fit-sum.md) | Variational fitting of compatible TreeTN sums without exact direct-sum materialization |

## Automatic Differentiation

| Document | Description |
|----------|-------------|
| [three_mode_ad_design.md](./three_mode_ad_design.md) | Three-mode automatic differentiation design |

## Julia Compatibility

| Document | Description |
|----------|-------------|
| [quanticstransform_julia_comparison.md](./quanticstransform_julia_comparison.md) | Quantics transform Julia compatibility analysis |
