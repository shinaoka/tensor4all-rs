# Issue #566 PR 3 (typed errors) — foundation checkpoint

**Status:** PR 3 の基礎スライス（設計 step 1: trait 層の typed 化）が完了・
レビュー済み。ブランチ `chore/issue-566-pr3-errors`（origin/main からの
12 コミット、未 push）。

## 完了したスライス

### Shared-rules prerequisite（別リポジトリ、完了・マージ済み）
- tensor4all-agent-rules PR #11 マージ（rules: generalize cross-repository
  rules from tenferro-rs, issue #6 をクローズ）。#7/#8 は既存。
- 追加内容: Invariant Markers / API Evolution / Output-Update Naming /
  File Organization / Work Logs（PR-body リンク要件 + rule-inventory
  meta-rule）/ Final Cross-Phase Multi-Agent Audit（common/repository.md）、
  Performance-Gated Experiment Protocol / Cache Ownership / Complexity
  Budget（common/performance.md）、Doc Examples / Public Result Error-Doc
  Gate（common/docs-and-tests.md）、Unsafe Code Boundary / Unit Test Org /
  Debug-Enum Hygiene（rust/index.md）、Uninitialized Scratch / Threading
  （rust/performance.md）、Typed Errors（rust/numerical.md）。

### PR 3 Slice 1 — trait 層の typed 化（commit 41e5c35 〜 5361e90）
- `TensorIndex` に `type Error: std::error::Error + Send + Sync + From<anyhow::Error>` を追加。
  replaceind / replaceinds / replaceinds_pairs を typed 化。
- `TensorVectorSpace` の axpby / scale / inner_product / sub / neg /
  validate / isapprox を `Self::Error` で typed 化（既存の norm_squared /
  norm / maxabs と統一）。
- `TensorContractionLike`（direct_sum / outer_product / permuteinds /
  fuse_indices / contract / contract_pair / validate）と
  `TensorConstructionLike`（diagonal / delta / scalar_one / ones /
  select_indices / onehot）を typed 化。
- 実装者: TensorDynLen → TensorDynLenError、BlockTensor →
  TensorVectorSpaceError、TensorTrain（itensorlike）→ TensorTrainError、
  TreeTN / LinearOperator（treetn）→ 新設 TreeTNOperationError。
- 下流呼び出し元（gse / tdvp / linsolve / contraction / cached_evaluator /
  partial_contraction / local_update_support / identity / benchmarks）の
  anyhow 変換を適用。
- trait メソッド全てに # Errors ドキュメント追加（changed-from ゲート pass）。
- 検証: ワークスペースビルド / nextest 2693+10 skip / core 841 / treetn 774
  / itensorlike 162 / doctests 320 — 全て green。
- レビュー: reviewer-gpt 3 ラウンド（Blocking/Important 全クローズ）。

## 残り（PR 3 完了まで）

1. **設計 step 2–6**: tensorbackend + core の公開 fn 型付け（54 + 82 件）、
   treetci/quanticstci、treetn、quanticstransform/HDF5、残り全公開面。
2. **# Errors ドキュメントバックログ 545 件 → 0**（core 124 / treetn 111 /
   tutorial-code 63 / tensorbackend 56 / simplett 53 ほか）— repository-wide
   ブロッキング切替に必要。
3. **ゲート切替**: `check-public-error-docs.py` を repository-wide モードへ +
   clippy missing_errors_doc / missing_panics_doc deny 有効化。
4. **レイヤリング（設計 c..i）**: tenferro 直接ルート除去（例外タプル撤去）、
   FullPivLuScalar 移動、core→tcicore 逆転解消、capi ID-only 文書化/削除 +
   union-find private 化、per-element 比較ループ置換、graph traversal
   正当化/重複解消/Euler-tour 再スキャン除去、work-log 規律。
5. 各スライスの reviewer-gpt レビュー → ローカル検証 green → push → PR
   （#566 参照）→ CI green → squash auto-merge。

## 注意
- スライス単位のコミットはブランチに蓄積済み（未 push）。1 つの PR として
  全スライス完了後に push する想定（設計の「crate 単位コミット・1 PR」方針）。
- tensordynlen.rs の # Errors バックログ 46 件が core の最大クラスタ。
