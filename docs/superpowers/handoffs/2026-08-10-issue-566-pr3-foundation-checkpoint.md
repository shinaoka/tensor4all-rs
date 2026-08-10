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

## 追記（2026-08-10 続行分）

- ブランチ: `chore/issue-566-pr3-errors`（origin/main から 22 コミット、未 push）。
- **バックログ燃焼**: # Errors ドキュメントバックログを 545 → 321 まで削減。
  対象: tensordynlen(46) / tenferro_bridge(26) / itensorlike tensortrain /
  treetn mod・ops・named_graph / storage / interpolation / affine / graph /
  contract / krylov / backend / site_index_network ほか。
- **教訓（繰り返す場合）**: ① doc 挿入は行番号シフトを避けるため降順処理。
  ② 継続行はファイルの実際のインデント（`    /// ` or `/// `）と完全一致させる
  （二重 `/// ///` 事故）。③ regex 語彙: `dimension`/`length`/`out-of-bounds`/
  `invalid-coefficient` は非対応。信頼語彙は shape/index/dtype/backend/graph/
  configuration + mismatch/failure/overflow/out of bounds 等。④ 既存の非具体
  # Errors セクション（箇条書き・"backend rejects"・"Returns Err if"）は
  属性より上にあり、挿入では置換されない → チェッカー抽出セクションを直接置換
  するか、列 0 の旧セクションを削除する。
- 残り 321 件 + 型付け（設計 step 2–6）+ レイヤリング項目 + ゲート切替。

## 追記（2026-08-10 続行分 2 — バックログ 0 + ゲート切替）

- **# Errors バックログ 545 → 0 達成**（crates/ 全公開面 + tutorial-code）。
  最後まで残ったのは pre-existing 箇条書き/`Returns Err if`/`backend rejects`
  セクションと `/// ///` 二重プレフィックス、`dimension`/`length`/
  `out-of-bounds` 語彙 — 全て解消。
- **PR 3-b 完了**: CI の `check-public-error-docs.py` を repository-wide
  ブロッキングに切替（--changed-from モードと base-SHA 解決ステップを削除）。
  clippy に `-D clippy::missing_errors_doc -D clippy::missing_panics_doc` を
  追加し、workspace --all-targets が green（StorageResult 4 件、
  doc_lazy_continuation 172 件、useless_conversion/Ok-? 8 件を修正）。
- 検証: nextest 2693+10 skip / hdf5 49 / doctests 840 / clippy 両 lint /
  public-error-docs-ok / crate-boundary-ok — 全 green。
- ブランチ: origin/main から 33 コミット。

### 残り（PR 3 完了まで）
1. **設計 step 2–6（公開 fn の typed-error 移行）**: tensorbackend/core →
   treetci/quanticstci → treetn → quanticstransform/HDF5 → 残り全公開面。
   anyhow::Result の公開 fn を typed enum に（trait 層は完了済み）。
2. **レイヤリング（c..i）**: tenferro 直接ルート除去（例外タプル撤去）、
   FullPivLuScalar 移動、core→tcicore 逆転、capi ID-only 文書化/削除 +
   union-find private 化、per-element 比較ループ置換、graph traversal
   正当化/重複解消/Euler-tour 再スキャン、work-log 規律。
3. 最終: 各スライスの reviewer-gpt → 全検証 → PR（#566）→ CI → merge。

## 追記（2026-08-10 続行分 3 — typed-error 移行 step 2 開始）

- **storage スライス**: from_dense_col_major / from_diag_col_major /
  permute_logical_axes / from_dense_f64|c64_col_major /
  from_diag_f64|c64_col_major を `StorageResult`（typed StorageError）に。
  StorageError に source 保存 `Operation` バリアント + `From<anyhow::Error>`
  を追加し、invalid_storage_error が source チェーンを保持。
- **matrix mul スライス**: mat_mul / mat_mul_owned /
  batched_mat_mul_same_shape(_owned) を `Result<_, MatrixMulError>` に
  （source 保存 thiserror ラッパー）。
- 検証: tensorbackend 316 / core+treetn nextest 1319 / clippy -D warnings /
  public-error-docs-ok — green。
- ブランチ: origin/main から 37 コミット。

### 残り（PR 3 完了まで）
- typed-error 移行 step 2–6: tensorbackend 残り（backend linalg wrappers,
  tenferro_bridge 等）+ core（krylov, contract, any_scalar 等）→ treetci/
  quanticstci → treetn → quanticstransform/HDF5 → 残り全公開面。
- レイヤリング（c..i）7 項目。
- 最終: 各スライス reviewer-gpt → 全検証 → PR（#566）→ CI → merge。

## Session 2026-08-12 (PR 3 typed-error step 2 継続)

### tensorbackend 完了（レビュー済み・コミット済み）
- `3281a4d` backend linalg dispatch → BackendLinalgError
- `082530a` tenferro_bridge 18 fn → BridgeError（source 保存、From<anyhow::Error>）
- `7763073` reviewer-gpt finding 4 件修正: (1) lib.rs で BackendLinalgError/BridgeError 再エクスポート, (2) triangular_solve_matrix(_owned) を BackendLinalgError 化, (3) axpby_native_tensor を BridgeError 化, (4) native_tensor_primal_to_storage の "native tensor snapshot materialization failed" コンテキスト復元
- 教訓: blanket `From<E: std::error::Error>` は E0119 自己衝突 → 使わない。trait メソッド（anyhow 返し）と tenferro エラーは map_err が異なる（前者 From 直、後者 anyhow::Error::new wrap）

### core contraction API 完了（レビュー済み・コミット済み）
- `f62c4af` defaults/contract.rs 9 fn + direct_sum.rs 1 fn → Result<_, TensorDynLenError>
- defaults/mod.rs で TensorDynLenError 再エクスポート追加
- trait impl（TensorContractionLike for TensorDynLen）の identity map_err 削除
- 呼び出し側: factorize.rs（anyhow::Error::new wrap）、treetn gse.rs 6 サイト（GseError::Algorithm source を anyhow::Error::new）、itensorlike tensortrain.rs 3 サイト（operation_source に anyhow::Error::new）
- 教訓: `TensorDynLenError::Operation.source` は `Arc<dyn Error + Send + Sync>`（anyhow ではない）→ std::io::Error::other で構築
- reviewer-gpt: finding 0（clean）

### 次のスライス候補
- defaults/index.rs + index_ops.rs（index 生成/置換系）
- any_scalar.rs（13 fn、内部に AnyScalarTensorError 下地あり）
- krylov.rs（9 fn、エラー enum 新設が必要）
- 残り ~100 fn の tensordynlen inherent メソッド

## Session 2026-08-12 後半（typed-error step 2 継続）

### core index 置換 API 完了（レビュー済み・コミット済み）
- `2c4720a` TensorDynLen::replaceind/replaceinds（inherent）→ Result<_, TensorDynLenError>（ShapeMismatch 構造化バリアント使用）、DynIndex::new_bond → TagSetError
- 呼び出し側: factorize（anyhow::Error::new wrap）、treetn gse 多数サイト、simplett_bridge collect 修正、itensorlike tensortrain
- tensor_basic.rs の 3 テストを構造化メッセージ（operation + "shape mismatch"）に更新
- `bfc4236` reviewer 指摘の rustdoc 修正: replaceind(s) の全等価マッチ記述・new_bond の TagSetError 記述・ShapeMismatch バリアント説明拡張

### core AnyScalar eager API 完了（レビュー済み・コミット済み）
- `7a7dec4` AnyScalarError（source 保存 thiserror struct + From<anyhow::Error>）新設、8 公開 fn（primal/enable_grad/grad/clear_grad/backward/detach/try_conj/compose_complex）を型付け
- fallback_result は anyhow のまま; conj() は `.map_err(|e| e.source)` で source を渡す
- モジュール内テストは error.source 経由で検査
- `06f958f` reviewer minor 2 件修正: lib.rs で AnyScalarError 再エクスポート + 実行可能 doc example 追加

### 現状
- ブランチ chore/issue-566-pr3-errors、origin/main より 47 コミット
- セッション内レビュー: 4 スライス（bridge/contraction/index/any_scalar）全て reviewer-gpt で finding 解消済み
- core 残り anyhow 公開 fn: 92 → ~80（tensordynlen inherent が大半）

### 次のスライス候補
- tensordynlen inherent メソッド群（sum/scale/add/axpby/inner_product/sub/neg/permute/select_indices/from_dense/from_diag/to_vec/scalar/zeros/fuse_indices/unfuse_index/stack_along_new_index/index_select/mask_index/from_native/as_native 等 ~80 fn）— 最大ブロック
- block_tensor.rs / col_major_array.rs / krylov.rs
- レイヤリング項目 c..i
