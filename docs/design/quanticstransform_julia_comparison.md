# Quantics Transform: Rust vs Julia (Quantics.jl) 実装比較

## 目的

Rust crate `tensor4all-quanticstransform` が Julia パッケージ `Quantics.jl` と同じアルゴリズムを実装しているか、
同じケースがテストでカバーされているかを調査する。

---

## 1. 対象コードベース

| | Rust | Julia |
|---|---|---|
| パス | `crates/tensor4all-quanticstransform/` | `../Quantics.jl/` |
| コード量 | ~3,138行 (8モジュール, binaryop.rs 削除済み) | ~2,276行 (9モジュール) |
| テンソル形式 | rank-3 (bond, fused_site, bond) | rank-4 (bond, out_site, in_site, bond) via ITensors |
| データ型 | `Complex64` 統一 | `Float64` / `BigFloat` 混合 |

### Rust モジュール構成

```
src/
├── lib.rs              # モジュールエクスポート
├── common.rs (453行)   # 共通型、TensorTrain変換ユーティリティ
├── flip.rs (250行)     # Flip: f(x) = g(2^R - x)
├── shift.rs (256行)    # Shift: f(x) = g(x + offset) mod 2^R
├── phase_rotation.rs (174行) # Phase rotation: f(x) = exp(iθx) g(x)
├── cumsum.rs (211行)   # Cumulative sum: y_i = Σ_{j<i} x_j
├── fourier.rs (411行)  # Quantics Fourier Transform (QFT)
└── affine.rs (1860行)  # Affine transform: y = Ax + b
```

> **Note (2026-04-18):** `binaryop.rs` was removed in issue #428. The binary-op
> functionality is a historical artifact predating general affine support.
> Two-variable interleaved operations are now handled by `affine.rs` /
> `affine_operator` at the Rust level, or by the Julia-side reimplementation
> using the affine pullback + tensor-level SVD workflow.

### Julia モジュール構成

```
src/
├── Quantics.jl         # メインモジュール
├── util.jl (591行)     # ユーティリティ (MPS/MPO操作)
├── tag.jl (64行)       # タグベースサイト検索
├── binaryop.jl (476行) # 二項演算 + アフィン変換
├── mul.jl (167行)      # 乗算演算
├── mps.jl (37行)       # MPS構築
├── fouriertransform.jl (155行) # Fourier変換
├── imaginarytime.jl (107行)    # 虚時間/松原周波数変換
├── transformer.jl (255行)      # 軸変換 (flip, shift, phase等)
└── affine.jl (400行)   # アフィン変換 (fused quantics)
```

---

## 2. オペレータ別アルゴリズム比較

### 2.1 Flip (`f(x) = g(2^R - x)`)

**アルゴリズム一致度: 一致**

- コア演算: 2の補数演算 + carry伝搬。単一サイトテンソルの算術は同一。
- carry方向: TT内の方向は逆(Rust: right-to-left, Julia: left-to-right)だが、物理的carry(LSB→MSB)は同じで結果は同等。
- 境界条件: **差異あり**
  - Julia: `bc ∈ {+1, -1}` (periodic / antisymmetric)
  - Rust: `BoundaryCondition::Periodic` / `BoundaryCondition::Open`
  - Antisymmetric BC (bc=-1) は Rust に存在しない
- `rev_carrydirec`: Julia のみサポート (MSB/LSB両方向のcarry伝搬)

### 2.2 Shift (`f(x) = g(x + offset)`)

**アルゴリズム一致度: 一致 (実装方法は異なる)**

- Julia: 汎用 `_binaryop_tensor` を使用 (carry states {-1,0,1}, bond dim=3)
- Rust: 専用実装 (carry states {0,1}, bond dim=2) — 加算のみなので最適化されている
- offset分解: 両方とも `offset = nbc * 2^R + offset_mod`, `bc^nbc` でスケーリング
- 境界条件: flip と同じ差異 (antisymmetric vs open)
- `rev_carrydirec`: Julia のみ

### 2.3 Phase Rotation (`f(x) = e^{iθx} g(x)`)

**アルゴリズム一致度: 一致**

- 数学: `e^{iθx} = Π_n e^{iθ·2^{R-1-n}·x_n}` を利用、bond dim=1 の対角演算
- **重要な精度差異**:
  - Julia: `BigFloat(256bit)` で `θ·2^{R-n} mod 2π` を計算
  - Rust: `f64` のみ
  - **R ≳ 50 で Rust は精度劣化の可能性あり** (`2^50 > 2^52` の精度限界に近づく)

### 2.4 Cumulative Sum (`y_i = Σ_{j<i} x_j`)

**アルゴリズム一致度: 一致**

- コア: 2状態オートマトン (「まだ等しい」/「比較済み」) による strict upper triangle
- semantics: 両方とも strict (対角線を含まない)
- 差異:
  - Julia: `:upper` / `:lower` 両方サポート、Rust は1方向のみ
  - Julia にバグあり: `cumsum()` が未定義の `upper_triangle_matrix` を呼び出す (テストでは `upper_lower_triangle_matrix` を直接使用して回避)

### 2.5 Fourier Transform (QFT)

**アルゴリズム一致度: 一致 (Chen & Lindsey, arXiv:2404.03182)**

- Chebyshev 補間グリッド: 同一 (K=25, 同じ barycentric weights)
- Core tensor: 同一の数式 `A[α,τ,σ,β] = P_α(x) · exp(2πi·sign·x·τ)`
- 正規化: `1/√2` per site (両方同じ)

差異:

| 項目 | Rust | Julia |
|------|------|-------|
| デフォルト forward sign | -1 | +1 (Quantics.jl wrapper経由) |
| 圧縮方法 | LU 分解 | SVD |
| 逆変換構築 | sign=+1 で再計算 | `conj(reverse(forward))` |
| Origin shift | 未サポート | サポート (BigFloat精度) |
| 2次圧縮 | なし | `truncate(cutoff=1e-25)` |
| 最小R | 1 | 2 |

### 2.6 Binary Operation (`g(ax+by, cx+dy)`)

> **注 (2026-04-19):** 以下は 2026-02-13 時点の記録。issue #428 で Rust 側の `binaryop_operator` は削除済み。issue #431 で `affine_pullback_operator` も削除し、pullback は `affine_operator(...).transpose()` で取得する方針に変更。Julia の binaryop.jl は `affine_operator + transpose + SVD` ベースで再実装する follow-up PR で対応予定。

**アルゴリズム一致度: 部分的一致**

- 単一テンソル演算: 一致 (carry states {-1,0,1}, 同じ算術)
- **重大な差異**:
  - Julia: N変数完全対応 (`binaryop_tensor_multisite`)。2出力変数の同時変換が可能。
  - **Rust: 1変数のみ** (`coeffs2` は未使用、コメントで "incomplete" と記載)
  - Julia: `(-1,-1)` 係数を flip 合成で対応。Rust はエラー。
  - Julia: 双方向 carry (`rev_carrydirec`)。Rust は固定1方向。
  - Julia: MPO 圧縮 (`truncate`)。Rust はなし。

### 2.7 Affine Transform (`y = Ax + b`)

**アルゴリズム一致度: 一致 (コア演算)**

- 有理数→整数変換: 同一 (LCM でスケーリング)
- Carry 伝搬: 同一 (LSB→MSB)
- bit 抽出: 方法は異なるが結果は同等 (Julia: 算術右シフト, Rust: 符号分離)

差異:

| 項目 | Julia | Rust |
|------|-------|------|
| `active_to_passive` (逆変換) | あり | **なし** |
| `M ≤ N` 制約 | 強制 | なし (M > N も許容) |
| 次元別 BC | 全次元同一 | **次元ごとに異なる BC 可** |
| 拡張ループ (大 `|b|`) | 行列乗算 | ベクトル縮約 (メモリ効率良) |
| Carry 順序 | Dict 挿入順 | ソート済み (決定的) |

---

## 3. Rust に存在しない Julia 機能

| モジュール/機能 | 説明 | 重要度 |
|----------------|------|--------|
| `imaginarytime.jl` | 虚時間/松原周波数変換 (`to_wn`, `to_tau`, `poletomps`) | 高 |
| `mul.jl` | 行列乗算/要素積 (`automul`, `MatrixMultiplier`) | 中 |
| `mps.jl` | MPS構築 (`onemps`, `expqtt`) | 低 |
| `transformer.jl` | `reverseaxis` (多変数flip), `flipop_to_negativedomain` | 中 |
| `binaryop.jl` | 多変数 `affinetransform`, tag付き高レベルAPI | 高 |
| `util.jl` | `unfuse_site_indices`, `makesitediagonal`, `rearrange_site_indices` 等 | 中 |
| `tag.jl` | タグベースのサイト検索 | 低 |
| 全般 | Antisymmetric BC (`bc = -1`) | 中 |
| 全般 | `rev_carrydirec` (carry方向切り替え) | 中 |
| Fourier | Origin shift (`originsrc`, `origindst`) | 中 |
| Phase rotation | BigFloat 精度 (大きな R で必要) | 中 |

---

## 4. テストカバレッジ ギャップ分析

### 4.1 Rust に不足しているテスト (優先度順)

> **注 (2026-04-18):** 下記のうち binaryop 関連の未解決項目は issue #428 で Rust 側の削除により moot 化した。項目 1（binaryop 数値正当性テスト皆無）と項目 4 の binaryop 部分は現時点では該当しない。

| # | ギャップ | 影響度 | 詳細 |
|---|---------|--------|------|
| 1 | **binaryop の数値正当性テストが皆無** | 致命的 | 全テストがスモークテスト(生成のみ)。Julia は81通りの (a,b,c,d) を検証 |
| 2 | **Fourier のフェーズ検証なし** | 高 | magnitude のみ確認。位相誤りが検出不能 |
| 3 | **Random MPS テストなし** | 高 | 全て product state。線形性の検証が不十分 |
| 4 | **多変数演算テストなし** | 高 | 2D/3D の flip, shift, binaryop, affine が未テスト |
| 5 | **Antisymmetric BC テストなし** | 中 | そもそも未実装 |
| 6 | **Carry方向テストなし** | 中 | `rev_carrydirec` 未実装のため |
| 7 | **Fourier R=2, R=4 テストなし** | 低 | R=3 のみ |

### 4.2 Julia に不足しているテスト

| # | ギャップ |
|---|---------|
| 1 | Open BC テスト (affine以外) |
| 2 | R=0, R=1 のエッジケース |
| 3 | エラーハンドリングテスト |
| 4 | R=8, R=16 等の大規模テスト |

### 4.3 テスト R 値の比較

| オペレータ | Julia | Rust |
|-----------|-------|------|
| Flip | 2, 3 | 0(error), 1, 3, 4 |
| Shift | 3 | 0(error), 1, 3, 4 |
| Phase rotation | 3 | 3 |
| Cumsum | 3 | 3 |
| Fourier | 2, 3, 4 | 3 |
| Binaryop | 2, 3 | 0(error), 2, 4 (削除済み — issue #428) |
| Affine | 1, 2, 3, 4, 5, 6 | 0(error), 1, 2, 3, 4, 5, 6, 8, 16 |

### 4.4 テスト境界条件の比較

| オペレータ | Julia | Rust |
|-----------|-------|------|
| Flip | Periodic, Antisymmetric | Periodic, Open |
| Shift | Periodic, Antisymmetric | Periodic, Open |
| Binaryop | Periodic, Antisymmetric | Periodic (削除済み — issue #428) |
| Affine | Periodic, Antisymmetric, Open | Periodic, Open |

---

## 5. 結論

> **注 (2026-04-18):** 項目 1（binaryop 数値正当性テスト）と項目 2（binaryop multi-var 対応）は issue #428 で Rust 側の削除により resolved。本 conclusion 上位に残るべきは項目 3 以降。

### アルゴリズム一致度

コア演算 (flip, shift, phase_rotation, cumsum, fourier, affine の核心部分) は
Julia と**同じアルゴリズムを忠実に実装**している。

### 主な実装ギャップ (対応推奨順)

1. **binaryop の多変数対応が未完成** — `coeffs2` 未使用、`(-1,-1)` 非対応
2. **binaryop のテスト不足** — 数値正当性テストが皆無 (最大のリスク)
3. **Fourier テストのフェーズ検証不足** — magnitude のみでは不十分
4. **Antisymmetric BC** が `BoundaryCondition` enum に存在しない
5. **imaginarytime モジュール** が完全に欠落
6. **Phase rotation の BigFloat 精度** が未実装 (大きな R で問題)
7. **Fourier の origin shift** 機能なし

### Rust 側の優位点

- R=0, R=1 のエッジケースとエラーハンドリング
- Open BC の広範なテスト
- Affine の次元別 BC サポート
- Affine の `M > N` 対応 (Julia は `M ≤ N` のみ)
- Affine 拡張ループのメモリ効率

---

*調査日: 2026-02-13*
*比較対象: tensor4all-quanticstransform (Rust) vs Quantics.jl v0.4.7 (Julia)*
