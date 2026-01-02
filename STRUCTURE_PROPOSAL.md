# Tensor4All-RS 階層構造提案

## NDTensors.jlの構造分析

NDTensors.jlの主要な階層構造：

```
NDTensors/
├── src/
│   ├── tensorstorage/          # ストレージ抽象型と実装
│   │   ├── tensorstorage.jl    # TensorStorage{ElT} 抽象型
│   │   ├── default_storage.jl  # デフォルトストレージ選択
│   │   ├── set_types.jl        # 型変換
│   │   └── similar.jl          # 類似テンソル作成
│   │
│   ├── tensor/                 # テンソル抽象型
│   │   ├── tensor.jl           # Tensor{ElT,N,StoreT,IndsT} 型
│   │   ├── set_types.jl
│   │   └── similar.jl
│   │
│   ├── dense/                  # Denseストレージ実装
│   │   ├── dense.jl            # Dense{ElT,DataT} 型
│   │   ├── densetensor.jl     # DenseTensor型エイリアス
│   │   ├── tensoralgebra/
│   │   │   ├── contract.jl    # 縮約実装
│   │   │   └── outer.jl       # 外積実装
│   │   └── linearalgebra/
│   │       └── decompositions.jl
│   │
│   ├── diag/                   # Diagストレージ実装
│   │   ├── diag.jl
│   │   ├── diagtensor.jl
│   │   └── tensoralgebra/
│   │
│   ├── blocksparse/            # BlockSparseストレージ実装
│   │   ├── blocksparse.jl
│   │   ├── blocksparsetensor.jl
│   │   └── contract.jl
│   │
│   ├── empty/                  # Emptyストレージ実装
│   │   └── EmptyTensor.jl
│   │
│   ├── abstractarray/          # 抽象配列インターフェース
│   │   ├── mul.jl
│   │   ├── permutedims.jl
│   │   └── tensoralgebra/
│   │       └── contract.jl
│   │
│   ├── tensoroperations/       # テンソル操作ロジック
│   │   ├── contraction_logic.jl
│   │   └── generic_tensor_operations.jl
│   │
│   └── linearalgebra/          # 線形代数操作
│       ├── linearalgebra.jl
│       ├── svd.jl
│       └── symmetric.jl
│
└── ext/                        # 拡張機能（GPUなど）
    ├── NDTensorsCUDAExt/
    ├── NDTensorsAMDGPUExt/
    └── ...
```

## Tensor4All-RS 提案構造

現在の構造をNDTensorsに近づける提案：

```
tensor4all-core/
├── src/
│   ├── storage/                # ストレージ抽象型と実装
│   │   ├── mod.rs              # Storage enum (現在のstorage.rs)
│   │   ├── dense.rs            # DenseStorage実装
│   │   ├── diag.rs             # DiagStorage実装 (将来)
│   │   ├── blocksparse.rs      # BlockSparseStorage実装 (将来)
│   │   ├── empty.rs            # EmptyStorage実装 (将来)
│   │   ├── traits.rs           # Storage traits
│   │   └── default.rs          # デフォルトストレージ選択
│   │
│   ├── tensor/                 # テンソル型と実装
│   │   ├── mod.rs              # Tensor型定義と基本実装
│   │   ├── tensor_dyn_len.rs   # TensorDynLen実装
│   │   ├── tensor_static_len.rs # TensorStaticLen実装
│   │   ├── traits.rs           # Tensor traits
│   │   └── similar.rs          # similar実装
│   │
│   ├── tensoralgebra/          # テンソル代数操作
│   │   ├── mod.rs
│   │   ├── contract.rs         # 縮約実装
│   │   ├── outer.rs            # 外積実装 (将来)
│   │   └── contraction_logic.rs # 縮約ロジック
│   │
│   ├── linearalgebra/          # 線形代数操作
│   │   ├── mod.rs
│   │   ├── svd.rs              # SVD分解 (将来)
│   │   └── decompositions.rs   # その他の分解 (将来)
│   │
│   ├── operations/             # テンソル操作
│   │   ├── mod.rs
│   │   ├── permute.rs          # 置換操作
│   │   ├── mul.rs              # 乗算操作 (将来)
│   │   └── generic.rs          # 汎用操作
│   │
│   ├── index.rs                # Index型 (現在のまま)
│   ├── tagset.rs               # TagSet型 (現在のまま)
│   ├── smallstring.rs          # SmallString型 (現在のまま)
│   └── lib.rs                  # メインモジュール
│
└── ext/                        # 拡張機能 (将来)
    └── ...
```

## 実装方針

### 1. Storage階層の再構成

現在の`Storage` enumを抽象トレイトベースに変更：

```rust
// storage/traits.rs
pub trait TensorStorage: Clone + Send + Sync {
    type Element;
    fn len(&self) -> usize;
    fn eltype(&self) -> TypeId;
    // ...
}

// storage/dense.rs
pub struct DenseStorage<T> {
    data: Vec<T>,
}

impl<T> TensorStorage for DenseStorage<T> {
    type Element = T;
    // ...
}

// storage/mod.rs
pub enum Storage {
    DenseF64(DenseStorage<f64>),
    DenseC64(DenseStorage<Complex64>),
    // 将来的に追加
    // Diag(DiagStorage<...>),
    // BlockSparse(BlockSparseStorage<...>),
}
```

### 2. Tensor階層の再構成

```rust
// tensor/mod.rs
pub trait Tensor: Clone {
    type Storage: TensorStorage;
    type Indices;
    
    fn storage(&self) -> &Self::Storage;
    fn indices(&self) -> &Self::Indices;
    fn dims(&self) -> &[usize];
}

// tensor/tensor_dyn_len.rs
pub struct TensorDynLen<Id, T, Symm = NoSymmSpace> {
    indices: Vec<Index<Id, Symm>>,
    dims: Vec<usize>,
    storage: Arc<Storage>,
    _phantom: PhantomData<T>,
}

impl<Id, T, Symm> Tensor for TensorDynLen<Id, T, Symm> {
    // ...
}
```

### 3. TensorAlgebraの分離

```rust
// tensoralgebra/contract.rs
pub fn contract<A, B>(
    a: &A,
    b: &B,
) -> Result<impl Tensor, ContractionError>
where
    A: Tensor,
    B: Tensor,
{
    // 共通インデックス検出
    let common = common_inds(a.indices(), b.indices());
    
    // ストレージタイプに応じた縮約
    match (a.storage(), b.storage()) {
        (Storage::DenseF64(_), Storage::DenseF64(_)) => {
            contract_dense(a, b, &common)
        }
        // 他のストレージタイプ
        _ => Err(ContractionError::UnsupportedStorage),
    }
}
```

## 移行計画

### Phase 1: Storage階層の再構成
1. `storage/`ディレクトリを作成
2. `Storage` enumを`storage/mod.rs`に移動
3. `DenseStorage`構造体を作成
4. 既存のコードを段階的に移行

### Phase 2: Tensor階層の再構成
1. `tensor/`ディレクトリを作成
2. `Tensor`トレイトを定義
3. `TensorDynLen`と`TensorStaticLen`を`tensor/`に移動
4. 既存のコードを段階的に移行

### Phase 3: TensorAlgebraの分離
1. `tensoralgebra/`ディレクトリを作成
2. `contract`メソッドを`tensoralgebra/contract.rs`に移動
3. ストレージタイプ別の実装を分離

### Phase 4: その他の操作の分離
1. `linearalgebra/`ディレクトリを作成
2. `operations/`ディレクトリを作成
3. 各操作を適切なモジュールに移動

## 利点

1. **モジュール性**: 各機能が明確に分離され、理解しやすい
2. **拡張性**: 新しいストレージタイプや操作を追加しやすい
3. **保守性**: 関連するコードが同じディレクトリに集約される
4. **一貫性**: NDTensors.jlと同様の構造で、既存の知識を活用できる

## 注意点

1. **段階的移行**: 一度にすべてを変更せず、段階的に移行する
2. **後方互換性**: 既存のAPIを維持しながら移行する
3. **テスト**: 各段階でテストを実行し、動作を確認する

