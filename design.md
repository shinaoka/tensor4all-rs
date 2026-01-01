# tensor4all-rs Design Notes

This document captures the current minimal Rust design for indices and tensors, inspired by ITensors.jl.

## Goals

- Keep compatibility with **ITensors.jl-style dynamic identity** (runtime IDs).
- Also support **strong compile-time checks** where possible (e.g. fixed tensor rank for MPO core tensors).
- Keep **index sizes dynamic** (runtime `usize` per index).

## Implemented crate

- `crates/tensor4all-core`

## Index

Identity is parameterized by the `Id` type.

- `Id = DynId` for ITensors-like runtime identity
- `Id = ZST marker type` (a singleton-like empty type) for compile-time-known identity

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DynId(pub u64);

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Index<Id> {
    pub id: Id,
    pub size: usize,
}
```

Runtime IDs are generated with a thread-safe monotonic counter:

```rust
use std::sync::atomic::{AtomicU64, Ordering};

static NEXT_ID: AtomicU64 = AtomicU64::new(1);

pub fn generate_id() -> u64 {
    NEXT_ID.fetch_add(1, Ordering::Relaxed)
}
```

## Storage

For now we start with a single storage kind:

```rust
#[derive(Debug, Clone)]
pub enum Storage {
    DenseF64(Vec<f64>),
}
```

## Tensor (two variants)

We want two tensor types:

1. **Index sizes: dynamic, Length(rank): dynamic**
2. **Index sizes: dynamic, Length(rank): static** (useful for MPO core tensors etc.)

```rust
use std::sync::Arc;

pub struct TensorDynLen<Id, T> {
    pub indices: Vec<Index<Id>>,
    pub dims: Vec<usize>,
    pub storage: Arc<Storage>,
    _phantom: std::marker::PhantomData<T>,
}

pub struct TensorStaticLen<const N: usize, Id, T> {
    pub indices: [Index<Id>; N],
    pub dims: [usize; N],
    pub storage: Arc<Storage>,
    _phantom: std::marker::PhantomData<T>,
}
```

### Element types

- **Static element type**: pick a concrete `T` and a matching concrete storage strategy.
- **Dynamic element type**: use a marker type `AnyScalar` and dispatch once per operation by matching on `Storage`.

## Tensor networks (MPS/MPO) and shared storage (Arc + COW)

Tensor network objects (like an `MPS`) are highly dynamic and repeatedly update local tensors.

We prefer **shared tensor storage without exclusive ownership**:

- Tensor data is shared via `Arc<Storage>` (thread-safe shared ownership).
- Updates use **copy-on-write (COW)** via `Arc::make_mut(&mut arc_storage)`:
  - If uniquely owned, mutate in place.
  - If shared, clone then mutate.
