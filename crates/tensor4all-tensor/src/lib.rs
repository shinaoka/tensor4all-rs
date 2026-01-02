pub mod storage;
pub mod tensor;

pub use storage::{AnyScalar, DenseStorageFactory, Storage, StorageScalar, SumFromStorage, make_mut_storage, mindim};
pub use tensor::{TensorDynLen, TensorStaticLen, compute_permutation_from_indices, is_diag_tensor, is_diag_tensor_static, diag_tensor_dyn_len, diag_tensor_dyn_len_c64, diag_tensor_static_len, diag_tensor_static_len_c64, unfold_split};

