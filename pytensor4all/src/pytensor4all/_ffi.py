"""CFFI definitions for tensor4all C API."""

import cffi

ffi = cffi.FFI()

# C API declarations
ffi.cdef("""
    // Status codes
    typedef int StatusCode;

    // Opaque types
    typedef struct { void* _private; } t4a_index;
    typedef struct { void* _private; } t4a_tensor;

    // Storage kind enum
    typedef enum {
        DenseF64 = 0,
        DenseC64 = 1,
        DiagF64 = 2,
        DiagC64 = 3,
    } t4a_storage_kind;

    // ========================================================================
    // Index functions
    // ========================================================================

    // Lifecycle
    t4a_index* t4a_index_new(size_t dim);
    t4a_index* t4a_index_new_with_tags(size_t dim, const char* tags_csv);
    t4a_index* t4a_index_new_with_id(size_t dim, uint64_t id_hi, uint64_t id_lo, const char* tags_csv);
    void t4a_index_release(t4a_index* ptr);
    t4a_index* t4a_index_clone(const t4a_index* ptr);
    int t4a_index_is_assigned(const t4a_index* ptr);

    // Accessors
    StatusCode t4a_index_dim(const t4a_index* ptr, size_t* out_dim);
    StatusCode t4a_index_id_u128(const t4a_index* ptr, uint64_t* out_hi, uint64_t* out_lo);
    StatusCode t4a_index_get_tags(const t4a_index* ptr, uint8_t* buf, size_t buf_len, size_t* out_len);
    int t4a_index_has_tag(const t4a_index* ptr, const char* tag);

    // Modifiers
    StatusCode t4a_index_add_tag(t4a_index* ptr, const char* tag);
    StatusCode t4a_index_set_tags_csv(t4a_index* ptr, const char* tags_csv);

    // ========================================================================
    // Tensor functions
    // ========================================================================

    // Lifecycle
    void t4a_tensor_release(t4a_tensor* ptr);
    t4a_tensor* t4a_tensor_clone(const t4a_tensor* ptr);
    int t4a_tensor_is_assigned(const t4a_tensor* ptr);

    // Constructors
    t4a_tensor* t4a_tensor_new_dense_f64(
        size_t rank,
        const t4a_index** index_ptrs,
        const size_t* dims,
        const double* data,
        size_t data_len
    );
    t4a_tensor* t4a_tensor_new_dense_c64(
        size_t rank,
        const t4a_index** index_ptrs,
        const size_t* dims,
        const double* data_re,
        const double* data_im,
        size_t data_len
    );

    // Accessors
    StatusCode t4a_tensor_get_rank(const t4a_tensor* ptr, size_t* out_rank);
    StatusCode t4a_tensor_get_dims(const t4a_tensor* ptr, size_t* out_dims, size_t buf_len);
    StatusCode t4a_tensor_get_indices(const t4a_tensor* ptr, t4a_index** out_indices, size_t buf_len);
    StatusCode t4a_tensor_get_storage_kind(const t4a_tensor* ptr, t4a_storage_kind* out_kind);
    StatusCode t4a_tensor_get_data_f64(const t4a_tensor* ptr, double* buf, size_t buf_len, size_t* out_len);
    StatusCode t4a_tensor_get_data_c64(const t4a_tensor* ptr, double* buf_re, double* buf_im, size_t buf_len, size_t* out_len);
""")
