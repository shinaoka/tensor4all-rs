//! Common macros for opaque type implementations
//!
//! This module provides macros to generate common C API functions for opaque types,
//! following the patterns from `sparse-ir-capi`.

/// Generate common opaque type functions: release, clone, is_assigned
///
/// This macro implements the standard lifecycle functions for opaque C API types.
///
/// # Requirements
/// - The type must implement `Clone`
/// - The type name should follow the pattern `t4a_*`
///
/// # Generated functions
/// - `t4a_<TYPE>_release()` - Drops the object
/// - `t4a_<TYPE>_clone()` - Creates a clone
/// - `t4a_<TYPE>_is_assigned()` - Checks if pointer is valid
///
/// # Example
/// ```ignore
/// impl_opaque_type_common!(index);
/// // Generates: t4a_index_release, t4a_index_clone, t4a_index_is_assigned
/// ```
#[macro_export]
macro_rules! impl_opaque_type_common {
    ($type_name:ident) => {
        paste::paste! {
            /// Release the object by dropping it
            ///
            /// # Safety
            /// The caller must ensure that the pointer is valid and not used after this call.
            #[unsafe(no_mangle)]
            pub extern "C" fn [<t4a_ $type_name _release>](obj: *mut [<t4a_ $type_name>]) {
                if obj.is_null() {
                    return;
                }
                // Convert back to Box and drop
                unsafe {
                    let _ = Box::from_raw(obj);
                }
            }

            /// Clone the object
            ///
            /// # Safety
            /// The caller must ensure that the source pointer is valid.
            /// The returned pointer must be freed with `t4a_<type>_release()`.
            ///
            /// # Returns
            /// A new pointer to a cloned object, or null if input is null or panic occurs.
            #[unsafe(no_mangle)]
            pub extern "C" fn [<t4a_ $type_name _clone>](
                src: *const [<t4a_ $type_name>]
            ) -> *mut [<t4a_ $type_name>] {
                if src.is_null() {
                    return std::ptr::null_mut();
                }

                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| unsafe {
                    let src_ref = &*src;
                    let cloned = (*src_ref).clone();
                    Box::into_raw(Box::new(cloned))
                }));

                result.unwrap_or(std::ptr::null_mut())
            }

            /// Check if the object pointer is valid (non-null and dereferenceable)
            ///
            /// # Returns
            /// 1 if the object is valid, 0 otherwise
            #[unsafe(no_mangle)]
            pub extern "C" fn [<t4a_ $type_name _is_assigned>](
                obj: *const [<t4a_ $type_name>]
            ) -> i32 {
                if obj.is_null() {
                    return 0;
                }

                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| unsafe {
                    let _ = &*obj;
                    1
                }));

                result.unwrap_or(0)
            }
        }
    };
}
