use super::*;

#[test]
fn test_generated_header_uses_named_status_enum() {
    let header = include_str!("../../include/tensor4all_capi.h");

    assert!(header.contains("typedef enum t4a_status_code {"));
    assert!(header.contains("} t4a_status_code;"));
    assert!(header.contains("enum t4a_status_code t4a_last_error_message("));
    assert!(!header.contains("typedef int StatusCode;"));
    assert!(!header.contains("StatusCode"));
}

/// Helper: read the last error message as a Rust String.
fn read_last_error() -> String {
    let mut out_len: libc::size_t = 0;
    // Query required length
    let status = t4a_last_error_message(std::ptr::null_mut(), 0, &mut out_len as *mut libc::size_t);
    assert_eq!(status, T4A_SUCCESS);
    if out_len <= 1 {
        return String::new();
    }
    let mut buf = vec![0u8; out_len];
    let status = t4a_last_error_message(
        buf.as_mut_ptr(),
        buf.len(),
        &mut out_len as *mut libc::size_t,
    );
    assert_eq!(status, T4A_SUCCESS);
    // Exclude null terminator
    String::from_utf8(buf[..out_len - 1].to_vec()).unwrap()
}

#[test]
fn test_last_error_message_null_out_len() {
    let status = t4a_last_error_message(std::ptr::null_mut(), 0, std::ptr::null_mut());
    assert_eq!(status, T4A_NULL_POINTER);
    assert_eq!(read_last_error(), "out_len is null");
}

#[test]
fn test_last_error_message_roundtrip() {
    set_last_error("test error message");
    let msg = read_last_error();
    assert_eq!(msg, "test error message");
}

#[test]
fn test_last_error_message_buffer_too_small_preserves_message_for_retry() {
    set_last_error("hello");
    let mut out_len: libc::size_t = 0;
    let mut short_buf = [0u8; 2]; // too small for "hello\0"
    let status = t4a_last_error_message(
        short_buf.as_mut_ptr(),
        short_buf.len(),
        &mut out_len as *mut libc::size_t,
    );
    assert_eq!(status, T4A_BUFFER_TOO_SMALL);
    assert_eq!(out_len, 6); // "hello" + null

    let required_len = out_len;
    let mut buf = vec![0u8; required_len];
    let status = t4a_last_error_message(
        buf.as_mut_ptr(),
        buf.len(),
        &mut out_len as *mut libc::size_t,
    );
    assert_eq!(status, T4A_SUCCESS);
    assert_eq!(out_len, required_len);
    assert_eq!(buf, b"hello\0");
}

#[test]
fn test_tensor_handle_is_one_pointer_word() {
    assert_eq!(
        std::mem::size_of::<t4a_tensor>(),
        std::mem::size_of::<*const ()>()
    );
    assert_eq!(
        std::mem::align_of::<t4a_tensor>(),
        std::mem::align_of::<*const ()>()
    );
}

#[test]
fn test_last_error_message_query_length_only() {
    set_last_error("abc");
    let mut out_len: libc::size_t = 0;
    let status = t4a_last_error_message(std::ptr::null_mut(), 0, &mut out_len as *mut libc::size_t);
    assert_eq!(status, T4A_SUCCESS);
    assert_eq!(out_len, 4); // "abc" + null
}

#[test]
fn test_err_status_stores_message() {
    let code = err_status("something failed", T4A_INVALID_ARGUMENT);
    assert_eq!(code, T4A_INVALID_ARGUMENT);
    assert_eq!(read_last_error(), "something failed");
}

#[test]
fn test_err_null_stores_message() {
    let ptr: *mut u8 = err_null("null error");
    assert!(ptr.is_null());
    assert_eq!(read_last_error(), "null error");
}

#[test]
fn test_run_catching_null_out_stores_message() {
    let status = run_catching::<u8, _>(std::ptr::null_mut(), || Ok(1));
    assert_eq!(status, T4A_NULL_POINTER);
    assert_eq!(read_last_error(), "out is null");
}

#[test]
fn test_clone_opaque_null_src_stores_message() {
    let mut out = std::ptr::null_mut();
    let status = clone_opaque::<u8>(std::ptr::null(), &mut out);
    assert_eq!(status, T4A_NULL_POINTER);
    assert!(out.is_null());
    assert_eq!(read_last_error(), "source handle is null");
}

#[test]
fn test_unwrap_catch_ok() {
    let result: std::thread::Result<t4a_status_code> = Ok(T4A_SUCCESS);
    assert_eq!(unwrap_catch(result), T4A_SUCCESS);
}

#[test]
fn test_unwrap_catch_panic() {
    let result: std::thread::Result<t4a_status_code> =
        std::panic::catch_unwind(|| panic!("test panic"));
    assert_eq!(unwrap_catch(result), T4A_INTERNAL_ERROR);
    assert_eq!(read_last_error(), "test panic");
}

#[test]
fn test_unwrap_catch_ptr_ok() {
    let mut val = 42u8;
    let result: std::thread::Result<*mut u8> = Ok(&mut val as *mut u8);
    let ptr = unwrap_catch_ptr(result);
    assert!(!ptr.is_null());
}

#[test]
fn test_unwrap_catch_ptr_panic() {
    let result: std::thread::Result<*mut u8> = std::panic::catch_unwind(|| panic!("ptr panic"));
    let ptr = unwrap_catch_ptr(result);
    assert!(ptr.is_null());
    assert_eq!(read_last_error(), "ptr panic");
}

#[test]
fn test_panic_message_str() {
    let result = std::panic::catch_unwind(|| panic!("str msg"));
    let msg = panic_message(&*result.unwrap_err());
    assert_eq!(msg, "str msg");
}

#[test]
fn test_panic_message_string() {
    let result = std::panic::catch_unwind(|| panic!("{}", "string msg"));
    let msg = panic_message(&*result.unwrap_err());
    assert_eq!(msg, "string msg");
}

#[test]
fn test_panic_message_unknown() {
    let result = std::panic::catch_unwind(|| std::panic::panic_any(42i32));
    let msg = panic_message(&*result.unwrap_err());
    assert_eq!(msg, "Unknown panic");
}
