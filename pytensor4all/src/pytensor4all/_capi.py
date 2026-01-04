"""Low-level CFFI bindings to tensor4all C API."""

import os
import sys
from pathlib import Path

from ._ffi import ffi

# Status codes
T4A_SUCCESS = 0
T4A_NULL_POINTER = -1
T4A_INVALID_ARGUMENT = -2
T4A_TAG_OVERFLOW = -3
T4A_TAG_TOO_LONG = -4
T4A_BUFFER_TOO_SMALL = -5
T4A_INTERNAL_ERROR = -6

# Library handle (lazy loaded)
_lib = None


class T4AError(Exception):
    """Base exception for tensor4all errors."""
    pass


class NullPointerError(T4AError):
    """Null pointer error."""
    pass


class InvalidArgumentError(T4AError):
    """Invalid argument error."""
    pass


class TagOverflowError(T4AError):
    """Too many tags error."""
    pass


class TagTooLongError(T4AError):
    """Tag string too long error."""
    pass


class BufferTooSmallError(T4AError):
    """Buffer too small error."""
    pass


class InternalError(T4AError):
    """Internal error."""
    pass


def _get_lib_name() -> str:
    """Get the platform-specific library name."""
    if sys.platform == "darwin":
        return "libtensor4all_capi.dylib"
    elif sys.platform == "win32":
        return "tensor4all_capi.dll"
    else:
        return "libtensor4all_capi.so"


def _find_library() -> Path:
    """Find the shared library path."""
    # First, check in the _lib directory (bundled with package)
    pkg_dir = Path(__file__).parent
    lib_dir = pkg_dir / "_lib"
    lib_name = _get_lib_name()
    lib_path = lib_dir / lib_name

    if lib_path.exists():
        return lib_path

    # Check environment variable
    env_path = os.environ.get("T4A_CAPI_LIB")
    if env_path:
        env_lib = Path(env_path)
        if env_lib.exists():
            return env_lib

    # Check in Cargo target directory (for development)
    # Go up from pytensor4all/src/pytensor4all to tensor4all-rs
    rs_root = pkg_dir.parent.parent.parent
    for build_type in ["release", "debug"]:
        target_path = rs_root / "target" / build_type / lib_name
        if target_path.exists():
            return target_path

    raise OSError(
        f"Could not find {lib_name}. "
        "Please run 'python scripts/build_capi.py' or set T4A_CAPI_LIB environment variable."
    )


def get_lib():
    """Get the loaded library handle (lazy loading)."""
    global _lib
    if _lib is None:
        lib_path = _find_library()
        _lib = ffi.dlopen(str(lib_path))
    return _lib


def check_status(status: int, context: str = ""):
    """Check status code and raise appropriate exception."""
    if status == T4A_SUCCESS:
        return

    prefix = f"{context}: " if context else ""

    if status == T4A_NULL_POINTER:
        raise NullPointerError(f"{prefix}Null pointer")
    elif status == T4A_INVALID_ARGUMENT:
        raise InvalidArgumentError(f"{prefix}Invalid argument")
    elif status == T4A_TAG_OVERFLOW:
        raise TagOverflowError(f"{prefix}Too many tags")
    elif status == T4A_TAG_TOO_LONG:
        raise TagTooLongError(f"{prefix}Tag string too long")
    elif status == T4A_BUFFER_TOO_SMALL:
        raise BufferTooSmallError(f"{prefix}Buffer too small")
    elif status == T4A_INTERNAL_ERROR:
        raise InternalError(f"{prefix}Internal error")
    else:
        raise T4AError(f"{prefix}Unknown error code: {status}")
