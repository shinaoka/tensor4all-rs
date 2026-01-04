#!/usr/bin/env python3
"""Build tensor4all-capi and copy to the package."""

import shutil
import subprocess
import sys
from pathlib import Path


def get_lib_name() -> str:
    """Get the platform-specific library name."""
    if sys.platform == "darwin":
        return "libtensor4all_capi.dylib"
    elif sys.platform == "win32":
        return "tensor4all_capi.dll"
    else:
        return "libtensor4all_capi.so"


def main():
    # Find paths
    script_dir = Path(__file__).parent
    pkg_dir = script_dir.parent
    rs_root = pkg_dir.parent  # tensor4all-rs directory
    lib_dir = pkg_dir / "src" / "pytensor4all" / "_lib"

    lib_name = get_lib_name()

    # Check if we're in the right place
    if not (rs_root / "tensor4all-capi").exists():
        print(f"Error: tensor4all-capi not found in {rs_root}")
        print("Make sure you're running from the pytensor4all directory")
        sys.exit(1)

    # Build the Rust library
    print("Building tensor4all-capi...")
    result = subprocess.run(
        ["cargo", "build", "--release", "-p", "tensor4all-capi"],
        cwd=rs_root,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print("Error building tensor4all-capi:")
        print(result.stderr)
        sys.exit(1)

    print(result.stdout)

    # Find the built library
    target_lib = rs_root / "target" / "release" / lib_name
    if not target_lib.exists():
        print(f"Error: Built library not found at {target_lib}")
        sys.exit(1)

    # Ensure _lib directory exists
    lib_dir.mkdir(parents=True, exist_ok=True)

    # Copy the library
    dest_lib = lib_dir / lib_name
    print(f"Copying {target_lib} -> {dest_lib}")
    shutil.copy2(target_lib, dest_lib)

    print("Done!")
    print(f"Library installed at: {dest_lib}")


if __name__ == "__main__":
    main()
