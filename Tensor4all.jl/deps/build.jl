# Build script for tensor4all-capi Rust library
#
# This script builds the Rust library and copies it to the deps directory.
# Run via `Pkg.build("Tensor4all")` or `julia deps/build.jl`.

using Libdl

# Paths
const SCRIPT_DIR = @__DIR__
const PACKAGE_DIR = dirname(SCRIPT_DIR)
const WORKSPACE_DIR = dirname(PACKAGE_DIR)  # tensor4all-rs
const CAPI_DIR = joinpath(WORKSPACE_DIR, "tensor4all-capi")

# Output library name (platform-specific)
const LIB_NAME = "libtensor4all_capi." * Libdl.dlext

function find_cargo()
    # Try common cargo locations
    cargo = Sys.which("cargo")
    if cargo !== nothing
        return cargo
    end

    # Try rustup default location
    home = get(ENV, "HOME", get(ENV, "USERPROFILE", ""))
    if !isempty(home)
        rustup_cargo = joinpath(home, ".cargo", "bin", "cargo")
        if isfile(rustup_cargo)
            return rustup_cargo
        end
    end

    error("""
        Could not find cargo (Rust package manager).
        Please install Rust from https://rustup.rs/ and ensure cargo is in your PATH.
        """)
end

function build_library()
    println("Building tensor4all-capi...")

    cargo = find_cargo()
    println("Using cargo: $cargo")

    # Build the library
    cd(WORKSPACE_DIR) do
        run(`$cargo build --release -p tensor4all-capi`)
    end

    # Find the built library
    target_dir = joinpath(WORKSPACE_DIR, "target", "release")
    src_lib = joinpath(target_dir, LIB_NAME)

    if !isfile(src_lib)
        error("Built library not found at: $src_lib")
    end

    # Copy to deps directory
    dst_lib = joinpath(SCRIPT_DIR, LIB_NAME)
    println("Copying $src_lib -> $dst_lib")
    cp(src_lib, dst_lib; force=true)

    println("Build complete!")
    println("Library installed to: $dst_lib")
end

# Run build
build_library()
