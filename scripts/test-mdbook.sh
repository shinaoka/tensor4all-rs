#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

cargo_profile="${TENSOR4ALL_CARGO_PROFILE:-release}"

probe_log="$(mktemp)"
rustdoc_wrapper_dir="$(mktemp -d)"
trap 'rm -f "$probe_log"; rm -rf "$rustdoc_wrapper_dir"' EXIT

# Force a fresh book-tests rustc invocation so cargo prints the exact --extern
# paths that its doctest harness resolves for the guide snippets.
probe_metadata="mdbook_probe_$(date +%s%N)"
cargo rustc -p book-tests --profile "$cargo_profile" --lib -vv -- -Cmetadata="$probe_metadata" >"$probe_log" 2>&1

rustc_line="$(grep -- '--crate-name book_tests' "$probe_log" | tail -n 1 || true)"
if [[ -z "$rustc_line" ]]; then
    echo "failed to locate the book-tests rustc command" >&2
    tail -n 200 "$probe_log" >&2 || true
    exit 1
fi

extern_args="$(printf '%s\n' "$rustc_line" | grep -oE -- '--extern [^ ]+' | sed 's/^--extern //')"
if [[ -z "$extern_args" ]]; then
    echo "failed to extract --extern flags from the book-tests rustc command" >&2
    tail -n 200 "$probe_log" >&2 || true
    exit 1
fi

real_rustdoc="$(rustup which rustdoc)"
{
    echo '#!/usr/bin/env bash'
    echo 'set -euo pipefail'
    printf 'exec %q ' "$real_rustdoc"
    while IFS= read -r extern_arg; do
        [[ -n "$extern_arg" ]] || continue
        crate_name="${extern_arg%%=*}"
        crate_path="${extern_arg#*=}"
        if [[ "$crate_path" == *.rmeta ]]; then
            rlib_path="${crate_path%.rmeta}.rlib"
            if [[ -f "$rlib_path" ]]; then
                crate_path="$rlib_path"
            fi
        fi
        printf '%q %q ' --extern "${crate_name}=${crate_path}"
    done <<< "$extern_args"
    echo '"$@"'
} > "$rustdoc_wrapper_dir/rustdoc"
chmod +x "$rustdoc_wrapper_dir/rustdoc"

PATH="$rustdoc_wrapper_dir:$PATH" mdbook test docs/book -L "$repo_root/target/$cargo_profile/deps" "$@"
