#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
expected_version="cbindgen 0.29.2"
actual_version=$(cbindgen --version)
if [[ "$actual_version" != "$expected_version" ]]; then
    echo "expected $expected_version, found $actual_version" >&2
    exit 1
fi

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
generated="$tmp/tensor4all_capi.h"
committed="$root/crates/tensor4all-capi/include/tensor4all_capi.h"

cbindgen "$root/crates/tensor4all-capi" \
    --config "$root/crates/tensor4all-capi/cbindgen.toml" \
    --output "$generated"

grep -Fq 'Generated with cbindgen:0.29.2' "$generated"
diff -u "$committed" "$generated"

cat >"$tmp/header.c" <<'EOF'
#include "crates/tensor4all-capi/include/tensor4all_capi.h"
int main(void) { return 0; }
EOF
cat >"$tmp/header.cpp" <<'EOF'
#include "crates/tensor4all-capi/include/tensor4all_capi.h"
int main() { return 0; }
EOF

"${CC:-cc}" -std=c11 -Wall -Wextra -Werror -I"$root" -fsyntax-only "$tmp/header.c"
"${CXX:-c++}" -std=c++17 -Wall -Wextra -Werror -I"$root" -fsyntax-only "$tmp/header.cpp"
