# TreeACI per-edge rank stability (#741)

## Problem and decision

TreeACI previously used the maximum output-edge rank as its convergence
stability signal. On the low-temperature CTTN reported in #741, a smaller cut
could continue growing while the network maximum stayed unchanged, allowing a
large localized reconstruction error to be returned as `Converged`.

The scheduler now retains one previous rank per output edge and increments a
stability counter only when no edge grows during a pass. Any growth restarts
the counter; rank decreases are allowed. The existing local residual and
global-guard criteria remain required. This keeps the check O(edges) in storage
and work without extra contractions, samples, or allocations.

`max_ranks` remains a diagnostic summary. `Converged` documents an algorithmic
stopping condition, not a full-grid error certificate, because local residuals
and bounded guard searches do not exhaustively validate every coordinate.

## Reading and verification

The relevant TreeACI scheduler, options, result, tests, design document, and
the repository README/rules were reviewed. The regression fixture exercises the
low-temperature Green's function on CTTN, swapped, and non-block topologies;
the executable reproducer is retained for independent investigation.

Verification completed:

- `cargo fmt --all -- --check`
- `git diff --check`
- `cargo test -p tensor4all-treeaci --release`
- `cargo clippy -p tensor4all-treeaci --all-targets --release -- -D warnings -D clippy::missing_errors_doc -D clippy::missing_panics_doc`

The TreeACI release suite passed, including the two #741 regression tests and
all crate doctests. No tolerance or existing test was weakened.
