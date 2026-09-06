---
name: audit-performance
description: "Use when reviewing or auditing tensor4all-rs code for performance rule violations: before merging changes that touch tensor-train evaluation, interpolation, caches, or contraction hot paths, when a workload is unexpectedly slow, or when scanning the repository for latent performance anti-patterns. Static audit that reports violations of PERFORMANCE_TIPS.md with file and line and never claims a speedup or slowdown without measurement."
license: MIT
---

# Audit performance

Thin launcher with no rule content. The canonical contracts, failure-mode
catalog, and audit procedure live in
[`PERFORMANCE_TIPS.md`](../../PERFORMANCE_TIPS.md).

1. Read `PERFORMANCE_TIPS.md` in full.
2. Follow its audit procedure. Pass any argument (`full` or a path) as the
   scope; with no argument, audit the pending diff.
3. Report each finding as `file:line`, violated section, evidence,
   remediation. Findings are rule violations, not measured regressions; do not
   claim a speedup or slowdown without measurement.
