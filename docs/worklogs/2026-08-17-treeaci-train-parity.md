# Is TreeACI doing the right amount of work at realistic bond dimension?

Work log. The pre-registration below was fixed before the benchmark was written;
the amendment, result and verdict follow it.

## Pre-registration

Filed before the benchmark is written or run.

## Why

The #646 review asks for a committed reproducible parity benchmark, and asks
that the algorithm be checked at realistic bond dimension before any effort goes
into removing interface overhead. That ordering is right and this measurement
implements it: at chi = 2 essentially all of the cost is the per-call constant,
so an algorithmic defect — a worse contraction order, a higher complexity class,
redundant work per evaluation — would be invisible there.

Existing evidence bears on evaluations but not on work per evaluation. At 18
sites TreeACI converges at rank 196 against the `TensorTrain` arm's 256, with
held-out error 2.0e-5 against 6.1e-5, for 0.74x the operator calls. That says the
interpolation does not request more work; it says nothing about how much work
each request performs.

## Hypotheses

- **H-same**: both implementations perform the same asymptotic work per
  evaluation. Predicts equal fitted scaling exponents against bond dimension,
  within the measurement's resolution.
- **H-worse**: TreeACI performs asymptotically more work — a higher complexity
  class or a systematically worse contraction order. Predicts a materially
  larger exponent.

A message contraction against a chain core is `O(chi^2 d)`, so both are expected
near 2 if H-same holds; the test is whether they agree with each other, not
whether either hits a particular number.

## What is measured

For each implementation, on identical inputs: wall-clock time for one
`elementwise` / `tree_elementwise` call, the final maximum rank, and the
held-out reconstruction error against a dense oracle.

## Cases

Complete list, fixed now. All are run; none is added, dropped or re-run after
results are seen.

- topology: chain of 16 nodes, local dimension 2
- input bond dimension: {16, 32, 64, 128}
- two random inputs at each bond dimension, seed 2026, elementwise product
- tolerance 1e-8, `max_bond_dim` 4096 so neither arm saturates
- both arms seeded from the same initial guess, built once per case
- scalar `f64`, single-threaded

Bond 16 and 32 are included below the range under discussion so the exponent is
fitted across a range rather than between two points; the verdict is read from
the fit over all four.

## Build and host

- Source: `crates/tensor4all-aci/benches/treeaci_parity.rs`, added by this work.
  It lives on the train side because `tensor4all-treeaci`'s manifest is asserted
  by its own test not to mention `tensor4all-aci`.
- Baseline commit: `bc8539f5` on `treeaci-crate`, plus that benchmark.
- Build: `cargo bench -p tensor4all-aci`, release.
- Criterion defaults; noise handled by its confidence intervals.

## Statistic and decision rule, fixed in advance

For each implementation fit an ordinary least-squares slope of `log(median
time)` against `log(bond dimension)` over the four cases. Call them `s_tree` and
`s_train`.

- **`s_tree - s_train <= 0.3`**: consistent with H-same. Conclusion: the
  algorithm is not the problem, and the remaining gap is the per-call constant.
  Whether to remove that constant is then a cost/benefit question at realistic
  chi, decided on the measured constant share, not on principle.
- **`s_tree - s_train > 0.3`**: consistent with H-worse. Conclusion: there is an
  algorithmic defect that interface work would not fix. Stop the overhead work
  and find it.

Accuracy is a gate, not a comparison: if either arm's held-out error exceeds 1%
of the largest held-out truth magnitude, that arm did not converge and the case
is reported as invalid rather than compared.

The result is recorded in both branches, including when it contradicts the
hypothesis. No case is re-run to move a result across the boundary.

## What this does not establish

A synthetic chain with random inputs, not a physical workload. It compares work
per call at fixed bond dimension; it does not re-establish the accuracy and rank
parity already measured, and it does not speak to branched topologies, where
there is no train implementation to compare against.

## Amendment, before any timing was produced

The registered configuration seeds both arms from the same converted initial
guess. It does not run: `tree_elementwise` rejects a guess whose rank matches the
inputs' with `canonicalize: factorization failed`, from a singular fixed-pivot
solve, while `tensor4all-aci` accepts the same object converted from the same
tensor train. Guesses of rank 1 and 2 are accepted and an actual input used as a
guess is not, so the failure is rank-dependent rather than specific to the
generator.

Both arms therefore run unseeded, which keeps the comparison paired. The
amendment is driven by an error, not by a timing: no timing had been produced
when it was made. The rejection is recorded as a defect in its own right.

## Result

`cargo bench -p tensor4all-aci --bench treeaci_parity`, 16 sites, local
dimension 2, tolerance 1e-8, `max_bond_dim` 4096, guard off, criterion medians:

| input chi | train rank | tree rank | train error | tree error | train | tree | ratio | tree us / evaluated point |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | 32 | 32 | 9.9e-9 | 8.7e-9 | 18.5 ms | 488 ms | 26x | 8.2 |
| 32 | 24 | 24 | 9.4e-9 | 9.7e-9 | 14.1 ms | 1970 ms | 139x | 35.9 |
| 64 | 19 | 19 | 8.8e-9 | 9.6e-9 | 24.3 ms | 7354 ms | 303x | 173.0 |
| 128 | 17 | 16 | 9.1e-9 | 9.0e-9 | 32.3 ms | 21855 ms | 677x | 545.1 |

Both arms converge at every case, to matching rank and comparable error, so the
accuracy gate passes and the cases are comparable.

Fitted slopes of `log(median time)` against `log(input chi)`: **train 0.32, tree
1.84**, difference **1.52**.

### Verdict against the registered rule

`s_tree - s_train = 1.52 > 0.3`, which is the second branch: **consistent with
H-worse**. There is an algorithmic difference, not a constant factor, and no
amount of interface work would remove it.

### What the difference is

It is not more evaluations. TreeACI evaluates *fewer* points as chi rises —
59488, 54888, 42496, 40096 — while its time rises 45x. Cost per evaluated point
is 8.2, 35.9, 173.0, 545.1 us, a fitted slope of **2.04** against chi.

So TreeACI pays `O(chi^2)` per evaluated point and `tensor4all-aci` does not.
Evaluating a chain from scratch is `O(n chi^2)`; the train path reuses cached
partial contractions, so an evaluation that differs from the last one in a few
sites costs only the changed part. TreeACI rebuilds what it needs each time.

### Correction to the earlier conclusion

`docs/worklogs/2026-08-17-treeaci-per-evaluation-cost.md` concluded that the gap
is a per-call constant and recommended deferring the cross-call cache the #646
review asks for. That conclusion holds at chi = 2, where the constant is
essentially all of the cost, and **it does not hold at realistic chi**: at 128
the constant is under a third of TreeACI's own time and the gap is dominated by
the `O(chi^2)` term. The recommendation to defer the cache was wrong, and the
review's ordering — parity at realistic bond dimension before any interface work
— is what surfaced it.
