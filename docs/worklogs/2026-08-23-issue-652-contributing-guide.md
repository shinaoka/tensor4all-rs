# Issue #652 contributing guide worklog

Date: 2026-08-23

## Scope

Add the missing root `CONTRIBUTING.md`, restore the existing `llms.txt` link, and provide a minimal external-contribution intake path without duplicating the full repository policy.

## Contract read

- bug and feature issue forms;
- `AGENTS.md` development, documentation, testing, API, and Git workflow requirements;
- `REPOSITORY_RULES.md` and the architecture vocabulary;
- README and `llms.txt` documentation indexes;
- workspace CI and maintenance commands.

## Implementation

- direct small documentation fixes to pull requests;
- direct bugs and feature/API/cross-repository proposals through the existing issue forms;
- ask nontrivial contributors to align scope and dependency boundaries before implementation;
- link canonical repository rules instead of copying them wholesale;
- list the CI-equivalent local validation and PR evidence expected;
- add the contribution guide to the README documentation index.

## Verification

- all relative links in `CONTRIBUTING.md`, README, and `llms.txt` resolve;
- `scripts/test-mdbook.sh` passed;
- independent external-contributor/read-only review returned **Correct-to-merge**;
- review findings corrected the documented clippy lints, exact nextest/HDF5 split, tool prerequisites, and the matching `AGENTS.md` pre-PR commands.

Exact-state CI-equivalent validation passed after rebasing onto current main:

- explicit-context check and rustdoc;
- formatting and clippy with the two blocking documentation lints;
- non-HDF5 nextest: 3141 passed, 13 skipped;
- HDF5 tests: 57 passed, 4 ignored;
- workspace doctests: 933 passed;
- mdBook, workspace rustdoc, C header, and public API inventory;
- repository-rules, library-panic, public-error-doc, crate-boundary, and coverage-checker test suites.

Repository-rules diff review passed with no findings. GitHub CI remains pending.
