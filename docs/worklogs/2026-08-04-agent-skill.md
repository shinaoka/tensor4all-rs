# Agent skill contribution

## Summary

Added a portable Agent Skills specification-compatible guide for using tensor4all-rs. The repository now owns the canonical copy under `skills/use-tensor4all-rs/`, and the README provides install and update commands.

## Material reviewed

- `README.md`, `AGENTS.md`, and `REPOSITORY_RULES.md`
- Current crate manifests, rustdoc, mdBook guides, and relevant public API implementations
- Agent Skills specification: <https://agentskills.io/specification>
- `skills` CLI discovery and update behavior: <https://github.com/vercel-labs/skills>
- The pre-existing local skill and two independent review passes

## Decisions

- Use the vendor-neutral `skills/<name>/SKILL.md` layout. It is discoverable by Agent Skills tooling and avoids maintaining client-specific copies under `.claude/`, `.pi/`, or similar directories.
- Keep the skill with the library so API and convention changes can update it in the same PR. `AGENTS.md` and `REPOSITORY_RULES.md` now include it in public-surface drift checks.
- Link it from the README and use `npx skills update` for installed-copy updates.
- Do not add dedicated CI automation yet. The skill has no executable scripts, and existing documentation/API checks remain the source of truth; format/discovery and links are checked during this PR.

## Review fixes

The imported draft was corrected before publication:

- fixed the column-major rule: the first listed index varies fastest;
- removed a stale dependency-layer diagram;
- replaced the stale fixed release example and incorrect “floating rev” wording;
- clarified that Rust code needs direct dependencies for every imported crate;
- distinguished SVD `rtol` from algorithm option fields named `tolerance`;
- fixed the random-tensor import, TreeTN edge count, HDF5 argument order, and a broken skill-relative link in the recipes;
- removed an unsupported ACI “pivot-search” option claim.

## Verification

- `npx skills add . --list` validated the local source and discovered exactly `use-tensor4all-rs`.
- A local Markdown link check resolved every relative link in the README and skill.
- Thirteen executable recipe blocks compiled and ran against the current workspace. The HDF5 recipe signature was checked against source; its local build is blocked by the installed Homebrew HDF5 2.2.0, which `hdf5-metno-sys` does not recognize.
- `cargo fmt --all -- --check`, `git diff --check`, and the repository-rules dry run passed.
- Clippy, release workspace tests, and rustdoc passed with `tensor4all-hdf5` excluded for the same local HDF5 issue. The full clippy command reached only that external build failure.
- Two independent read-only reviewers found no blockers in the final API content or integration strategy.

## Remaining risk

The detailed crate reference and recipes intentionally track `main` and can drift as public APIs evolve. The repository rules now make that review obligation explicit; volatile signatures should continue to defer to rustdoc rather than gain more duplicated detail.
