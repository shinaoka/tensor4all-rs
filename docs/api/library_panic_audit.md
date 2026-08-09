# library-panic-audit

## src/audit.rs

### ` fn entry(&self) -> String` (impl Finding)

### `pub(crate) fn exit_code_and_print(&self) -> i32` (impl AuditReport)

### ` fn plural(count: usize, singular: & str, plural: Option < & str >) -> String`

### ` fn key(&self) -> TargetKey` (impl ProductionTarget)

### ` fn load(root: & Path) -> Result < Self >` (impl ProductionSelection)

### ` fn expected_keys(&self) -> HashSet < TargetKey >` (impl ProductionSelection)

### ` fn cargo_target_kind(kinds: & [Value]) -> Result < Option < String > >`

### ` fn canonical_root(root: & Path) -> Result < PathBuf >`

### ` fn ensure_inside(root: & Path, path: & Path, description: & str) -> Result < PathBuf >`

### ` fn command_detail(stdout: & [u8], stderr: & [u8]) -> String`

### ` fn run_clippy(selection: & ProductionSelection, all_features: bool) -> Result < (BTreeSet < PathBuf > , BTreeSet < Finding >) >`

### ` fn command_output_with_timeout(command: & mut Command, timeout: Duration) -> Result < Output >`

### ` fn parse_clippy_output(stdout: & [u8], root: & Path, expected: & HashSet < TargetKey >) -> Result < ClippyParse >`

### ` fn selected_target_key(message: & Value, expected: & HashSet < TargetKey >) -> Result < Option < TargetKey > >`

### ` fn parse_compiler_message(message: & Value, root: & Path) -> Result < Option < Finding > >`

### ` fn resolve_call_site(span: & Value, root: & Path) -> Result < Option < (String , usize) > >`

### ` fn dep_info_sources(root: & Path, artifacts: & CompilerArtifacts, expected: & HashSet < TargetKey >) -> Result < BTreeSet < PathBuf > >`

### ` fn artifact_identity(artifact: & CompilerArtifact) -> String`

### ` fn artifact_source_path(root: & Path, artifact: & CompilerArtifact) -> Result < Option < PathBuf > >`

### ` fn locate_dep_infos(root: & Path, artifact: & CompilerArtifact) -> Result < Vec < (PathBuf , ParsedDepInfo) > >`

### ` fn normalize_path(root: & Path, path: & Path) -> PathBuf`

### ` fn dep_info_candidates(artifact: & Path) -> Vec < PathBuf >`

### ` fn parse_make_dep_info(path: & Path, root: & Path) -> Result < ParsedDepInfo >`

### ` fn join_make_lines(source: & str) -> Result < String >`

### ` fn make_rule_separator(line: & str) -> Result < usize >`

### ` fn parse_make_words(source: & str) -> Result < Vec < String > >`

### ` fn not(self) -> Self` (impl CfgState)

### ` fn cfg_state(meta: & Meta) -> Result < CfgState >`

### ` fn parse_meta_list(list: & syn :: MetaList) -> Result < Vec < Meta > >`

### ` fn definitely_test_only(attrs: & [Attribute]) -> Result < bool >`

### ` fn cfg_attr_definitely_false(meta: & Meta) -> Result < bool >`

### ` fn assertion_kind(name: & str) -> Option < & 'static str >`

### ` fn record_assertion(path: & str, line: usize, kind: & str, findings: & mut BTreeSet < Finding >)`

### ` fn scan_assertion_tokens(path: & str, tokens: TokenStream, findings: & mut BTreeSet < Finding >)`

### ` fn scan_macro_rules(path: & str, tokens: TokenStream, findings: & mut BTreeSet < Finding >)`

### ` fn scan_item_macro(path: & str, item: & syn :: ItemMacro, findings: & mut BTreeSet < Finding >)`

### ` fn line_starts(source: & str) -> Vec < usize >`

### ` fn source_line(starts: & [usize], offset: usize) -> usize`

### ` fn skip_quoted(bytes: & [u8], start: usize, quote: u8) -> usize`

### ` fn raw_string_end(bytes: & [u8], start: usize) -> Option < usize >`

### ` fn skip_block_comment(bytes: & [u8], start: usize) -> usize`

### ` fn skip_space_and_comments(bytes: & [u8], index: usize) -> usize`

### ` fn scan_literal_assertions_text(source: & str, path: & str, findings: & mut BTreeSet < Finding >)`

### ` fn scan_items(&mut self, items: & [Item]) -> Result < () >` (impl PublicAssertionVisitor < '_ >)

### ` fn scan_block(&mut self, block: & syn :: Block) -> Result < () >` (impl PublicAssertionVisitor < '_ >)

### ` fn skip_attrs(&mut self, attrs: & [Attribute]) -> bool` (impl PublicAssertionVisitor < '_ >)

### ` fn visit_item(&mut self, node: & 'ast Item)` (impl PublicAssertionVisitor < '_ >)

### ` fn visit_stmt(&mut self, node: & 'ast Stmt)` (impl PublicAssertionVisitor < '_ >)

### ` fn visit_expr(&mut self, node: & 'ast Expr)` (impl PublicAssertionVisitor < '_ >)

### ` fn visit_arm(&mut self, node: & 'ast Arm)` (impl PublicAssertionVisitor < '_ >)

### ` fn visit_item_macro(&mut self, node: & 'ast syn :: ItemMacro)` (impl PublicAssertionVisitor < '_ >)

### ` fn visit_macro(&mut self, node: & 'ast syn :: Macro)` (impl PublicAssertionVisitor < '_ >)

### ` fn scan_assertions(root: & Path, source_files: & BTreeSet < PathBuf >) -> Result < BTreeSet < Finding > >`

### ` fn expr_attrs(expr: & Expr) -> & [Attribute]`

### ` fn item_attrs(item: & Item) -> & [Attribute]`

### ` fn trait_item_attrs(item: & TraitItem) -> & [Attribute]`

### ` fn impl_item_attrs(item: & ImplItem) -> & [Attribute]`

### ` fn is_public(visibility: & Visibility) -> bool`

### ` fn parse_entry(value: & str) -> Result < Finding >`

### ` fn load_baseline(path: & Path) -> Result < BTreeSet < Finding > >`

### ` fn build_report(actual: & BTreeSet < Finding >, baseline: & BTreeSet < Finding >) -> AuditReport`

### `pub(crate) fn audit(root_arg: & Path, baseline_path: & Path) -> Result < AuditReport >`

### ` fn new() -> Self` (impl TempRoot)

### ` fn drop(&mut self)` (impl TempRoot)

### ` fn diagnostic(code: & str, span: Value) -> Value`

### ` fn local_span(file_name: & str, line: u64) -> Value`

### ` fn artifact() -> Value`

### ` fn cargo_json_boundary_is_strict_and_tracks_lib_like_targets()`

### ` fn compiler_json_accepts_only_exact_codes()`

### ` fn compiler_json_follows_expansion_to_local_call_site()`

### ` fn compiler_json_rejects_missing_or_outside_local_span()`

### ` fn dep_info_parser_handles_make_escapes_and_filters_external_sources()`

### ` fn dep_info_parser_handles_rustc_encoded_paths()`

### ` fn dep_info_continuation_separates_adjacent_paths_without_spaces()`

### ` fn dep_info_candidates_cover_hashed_artifact_names()`

### ` fn dep_info_lookup_rejects_stale_heuristic_targets()`

### ` fn dep_info_lookup_requires_artifact_source_in_selected_sources()`

### ` fn cargo_json_preserves_each_artifact_record()`

### ` fn dep_info_parser_and_artifact_lookup_fail_closed()`

### ` fn assertion_scan_handles_fragments_and_macro_tokens()`

### ` fn cfg_evaluator_only_skips_definitely_test_only_content()`

### ` fn baseline_lines_reject_noncanonical_numbers()`

### ` fn report_marks_new_and_stale_entries_without_baselining_raw_paths()`

## src/main.rs

### ` fn parse_args() -> Result < Option < Args > >`

### ` fn print_help()`

### ` fn main()`
