# library-panic-audit

## src/audit.rs

### ` fn entry(&self) -> String` (impl Finding)

### `pub(crate) fn exit_code_and_print(&self) -> i32` (impl AuditReport)

### ` fn plural(count: usize, singular: & str, plural: Option < & str >) -> String`

### ` fn key(&self) -> TargetKey` (impl ProductionTarget)

### ` fn load(root: & Path) -> Result < Self >` (impl ProductionSelection)

### ` fn expected_keys(&self) -> HashSet < TargetKey >` (impl ProductionSelection)

### ` fn cargo_default_features(feature_map: & serde_json :: Map < String , Value >) -> Result < BTreeSet < String > >`

### ` fn cargo_target_kind(kinds: & [Value]) -> Result < Option < String > >`

### ` fn canonical_root(root: & Path) -> Result < PathBuf >`

### ` fn ensure_inside(root: & Path, path: & Path, description: & str) -> Result < PathBuf >`

### ` fn command_detail(stdout: & [u8], stderr: & [u8]) -> String`

### ` fn run_clippy(selection: & ProductionSelection, all_features: bool) -> Result < BTreeSet < Finding > >`

### ` fn command_output_with_timeout(command: & mut Command, timeout: Duration) -> Result < Output >`

### ` fn parse_clippy_output(stdout: & [u8], root: & Path, expected: & HashSet < TargetKey >) -> Result < (HashSet < TargetKey > , BTreeSet < Finding >) >`

### ` fn selected_target_key(message: & Value, expected: & HashSet < TargetKey >) -> Result < Option < TargetKey > >`

### ` fn parse_compiler_message(message: & Value, root: & Path) -> Result < Option < Finding > >`

### ` fn resolve_call_site(span: & Value, root: & Path) -> Result < Option < (String , usize) > >`

### ` fn scan(root: & Path, targets: & [ProductionTarget], all_features: bool) -> Result < BTreeSet < Finding > >` (impl AssertionScanner)

### ` fn scan_file(&mut self, path: & Path, module_dir: & Path, target: & TargetKey, features: & BTreeSet < String >) -> Result < () >` (impl AssertionScanner)

### ` fn scan_items(&mut self, items: & [Item], module_dir: & Path, source_path: & str, target: & TargetKey, features: & BTreeSet < String >) -> Result < () >` (impl AssertionScanner)

### ` fn scan_block(&mut self, block: & syn :: Block, source_path: & str, features: & BTreeSet < String >) -> Result < () >` (impl AssertionScanner)

### ` fn skip_attrs(&mut self, attrs: & [Attribute]) -> bool` (impl AssertionVisitor < '_ >)

### ` fn visit_item(&mut self, node: & 'ast Item)` (impl AssertionVisitor < '_ >)

### ` fn visit_stmt(&mut self, node: & 'ast Stmt)` (impl AssertionVisitor < '_ >)

### ` fn visit_expr(&mut self, node: & 'ast Expr)` (impl AssertionVisitor < '_ >)

### ` fn visit_item_macro(&mut self, _node: & 'ast syn :: ItemMacro)` (impl AssertionVisitor < '_ >)

### ` fn visit_macro(&mut self, node: & 'ast syn :: Macro)` (impl AssertionVisitor < '_ >)

### ` fn expr_attrs(expr: & Expr) -> & [Attribute]`

### ` fn module_directory(path: & Path, crate_root: bool) -> PathBuf`

### ` fn resolve_module_file(root: & Path, module_dir: & Path, module: & syn :: ItemMod, features: & BTreeSet < String >) -> Result < Option < PathBuf > >`

### ` fn lexical_inside(root: & Path, path: & Path, description: & str) -> Result < PathBuf >`

### ` fn parse_meta_list(list: & syn :: MetaList) -> Result < Vec < Meta > >`

### ` fn cfg_enabled(attrs: & [Attribute], features: & BTreeSet < String >) -> Result < bool >`

### ` fn syn_path_name(path: & syn :: Path) -> String`

### ` fn eval_cfg_meta(meta: & Meta, features: & BTreeSet < String >) -> Result < bool >`

### ` fn meta_affects_cfg(meta: & Meta) -> bool`

### ` fn meta_affects_path(meta: & Meta) -> bool`

### ` fn path_meta_value(meta: & Meta, features: & BTreeSet < String >) -> Result < Option < String > >`

### ` fn path_attribute(attrs: & [Attribute], features: & BTreeSet < String >) -> Result < Option < String > >`

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

### ` fn cargo_json_boundary_is_strict_and_tracks_lib_like_targets()`

### ` fn compiler_json_accepts_only_exact_codes()`

### ` fn compiler_json_follows_expansion_to_local_call_site()`

### ` fn compiler_json_rejects_missing_or_outside_local_span()`

### ` fn baseline_lines_reject_noncanonical_numbers()`

### ` fn report_marks_new_and_stale_entries_without_baselining_raw_paths()`

## src/main.rs

### ` fn parse_args() -> Result < Option < Args > >`

### ` fn print_help()`

### ` fn main()`
