//! Compiler-backed production panic audit.
//!
//! Raw panic findings come from Clippy diagnostics emitted while Cargo checks
//! the workspace's production library and binary targets. `syn` is retained
//! only for the reviewed public assertion baseline; it does not resolve names,
//! types, imports, or macros.

use anyhow::{anyhow, bail, Context, Result};
use serde_json::Value;
use std::collections::{BTreeSet, HashSet};
use std::fs;
use std::path::{Component, Path, PathBuf};
use std::process::Command;
use syn::spanned::Spanned;
use syn::visit::Visit;
use syn::{Attribute, Expr, ExprLit, ImplItem, Item, Lit, Meta, TraitItem, Visibility};

const RAW_CODES: [(&str, &str); 4] = [
    ("clippy::panic", "panic"),
    ("clippy::unreachable", "unreachable"),
    ("clippy::unwrap_used", "unwrap"),
    ("clippy::expect_used", "expect"),
];
const RAW_KINDS: [&str; 4] = ["panic", "unreachable", "unwrap", "expect"];
const ASSERTION_KINDS: [&str; 2] = ["assert", "debug_assert"];

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub(crate) struct Finding {
    path: String,
    line: usize,
    kind: String,
}

impl Finding {
    fn entry(&self) -> String {
        format!("{}:{}:{}", self.path, self.line, self.kind)
    }
}

#[derive(Debug)]
pub(crate) struct AuditReport {
    matched: Vec<Finding>,
    unbaselined: Vec<Finding>,
    stale: Vec<Finding>,
    forbidden_baseline: Vec<Finding>,
}

impl AuditReport {
    pub(crate) fn exit_code_and_print(&self) -> i32 {
        for finding in &self.matched {
            println!("Baseline matched: {}", finding.entry());
        }
        for finding in &self.unbaselined {
            println!("{}", finding.entry());
        }
        for finding in &self.stale {
            println!("Stale baseline: {}", finding.entry());
        }
        for finding in &self.forbidden_baseline {
            eprintln!(
                "Baseline contains forbidden raw panic-style entry: {}",
                finding.entry()
            );
        }

        if !self.unbaselined.is_empty()
            || !self.stale.is_empty()
            || !self.forbidden_baseline.is_empty()
        {
            eprintln!(
                "Audit failed: {} {}, {} {}.",
                self.unbaselined.len(),
                plural(self.unbaselined.len(), "unbaselined finding", None),
                self.stale.len(),
                plural(
                    self.stale.len(),
                    "stale baseline entry",
                    Some("stale baseline entries")
                )
            );
            1
        } else {
            println!("Audit passed: 0 unbaselined findings, 0 stale baseline entries");
            0
        }
    }
}

fn plural(count: usize, singular: &str, plural: Option<&str>) -> String {
    if count == 1 {
        singular.to_owned()
    } else {
        plural
            .map(str::to_owned)
            .unwrap_or_else(|| format!("{singular}s"))
    }
}

#[derive(Clone, Debug)]
struct ProductionTarget {
    package_id: String,
    package_name: String,
    target_name: String,
    kind: String,
    source_path: PathBuf,
}

impl ProductionTarget {
    fn key(&self) -> TargetKey {
        TargetKey {
            package_id: self.package_id.clone(),
            target_name: self.target_name.clone(),
            kind: self.kind.clone(),
        }
    }
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct TargetKey {
    package_id: String,
    target_name: String,
    kind: String,
}

#[derive(Debug)]
struct ProductionSelection {
    root: PathBuf,
    targets: Vec<ProductionTarget>,
    excluded_packages: Vec<String>,
    all_features: bool,
}

impl ProductionSelection {
    fn load(root: &Path) -> Result<Self> {
        let root = canonical_root(root)?;
        let output = Command::new("cargo")
            .args(["metadata", "--format-version", "1", "--no-deps"])
            .current_dir(&root)
            .output()
            .context("failed to execute cargo metadata")?;
        if !output.status.success() {
            bail!(
                "cargo metadata failed: {}",
                command_detail(&output.stdout, &output.stderr)
            );
        }
        let metadata: Value = serde_json::from_slice(&output.stdout)
            .context("cargo metadata returned malformed JSON")?;
        let workspace_members = metadata
            .get("workspace_members")
            .and_then(Value::as_array)
            .ok_or_else(|| anyhow!("cargo metadata omitted workspace_members"))?;
        let member_ids = workspace_members
            .iter()
            .map(|value| {
                value
                    .as_str()
                    .map(str::to_owned)
                    .ok_or_else(|| anyhow!("cargo metadata contains a non-string workspace member"))
            })
            .collect::<Result<HashSet<_>>>()?;
        let packages = metadata
            .get("packages")
            .and_then(Value::as_array)
            .ok_or_else(|| anyhow!("cargo metadata omitted packages"))?;

        let crates_root = root.join("crates");
        let mut targets = Vec::new();
        let mut all_workspace_packages = Vec::new();
        for package in packages {
            let package_id = package
                .get("id")
                .and_then(Value::as_str)
                .ok_or_else(|| anyhow!("cargo metadata package omitted id"))?;
            if !member_ids.contains(package_id) {
                continue;
            }
            let package_name = package
                .get("name")
                .and_then(Value::as_str)
                .ok_or_else(|| anyhow!("cargo metadata package omitted name"))?;
            all_workspace_packages.push(package_name.to_owned());
            let manifest = package
                .get("manifest_path")
                .and_then(Value::as_str)
                .ok_or_else(|| anyhow!("cargo metadata package omitted manifest_path"))?;
            let manifest = ensure_inside(&root, Path::new(manifest), "package manifest")?;
            if !manifest.starts_with(&crates_root) {
                continue;
            }
            let package_targets = package
                .get("targets")
                .and_then(Value::as_array)
                .ok_or_else(|| anyhow!("cargo metadata package omitted targets"))?;
            for target in package_targets {
                let target_kind = target
                    .get("kind")
                    .and_then(Value::as_array)
                    .ok_or_else(|| anyhow!("cargo metadata target omitted kind"))?;
                let kind = if target_kind.iter().any(|value| value == "lib") {
                    Some("lib")
                } else if target_kind.iter().any(|value| value == "bin") {
                    Some("bin")
                } else {
                    None
                };
                let Some(kind) = kind else {
                    continue;
                };
                let target_name = target
                    .get("name")
                    .and_then(Value::as_str)
                    .ok_or_else(|| anyhow!("cargo metadata target omitted name"))?;
                let source_path = target
                    .get("src_path")
                    .and_then(Value::as_str)
                    .ok_or_else(|| anyhow!("cargo metadata target omitted src_path"))?;
                let source_path = ensure_inside(&root, Path::new(source_path), "target source")?;
                targets.push(ProductionTarget {
                    package_id: package_id.to_owned(),
                    package_name: package_name.to_owned(),
                    target_name: target_name.to_owned(),
                    kind: kind.to_owned(),
                    source_path,
                });
            }
        }

        if targets.is_empty() {
            bail!("cargo metadata selected no production lib/bin targets under crates/");
        }
        let selected_package_names = targets
            .iter()
            .map(|target| target.package_name.as_str())
            .collect::<HashSet<_>>();
        let excluded_packages = all_workspace_packages
            .into_iter()
            .filter(|name| !selected_package_names.contains(name.as_str()))
            .collect::<Vec<_>>();
        let all_features = packages.iter().any(|package| {
            let Some(package_id) = package.get("id").and_then(Value::as_str) else {
                return false;
            };
            if !member_ids.contains(package_id) {
                return false;
            }
            let Some(manifest) = package.get("manifest_path").and_then(Value::as_str) else {
                return false;
            };
            let Ok(manifest) = ensure_inside(&root, Path::new(manifest), "package manifest") else {
                return false;
            };
            if !manifest.starts_with(&crates_root) {
                return false;
            }
            package
                .get("features")
                .and_then(Value::as_object)
                .is_some_and(|features| features.keys().any(|name| name != "default"))
        });

        Ok(Self {
            root,
            targets,
            excluded_packages,
            all_features,
        })
    }

    fn expected_keys(&self) -> HashSet<TargetKey> {
        self.targets.iter().map(ProductionTarget::key).collect()
    }
}

fn canonical_root(root: &Path) -> Result<PathBuf> {
    let root = fs::canonicalize(root)
        .with_context(|| format!("audit root does not exist: {}", root.display()))?;
    if !root.is_dir() {
        bail!("audit root is not a directory: {}", root.display());
    }
    Ok(root)
}

fn ensure_inside(root: &Path, path: &Path, description: &str) -> Result<PathBuf> {
    let candidate = if path.is_absolute() {
        path.to_path_buf()
    } else {
        root.join(path)
    };
    let canonical = fs::canonicalize(&candidate)
        .with_context(|| format!("{description} does not exist: {}", candidate.display()))?;
    if !canonical.starts_with(root) {
        bail!(
            "{description} is outside audit root: {}",
            canonical.display()
        );
    }
    Ok(canonical)
}

fn command_detail(stdout: &[u8], stderr: &[u8]) -> String {
    let stderr = String::from_utf8_lossy(stderr).trim().to_owned();
    if !stderr.is_empty() {
        return stderr;
    }
    String::from_utf8_lossy(stdout).trim().to_string()
}

fn run_clippy(selection: &ProductionSelection, all_features: bool) -> Result<BTreeSet<Finding>> {
    let mut command = Command::new("cargo");
    command.args(["clippy", "--workspace"]);
    for package in &selection.excluded_packages {
        command.args(["--exclude", package]);
    }
    command.args(["--lib", "--bins", "--message-format=json"]);
    if all_features {
        command.arg("--all-features");
    }
    command.args(["--"]);
    command.args(["-A", "clippy::all", "--cap-lints", "warn"]);
    for (code, _) in RAW_CODES {
        // `--force-warn` is deliberately stronger than `-W`: a source-level
        // `allow` must not hide a production panic finding.
        command.args(["--force-warn", code]);
    }

    let output = command
        .current_dir(&selection.root)
        .output()
        .context("failed to execute cargo clippy")?;
    if !output.status.success() {
        bail!(
            "cargo clippy {} failed: {}",
            if all_features {
                "(--all-features)"
            } else {
                "(default features)"
            },
            command_detail(&output.stdout, &output.stderr)
        );
    }

    let expected = selection.expected_keys();
    let mut seen_targets = HashSet::new();
    let mut findings = BTreeSet::new();
    let mut build_finished = false;
    for (line_number, line) in String::from_utf8_lossy(&output.stdout).lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let message: Value = serde_json::from_str(line).with_context(|| {
            format!(
                "cargo clippy emitted malformed JSON on line {}",
                line_number + 1
            )
        })?;
        match message.get("reason").and_then(Value::as_str) {
            Some("compiler-artifact") => {
                if let Some(key) = selected_target_key(&message, &expected)? {
                    seen_targets.insert(key);
                }
            }
            Some("compiler-message") => {
                if selected_target_key(&message, &expected)?.is_some() {
                    if let Some(finding) = parse_compiler_message(&message, &selection.root)? {
                        findings.insert(finding);
                    }
                }
            }
            Some("build-finished") => {
                build_finished = true;
                if message.get("success").and_then(Value::as_bool) != Some(true) {
                    bail!("cargo clippy reported an incomplete build");
                }
            }
            _ => {}
        }
    }
    if !build_finished {
        bail!("cargo clippy output did not contain a successful build-finished record");
    }
    if seen_targets != expected {
        let missing = expected
            .difference(&seen_targets)
            .map(|key| format!("{}:{}", key.target_name, key.kind))
            .collect::<Vec<_>>();
        bail!("cargo clippy did not compile every production target: {missing:?}");
    }
    Ok(findings)
}

fn selected_target_key(
    message: &Value,
    expected: &HashSet<TargetKey>,
) -> Result<Option<TargetKey>> {
    let package_id = message
        .get("package_id")
        .and_then(Value::as_str)
        .ok_or_else(|| anyhow!("cargo JSON record omitted package_id"))?;
    let target = message
        .get("target")
        .and_then(Value::as_object)
        .ok_or_else(|| anyhow!("cargo JSON record omitted target"))?;
    let target_name = target
        .get("name")
        .and_then(Value::as_str)
        .ok_or_else(|| anyhow!("cargo JSON target omitted name"))?;
    let kinds = target
        .get("kind")
        .and_then(Value::as_array)
        .ok_or_else(|| anyhow!("cargo JSON target omitted kind"))?;
    let kind = if kinds.iter().any(|value| value == "lib") {
        "lib"
    } else if kinds.iter().any(|value| value == "bin") {
        "bin"
    } else {
        return Ok(None);
    };
    let key = TargetKey {
        package_id: package_id.to_owned(),
        target_name: target_name.to_owned(),
        kind: kind.to_owned(),
    };
    Ok(expected.get(&key).cloned())
}

fn parse_compiler_message(message: &Value, root: &Path) -> Result<Option<Finding>> {
    let diagnostic = message
        .get("message")
        .ok_or_else(|| anyhow!("cargo compiler-message record omitted message"))?;
    let code = diagnostic
        .get("code")
        .and_then(Value::as_object)
        .and_then(|code| code.get("code"))
        .and_then(Value::as_str);
    let Some(code) = code else {
        return Ok(None);
    };
    let Some((_, kind)) = RAW_CODES.iter().find(|(known, _)| *known == code) else {
        if code.starts_with("clippy::") {
            bail!("unsupported Clippy diagnostic code: {code}");
        }
        return Ok(None);
    };
    let spans = diagnostic
        .get("spans")
        .and_then(Value::as_array)
        .ok_or_else(|| anyhow!("{code:?} diagnostic omitted spans"))?;
    let primary_spans = spans
        .iter()
        .filter(|span| span.get("is_primary").and_then(Value::as_bool) == Some(true))
        .collect::<Vec<_>>();
    if primary_spans.is_empty() {
        bail!("{code:?} diagnostic has no primary span");
    }
    for span in primary_spans {
        if let Some((path, line)) = resolve_call_site(span, root)? {
            return Ok(Some(Finding {
                path,
                line,
                kind: (*kind).to_owned(),
            }));
        }
    }
    bail!("{code:?} diagnostic has no local primary or expansion call-site span")
}

fn resolve_call_site(span: &Value, root: &Path) -> Result<Option<(String, usize)>> {
    if let Some(expansion) = span.get("expansion").and_then(Value::as_object) {
        if let Some(call_site) = expansion.get("span") {
            if let Some(found) = resolve_call_site(call_site, root)? {
                return Ok(Some(found));
            }
        }
    }
    let Some(file_name) = span.get("file_name").and_then(Value::as_str) else {
        return Ok(None);
    };
    let Some(line) = span.get("line_start").and_then(Value::as_u64) else {
        return Ok(None);
    };
    if line == 0 || line > usize::MAX as u64 {
        bail!("diagnostic span has invalid line number: {line}");
    }
    let path = Path::new(file_name);
    let candidate = if path.is_absolute() {
        path.to_path_buf()
    } else {
        root.join(path)
    };
    let Ok(canonical) = fs::canonicalize(&candidate) else {
        return Ok(None);
    };
    if !canonical.starts_with(root) {
        return Ok(None);
    }
    let relative = canonical
        .strip_prefix(root)
        .map_err(|_| anyhow!("diagnostic span path escaped audit root"))?;
    let relative = relative
        .to_str()
        .ok_or_else(|| anyhow!("diagnostic span path is not UTF-8"))?
        .replace(std::path::MAIN_SEPARATOR, "/");
    let source = fs::read_to_string(&canonical)
        .with_context(|| format!("failed to read diagnostic source {}", canonical.display()))?;
    let line = line as usize;
    if source.lines().count() < line {
        bail!("diagnostic span line is outside source file: {relative}:{line}");
    }
    Ok(Some((relative, line)))
}

struct AssertionScanner {
    root: PathBuf,
    visited: BTreeSet<PathBuf>,
    findings: BTreeSet<Finding>,
}

impl AssertionScanner {
    fn scan(root: &Path, targets: &[ProductionTarget]) -> Result<BTreeSet<Finding>> {
        let mut scanner = Self {
            root: root.to_owned(),
            visited: BTreeSet::new(),
            findings: BTreeSet::new(),
        };
        for target in targets {
            scanner.scan_file(&target.source_path, true)?;
        }
        Ok(scanner.findings)
    }

    fn scan_file(&mut self, path: &Path, crate_root: bool) -> Result<()> {
        let path = ensure_inside(&self.root, path, "production source")?;
        if !self.visited.insert(path.clone()) {
            return Ok(());
        }
        let source = fs::read_to_string(&path)
            .with_context(|| format!("failed to read production source {}", path.display()))?;
        let file = syn::parse_file(&source)
            .with_context(|| format!("failed to parse production source {}", path.display()))?;
        let relative = path
            .strip_prefix(&self.root)
            .map_err(|_| anyhow!("production source escaped audit root"))?
            .to_string_lossy()
            .replace(std::path::MAIN_SEPARATOR, "/");
        let module_dir = module_directory(&path, crate_root);
        self.scan_items(&file.items, &module_dir, &relative)
    }

    fn scan_items(&mut self, items: &[Item], module_dir: &Path, source_path: &str) -> Result<()> {
        for item in items {
            if has_cfg_test(item_attrs(item)) {
                continue;
            }
            match item {
                Item::Fn(function) if is_public(&function.vis) => {
                    self.scan_block(&function.block, source_path);
                }
                Item::Trait(trait_item) if is_public(&trait_item.vis) => {
                    for trait_item in &trait_item.items {
                        if has_cfg_test(trait_item_attrs(trait_item)) {
                            continue;
                        }
                        if let TraitItem::Fn(function) = trait_item {
                            if let Some(block) = &function.default {
                                self.scan_block(block, source_path);
                            }
                        }
                    }
                }
                Item::Impl(impl_item) => {
                    let trait_impl = impl_item.trait_.is_some();
                    for impl_item in &impl_item.items {
                        if has_cfg_test(impl_item_attrs(impl_item)) {
                            continue;
                        }
                        if let ImplItem::Fn(function) = impl_item {
                            if trait_impl || is_public(&function.vis) {
                                self.scan_block(&function.block, source_path);
                            }
                        }
                    }
                }
                Item::Mod(module) => {
                    let child_dir = module_dir.join(module.ident.to_string());
                    if let Some((_, items)) = &module.content {
                        self.scan_items(items, &child_dir, source_path)?;
                    } else if let Some(path) = resolve_module_file(&self.root, module_dir, module)?
                    {
                        self.scan_file(&path, false)?;
                    }
                }
                // Macro definitions and invocations are intentionally opaque:
                // Clippy is the production expansion boundary for raw paths.
                Item::Macro(_) => {}
                _ => {}
            }
        }
        Ok(())
    }

    fn scan_block(&mut self, block: &syn::Block, source_path: &str) {
        let mut visitor = AssertionVisitor {
            path: source_path,
            findings: &mut self.findings,
        };
        visitor.visit_block(block);
    }
}

struct AssertionVisitor<'a> {
    path: &'a str,
    findings: &'a mut BTreeSet<Finding>,
}

impl<'ast> Visit<'ast> for AssertionVisitor<'_> {
    fn visit_item_macro(&mut self, _node: &'ast syn::ItemMacro) {
        // Do not inspect dormant macro token text.
    }

    fn visit_macro(&mut self, node: &'ast syn::Macro) {
        let Some(segment) = node.path.segments.last() else {
            return;
        };
        let name = segment.ident.to_string();
        let kind = match name.as_str() {
            "assert" => "assert",
            "debug_assert" => "debug_assert",
            _ => return,
        };
        let line = node.span().start().line;
        if line == 0 {
            return;
        }
        self.findings.insert(Finding {
            path: self.path.to_owned(),
            line,
            kind: kind.to_owned(),
        });
    }
}

fn module_directory(path: &Path, crate_root: bool) -> PathBuf {
    let parent = path.parent().unwrap_or_else(|| Path::new("/")).to_owned();
    if crate_root || path.file_name().and_then(|name| name.to_str()) == Some("mod.rs") {
        parent
    } else {
        parent.join(path.file_stem().unwrap_or_default())
    }
}

fn resolve_module_file(
    root: &Path,
    module_dir: &Path,
    module: &syn::ItemMod,
) -> Result<Option<PathBuf>> {
    let relative = path_attribute(&module.attrs)?
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(module.ident.to_string()));
    let candidate = lexical_inside(root, &module_dir.join(relative), "module source")?;
    if path_attribute(&module.attrs)?.is_some() {
        return if candidate.is_file() {
            Ok(Some(ensure_inside(root, &candidate, "module source")?))
        } else {
            Ok(None)
        };
    }
    let file = candidate.with_extension("rs");
    if file.is_file() {
        return Ok(Some(ensure_inside(root, &file, "module source")?));
    }
    let nested = candidate.join("mod.rs");
    if nested.is_file() {
        return Ok(Some(ensure_inside(root, &nested, "module source")?));
    }
    Ok(None)
}

fn lexical_inside(root: &Path, path: &Path, description: &str) -> Result<PathBuf> {
    let mut normalized = PathBuf::new();
    for component in path.components() {
        match component {
            Component::Prefix(prefix) => normalized.push(prefix.as_os_str()),
            Component::RootDir => normalized.push(Path::new(std::path::MAIN_SEPARATOR_STR)),
            Component::CurDir => {}
            Component::ParentDir => {
                if !normalized.pop() {
                    bail!("{description} is outside audit root: {}", path.display());
                }
            }
            Component::Normal(value) => normalized.push(value),
        }
    }
    if !normalized.starts_with(root) {
        bail!("{description} is outside audit root: {}", path.display());
    }
    Ok(normalized)
}

fn path_attribute(attrs: &[Attribute]) -> Result<Option<String>> {
    for attr in attrs {
        if !attr.path().is_ident("path") {
            continue;
        }
        let Meta::NameValue(name_value) = &attr.meta else {
            bail!("#[path] must have a string value");
        };
        let Expr::Lit(ExprLit {
            lit: Lit::Str(value),
            ..
        }) = &name_value.value
        else {
            bail!("#[path] must have a string value");
        };
        return Ok(Some(value.value()));
    }
    Ok(None)
}

fn has_cfg_test(attrs: &[Attribute]) -> bool {
    attrs.iter().any(|attr| {
        let path = attr.path();
        if path.is_ident("cfg") {
            return match &attr.meta {
                Meta::List(list) => list.tokens.to_string().replace(' ', "") == "test",
                _ => false,
            };
        }
        false
    })
}

fn item_attrs(item: &Item) -> &[Attribute] {
    match item {
        Item::Const(item) => &item.attrs,
        Item::Enum(item) => &item.attrs,
        Item::ExternCrate(item) => &item.attrs,
        Item::Fn(item) => &item.attrs,
        Item::ForeignMod(item) => &item.attrs,
        Item::Impl(item) => &item.attrs,
        Item::Macro(item) => &item.attrs,
        Item::Mod(item) => &item.attrs,
        Item::Static(item) => &item.attrs,
        Item::Struct(item) => &item.attrs,
        Item::Trait(item) => &item.attrs,
        Item::TraitAlias(item) => &item.attrs,
        Item::Type(item) => &item.attrs,
        Item::Union(item) => &item.attrs,
        Item::Use(item) => &item.attrs,
        Item::Verbatim(_) => &[],
        _ => &[],
    }
}

fn trait_item_attrs(item: &TraitItem) -> &[Attribute] {
    match item {
        TraitItem::Const(item) => &item.attrs,
        TraitItem::Fn(item) => &item.attrs,
        TraitItem::Type(item) => &item.attrs,
        TraitItem::Macro(item) => &item.attrs,
        TraitItem::Verbatim(_) => &[],
        _ => &[],
    }
}

fn impl_item_attrs(item: &ImplItem) -> &[Attribute] {
    match item {
        ImplItem::Const(item) => &item.attrs,
        ImplItem::Fn(item) => &item.attrs,
        ImplItem::Type(item) => &item.attrs,
        ImplItem::Macro(item) => &item.attrs,
        ImplItem::Verbatim(_) => &[],
        _ => &[],
    }
}

fn is_public(visibility: &Visibility) -> bool {
    matches!(visibility, Visibility::Public(_))
}

fn parse_entry(value: &str) -> Result<Finding> {
    let parts = value.split(':').collect::<Vec<_>>();
    if parts.len() != 3 {
        bail!("invalid baseline entry: {value:?}");
    }
    let path = parts[0];
    let line = parts[1];
    let kind = parts[2];
    if !path.starts_with("crates/")
        || path.contains('\\')
        || path.contains("//")
        || path
            .split('/')
            .any(|part| part.is_empty() || part == "." || part == "..")
        || Path::new(path).is_absolute()
    {
        bail!("baseline path is not normalized: {value:?}");
    }
    let parsed_line = line.parse::<usize>().ok();
    if parsed_line.filter(|line| *line > 0).is_none()
        || parsed_line.map(|line| line.to_string()).as_deref() != Some(line)
    {
        bail!("invalid baseline entry: {value:?}");
    }
    if !RAW_KINDS.contains(&kind) && !ASSERTION_KINDS.contains(&kind) {
        bail!("invalid baseline entry: {value:?}");
    }
    Ok(Finding {
        path: path.to_owned(),
        line: parsed_line.ok_or_else(|| anyhow!("invalid baseline entry: {value:?}"))?,
        kind: kind.to_owned(),
    })
}

fn load_baseline(path: &Path) -> Result<BTreeSet<Finding>> {
    let source = fs::read_to_string(path)
        .with_context(|| format!("failed to read panic baseline {}", path.display()))?;
    let values: Vec<String> = serde_json::from_str(&source).with_context(|| {
        format!(
            "panic baseline is not a JSON array of strings: {}",
            path.display()
        )
    })?;
    let mut baseline = BTreeSet::new();
    for value in values {
        let finding = parse_entry(&value)?;
        if !baseline.insert(finding) {
            bail!("panic baseline contains duplicate entries");
        }
    }
    Ok(baseline)
}

fn build_report(actual: &BTreeSet<Finding>, baseline: &BTreeSet<Finding>) -> AuditReport {
    let matched = baseline.intersection(actual).cloned().collect::<Vec<_>>();
    let unbaselined = actual.difference(baseline).cloned().collect::<Vec<_>>();
    let stale = baseline.difference(actual).cloned().collect::<Vec<_>>();
    let forbidden_baseline = baseline
        .iter()
        .filter(|finding| RAW_KINDS.contains(&finding.kind.as_str()))
        .cloned()
        .collect::<Vec<_>>();
    AuditReport {
        matched,
        unbaselined,
        stale,
        forbidden_baseline,
    }
}

pub(crate) fn audit(root_arg: &Path, baseline_path: &Path) -> Result<AuditReport> {
    let selection = ProductionSelection::load(root_arg)?;
    let baseline = load_baseline(baseline_path)?;
    let mut actual = run_clippy(&selection, false)?;
    if selection.all_features {
        actual.extend(run_clippy(&selection, true)?);
    }
    actual.extend(AssertionScanner::scan(&selection.root, &selection.targets)?);
    Ok(build_report(&actual, &baseline))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::time::{SystemTime, UNIX_EPOCH};

    struct TempRoot {
        path: PathBuf,
    }

    impl TempRoot {
        fn new() -> Self {
            let nonce = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("clock before epoch")
                .as_nanos();
            let path = std::env::temp_dir().join(format!("t4a-panic-audit-{nonce}"));
            fs::create_dir_all(path.join("crates/demo/src")).expect("create fixture root");
            fs::write(path.join("crates/demo/src/lib.rs"), "pub fn demo() {}\n")
                .expect("write fixture source");
            Self { path }
        }
    }

    impl Drop for TempRoot {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.path);
        }
    }

    fn diagnostic(code: &str, span: Value) -> Value {
        json!({
            "reason": "compiler-message",
            "message": {
                "code": {"code": code},
                "spans": [span]
            }
        })
    }

    fn local_span(file_name: &str, line: u64) -> Value {
        json!({
            "file_name": file_name,
            "line_start": line,
            "line_end": line,
            "is_primary": true
        })
    }

    #[test]
    fn compiler_json_accepts_only_exact_codes() {
        let root = TempRoot::new();
        let message = diagnostic(
            "clippy::unwrap_used",
            local_span("crates/demo/src/lib.rs", 1),
        );
        let finding = parse_compiler_message(&message, &root.path)
            .expect("valid diagnostic")
            .expect("known code");
        assert_eq!(finding.entry(), "crates/demo/src/lib.rs:1:unwrap");
        let unknown = diagnostic("clippy::unwrap", local_span("crates/demo/src/lib.rs", 1));
        assert!(parse_compiler_message(&unknown, &root.path).is_err());
    }

    #[test]
    fn compiler_json_follows_expansion_to_local_call_site() {
        let root = TempRoot::new();
        let message = diagnostic(
            "clippy::panic",
            json!({
                "file_name": "/rustc/library/std/src/panic.rs",
                "line_start": 7,
                "is_primary": true,
                "expansion": {
                    "span": local_span("crates/demo/src/lib.rs", 1)
                }
            }),
        );
        let finding = parse_compiler_message(&message, &root.path)
            .expect("valid expansion")
            .expect("known code");
        assert_eq!(finding.entry(), "crates/demo/src/lib.rs:1:panic");
    }

    #[test]
    fn compiler_json_rejects_missing_or_outside_local_span() {
        let root = TempRoot::new();
        for span in [
            json!({"file_name": "crates/demo/src/lib.rs", "is_primary": false}),
            local_span("/tmp/outside.rs", 1),
        ] {
            let message = diagnostic("clippy::expect_used", span);
            assert!(parse_compiler_message(&message, &root.path).is_err());
        }
    }

    #[test]
    fn baseline_lines_reject_noncanonical_numbers() {
        assert!(parse_entry("crates/demo/src/lib.rs:01:assert").is_err());
        assert!(parse_entry("crates/demo/src/lib.rs:0:assert").is_err());
        assert!(parse_entry("crates/demo/src/lib.rs:1:unknown").is_err());
        assert!(parse_entry("demo/src/lib.rs:1:assert").is_err());
    }

    #[test]
    fn report_marks_new_and_stale_entries_without_baselining_raw_paths() {
        let matched = parse_entry("crates/demo/src/lib.rs:1:assert").expect("matched");
        let stale = parse_entry("crates/demo/src/lib.rs:2:assert").expect("stale");
        let raw = parse_entry("crates/demo/src/lib.rs:3:panic").expect("raw");
        let new = parse_entry("crates/demo/src/lib.rs:4:unwrap").expect("new");
        let actual = [matched.clone(), new].into_iter().collect::<BTreeSet<_>>();
        let baseline = [matched, stale, raw].into_iter().collect::<BTreeSet<_>>();
        let report = build_report(&actual, &baseline);
        assert_eq!(
            report.matched,
            vec![parse_entry("crates/demo/src/lib.rs:1:assert").unwrap()]
        );
        assert_eq!(
            report.unbaselined,
            vec![parse_entry("crates/demo/src/lib.rs:4:unwrap").unwrap()]
        );
        assert_eq!(
            report.stale,
            vec![
                parse_entry("crates/demo/src/lib.rs:2:assert").unwrap(),
                parse_entry("crates/demo/src/lib.rs:3:panic").unwrap()
            ]
        );
        assert_eq!(
            report.forbidden_baseline,
            vec![parse_entry("crates/demo/src/lib.rs:3:panic").unwrap()]
        );
    }
}
