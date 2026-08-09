use anyhow::{anyhow, bail, Context, Result};
use proc_macro2::{Delimiter, Group, Ident, Span, TokenStream, TokenTree};
use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::fs;
use std::path::{Component, Path, PathBuf};
use syn::parse::Parser;
use syn::punctuated::Punctuated;
use syn::spanned::Spanned;
use syn::visit::{self, Visit};
use syn::{
    Attribute, Expr, ExprCall, ExprMethodCall, ExprPath, File, ImplItem, Item, Meta, TraitItem,
    Type, Visibility,
};

const RAW_KINDS: [&str; 4] = ["panic", "unreachable", "unwrap", "expect"];
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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum PanicKind {
    Panic,
    Unreachable,
    Unwrap,
    Expect,
    Assert,
    DebugAssert,
}

impl PanicKind {
    fn name(self) -> &'static str {
        match self {
            Self::Panic => "panic",
            Self::Unreachable => "unreachable",
            Self::Unwrap => "unwrap",
            Self::Expect => "expect",
            Self::Assert => "assert",
            Self::DebugAssert => "debug_assert",
        }
    }

    fn from_name(name: &str) -> Option<Self> {
        Some(match name {
            "panic" => Self::Panic,
            "unreachable" => Self::Unreachable,
            "unwrap" => Self::Unwrap,
            "expect" => Self::Expect,
            "assert" => Self::Assert,
            "debug_assert" => Self::DebugAssert,
            _ => return None,
        })
    }

    fn is_raw(self) -> bool {
        matches!(
            self,
            Self::Panic | Self::Unreachable | Self::Unwrap | Self::Expect
        )
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum StandardType {
    Option,
    Result,
}

struct SourceNode {
    source: String,
    file: File,
    initial_source: bool,
    crate_root: bool,
    edges: Vec<ModuleEdge>,
}

#[derive(Clone, Debug)]
struct ModuleEdge {
    target: PathBuf,
    test_only: bool,
}

#[derive(Clone, Debug)]
struct DiscoveredSource {
    path: PathBuf,
    crate_root: bool,
}

struct Project {
    root: PathBuf,
    nodes: BTreeMap<PathBuf, SourceNode>,
}

impl Project {
    fn load(root: &Path) -> Result<Self> {
        let discovered = discover_sources(root)?;
        if discovered.is_empty() {
            bail!(
                "no Rust crate sources found under {}",
                root.join("crates").display()
            );
        }

        let mut nodes = BTreeMap::new();
        for source in discovered {
            let (source_text, file) = parse_source(&source.path)?;
            nodes
                .entry(source.path.clone())
                .and_modify(|node: &mut SourceNode| {
                    node.crate_root |= source.crate_root;
                })
                .or_insert(SourceNode {
                    source: source_text,
                    file,
                    initial_source: true,
                    crate_root: source.crate_root,
                    edges: Vec::new(),
                });
        }

        let mut queue: VecDeque<PathBuf> = nodes.keys().cloned().collect();
        while let Some(path) = queue.pop_front() {
            let edges = {
                let node = nodes
                    .get(&path)
                    .ok_or_else(|| anyhow!("internal source graph error for {}", path.display()))?;
                collect_module_edges(&node.file, &path, root)?
            };
            for edge in edges {
                if !nodes.contains_key(&edge.target) {
                    let (source_text, file) = parse_source(&edge.target)?;
                    nodes.insert(
                        edge.target.clone(),
                        SourceNode {
                            source: source_text,
                            file,
                            initial_source: false,
                            crate_root: false,
                            edges: Vec::new(),
                        },
                    );
                    queue.push_back(edge.target.clone());
                }
                nodes
                    .get_mut(&path)
                    .ok_or_else(|| anyhow!("internal source graph error for {}", path.display()))?
                    .edges
                    .push(edge);
            }
        }

        Ok(Self {
            root: root.to_owned(),
            nodes,
        })
    }

    fn production_reachable(&self) -> BTreeSet<PathBuf> {
        let mut incoming = BTreeMap::<PathBuf, usize>::new();
        for path in self.nodes.keys() {
            incoming.insert(path.clone(), 0);
        }
        for node in self.nodes.values() {
            for edge in &node.edges {
                *incoming.entry(edge.target.clone()).or_default() += 1;
            }
        }

        let mut reachable = BTreeSet::new();
        let mut queue = VecDeque::new();
        for (path, node) in &self.nodes {
            if node.crate_root || (node.initial_source && incoming[path] == 0) {
                queue.push_back(path.clone());
            }
        }
        while let Some(path) = queue.pop_front() {
            if !reachable.insert(path.clone()) {
                continue;
            }
            if let Some(node) = self.nodes.get(&path) {
                for edge in &node.edges {
                    if !edge.test_only {
                        queue.push_back(edge.target.clone());
                    }
                }
            }
        }
        reachable
    }

    fn test_reachable(&self) -> BTreeSet<PathBuf> {
        let mut reachable = BTreeSet::new();
        let mut queue = VecDeque::new();
        for node in self.nodes.values() {
            for edge in &node.edges {
                if edge.test_only {
                    queue.push_back(edge.target.clone());
                }
            }
        }
        while let Some(path) = queue.pop_front() {
            if !reachable.insert(path.clone()) {
                continue;
            }
            if let Some(node) = self.nodes.get(&path) {
                for edge in &node.edges {
                    queue.push_back(edge.target.clone());
                }
            }
        }
        reachable
    }

    fn scan(&self) -> BTreeSet<Finding> {
        let production = self.production_reachable();
        let test = self.test_reachable();
        let mut findings = BTreeSet::new();
        for (path, node) in &self.nodes {
            if test.contains(path) && !production.contains(path) {
                continue;
            }
            let relative = relative_path(&self.root, path);
            let mut visitor = SourceVisitor::new(relative, &node.source);
            visitor.visit_file(&node.file);
            findings.extend(visitor.findings);
        }
        findings
    }
}

pub(crate) fn audit(root_arg: &Path, baseline_path: &Path) -> Result<AuditReport> {
    let root = fs::canonicalize(root_arg)
        .with_context(|| format!("cannot resolve audit root {}", root_arg.display()))?;
    if !root.is_dir() {
        bail!("audit root is not a directory: {}", root.display());
    }
    let project = Project::load(&root)?;
    let findings = project.scan();
    let baseline = load_baseline(baseline_path)?;

    let matched = findings.intersection(&baseline).cloned().collect();
    let unbaselined = findings.difference(&baseline).cloned().collect();
    let stale = baseline.difference(&findings).cloned().collect();
    let forbidden_baseline = baseline
        .iter()
        .filter(|finding| RAW_KINDS.contains(&finding.kind.as_str()))
        .cloned()
        .collect();

    Ok(AuditReport {
        matched,
        unbaselined,
        stale,
        forbidden_baseline,
    })
}

fn parse_source(path: &Path) -> Result<(String, File)> {
    let source = fs::read_to_string(path)
        .with_context(|| format!("cannot read Rust source {}", path.display()))?;
    let file = syn::parse_file(&source)
        .map_err(|error| anyhow!("cannot parse Rust source {}: {}", path.display(), error))?;
    Ok((source, file))
}

fn relative_path(root: &Path, path: &Path) -> String {
    path.strip_prefix(root)
        .unwrap_or(path)
        .to_string_lossy()
        .replace(std::path::MAIN_SEPARATOR, "/")
}

fn ensure_inside(root: &Path, path: &Path, description: &str) -> Result<PathBuf> {
    if !path.starts_with(root) {
        bail!("{} escapes audit root: {}", description, path.display());
    }
    Ok(path.to_owned())
}

fn normalize_absolute(path: &Path) -> PathBuf {
    let mut normalized = PathBuf::new();
    for component in path.components() {
        match component {
            Component::CurDir => {}
            Component::ParentDir => {
                normalized.pop();
            }
            Component::RootDir | Component::Prefix(_) => normalized.push(component.as_os_str()),
            Component::Normal(value) => normalized.push(value),
        }
    }
    normalized
}

fn canonical_module_path(root: &Path, candidate: &Path, description: &str) -> Result<PathBuf> {
    let lexical = normalize_absolute(candidate);
    ensure_inside(root, &lexical, description)?;
    let resolved = fs::canonicalize(candidate)
        .with_context(|| format!("cannot resolve module source {}", candidate.display()))?;
    ensure_inside(root, &resolved, description)
}

fn discover_sources(root: &Path) -> Result<Vec<DiscoveredSource>> {
    let crates = root.join("crates");
    if !crates.is_dir() {
        bail!(
            "audit root does not contain a crates directory: {}",
            crates.display()
        );
    }

    let mut sources = BTreeMap::<PathBuf, bool>::new();
    for entry in
        fs::read_dir(&crates).with_context(|| format!("cannot read {}", crates.display()))?
    {
        let entry = entry?;
        let logical_crate = entry.path();
        let metadata = fs::symlink_metadata(&logical_crate)?;
        if !metadata.file_type().is_dir() && !metadata.file_type().is_symlink() {
            continue;
        }
        let crate_path = fs::canonicalize(&logical_crate).with_context(|| {
            format!("cannot resolve crate directory {}", logical_crate.display())
        })?;
        ensure_inside(root, &crate_path, "crate directory")?;
        let logical_src = logical_crate.join("src");
        if !logical_src.exists() {
            continue;
        }
        let src = fs::canonicalize(&logical_src).with_context(|| {
            format!("cannot resolve source directory {}", logical_src.display())
        })?;
        ensure_inside(root, &src, "source directory")?;
        collect_source_files(
            root,
            &logical_src,
            &logical_src,
            &mut sources,
            &mut BTreeSet::new(),
        )?;
    }

    Ok(sources
        .into_iter()
        .map(|(path, crate_root)| DiscoveredSource { path, crate_root })
        .collect())
}

fn collect_source_files(
    root: &Path,
    crate_src: &Path,
    logical_dir: &Path,
    sources: &mut BTreeMap<PathBuf, bool>,
    visited_dirs: &mut BTreeSet<PathBuf>,
) -> Result<()> {
    let canonical_dir = fs::canonicalize(logical_dir)
        .with_context(|| format!("cannot resolve source directory {}", logical_dir.display()))?;
    ensure_inside(root, &canonical_dir, "source path")?;
    if !visited_dirs.insert(canonical_dir.clone()) {
        return Ok(());
    }

    for entry in fs::read_dir(logical_dir)
        .with_context(|| format!("cannot read source directory {}", logical_dir.display()))?
    {
        let entry = entry?;
        let logical = entry.path();
        let canonical = fs::canonicalize(&logical)
            .with_context(|| format!("cannot resolve source path {}", logical.display()))?;
        ensure_inside(root, &canonical, "source path")?;
        if canonical.is_dir() {
            collect_source_files(root, crate_src, &logical, sources, visited_dirs)?;
        } else if logical.extension().and_then(|extension| extension.to_str()) == Some("rs")
            && canonical.is_file()
        {
            // Rust follows the logical directory entry. A `.rs` symlink is a
            // source even when its in-root target has no `.rs` suffix.
            let crate_root = is_crate_root_source(crate_src, &logical);
            sources
                .entry(canonical)
                .and_modify(|is_root| *is_root |= crate_root)
                .or_insert(crate_root);
        }
    }
    Ok(())
}

fn is_crate_root_source(src: &Path, logical: &Path) -> bool {
    let Ok(relative) = logical.strip_prefix(src) else {
        return false;
    };
    let components: Vec<_> = relative.components().collect();
    match components.as_slice() {
        [Component::Normal(name)] => {
            *name == std::ffi::OsStr::new("lib.rs") || *name == std::ffi::OsStr::new("main.rs")
        }
        [Component::Normal(bin), Component::Normal(name)] => {
            *bin == std::ffi::OsStr::new("bin")
                && Path::new(name).extension().and_then(|e| e.to_str()) == Some("rs")
        }
        _ => false,
    }
}

fn module_base_for_file(path: &Path) -> PathBuf {
    let parent = path.parent().unwrap_or_else(|| Path::new("/"));
    match path.file_name().and_then(|name| name.to_str()) {
        Some("lib.rs") | Some("main.rs") | Some("mod.rs") => parent.to_owned(),
        Some(name) => parent.join(name.trim_end_matches(".rs")),
        None => parent.to_owned(),
    }
}

fn collect_module_edges(file: &File, path: &Path, root: &Path) -> Result<Vec<ModuleEdge>> {
    let mut edges = Vec::new();
    collect_module_items(
        &file.items,
        &module_base_for_file(path),
        path.parent().unwrap_or_else(|| Path::new("/")),
        false,
        root,
        &mut edges,
    )?;
    Ok(edges)
}

fn collect_module_items(
    items: &[Item],
    module_dir: &Path,
    source_dir: &Path,
    inherited_test_only: bool,
    root: &Path,
    edges: &mut Vec<ModuleEdge>,
) -> Result<()> {
    for item in items {
        let Item::Mod(module) = item else {
            continue;
        };
        let test_only = inherited_test_only || has_exact_cfg_test(&module.attrs);
        let child_dir = module_dir.join(ident_name(&module.ident));
        if let Some((_, inner)) = &module.content {
            // Inline modules contribute to the path ancestry used by both
            // implicit children and explicit `#[path]` attributes.
            collect_module_items(inner, &child_dir, &child_dir, test_only, root, edges)?;
            continue;
        }

        let path_attr = path_attribute(&module.attrs)?;
        let target = if let Some(path_attr) = path_attr {
            let candidate = if Path::new(&path_attr).is_absolute() {
                PathBuf::from(path_attr)
            } else {
                source_dir.join(path_attr)
            };
            canonical_module_path(root, &candidate, "module path")?
        } else {
            let candidates = [child_dir.with_extension("rs"), child_dir.join("mod.rs")];
            let existing = candidates
                .iter()
                .find(|candidate| fs::symlink_metadata(candidate).is_ok());
            let candidate = existing.ok_or_else(|| {
                anyhow!(
                    "module {} source not found (looked for {} and {})",
                    module.ident,
                    candidates[0].display(),
                    candidates[1].display()
                )
            })?;
            canonical_module_path(root, candidate, "module path")?
        };
        edges.push(ModuleEdge { target, test_only });
    }
    Ok(())
}

fn path_attribute(attrs: &[Attribute]) -> Result<Option<String>> {
    for attr in attrs {
        if !attr.path().is_ident("path") {
            continue;
        }
        let Meta::NameValue(name_value) = &attr.meta else {
            bail!("#[path] must contain a string literal");
        };
        let Expr::Lit(expression) = &name_value.value else {
            bail!("#[path] must contain a string literal");
        };
        let syn::Lit::Str(value) = &expression.lit else {
            bail!("#[path] must contain a string literal");
        };
        return Ok(Some(value.value()));
    }
    Ok(None)
}

fn has_exact_cfg_test(attrs: &[Attribute]) -> bool {
    attrs.iter().any(|attr| {
        if !attr.path().is_ident("cfg") {
            return false;
        }
        let Meta::List(list) = &attr.meta else {
            return false;
        };
        list.parse_args::<Ident>()
            .map(|ident| ident_name(&ident) == "test")
            .unwrap_or(false)
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
        TraitItem::Macro(item) => &item.attrs,
        TraitItem::Type(item) => &item.attrs,
        TraitItem::Verbatim(_) => &[],
        _ => &[],
    }
}

fn impl_item_attrs(item: &ImplItem) -> &[Attribute] {
    match item {
        ImplItem::Const(item) => &item.attrs,
        ImplItem::Fn(item) => &item.attrs,
        ImplItem::Macro(item) => &item.attrs,
        ImplItem::Type(item) => &item.attrs,
        ImplItem::Verbatim(_) => &[],
        _ => &[],
    }
}

fn is_public(visibility: &Visibility) -> bool {
    matches!(visibility, Visibility::Public(_))
}

fn ident_name(ident: &Ident) -> String {
    let name = ident.to_string();
    name.strip_prefix("r#").unwrap_or(&name).to_owned()
}

fn macro_kind(path: &syn::Path) -> Option<PanicKind> {
    let ident = &path.segments.last()?.ident;
    let name = ident_name(ident);
    PanicKind::from_name(&name)
}

#[derive(Clone, Default)]
struct TypeScope {
    bindings: BTreeMap<String, Option<StandardType>>,
    aliases: BTreeMap<String, Type>,
}

fn canonical_standard_type(names: &[String]) -> Option<StandardType> {
    match names {
        [root, module, name]
            if (root == "std" || root == "core")
                && ((module == "option" && name == "Option")
                    || (module == "result" && name == "Result")) =>
        {
            (name == "Option")
                .then_some(StandardType::Option)
                .or_else(|| (name == "Result").then_some(StandardType::Result))
        }
        _ => None,
    }
}

fn resolve_alias(
    scope_index: usize,
    name: &str,
    scopes: &[TypeScope],
    visiting: &mut BTreeSet<(usize, String)>,
) -> Option<StandardType> {
    if !visiting.insert((scope_index, name.to_owned())) {
        return None;
    }
    let result = scopes
        .get(scope_index)
        .and_then(|scope| scope.aliases.get(name))
        .and_then(|ty| standard_type(ty, scopes, visiting));
    visiting.remove(&(scope_index, name.to_owned()));
    result
}

fn resolve_type_name(
    name: &str,
    scopes: &[TypeScope],
    visiting: &mut BTreeSet<(usize, String)>,
) -> Option<StandardType> {
    for index in (0..scopes.len()).rev() {
        let scope = &scopes[index];
        if scope.aliases.contains_key(name) {
            return resolve_alias(index, name, scopes, visiting);
        }
        if let Some(binding) = scope.bindings.get(name) {
            return *binding;
        }
    }
    match name {
        "Option" => Some(StandardType::Option),
        "Result" => Some(StandardType::Result),
        _ => None,
    }
}

fn standard_type_names(
    names: &[String],
    leading_colon: bool,
    scopes: &[TypeScope],
    visiting: &mut BTreeSet<(usize, String)>,
) -> Option<StandardType> {
    if leading_colon {
        return canonical_standard_type(names);
    }
    match names {
        [name] => resolve_type_name(name, scopes, visiting),
        [prefix, name] if prefix == "self" => scopes.last().and_then(|scope| {
            if scope.aliases.contains_key(name) {
                resolve_alias(scopes.len() - 1, name, scopes, visiting)
            } else {
                scope.bindings.get(name).copied().flatten()
            }
        }),
        [prefix, name] if prefix == "super" => {
            scopes
                .get(scopes.len().saturating_sub(2))
                .and_then(|scope| {
                    if scope.aliases.contains_key(name) {
                        resolve_alias(scopes.len().saturating_sub(2), name, scopes, visiting)
                    } else {
                        scope.bindings.get(name).copied().flatten()
                    }
                })
        }
        _ => canonical_standard_type(names),
    }
}

fn standard_type(
    ty: &Type,
    scopes: &[TypeScope],
    visiting: &mut BTreeSet<(usize, String)>,
) -> Option<StandardType> {
    match ty {
        Type::Path(type_path) if type_path.qself.is_none() => standard_type_names(
            &type_path
                .path
                .segments
                .iter()
                .map(|segment| ident_name(&segment.ident))
                .collect::<Vec<_>>(),
            type_path.path.leading_colon.is_some(),
            scopes,
            visiting,
        ),
        Type::Paren(type_paren) => standard_type(&type_paren.elem, scopes, visiting),
        Type::Group(type_group) => standard_type(&type_group.elem, scopes, visiting),
        _ => None,
    }
}

fn associated_method(path: &ExprPath, scopes: &[TypeScope]) -> Option<PanicKind> {
    let method = path.path.segments.last()?;
    let method_name = ident_name(&method.ident);
    let kind = PanicKind::from_name(&method_name)?;
    if !matches!(kind, PanicKind::Unwrap | PanicKind::Expect) {
        return None;
    }

    let standard = if let Some(qself) = &path.qself {
        standard_type(&qself.ty, scopes, &mut BTreeSet::new())
    } else {
        let names: Vec<String> = path
            .path
            .segments
            .iter()
            .map(|segment| ident_name(&segment.ident))
            .collect();
        if names.len() < 2 {
            None
        } else {
            standard_type_names(
                &names[..names.len() - 1],
                path.path.leading_colon.is_some(),
                scopes,
                &mut BTreeSet::new(),
            )
        }
    }?;
    let _ = standard;
    Some(kind)
}

fn collect_use_tree(
    tree: &syn::UseTree,
    prefix: &[String],
    bindings: &mut BTreeMap<String, Option<StandardType>>,
) {
    match tree {
        syn::UseTree::Path(path) => {
            let mut next = prefix.to_vec();
            next.push(ident_name(&path.ident));
            collect_use_tree(&path.tree, &next, bindings);
        }
        syn::UseTree::Name(name) => {
            let mut path = prefix.to_vec();
            path.push(ident_name(&name.ident));
            bindings.insert(ident_name(&name.ident), canonical_standard_type(&path));
        }
        syn::UseTree::Rename(rename) => {
            let mut path = prefix.to_vec();
            path.push(ident_name(&rename.ident));
            bindings.insert(ident_name(&rename.rename), canonical_standard_type(&path));
        }
        syn::UseTree::Group(group) => {
            for item in &group.items {
                collect_use_tree(item, prefix, bindings);
            }
        }
        syn::UseTree::Glob { .. } => {}
    }
}

fn collect_scope_binding(item: &Item, scope: &mut TypeScope) {
    match item {
        Item::Use(item) => collect_use_tree(&item.tree, &[], &mut scope.bindings),
        Item::Type(item) => {
            scope
                .aliases
                .insert(ident_name(&item.ident), (*item.ty).clone());
        }
        Item::Enum(item) => {
            scope.bindings.insert(ident_name(&item.ident), None);
        }
        Item::Struct(item) => {
            scope.bindings.insert(ident_name(&item.ident), None);
        }
        Item::Union(item) => {
            scope.bindings.insert(ident_name(&item.ident), None);
        }
        Item::Trait(item) => {
            scope.bindings.insert(ident_name(&item.ident), None);
        }
        Item::TraitAlias(item) => {
            scope.bindings.insert(ident_name(&item.ident), None);
        }
        Item::Mod(item) => {
            scope.bindings.insert(ident_name(&item.ident), None);
        }
        Item::ExternCrate(item) => {
            scope.bindings.insert(ident_name(&item.ident), None);
        }
        _ => {}
    }
}

fn collect_scope_bindings(items: &[Item]) -> TypeScope {
    let mut scope = TypeScope::default();
    for item in items {
        collect_scope_binding(item, &mut scope);
    }
    scope
}

fn collect_block_bindings(block: &syn::Block) -> TypeScope {
    let mut scope = TypeScope::default();
    for statement in &block.stmts {
        if let syn::Stmt::Item(item) = statement {
            collect_scope_binding(item, &mut scope);
        }
    }
    scope
}

fn sanitize_macro_variables(tokens: TokenStream) -> TokenStream {
    let trees: Vec<TokenTree> = tokens.into_iter().collect();
    let mut result = TokenStream::new();
    let mut index = 0;
    while index < trees.len() {
        if matches!(&trees[index], TokenTree::Punct(punct) if punct.as_char() == '$')
            && matches!(trees.get(index + 1), Some(TokenTree::Ident(_)))
        {
            if let TokenTree::Ident(ident) = &trees[index + 1] {
                result.extend([TokenTree::Ident(Ident::new(
                    "__t4a_macro_variable",
                    ident.span(),
                ))]);
            }
            index += 2;
            continue;
        }
        let tree = match &trees[index] {
            TokenTree::Group(group) => {
                let mut replacement =
                    Group::new(group.delimiter(), sanitize_macro_variables(group.stream()));
                replacement.set_span(group.span());
                TokenTree::Group(replacement)
            }
            other => other.clone(),
        };
        result.extend([tree]);
        index += 1;
    }
    result
}

struct SourceVisitor<'source> {
    path: String,
    source: &'source str,
    line_starts: Vec<usize>,
    findings: BTreeSet<Finding>,
    public_context: bool,
    trait_impl_context: bool,
    type_scopes: Vec<TypeScope>,
}

impl<'source> SourceVisitor<'source> {
    fn new(path: String, source: &'source str) -> Self {
        let mut line_starts = vec![0];
        line_starts.extend(
            source
                .bytes()
                .enumerate()
                .filter_map(|(offset, byte)| (byte == b'\n').then_some(offset + 1)),
        );
        Self {
            path,
            source,
            line_starts,
            findings: BTreeSet::new(),
            public_context: false,
            trait_impl_context: false,
            type_scopes: Vec::new(),
        }
    }

    fn add(&mut self, kind: PanicKind, span: Span) {
        self.findings.insert(Finding {
            path: self.path.clone(),
            line: span.start().line,
            kind: kind.name().to_owned(),
        });
    }

    fn push_scope(&mut self, scope: TypeScope) {
        self.type_scopes.push(scope);
    }

    fn push_generic_scope(&mut self, generics: &syn::Generics) {
        let mut scope = TypeScope::default();
        for parameter in &generics.params {
            if let syn::GenericParam::Type(parameter) = parameter {
                scope.bindings.insert(ident_name(&parameter.ident), None);
            }
        }
        self.push_scope(scope);
    }

    fn method_call_line(&self, node: &ExprMethodCall) -> usize {
        let receiver_end = node.receiver.span().end();
        let method_start = node.method.span().start();
        for line in receiver_end.line..=method_start.line {
            let Some(&line_start) = self.line_starts.get(line.saturating_sub(1)) else {
                continue;
            };
            let line_end = self
                .line_starts
                .get(line)
                .copied()
                .unwrap_or(self.source.len());
            let text = &self.source[line_start..line_end];
            let start = if line == receiver_end.line {
                receiver_end.column.min(text.len())
            } else {
                0
            };
            let end = if line == method_start.line {
                method_start.column.min(text.len())
            } else {
                text.len()
            };
            if start <= end && text.as_bytes()[start..end].contains(&b'.') {
                return line;
            }
        }
        node.span().start().line
    }

    fn try_parse_macro_stream(&mut self, tokens: TokenStream) -> bool {
        if let Ok(expression) = syn::parse2::<Expr>(tokens.clone()) {
            self.visit_expr(&expression);
            return true;
        }
        if let Ok(expressions) =
            Punctuated::<Expr, syn::Token![,]>::parse_terminated.parse2(tokens.clone())
        {
            if !expressions.is_empty() {
                for expression in expressions {
                    self.visit_expr(&expression);
                }
                return true;
            }
        }
        let mut wrapped = Group::new(Delimiter::Brace, tokens.clone());
        wrapped.set_span(Span::call_site());
        if let Ok(block) = syn::parse2::<syn::Block>(TokenStream::from(TokenTree::Group(wrapped))) {
            self.visit_block(&block);
            return true;
        }
        if let Ok(file) = syn::parse2::<File>(tokens) {
            self.visit_file(&file);
            return true;
        }
        false
    }

    fn scan_literal_token_patterns(&mut self, tokens: &TokenStream) {
        let trees: Vec<TokenTree> = tokens.clone().into_iter().collect();
        for index in 0..trees.len() {
            if let TokenTree::Ident(ident) = &trees[index] {
                let name = ident_name(ident);
                let metavariable = index > 0
                    && matches!(&trees[index - 1], TokenTree::Punct(punct) if punct.as_char() == '$');
                if !metavariable
                    && RAW_KINDS.contains(&name.as_str())
                    && matches!(trees.get(index + 1), Some(TokenTree::Punct(punct)) if punct.as_char() == '!')
                {
                    if let Some(kind) = PanicKind::from_name(&name) {
                        self.add(kind, ident.span());
                    }
                }
                if matches!(name.as_str(), "unwrap" | "expect")
                    && index >= 3
                    && matches!(&trees[index - 1], TokenTree::Punct(punct) if punct.as_char() == ':')
                    && matches!(&trees[index - 2], TokenTree::Punct(punct) if punct.as_char() == ':')
                {
                    let mut names = Vec::new();
                    let mut cursor = index;
                    while cursor >= 3
                        && matches!(&trees[cursor - 1], TokenTree::Punct(punct) if punct.as_char() == ':')
                        && matches!(&trees[cursor - 2], TokenTree::Punct(punct) if punct.as_char() == ':')
                    {
                        let Some(TokenTree::Ident(base)) = trees.get(cursor - 3) else {
                            break;
                        };
                        names.push(ident_name(base));
                        cursor -= 3;
                    }
                    names.reverse();
                    if !names.is_empty()
                        && standard_type_names(
                            &names,
                            false,
                            &self.type_scopes,
                            &mut BTreeSet::new(),
                        )
                        .is_some()
                    {
                        let kind = PanicKind::from_name(&name).expect("method name checked");
                        self.add(kind, ident.span());
                    }
                }
            }
            if let TokenTree::Punct(dot) = &trees[index] {
                if dot.as_char() == '.'
                    && matches!(
                        trees.get(index + 1),
                        Some(TokenTree::Ident(ident))
                            if matches!(ident_name(ident).as_str(), "unwrap" | "expect")
                    )
                {
                    if let Some(TokenTree::Ident(method)) = trees.get(index + 1) {
                        let kind =
                            PanicKind::from_name(&ident_name(method)).expect("method name checked");
                        self.add(kind, dot.span());
                    }
                }
            }
            if let TokenTree::Group(group) = &trees[index] {
                self.scan_literal_token_patterns(&group.stream());
            }
        }
    }

    fn scan_macro_groups(&mut self, tokens: &TokenStream) {
        for tree in tokens.clone() {
            let TokenTree::Group(group) = tree else {
                continue;
            };
            let inner = group.stream();
            let parsed = self.try_parse_macro_stream(inner.clone())
                || self.try_parse_macro_stream(sanitize_macro_variables(inner.clone()));
            if !parsed {
                self.scan_literal_token_patterns(&inner);
                self.scan_macro_groups(&inner);
            }
        }
    }

    /// Scan literal macro tokens without attempting expansion. Metavariables
    /// are replaced only for parsing (`$panic!` is not a literal call), while
    /// literal calls in invocation arguments and macro definitions remain
    /// visible. Full macro name resolution/expansion is intentionally outside
    /// this audit's conservative, source-only boundary.
    fn scan_macro_tokens(&mut self, tokens: TokenStream) {
        let parsed = self.try_parse_macro_stream(tokens.clone())
            || self.try_parse_macro_stream(sanitize_macro_variables(tokens.clone()));
        if !parsed {
            self.scan_literal_token_patterns(&tokens);
            self.scan_macro_groups(&tokens);
        }
    }
}

impl<'ast, 'source> Visit<'ast> for SourceVisitor<'source> {
    fn visit_file(&mut self, node: &'ast File) {
        self.push_scope(collect_scope_bindings(&node.items));
        visit::visit_file(self, node);
        self.type_scopes.pop();
    }

    fn visit_item(&mut self, node: &'ast Item) {
        if has_exact_cfg_test(item_attrs(node)) {
            return;
        }
        visit::visit_item(self, node);
    }

    fn visit_item_mod(&mut self, node: &'ast syn::ItemMod) {
        let Some((_, items)) = &node.content else {
            return;
        };
        self.push_scope(collect_scope_bindings(items));
        for item in items {
            self.visit_item(item);
        }
        self.type_scopes.pop();
    }

    fn visit_block(&mut self, node: &'ast syn::Block) {
        self.push_scope(collect_block_bindings(node));
        visit::visit_block(self, node);
        self.type_scopes.pop();
    }

    fn visit_trait_item(&mut self, node: &'ast TraitItem) {
        if has_exact_cfg_test(trait_item_attrs(node)) {
            return;
        }
        visit::visit_trait_item(self, node);
    }

    fn visit_impl_item(&mut self, node: &'ast ImplItem) {
        if has_exact_cfg_test(impl_item_attrs(node)) {
            return;
        }
        visit::visit_impl_item(self, node);
    }

    fn visit_item_fn(&mut self, node: &'ast syn::ItemFn) {
        let previous = self.public_context;
        self.public_context |= is_public(&node.vis);
        self.push_generic_scope(&node.sig.generics);
        visit::visit_item_fn(self, node);
        self.type_scopes.pop();
        self.public_context = previous;
    }

    fn visit_item_trait(&mut self, node: &'ast syn::ItemTrait) {
        let previous = self.public_context;
        self.public_context |= is_public(&node.vis);
        self.push_generic_scope(&node.generics);
        visit::visit_item_trait(self, node);
        self.type_scopes.pop();
        self.public_context = previous;
    }

    fn visit_item_impl(&mut self, node: &'ast syn::ItemImpl) {
        let previous = self.trait_impl_context;
        self.trait_impl_context |= node.trait_.is_some();
        self.push_generic_scope(&node.generics);
        visit::visit_item_impl(self, node);
        self.type_scopes.pop();
        self.trait_impl_context = previous;
    }

    fn visit_impl_item_fn(&mut self, node: &'ast syn::ImplItemFn) {
        let previous = self.public_context;
        self.public_context |= self.trait_impl_context || is_public(&node.vis);
        self.push_generic_scope(&node.sig.generics);
        visit::visit_impl_item_fn(self, node);
        self.type_scopes.pop();
        self.public_context = previous;
    }

    fn visit_macro(&mut self, node: &'ast syn::Macro) {
        if let Some(kind) = macro_kind(&node.path) {
            if kind.is_raw() || self.public_context {
                self.add(
                    kind,
                    node.path
                        .segments
                        .last()
                        .map(|s| s.ident.span())
                        .unwrap_or_else(Span::call_site),
                );
            }
        }
        self.scan_macro_tokens(node.tokens.clone());
    }

    fn visit_expr_method_call(&mut self, node: &'ast ExprMethodCall) {
        let name = ident_name(&node.method);
        if let Some(kind) = PanicKind::from_name(&name) {
            if matches!(kind, PanicKind::Unwrap | PanicKind::Expect) {
                let line = self.method_call_line(node);
                self.findings.insert(Finding {
                    path: self.path.clone(),
                    line,
                    kind: kind.name().to_owned(),
                });
            }
        }
        visit::visit_expr_method_call(self, node);
    }

    fn visit_expr_call(&mut self, node: &'ast ExprCall) {
        if let Expr::Path(path) = node.func.as_ref() {
            if let Some(kind) = associated_method(path, &self.type_scopes) {
                self.add(kind, node.span());
            }
        }
        visit::visit_expr_call(self, node);
    }
}

fn quoted(value: &str) -> String {
    format!("'{}'", value.replace('\\', "\\\\").replace('\'', "\\'"))
}

fn parse_entry(value: &str) -> Result<Finding> {
    let mut parts = value.rsplitn(3, ':');
    let kind = parts.next().unwrap_or_default();
    let line = parts.next().unwrap_or_default();
    let path = parts.next().unwrap_or_default();
    let Some(kind) = PanicKind::from_name(kind) else {
        bail!("invalid baseline entry: {}", quoted(value));
    };
    if line.is_empty()
        || !line.bytes().all(|byte| byte.is_ascii_digit())
        || (line.len() > 1 && line.starts_with('0'))
    {
        bail!("invalid baseline entry: {}", quoted(value));
    }
    let line = line
        .parse::<usize>()
        .ok()
        .filter(|line| *line > 0)
        .ok_or_else(|| anyhow!("invalid baseline entry: {}", quoted(value)))?;
    if path.is_empty()
        || !path.starts_with("crates/")
        || path.contains('\\')
        || path.starts_with('/')
        || path
            .split('/')
            .any(|part| part.is_empty() || part == "." || part == "..")
        || Path::new(path)
            .components()
            .any(|component| !matches!(component, Component::Normal(_)))
    {
        bail!("baseline path is not normalized: {}", quoted(value));
    }
    Ok(Finding {
        path: path.to_owned(),
        line,
        kind: kind.name().to_owned(),
    })
}

fn load_baseline(path: &Path) -> Result<BTreeSet<Finding>> {
    let text = match fs::read_to_string(path) {
        Ok(text) => text,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(BTreeSet::new()),
        Err(error) => {
            return Err(error).with_context(|| format!("cannot read baseline {}", path.display()))
        }
    };
    let entries: Vec<String> = serde_json::from_str(&text)
        .with_context(|| format!("cannot parse panic baseline JSON {}", path.display()))?;
    let mut baseline = BTreeSet::new();
    for value in entries {
        let finding = parse_entry(&value)?;
        if !baseline.insert(finding) {
            bail!("panic baseline contains duplicate entries");
        }
    }
    Ok(baseline)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    struct Fixture {
        root: PathBuf,
    }

    impl Fixture {
        fn new() -> Self {
            let stamp = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("clock")
                .as_nanos();
            let root = std::env::temp_dir().join(format!("library-panic-audit-{stamp}"));
            fs::create_dir_all(&root).expect("fixture root");
            Self { root }
        }

        fn write(&self, relative: &str, source: &str) {
            let path = self.root.join(relative);
            fs::create_dir_all(path.parent().expect("parent")).expect("fixture parent");
            fs::write(path, source).expect("fixture source");
        }

        fn scan(&self) -> Vec<String> {
            let project = Project::load(&self.root).expect("project");
            project
                .scan()
                .into_iter()
                .map(|finding| finding.entry())
                .collect()
        }
    }

    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.root);
        }
    }

    #[test]
    fn ast_handles_raw_identifiers_and_operator_rich_generics() {
        let fixture = Fixture::new();
        fixture.write(
            "crates/demo/src/lib.rs",
            r#"fn calls(value: Option<bool>) {
    let _ = value.r#unwrap();
    let _ = Option::r#expect(value, "missing");
    let _ = Option::<fn() -> bool>::unwrap(value);
    let _ = <Option<[u8; (1 < 2) as usize]>>::expect(value, "missing");
}
pub fn public<T: Bound<{ 1 < 2 }>>() {
    assert!(true);
}
trait Bound<const N: usize> {}
"#,
        );
        assert_eq!(
            fixture.scan(),
            vec![
                "crates/demo/src/lib.rs:2:unwrap",
                "crates/demo/src/lib.rs:3:expect",
                "crates/demo/src/lib.rs:4:unwrap",
                "crates/demo/src/lib.rs:5:expect",
                "crates/demo/src/lib.rs:8:assert",
            ]
        );
    }

    #[test]
    fn macro_rules_templates_are_opaque_but_macro_items_are_scanned() {
        let fixture = Fixture::new();
        fixture.write(
            "crates/demo/src/lib.rs",
            r#"macro_rules! not_a_call {
    ($panic:ident) => { $panic!(); };
}
panic!();
"#,
        );
        assert_eq!(fixture.scan(), vec!["crates/demo/src/lib.rs:4:panic"]);
    }

    #[test]
    fn cfg_macro_items_are_skipped_structurally() {
        let fixture = Fixture::new();
        fixture.write(
            "crates/demo/src/lib.rs",
            "#[cfg(test)] sink! { panic!(\"test\"); }\n",
        );
        assert!(fixture.scan().is_empty());
    }

    #[test]
    fn macro_token_streams_scan_nested_calls_and_ignore_metavariables() {
        let fixture = Fixture::new();
        fixture.write(
            "crates/demo/src/lib.rs",
            r#"macro_rules! literal {
    () => { panic!("definition"); };
}
macro_rules! passthrough {
    ($panic:ident) => { $panic!(); };
}
fn calls(value: Option<bool>) {
    wrapper!(panic!("statement argument"));
    dbg!(value.unwrap());
    wrapper!(Option::expect(value, "associated argument"));
}
wrapper! {
    fn generated() {
        unreachable!("item body");
    }
}
"#,
        );
        assert_eq!(
            fixture.scan(),
            vec![
                "crates/demo/src/lib.rs:2:panic",
                "crates/demo/src/lib.rs:8:panic",
                "crates/demo/src/lib.rs:9:unwrap",
                "crates/demo/src/lib.rs:10:expect",
                "crates/demo/src/lib.rs:14:unreachable",
            ]
        );
    }

    #[test]
    fn macro_definitions_scan_generic_associated_calls_with_metavariables() {
        let fixture = Fixture::new();
        fixture.write(
            "crates/demo/src/lib.rs",
            r#"macro_rules! generic {
    ($value:expr) => { Option::<bool>::unwrap($value); };
}
"#,
        );
        assert_eq!(fixture.scan(), vec!["crates/demo/src/lib.rs:2:unwrap"]);
    }

    #[test]
    fn ufcs_aliases_resolve_and_local_type_shadows_are_not_reported() {
        let fixture = Fixture::new();
        fixture.write(
            "crates/demo/src/lib.rs",
            r#"use std::option::Option as Maybe;
use core::result::Result as R;
use user::Option as LocalOption;
type Alias<T> = Maybe<T>;
mod local {
    struct Option<T>(T);
    fn shadow(value: Option<bool>) {
        let _ = Option::unwrap(value);
    }
}
fn calls(value: Maybe<bool>, result: R<bool, ()>) {
    let _ = Maybe::unwrap(value);
    let _ = R::r#expect(result, "missing");
    let _ = Alias::<bool>::unwrap(value);
    let _ = <Maybe<bool>>::unwrap(value);
    let _ = <Alias<bool>>::r#expect(value, "missing");
    let _ = LocalOption::unwrap(value);
}
"#,
        );
        assert_eq!(
            fixture.scan(),
            vec![
                "crates/demo/src/lib.rs:12:unwrap",
                "crates/demo/src/lib.rs:13:expect",
                "crates/demo/src/lib.rs:14:unwrap",
                "crates/demo/src/lib.rs:15:unwrap",
                "crates/demo/src/lib.rs:16:expect",
            ]
        );
    }

    #[test]
    fn nested_inline_path_attributes_use_ancestry_and_raw_strings() {
        let fixture = Fixture::new();
        fixture.write(
            "crates/demo/src/lib.rs",
            r#"#[path = r"tests_root.rs"]
#[cfg(test)]
mod tests_root;
#[cfg(test)]
mod inline_tests {
    #[path = r"custom/helper.rs"]
    mod helper;
}
"#,
        );
        fixture.write(
            "crates/demo/src/tests_root.rs",
            "pub fn test_only() { panic!(\"test\"); }\n",
        );
        fixture.write(
            "crates/demo/src/inline_tests/custom/helper.rs",
            "pub fn test_only() { unreachable!(\"test\"); }\n",
        );
        assert!(fixture.scan().is_empty());
    }

    #[test]
    fn baseline_line_numbers_must_be_canonical_decimal() {
        for line in ["01", "001", "+1", " 1"] {
            assert!(
                parse_entry(&format!("crates/demo/src/lib.rs:{line}:assert")).is_err(),
                "accepted non-canonical baseline line {line:?}"
            );
        }
    }

    #[test]
    fn production_alias_overrides_test_alias_and_inline_paths_are_ancestral() {
        let fixture = Fixture::new();
        fixture.write(
            "crates/demo/src/lib.rs",
            r#"#[cfg(test)]
#[path = r"tests_root.rs"]
mod tests_root;
#[path = "shared.rs"]
mod production;
#[cfg(test)]
#[path = r"shared.rs"]
mod test_alias;
#[cfg(test)]
mod inline {
    mod helper;
}
"#,
        );
        fixture.write(
            "crates/demo/src/tests_root.rs",
            "mod nested;\npub fn test_only() { panic!(\"test\"); }\n",
        );
        fixture.write(
            "crates/demo/src/tests_root/nested.rs",
            "pub fn test_only() { unreachable!(\"test\"); }\n",
        );
        fixture.write(
            "crates/demo/src/shared.rs",
            "pub fn production() { panic!(\"production\"); }\n",
        );
        fixture.write(
            "crates/demo/src/inline/helper.rs",
            "pub fn test_only() { panic!(\"inline\"); }\n",
        );
        assert_eq!(fixture.scan(), vec!["crates/demo/src/shared.rs:1:panic"]);
    }

    #[test]
    fn orphan_cycles_are_scanned_as_production() {
        let fixture = Fixture::new();
        fixture.write("crates/demo/src/lib.rs", "pub fn root() {}\n");
        fixture.write(
            "crates/demo/src/a.rs",
            "mod b;\npub fn a() { panic!(\"a\"); }\n",
        );
        fixture.write(
            "crates/demo/src/a/b.rs",
            "#[path = \"../a.rs\"] mod a;\npub fn b() { unreachable!(\"b\"); }\n",
        );
        assert_eq!(
            fixture.scan(),
            vec![
                "crates/demo/src/a.rs:2:panic",
                "crates/demo/src/a/b.rs:2:unreachable",
            ]
        );
    }

    #[test]
    fn all_trait_impl_methods_and_external_public_defaults_are_assertion_contexts() {
        let fixture = Fixture::new();
        fixture.write("crates/demo/src/lib.rs", "mod api;\nmod impls;\n");
        fixture.write(
            "crates/demo/src/api.rs",
            "pub trait PublicApi { fn default_method(&self) { assert!(true); } }\n",
        );
        fixture.write(
            "crates/demo/src/impls.rs",
            r#"struct T;
impl crate::api::PublicApi for T {
    fn implementation(&self) { debug_assert!(true); }
}
impl std::fmt::Display for T {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        assert!(true);
        write!(f, "T")
    }
}
"#,
        );
        assert_eq!(
            fixture.scan(),
            vec![
                "crates/demo/src/api.rs:1:assert",
                "crates/demo/src/impls.rs:3:debug_assert",
                "crates/demo/src/impls.rs:7:assert",
            ]
        );
    }

    #[test]
    fn production_path_targets_outside_src_are_scanned_and_escaping_paths_fail_closed() {
        let fixture = Fixture::new();
        fixture.write(
            "crates/demo/src/lib.rs",
            "#[path = \"../shared.rs\"] mod shared;\n",
        );
        fixture.write(
            "crates/demo/shared.rs",
            "pub fn bad() { panic!(\"outside src\"); }\n",
        );
        assert_eq!(fixture.scan(), vec!["crates/demo/shared.rs:1:panic"]);

        let escaping = Fixture::new();
        escaping.write(
            "crates/demo/src/lib.rs",
            "#[path = \"../../../../outside.rs\"] mod outside;\n",
        );
        escaping.write("outside.rs", "pub fn bad() { panic!(\"outside root\"); }\n");
        assert!(Project::load(&escaping.root).is_err());

        let wrong = std::env::temp_dir().join("library-panic-audit-wrong-root");
        let _ = fs::remove_dir_all(&wrong);
        fs::create_dir_all(&wrong).expect("wrong root");
        let result = audit(&wrong, &wrong.join("baseline.json"));
        assert!(result.is_err());
        let _ = fs::remove_dir_all(wrong);
    }

    #[cfg(unix)]
    #[test]
    fn logical_rs_symlink_entry_discovers_extensionless_target() {
        let fixture = Fixture::new();
        fixture.write("crates/demo/src/lib.rs", "pub fn root() {}\n");
        fixture.write(
            "crates/demo/shared_target",
            "pub fn bad() { panic!(\"extensionless target\"); }\n",
        );
        let link = fixture.root.join("crates/demo/src/bin/tool.rs");
        fs::create_dir_all(link.parent().expect("parent")).expect("bin dir");
        std::os::unix::fs::symlink(fixture.root.join("crates/demo/shared_target"), &link)
            .expect("symlink");
        assert_eq!(fixture.scan(), vec!["crates/demo/shared_target:1:panic"]);
    }

    #[cfg(unix)]
    #[test]
    fn escaping_source_symlinks_fail_closed() {
        let fixture = Fixture::new();
        let outside = fixture.root.join("outside.rs");
        fs::write(&outside, "pub fn bad() { panic!(\"outside\"); }\n").expect("outside");
        let link = fixture.root.join("crates/demo/src/link.rs");
        fs::create_dir_all(link.parent().expect("parent")).expect("source dir");
        std::os::unix::fs::symlink(&outside, &link).expect("symlink");
        assert!(discover_sources(&fixture.root).is_ok());

        let outside_root = std::env::temp_dir().join(format!(
            "library-panic-audit-outside-root-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("clock")
                .as_nanos()
        ));
        fs::create_dir_all(&outside_root).expect("outside root");
        let escaping = fixture.root.join("crates/demo/src/escaping.rs");
        std::os::unix::fs::symlink(outside_root.join("missing.rs"), &escaping)
            .expect("dangling symlink");
        assert!(discover_sources(&fixture.root).is_err());
        let _ = fs::remove_dir_all(outside_root);
    }
}
