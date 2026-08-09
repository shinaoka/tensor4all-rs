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

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct NodeKey {
    /// Canonical read/containment identity.
    identity: PathBuf,
    /// Logical root identity. Cargo targets and orphan source roots each get
    /// their own value, even when they read the same canonical file.
    root_id: PathBuf,
    /// Logical directory-entry path used by Rust module lookup.
    logical_path: PathBuf,
    module_path: Vec<String>,
}

struct SourceNode {
    identity: PathBuf,
    source: String,
    file: File,
    logical_path: PathBuf,
    root_id: PathBuf,
    module_path: Vec<String>,
    /// True for an initially discovered source. Orphan sources retain their
    /// own registry identity but use Rust's file-stem module base.
    initial_source: bool,
    /// True when this context is a Cargo target root and therefore resolves
    /// child modules from its containing directory.
    logical_root: bool,
    crate_root: bool,
    edges: Vec<ModuleEdge>,
}

#[derive(Clone, Debug)]
struct ModuleEdge {
    /// Canonical read/containment identity.
    target: PathBuf,
    /// The path as written through the containing module's logical ancestry.
    logical_target: PathBuf,
    root_id: PathBuf,
    module_path: Vec<String>,
    test_only: bool,
}

#[derive(Clone, Debug)]
struct DiscoveredSource {
    /// Canonical read/containment identity.
    identity: PathBuf,
    /// Logical directory-entry path used by Rust module lookup.
    logical_path: PathBuf,
    root_id: PathBuf,
    crate_root: bool,
}

struct Project {
    root: PathBuf,
    nodes: BTreeMap<NodeKey, SourceNode>,
    modules: ModuleRegistry,
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

        let mut parsed = BTreeMap::<PathBuf, (String, File)>::new();
        let mut nodes = BTreeMap::new();
        let mut queue = VecDeque::new();
        let mut orphan_queue = VecDeque::new();
        for source in discovered {
            let (source_text, file) = if let Some((text, file)) = parsed.get(&source.identity) {
                (text.clone(), file.clone())
            } else {
                let parsed_source = parse_source(&source.identity)?;
                parsed.insert(source.identity.clone(), parsed_source.clone());
                parsed_source
            };
            let key = NodeKey {
                identity: source.identity.clone(),
                root_id: source.root_id.clone(),
                logical_path: source.logical_path.clone(),
                module_path: Vec::new(),
            };
            nodes.insert(
                key.clone(),
                SourceNode {
                    identity: source.identity,
                    source: source_text,
                    file,
                    logical_path: source.logical_path,
                    root_id: source.root_id,
                    module_path: Vec::new(),
                    initial_source: true,
                    logical_root: source.crate_root,
                    crate_root: source.crate_root,
                    edges: Vec::new(),
                },
            );
            if source.crate_root {
                queue.push_back(key);
            } else {
                orphan_queue.push_back(key);
            }
        }

        // Keep every logical root as a separate graph context. Canonical
        // paths deduplicate reads, not module ancestry or registry identity.
        let mut expanded = BTreeSet::<(PathBuf, PathBuf, PathBuf)>::new();
        let mut referenced_test = BTreeSet::new();
        let mut referenced_production = BTreeSet::new();
        let mut root_phase = true;
        loop {
            if queue.is_empty() {
                if root_phase {
                    root_phase = false;
                    queue = std::mem::take(&mut orphan_queue);
                    continue;
                }
                break;
            }
            let key = queue.pop_front().expect("queue checked non-empty");
            let (skip_orphan, expansion_key) = {
                let node = nodes
                    .get(&key)
                    .ok_or_else(|| anyhow!("internal source graph error"))?;
                (
                    !node.crate_root
                        && node.initial_source
                        && referenced_test.contains(&node.identity)
                        && !referenced_production.contains(&node.identity),
                    (
                        node.root_id.clone(),
                        node.identity.clone(),
                        node.logical_path.clone(),
                    ),
                )
            };
            if skip_orphan {
                continue;
            }
            if !expanded.insert(expansion_key) {
                continue;
            }
            let edges = {
                let node = nodes
                    .get(&key)
                    .ok_or_else(|| anyhow!("internal source graph error"))?;
                collect_module_edges(
                    &node.file,
                    &node.logical_path,
                    &node.module_path,
                    &node.root_id,
                    node.logical_root,
                    root,
                )?
            };
            for edge in edges {
                if edge.test_only {
                    referenced_test.insert(edge.target.clone());
                } else {
                    referenced_production.insert(edge.target.clone());
                }
                let child_key = NodeKey {
                    identity: edge.target.clone(),
                    root_id: edge.root_id.clone(),
                    logical_path: edge.logical_target.clone(),
                    module_path: edge.module_path.clone(),
                };
                if !nodes.contains_key(&child_key) {
                    let (source_text, file) = if let Some((text, file)) = parsed.get(&edge.target) {
                        (text.clone(), file.clone())
                    } else {
                        let parsed_source = parse_source(&edge.target)?;
                        parsed.insert(edge.target.clone(), parsed_source.clone());
                        parsed_source
                    };
                    nodes.insert(
                        child_key.clone(),
                        SourceNode {
                            identity: edge.target.clone(),
                            source: source_text,
                            file,
                            logical_path: edge.logical_target.clone(),
                            root_id: edge.root_id.clone(),
                            module_path: edge.module_path.clone(),
                            initial_source: false,
                            logical_root: false,
                            crate_root: false,
                            edges: Vec::new(),
                        },
                    );
                }
                nodes
                    .get_mut(&key)
                    .ok_or_else(|| anyhow!("internal source graph error"))?
                    .edges
                    .push(edge);
                queue.push_back(child_key);
            }
        }

        let modules = ModuleRegistry::build(&nodes);
        Ok(Self {
            root: root.to_owned(),
            nodes,
            modules,
        })
    }

    fn production_reachable(&self) -> BTreeSet<NodeKey> {
        let mut incoming = BTreeMap::<NodeKey, usize>::new();
        for key in self.nodes.keys() {
            incoming.insert(key.clone(), 0);
        }
        for node in self.nodes.values() {
            for edge in &node.edges {
                let target = NodeKey {
                    identity: edge.target.clone(),
                    root_id: edge.root_id.clone(),
                    logical_path: edge.logical_target.clone(),
                    module_path: edge.module_path.clone(),
                };
                *incoming.entry(target).or_default() += 1;
            }
        }

        let mut reachable = BTreeSet::new();
        let mut queue = VecDeque::new();
        for (key, node) in &self.nodes {
            if node.crate_root || (node.initial_source && incoming[key] == 0) {
                queue.push_back(key.clone());
            }
        }
        while let Some(key) = queue.pop_front() {
            if !reachable.insert(key.clone()) {
                continue;
            }
            if let Some(node) = self.nodes.get(&key) {
                for edge in &node.edges {
                    if !edge.test_only {
                        queue.push_back(NodeKey {
                            identity: edge.target.clone(),
                            root_id: edge.root_id.clone(),
                            logical_path: edge.logical_target.clone(),
                            module_path: edge.module_path.clone(),
                        });
                    }
                }
            }
        }
        reachable
    }

    fn test_reachable(&self) -> BTreeSet<NodeKey> {
        let mut reachable = BTreeSet::new();
        let mut queue = VecDeque::new();
        for node in self.nodes.values() {
            for edge in &node.edges {
                if edge.test_only {
                    queue.push_back(NodeKey {
                        identity: edge.target.clone(),
                        root_id: edge.root_id.clone(),
                        logical_path: edge.logical_target.clone(),
                        module_path: edge.module_path.clone(),
                    });
                }
            }
        }
        while let Some(key) = queue.pop_front() {
            if !reachable.insert(key.clone()) {
                continue;
            }
            if let Some(node) = self.nodes.get(&key) {
                for edge in &node.edges {
                    queue.push_back(NodeKey {
                        identity: edge.target.clone(),
                        root_id: edge.root_id.clone(),
                        logical_path: edge.logical_target.clone(),
                        module_path: edge.module_path.clone(),
                    });
                }
            }
        }
        reachable
    }

    fn scan(&self) -> BTreeSet<Finding> {
        let production = self.production_reachable();
        let test = self.test_reachable();
        let production_identities: BTreeSet<_> = production
            .iter()
            .filter_map(|key| self.nodes.get(key))
            .filter(|node| node.crate_root || !node.initial_source)
            .map(|node| node.identity.clone())
            .collect();
        let test_identities: BTreeSet<_> = test
            .iter()
            .filter_map(|key| self.nodes.get(key).map(|node| node.identity.clone()))
            .collect();
        let mut findings = BTreeSet::new();
        for (key, node) in &self.nodes {
            let orphan_test_only = node.initial_source
                && !node.crate_root
                && test_identities.contains(&node.identity)
                && !production_identities.contains(&node.identity);
            if (test.contains(key) && !production.contains(key)) || orphan_test_only {
                continue;
            }
            let relative = relative_path(&self.root, &node.identity);
            let module = module_key(&node.root_id, &node.module_path);
            let mut visitor = SourceVisitor::new(relative, &node.source, &self.modules, module);
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

    let mut crate_entries = fs::read_dir(&crates)
        .with_context(|| format!("cannot read {}", crates.display()))?
        .collect::<std::io::Result<Vec<_>>>()?;
    crate_entries.sort_by_key(|entry| entry.path());
    let mut sources = Vec::new();
    for entry in crate_entries {
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
            &mut BTreeSet::new(),
        )?;
    }
    sources.sort_by(|left, right| {
        left.logical_path
            .cmp(&right.logical_path)
            .then_with(|| left.identity.cmp(&right.identity))
    });
    Ok(sources)
}

fn collect_source_files(
    root: &Path,
    crate_src: &Path,
    logical_dir: &Path,
    sources: &mut Vec<DiscoveredSource>,
    visited_dirs: &mut BTreeSet<PathBuf>,
    active_dirs: &mut BTreeSet<PathBuf>,
) -> Result<()> {
    let logical_key = normalize_absolute(logical_dir);
    if !visited_dirs.insert(logical_key) {
        return Ok(());
    }
    let canonical_dir = fs::canonicalize(logical_dir)
        .with_context(|| format!("cannot resolve source directory {}", logical_dir.display()))?;
    ensure_inside(root, &canonical_dir, "source path")?;
    if !active_dirs.insert(canonical_dir.clone()) {
        return Ok(());
    }

    let mut entries = fs::read_dir(logical_dir)
        .with_context(|| format!("cannot read source directory {}", logical_dir.display()))?
        .collect::<std::io::Result<Vec<_>>>()?;
    entries.sort_by_key(|entry| entry.path());
    for entry in entries {
        let logical = entry.path();
        let canonical = fs::canonicalize(&logical)
            .with_context(|| format!("cannot resolve source path {}", logical.display()))?;
        ensure_inside(root, &canonical, "source path")?;
        if canonical.is_dir() {
            collect_source_files(
                root,
                crate_src,
                &logical,
                sources,
                visited_dirs,
                active_dirs,
            )?;
        } else if logical.extension().and_then(|extension| extension.to_str()) == Some("rs")
            && canonical.is_file()
        {
            // Rust follows the logical directory entry. Keep the logical path
            // even when several entries canonicalize to one source file.
            let logical_path = normalize_absolute(&logical);
            sources.push(DiscoveredSource {
                identity: canonical,
                root_id: logical_path.clone(),
                crate_root: is_crate_root_source(crate_src, &logical),
                logical_path,
            });
        }
    }
    active_dirs.remove(&canonical_dir);
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

fn module_base_for_file(path: &Path, logical_root: bool) -> PathBuf {
    let parent = path.parent().unwrap_or_else(|| Path::new("/"));
    if logical_root {
        return parent.to_owned();
    }
    match path.file_name().and_then(|name| name.to_str()) {
        Some("mod.rs") => parent.to_owned(),
        Some(name) => parent.join(name.trim_end_matches(".rs")),
        None => parent.to_owned(),
    }
}

struct ModuleCollection<'a> {
    root_id: &'a Path,
    root: &'a Path,
}

fn collect_module_edges(
    file: &File,
    logical_path: &Path,
    module_path: &[String],
    root_id: &Path,
    logical_root: bool,
    root: &Path,
) -> Result<Vec<ModuleEdge>> {
    let mut edges = Vec::new();
    let source_dir = logical_path.parent().unwrap_or_else(|| Path::new("/"));
    let context = ModuleCollection { root_id, root };
    collect_module_items(
        &file.items,
        &module_base_for_file(logical_path, logical_root),
        source_dir,
        module_path.to_vec(),
        false,
        &context,
        &mut edges,
    )?;
    Ok(edges)
}

fn collect_module_items(
    items: &[Item],
    module_dir: &Path,
    path_base: &Path,
    module_path: Vec<String>,
    inherited_test_only: bool,
    context: &ModuleCollection<'_>,
    edges: &mut Vec<ModuleEdge>,
) -> Result<()> {
    for item in items {
        let Item::Mod(module) = item else {
            continue;
        };
        let test_only = inherited_test_only || has_exact_cfg_test(&module.attrs);
        let name = ident_name(&module.ident);
        let mut child_module_path = module_path.clone();
        child_module_path.push(name.clone());
        let path_attr = path_attribute(&module.attrs)?;
        let default_child_dir = module_dir.join(&name);
        if let Some((_, inner)) = &module.content {
            // An inline `#[path]` changes the logical module directory used by
            // nested external declarations, just like an external module
            // source would. The virtual source path also handles `alt.rs`.
            let child_dir = if let Some(path_attr) = path_attr {
                let virtual_source =
                    logical_join(path_base, &path_attr, context.root, "inline module path")?;
                module_base_for_file(&virtual_source, false)
            } else {
                default_child_dir
            };
            collect_module_items(
                inner,
                &child_dir,
                &child_dir,
                child_module_path,
                test_only,
                context,
                edges,
            )?;
            continue;
        }

        let logical_target = if let Some(path_attr) = path_attr {
            logical_join(path_base, &path_attr, context.root, "module path")?
        } else {
            let candidates = [
                default_child_dir.with_extension("rs"),
                default_child_dir.join("mod.rs"),
            ];
            let existing = candidates
                .iter()
                .find(|candidate| fs::symlink_metadata(candidate).is_ok());
            existing.cloned().ok_or_else(|| {
                anyhow!(
                    "module {} source not found (looked for {} and {})",
                    module.ident,
                    candidates[0].display(),
                    candidates[1].display()
                )
            })?
        };
        let target = canonical_module_path(context.root, &logical_target, "module path")?;
        edges.push(ModuleEdge {
            target,
            logical_target,
            root_id: context.root_id.to_owned(),
            module_path: child_module_path,
            test_only,
        });
    }
    Ok(())
}

fn logical_join(base: &Path, value: &str, root: &Path, description: &str) -> Result<PathBuf> {
    let candidate = if Path::new(value).is_absolute() {
        PathBuf::from(value)
    } else {
        base.join(value)
    };
    let normalized = normalize_absolute(&candidate);
    ensure_inside(root, &normalized, description)
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

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct ModuleKey {
    crate_id: PathBuf,
    path: Vec<String>,
}

#[derive(Clone)]
enum ModuleBinding {
    Unknown,
    Alias(Box<Type>),
    Path(Vec<String>),
}

#[derive(Clone, Default)]
struct ModuleScope {
    bindings: BTreeMap<String, ModuleBinding>,
    macro_bindings: BTreeMap<String, ModuleBinding>,
    children: BTreeMap<String, ModuleKey>,
}

#[derive(Default)]
struct ModuleRegistry {
    scopes: BTreeMap<ModuleKey, ModuleScope>,
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

fn builtin_macro_path(names: &[String]) -> bool {
    matches!(
        names,
        [root, name]
            if (root == "std" || root == "core")
                && matches!(
                    name.as_str(),
                    "assert"
                        | "assert_eq"
                        | "assert_ne"
                        | "debug_assert"
                        | "debug_assert_eq"
                        | "debug_assert_ne"
                        | "eprint"
                        | "eprintln"
                        | "format"
                        | "panic"
                        | "print"
                        | "println"
                        | "unreachable"
                        | "unimplemented"
                        | "vec"
                        | "write"
                        | "writeln"
                )
    )
}

fn canonical_macro(names: &[String]) -> Option<PanicKind> {
    match names {
        [root, name] if (root == "std" || root == "core") && name == "panic" => {
            Some(PanicKind::Panic)
        }
        [root, name] if (root == "std" || root == "core") && name == "unreachable" => {
            Some(PanicKind::Unreachable)
        }
        _ => None,
    }
}

fn module_key(crate_id: &Path, path: &[String]) -> ModuleKey {
    ModuleKey {
        crate_id: crate_id.to_owned(),
        path: path.to_vec(),
    }
}

fn collect_use_paths(tree: &syn::UseTree, prefix: &[String], out: &mut Vec<(String, Vec<String>)>) {
    match tree {
        syn::UseTree::Path(path) => {
            let mut next = prefix.to_vec();
            next.push(ident_name(&path.ident));
            collect_use_paths(&path.tree, &next, out);
        }
        syn::UseTree::Name(name) => {
            let mut path = prefix.to_vec();
            let name = ident_name(&name.ident);
            path.push(name.clone());
            out.push((name, path));
        }
        syn::UseTree::Rename(rename) => {
            let mut path = prefix.to_vec();
            path.push(ident_name(&rename.ident));
            out.push((ident_name(&rename.rename), path));
        }
        syn::UseTree::Group(group) => {
            for item in &group.items {
                collect_use_paths(item, prefix, out);
            }
        }
        syn::UseTree::Glob { .. } => {}
    }
}

fn insert_module_binding(
    bindings: &mut BTreeMap<String, ModuleBinding>,
    name: String,
    binding: ModuleBinding,
) {
    match bindings.get(&name) {
        None => {
            bindings.insert(name, binding);
        }
        Some(ModuleBinding::Unknown) => {}
        Some(existing) if std::mem::discriminant(existing) == std::mem::discriminant(&binding) => {
            // Duplicate imports are harmless; conflicting declarations are
            // conservatively rejected below.
            if !matches!(existing, ModuleBinding::Path(left) if matches!(&binding, ModuleBinding::Path(right) if left == right))
            {
                bindings.insert(name, ModuleBinding::Unknown);
            }
        }
        Some(_) => {
            bindings.insert(name, ModuleBinding::Unknown);
        }
    }
}

fn module_scope_from_items(items: &[Item]) -> ModuleScope {
    let mut scope = ModuleScope::default();
    // First reserve all production declarations. A cfg(test) item is skipped
    // before binding collection so test-only shadows cannot affect production.
    for item in items {
        if has_exact_cfg_test(item_attrs(item)) {
            continue;
        }
        // Only the type namespace can shadow the prelude's Option/Result.
        // Functions, constants, statics, modules, and other value bindings
        // live in separate Rust namespaces.
        let name = match item {
            Item::Enum(item) => Some(ident_name(&item.ident)),
            Item::ExternCrate(item) => Some(ident_name(&item.ident)),
            Item::Struct(item) => Some(ident_name(&item.ident)),
            Item::Trait(item) => Some(ident_name(&item.ident)),
            Item::TraitAlias(item) => Some(ident_name(&item.ident)),
            Item::Type(item) => Some(ident_name(&item.ident)),
            Item::Union(item) => Some(ident_name(&item.ident)),
            _ => None,
        };
        if let Some(name) = name {
            match item {
                Item::Type(item) => insert_module_binding(
                    &mut scope.bindings,
                    name,
                    ModuleBinding::Alias(Box::new((*item.ty).clone())),
                ),
                _ => insert_module_binding(&mut scope.bindings, name, ModuleBinding::Unknown),
            }
        }
        if let Item::Macro(item) = item {
            if let Some(name) = &item.ident {
                insert_module_binding(
                    &mut scope.macro_bindings,
                    ident_name(name),
                    ModuleBinding::Unknown,
                );
            }
        }
    }
    for item in items {
        if has_exact_cfg_test(item_attrs(item)) {
            continue;
        }
        if let Item::Use(item) = item {
            let mut imports = Vec::new();
            collect_use_paths(&item.tree, &[], &mut imports);
            for (name, path) in imports {
                if !builtin_macro_path(&path) {
                    insert_module_binding(
                        &mut scope.bindings,
                        name.clone(),
                        ModuleBinding::Path(path.clone()),
                    );
                }
                insert_module_binding(&mut scope.macro_bindings, name, ModuleBinding::Path(path));
            }
        }
    }
    scope
}

impl ModuleRegistry {
    fn build(nodes: &BTreeMap<NodeKey, SourceNode>) -> Self {
        let mut registry = Self::default();
        for node in nodes.values() {
            let key = module_key(&node.root_id, &node.module_path);
            registry
                .scopes
                .entry(key.clone())
                .or_insert_with(|| module_scope_from_items(&node.file.items));
            register_inline_modules(&mut registry.scopes, &key, &node.file.items);
        }
        for node in nodes.values() {
            for edge in &node.edges {
                let Some(name) = edge.module_path.last() else {
                    continue;
                };
                let owner = module_key(
                    &node.root_id,
                    &edge.module_path[..edge.module_path.len() - 1],
                );
                let child = module_key(&node.root_id, &edge.module_path);
                registry
                    .scopes
                    .entry(owner)
                    .or_default()
                    .children
                    .insert(name.clone(), child);
            }
        }
        registry
    }

    fn scope(&self, key: &ModuleKey) -> Option<&ModuleScope> {
        self.scopes.get(key)
    }

    fn child(&self, key: &ModuleKey, name: &str) -> Option<ModuleKey> {
        self.scope(key)?.children.get(name).cloned()
    }

    fn canonical_standard_in_scope(
        &self,
        current: &ModuleKey,
        names: &[String],
    ) -> Option<StandardType> {
        let standard = canonical_standard_type(names)?;
        let root = names.first()?;
        let shadowed = self
            .scope(current)
            .map(|scope| scope.bindings.contains_key(root) || scope.children.contains_key(root))
            .unwrap_or(false);
        (!shadowed).then_some(standard)
    }

    fn canonical_macro_in_scope(
        &self,
        current: &ModuleKey,
        names: &[String],
        lexical_bindings: &BTreeMap<String, ModuleBinding>,
    ) -> Option<PanicKind> {
        let kind = canonical_macro(names)?;
        let root = names.first()?;
        if lexical_bindings.contains_key(root) {
            return None;
        }
        let shadowed = self
            .scope(current)
            .map(|scope| scope.bindings.contains_key(root) || scope.children.contains_key(root))
            .unwrap_or(false);
        (!shadowed).then_some(kind)
    }

    fn resolve_standard(
        &self,
        current: &ModuleKey,
        names: &[String],
        leading_colon: bool,
        lexical_bindings: &BTreeMap<String, ModuleBinding>,
    ) -> Option<StandardType> {
        if names.is_empty() {
            return None;
        }
        if leading_colon {
            return canonical_standard_type(names);
        }
        let mut visiting = BTreeSet::new();
        if let Some(binding) = lexical_bindings.get(&names[0]) {
            if names.len() == 1 {
                return self.resolve_binding_value(
                    current,
                    binding,
                    &mut visiting,
                    lexical_bindings,
                );
            }
            let module = self.resolve_module_binding(current, binding, &mut BTreeSet::new())?;
            return self.resolve_standard_in_module(&module, &names[1..], &mut visiting);
        }
        self.resolve_standard_in_module(current, names, &mut visiting)
    }

    fn resolve_standard_in_module(
        &self,
        current: &ModuleKey,
        names: &[String],
        visiting: &mut BTreeSet<(ModuleKey, String)>,
    ) -> Option<StandardType> {
        if names.len() == 1 {
            return self.resolve_binding(current, &names[0], visiting, &BTreeMap::new());
        }
        if let Some(standard) = self.canonical_standard_in_scope(current, names) {
            return Some(standard);
        }
        let (module, name) = self.resolve_module_prefix(current, names)?;
        self.resolve_binding(&module, name, visiting, &BTreeMap::new())
    }

    fn module_path_start(
        &self,
        current: &ModuleKey,
        names: &[String],
    ) -> Option<(ModuleKey, usize)> {
        if names.is_empty() {
            return None;
        }
        let mut module = current.clone();
        let mut index = 0;
        match names[0].as_str() {
            "crate" => {
                module.path.clear();
                index = 1;
            }
            "self" => index = 1,
            "super" => {
                while index < names.len() && names[index] == "super" {
                    module.path.pop();
                    index += 1;
                }
            }
            _ => {}
        }
        Some((module, index))
    }

    fn resolve_module_binding(
        &self,
        current: &ModuleKey,
        binding: &ModuleBinding,
        visiting: &mut BTreeSet<(ModuleKey, String)>,
    ) -> Option<ModuleKey> {
        let ModuleBinding::Path(path) = binding else {
            return None;
        };
        self.resolve_module_path(current, path, visiting)
    }

    fn resolve_module_path(
        &self,
        current: &ModuleKey,
        names: &[String],
        visiting: &mut BTreeSet<(ModuleKey, String)>,
    ) -> Option<ModuleKey> {
        let (mut module, mut index) = self.module_path_start(current, names)?;
        while index < names.len() {
            let name = &names[index];
            if let Some(child) = self.child(&module, name) {
                module = child;
                index += 1;
                continue;
            }
            let key = (module.clone(), name.clone());
            if !visiting.insert(key.clone()) {
                return None;
            }
            let binding = self.scope(&module)?.bindings.get(name).cloned();
            let next = match binding {
                Some(binding) => self.resolve_module_binding(&module, &binding, visiting),
                None => None,
            };
            visiting.remove(&key);
            module = next?;
            index += 1;
        }
        Some(module)
    }

    fn resolve_module_prefix<'a>(
        &self,
        current: &ModuleKey,
        names: &'a [String],
    ) -> Option<(ModuleKey, &'a str)> {
        if names.len() < 2 {
            return None;
        }
        let (mut module, mut index) = self.module_path_start(current, names)?;
        while index + 1 < names.len() {
            if let Some(child) = self.child(&module, &names[index]) {
                module = child;
            } else {
                let mut visiting = BTreeSet::new();
                let binding = self.scope(&module)?.bindings.get(&names[index]).cloned()?;
                module = self.resolve_module_binding(&module, &binding, &mut visiting)?;
            }
            index += 1;
        }
        Some((module, &names[index]))
    }

    fn resolve_binding(
        &self,
        module: &ModuleKey,
        name: &str,
        visiting: &mut BTreeSet<(ModuleKey, String)>,
        lexical_bindings: &BTreeMap<String, ModuleBinding>,
    ) -> Option<StandardType> {
        if !visiting.insert((module.clone(), name.to_owned())) {
            return None;
        }
        let binding = self.scope(module)?.bindings.get(name).cloned();
        let result = match binding {
            Some(binding) => {
                self.resolve_binding_value(module, &binding, visiting, lexical_bindings)
            }
            None if self.child(module, name).is_some() => None,
            None => match name {
                "Option" => Some(StandardType::Option),
                "Result" => Some(StandardType::Result),
                _ => None,
            },
        };
        visiting.remove(&(module.clone(), name.to_owned()));
        result
    }

    fn resolve_binding_value(
        &self,
        module: &ModuleKey,
        binding: &ModuleBinding,
        visiting: &mut BTreeSet<(ModuleKey, String)>,
        lexical_bindings: &BTreeMap<String, ModuleBinding>,
    ) -> Option<StandardType> {
        match binding {
            ModuleBinding::Unknown => None,
            ModuleBinding::Alias(ty) => self.resolve_type(module, ty, visiting, lexical_bindings),
            ModuleBinding::Path(path) => {
                if let Some(standard) = self.canonical_standard_in_scope(module, path) {
                    Some(standard)
                } else if path.len() == 1 {
                    if let Some(local) = lexical_bindings.get(&path[0]) {
                        self.resolve_binding_value(module, local, visiting, lexical_bindings)
                    } else {
                        self.resolve_binding(module, &path[0], visiting, lexical_bindings)
                    }
                } else {
                    let (target, last) = self.resolve_module_prefix(module, path)?;
                    self.resolve_binding(&target, last, visiting, &BTreeMap::new())
                }
            }
        }
    }

    fn resolve_type(
        &self,
        module: &ModuleKey,
        ty: &Type,
        visiting: &mut BTreeSet<(ModuleKey, String)>,
        lexical_bindings: &BTreeMap<String, ModuleBinding>,
    ) -> Option<StandardType> {
        match ty {
            Type::Path(type_path) if type_path.qself.is_none() => {
                let names: Vec<String> = type_path
                    .path
                    .segments
                    .iter()
                    .map(|segment| ident_name(&segment.ident))
                    .collect();
                if names.len() == 1 {
                    if let Some(binding) = lexical_bindings.get(&names[0]) {
                        self.resolve_binding_value(module, binding, visiting, lexical_bindings)
                    } else {
                        self.resolve_binding(module, &names[0], visiting, lexical_bindings)
                    }
                } else if type_path.path.leading_colon.is_some() {
                    canonical_standard_type(&names)
                } else if let Some(standard) = self.canonical_standard_in_scope(module, &names) {
                    Some(standard)
                } else {
                    let (target, name) = self.resolve_module_prefix(module, &names)?;
                    self.resolve_binding(&target, name, visiting, &BTreeMap::new())
                }
            }
            Type::Paren(type_paren) => {
                self.resolve_type(module, &type_paren.elem, visiting, lexical_bindings)
            }
            Type::Group(type_group) => {
                self.resolve_type(module, &type_group.elem, visiting, lexical_bindings)
            }
            _ => None,
        }
    }

    fn resolve_macro(
        &self,
        current: &ModuleKey,
        path: &syn::Path,
        lexical_bindings: &BTreeMap<String, ModuleBinding>,
    ) -> Option<PanicKind> {
        let names: Vec<String> = path
            .segments
            .iter()
            .map(|segment| ident_name(&segment.ident))
            .collect();
        self.resolve_macro_names(
            current,
            &names,
            path.leading_colon.is_some(),
            lexical_bindings,
        )
    }

    fn resolve_macro_names(
        &self,
        current: &ModuleKey,
        names: &[String],
        leading_colon: bool,
        lexical_bindings: &BTreeMap<String, ModuleBinding>,
    ) -> Option<PanicKind> {
        if names.is_empty() {
            return None;
        }
        if let Some(kind) = self.canonical_macro_in_scope(current, names, lexical_bindings) {
            return Some(kind);
        }
        let mut visiting = BTreeSet::new();
        if names.len() == 1 {
            if let Some(binding) = lexical_bindings.get(&names[0]) {
                return self.resolve_macro_value(current, binding, &mut visiting);
            }
            return self.resolve_macro_name(current, &names[0]);
        }
        if leading_colon && canonical_macro(names).is_none() {
            return None;
        }
        if let Some(binding) = lexical_bindings.get(&names[0]) {
            let module = self.resolve_module_binding(current, binding, &mut BTreeSet::new())?;
            return self.resolve_macro_tail(&module, &names[1..], &mut visiting);
        }
        let (module, last) = self.resolve_module_prefix(current, names)?;
        self.resolve_macro_binding(&module, last, &mut visiting)
    }

    fn resolve_macro_tail(
        &self,
        current: &ModuleKey,
        names: &[String],
        visiting: &mut BTreeSet<(ModuleKey, String)>,
    ) -> Option<PanicKind> {
        if names.len() == 1 {
            return self.resolve_macro_binding(current, &names[0], visiting);
        }
        let (module, last) = self.resolve_module_prefix(current, names)?;
        self.resolve_macro_binding(&module, last, visiting)
    }

    fn resolve_macro_name(&self, current: &ModuleKey, name: &str) -> Option<PanicKind> {
        if self
            .scope(current)
            .and_then(|scope| scope.macro_bindings.get(name))
            .is_some()
        {
            return self.resolve_macro_binding(current, name, &mut BTreeSet::new());
        }
        PanicKind::from_name(name)
    }

    fn resolve_macro_value(
        &self,
        current: &ModuleKey,
        binding: &ModuleBinding,
        visiting: &mut BTreeSet<(ModuleKey, String)>,
    ) -> Option<PanicKind> {
        let ModuleBinding::Path(path) = binding else {
            return None;
        };
        if let Some(kind) = self.canonical_macro_in_scope(current, path, &BTreeMap::new()) {
            return Some(kind);
        }
        if path.len() == 1 {
            return self.resolve_macro_binding(current, &path[0], visiting);
        }
        let (module, last) = self.resolve_module_prefix(current, path)?;
        self.resolve_macro_binding(&module, last, visiting)
    }

    fn resolve_macro_binding(
        &self,
        current: &ModuleKey,
        name: &str,
        visiting: &mut BTreeSet<(ModuleKey, String)>,
    ) -> Option<PanicKind> {
        if !visiting.insert((current.clone(), name.to_owned())) {
            return None;
        }
        let binding = self.scope(current)?.macro_bindings.get(name).cloned();
        let result = match binding {
            Some(binding) => self.resolve_macro_value(current, &binding, visiting),
            None => None,
        };
        visiting.remove(&(current.clone(), name.to_owned()));
        result
    }
}

#[derive(Clone, Default)]
struct LexicalScope {
    bindings: BTreeMap<String, ModuleBinding>,
    macro_bindings: BTreeMap<String, ModuleBinding>,
}

fn collect_block_scope(block: &syn::Block) -> LexicalScope {
    let mut items = Vec::new();
    for statement in &block.stmts {
        let syn::Stmt::Item(item) = statement else {
            continue;
        };
        if !has_exact_cfg_test(item_attrs(item)) {
            items.push(item.clone());
        }
    }
    let scope = module_scope_from_items(&items);
    LexicalScope {
        bindings: scope.bindings,
        macro_bindings: scope.macro_bindings,
    }
}

fn register_inline_modules(
    scopes: &mut BTreeMap<ModuleKey, ModuleScope>,
    parent: &ModuleKey,
    items: &[Item],
) {
    for item in items {
        let Item::Mod(module) = item else {
            continue;
        };
        if has_exact_cfg_test(&module.attrs) {
            continue;
        }
        let mut path = parent.path.clone();
        let name = ident_name(&module.ident);
        path.push(name.clone());
        let key = module_key(&parent.crate_id, &path);
        if let Some((_, inner)) = &module.content {
            scopes
                .entry(key.clone())
                .or_insert_with(|| module_scope_from_items(inner));
            if let Some(scope) = scopes.get_mut(parent) {
                scope.children.insert(name, key.clone());
            }
            register_inline_modules(scopes, &key, inner);
        }
    }
}

fn associated_method(
    path: &ExprPath,
    registry: &ModuleRegistry,
    current: &ModuleKey,
    lexical_bindings: &BTreeMap<String, ModuleBinding>,
) -> Option<PanicKind> {
    let method = path.path.segments.last()?;
    let method_name = ident_name(&method.ident);
    let kind = PanicKind::from_name(&method_name)?;
    if !matches!(kind, PanicKind::Unwrap | PanicKind::Expect) {
        return None;
    }
    let standard = if let Some(qself) = &path.qself {
        registry.resolve_type(current, &qself.ty, &mut BTreeSet::new(), lexical_bindings)
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
            registry.resolve_standard(
                current,
                &names[..names.len() - 1],
                path.path.leading_colon.is_some(),
                lexical_bindings,
            )
        }
    }?;
    let _ = standard;
    Some(kind)
}

fn is_punct(tree: Option<&TokenTree>, expected: char) -> bool {
    matches!(tree, Some(TokenTree::Punct(punct)) if punct.as_char() == expected)
}

fn matching_angle_open(trees: &[TokenTree], close: usize) -> Option<usize> {
    let mut depth = 0usize;
    for index in (0..=close).rev() {
        if is_punct(trees.get(index), '>') {
            depth += 1;
        } else if is_punct(trees.get(index), '<') {
            depth = depth.checked_sub(1)?;
            if depth == 0 {
                return Some(index);
            }
        }
    }
    None
}

fn matching_angle_close(trees: &[TokenTree], open: usize, end: usize) -> Option<usize> {
    let mut depth = 0usize;
    for index in open..end {
        if is_punct(trees.get(index), '<') {
            depth += 1;
        } else if is_punct(trees.get(index), '>') {
            depth = depth.checked_sub(1)?;
            if depth == 0 {
                return Some(index);
            }
        }
    }
    None
}

fn token_path_range(
    trees: &[TokenTree],
    start: usize,
    end: usize,
) -> Option<(Vec<String>, bool, usize)> {
    if start >= end {
        return None;
    }
    let mut index = start;
    let leading_colon =
        index + 1 < end && is_punct(trees.get(index), ':') && is_punct(trees.get(index + 1), ':');
    if leading_colon {
        index += 2;
    }
    let first_index = index;
    let mut names = Vec::new();
    loop {
        let Some(TokenTree::Ident(ident)) = trees.get(index) else {
            return None;
        };
        names.push(ident_name(ident));
        index += 1;
        if index < end && is_punct(trees.get(index), '<') {
            let close = matching_angle_close(trees, index, end)?;
            index = close + 1;
        }
        if index == end {
            break;
        }
        if index + 1 >= end
            || !is_punct(trees.get(index), ':')
            || !is_punct(trees.get(index + 1), ':')
        {
            return None;
        }
        index += 2;
    }
    Some((names, leading_colon, first_index))
}

fn token_call_group_after(trees: &[TokenTree], method_index: usize) -> bool {
    let mut next = method_index + 1;
    if is_punct(trees.get(next), '<') {
        let Some(close) = matching_angle_close(trees, next, trees.len()) else {
            return false;
        };
        next = close + 1;
    }
    matches!(
        trees.get(next),
        Some(TokenTree::Group(group)) if group.delimiter() == Delimiter::Parenthesis
    )
}

fn token_path_start(trees: &[TokenTree], end: usize) -> Option<usize> {
    let mut cursor = end.checked_sub(1)?;
    loop {
        let Some(TokenTree::Ident(_)) = trees.get(cursor) else {
            return None;
        };
        if cursor >= 3
            && is_punct(trees.get(cursor - 1), ':')
            && is_punct(trees.get(cursor - 2), ':')
            && matches!(trees.get(cursor - 3), Some(TokenTree::Ident(_)))
        {
            cursor -= 3;
            continue;
        }
        if cursor >= 2
            && is_punct(trees.get(cursor - 1), ':')
            && is_punct(trees.get(cursor - 2), ':')
        {
            return Some(cursor - 2);
        }
        return Some(cursor);
    }
}

/// Return the literal base path immediately before an associated `unwrap` or
/// `expect` token. Balanced angle groups cover both turbofish and QSelf forms;
/// their contents are skipped without expanding metavariables.
fn token_ufcs_path(trees: &[TokenTree], method_index: usize) -> Option<(Vec<String>, bool, usize)> {
    if method_index < 2
        || !is_punct(trees.get(method_index - 1), ':')
        || !is_punct(trees.get(method_index - 2), ':')
    {
        return None;
    }
    let end = method_index - 2;
    let last = end.checked_sub(1)?;
    if is_punct(trees.get(last), '>') {
        let open = matching_angle_open(trees, last)?;
        if open >= 2 && is_punct(trees.get(open - 1), ':') && is_punct(trees.get(open - 2), ':') {
            // `Option::<T>::unwrap`: parse the path before its turbofish.
            let path_end = open - 2;
            token_path_range(trees, token_path_start(trees, path_end)?, path_end)
        } else {
            // `<Option<T>>::unwrap`: parse the path inside the outer QSelf.
            token_path_range(trees, open + 1, last)
        }
    } else {
        token_path_range(trees, token_path_start(trees, end)?, end)
    }
}

fn token_macro_path(trees: &[TokenTree], macro_index: usize) -> Option<(Vec<String>, bool, usize)> {
    let mut first_ident = macro_index;
    while first_ident >= 3
        && is_punct(trees.get(first_ident - 1), ':')
        && is_punct(trees.get(first_ident - 2), ':')
    {
        first_ident -= 3;
        if !matches!(trees.get(first_ident), Some(TokenTree::Ident(_))) {
            return None;
        }
    }
    let leading_colon = first_ident >= 2
        && is_punct(trees.get(first_ident - 1), ':')
        && is_punct(trees.get(first_ident - 2), ':');
    let first_index = if leading_colon {
        first_ident - 2
    } else {
        first_ident
    };
    let mut index = if leading_colon {
        first_index + 2
    } else {
        first_index
    };
    let mut names = Vec::new();
    loop {
        let Some(TokenTree::Ident(ident)) = trees.get(index) else {
            return None;
        };
        names.push(ident_name(ident));
        if index == macro_index {
            break;
        }
        if index + 2 > macro_index
            || !is_punct(trees.get(index + 1), ':')
            || !is_punct(trees.get(index + 2), ':')
        {
            return None;
        }
        index += 3;
    }
    Some((names, leading_colon, first_index))
}

struct SourceVisitor<'source> {
    path: String,
    source: &'source str,
    line_starts: Vec<usize>,
    findings: BTreeSet<Finding>,
    public_context: bool,
    trait_impl_context: bool,
    registry: &'source ModuleRegistry,
    module_stack: Vec<ModuleKey>,
    lexical_scopes: Vec<LexicalScope>,
}

impl<'source> SourceVisitor<'source> {
    fn new(
        path: String,
        source: &'source str,
        registry: &'source ModuleRegistry,
        module: ModuleKey,
    ) -> Self {
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
            registry,
            module_stack: vec![module],
            lexical_scopes: Vec::new(),
        }
    }

    fn add(&mut self, kind: PanicKind, span: Span) {
        self.findings.insert(Finding {
            path: self.path.clone(),
            line: span.start().line,
            kind: kind.name().to_owned(),
        });
    }

    fn current_module(&self) -> &ModuleKey {
        self.module_stack
            .last()
            .expect("source visitor always has a module")
    }

    fn lexical_bindings(&self) -> BTreeMap<String, ModuleBinding> {
        let mut bindings = BTreeMap::new();
        for scope in &self.lexical_scopes {
            for (name, binding) in &scope.bindings {
                bindings.insert(name.clone(), binding.clone());
            }
        }
        bindings
    }

    fn lexical_macro_bindings(&self) -> BTreeMap<String, ModuleBinding> {
        let mut bindings = BTreeMap::new();
        for scope in &self.lexical_scopes {
            for (name, binding) in &scope.macro_bindings {
                bindings.insert(name.clone(), binding.clone());
            }
        }
        bindings
    }

    fn push_lexical_scope(&mut self, scope: LexicalScope) {
        self.lexical_scopes.push(scope);
    }

    fn push_generic_scope(&mut self, generics: &syn::Generics) {
        let mut scope = LexicalScope::default();
        for parameter in &generics.params {
            if let syn::GenericParam::Type(parameter) = parameter {
                scope
                    .bindings
                    .insert(ident_name(&parameter.ident), ModuleBinding::Unknown);
            }
        }
        self.push_lexical_scope(scope);
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
        let lexical_bindings = self.lexical_bindings();
        let lexical_macros = self.lexical_macro_bindings();
        for index in 0..trees.len() {
            if let TokenTree::Ident(ident) = &trees[index] {
                let name = ident_name(ident);
                if is_punct(trees.get(index + 1), '!') {
                    if let Some((names, leading_colon, first_index)) =
                        token_macro_path(&trees, index)
                    {
                        // A path assembled from `$` is intentionally outside
                        // this scanner's conservative literal boundary.
                        let has_metavariable = (first_index > 0
                            && is_punct(trees.get(first_index - 1), '$'))
                            || trees[first_index..=index].iter().any(
                                |tree| matches!(tree, TokenTree::Punct(punct) if punct.as_char() == '$'),
                            );
                        if !has_metavariable {
                            if let Some(kind) = self.registry.resolve_macro_names(
                                self.current_module(),
                                &names,
                                leading_colon,
                                &lexical_macros,
                            ) {
                                if kind.is_raw() || self.public_context {
                                    self.add(kind, ident.span());
                                }
                            }
                        }
                    }
                }
                if matches!(name.as_str(), "unwrap" | "expect")
                    && token_call_group_after(&trees, index)
                {
                    if let Some((names, leading_colon, first_index)) =
                        token_ufcs_path(&trees, index)
                    {
                        // A path assembled from `$` is intentionally outside
                        // this scanner's conservative literal boundary.
                        if first_index > 0 && is_punct(trees.get(first_index - 1), '$') {
                            continue;
                        }
                        if self
                            .registry
                            .resolve_standard(
                                self.current_module(),
                                &names,
                                leading_colon,
                                &lexical_bindings,
                            )
                            .is_some()
                        {
                            let kind =
                                PanicKind::from_name(&name).expect("method name checked above");
                            self.add(kind, ident.span());
                        }
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
                    && token_call_group_after(&trees, index + 1)
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

    fn scan_macro_rules(&mut self, tokens: TokenStream) {
        let trees: Vec<TokenTree> = tokens.into_iter().collect();
        let mut index = 0;
        while index < trees.len() {
            if !matches!(trees.get(index), Some(TokenTree::Group(_))) {
                index += 1;
                continue;
            }
            index += 1;
            if is_punct(trees.get(index), '=') && is_punct(trees.get(index + 1), '>') {
                index += 2;
                if let Some(TokenTree::Group(group)) = trees.get(index) {
                    // Matchers describe input and are never executable. Only
                    // the transcriber RHS is inspected.
                    self.scan_literal_token_patterns(&group.stream());
                }
                index += 1;
            }
            while index < trees.len() && is_punct(trees.get(index), ';') {
                index += 1;
            }
        }
    }

    /// Scan literal forbidden constructs in source macro arguments and macro
    /// transcribers. This does not expand arbitrary metavariable calls:
    /// `$panic!` and `$ty::unwrap` are intentionally outside the boundary.
    fn scan_macro_tokens(&mut self, tokens: TokenStream) {
        if self.try_parse_macro_stream(tokens.clone()) {
            return;
        }
        self.scan_literal_token_patterns(&tokens);
    }
}

impl<'ast, 'source> Visit<'ast> for SourceVisitor<'source> {
    fn visit_file(&mut self, node: &'ast File) {
        // Module declarations were collected before traversal. Lexical
        // function/block scopes are intentionally kept separate below.
        visit::visit_file(self, node);
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
        let parent = self.current_module().clone();
        let name = ident_name(&node.ident);
        let child = self.registry.child(&parent, &name).unwrap_or_else(|| {
            let mut path = parent.path.clone();
            path.push(name);
            module_key(&parent.crate_id, &path)
        });
        self.module_stack.push(child);
        for item in items {
            self.visit_item(item);
        }
        self.module_stack.pop();
    }

    fn visit_block(&mut self, node: &'ast syn::Block) {
        self.push_lexical_scope(collect_block_scope(node));
        visit::visit_block(self, node);
        self.lexical_scopes.pop();
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
        self.lexical_scopes.pop();
        self.public_context = previous;
    }

    fn visit_item_trait(&mut self, node: &'ast syn::ItemTrait) {
        let previous = self.public_context;
        self.public_context |= is_public(&node.vis);
        self.push_generic_scope(&node.generics);
        visit::visit_item_trait(self, node);
        self.lexical_scopes.pop();
        self.public_context = previous;
    }

    fn visit_item_impl(&mut self, node: &'ast syn::ItemImpl) {
        let previous = self.trait_impl_context;
        self.trait_impl_context |= node.trait_.is_some();
        self.push_generic_scope(&node.generics);
        visit::visit_item_impl(self, node);
        self.lexical_scopes.pop();
        self.trait_impl_context = previous;
    }

    fn visit_impl_item_fn(&mut self, node: &'ast syn::ImplItemFn) {
        let previous = self.public_context;
        self.public_context |= self.trait_impl_context || is_public(&node.vis);
        self.push_generic_scope(&node.sig.generics);
        visit::visit_impl_item_fn(self, node);
        self.lexical_scopes.pop();
        self.public_context = previous;
    }

    fn visit_item_macro(&mut self, node: &'ast syn::ItemMacro) {
        if node.ident.is_some() && node.mac.path.is_ident("macro_rules") {
            self.scan_macro_rules(node.mac.tokens.clone());
            return;
        }
        visit::visit_item_macro(self, node);
    }

    fn visit_macro(&mut self, node: &'ast syn::Macro) {
        let lexical_macros = self.lexical_macro_bindings();
        if let Some(kind) =
            self.registry
                .resolve_macro(self.current_module(), &node.path, &lexical_macros)
        {
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
            let lexical_bindings = self.lexical_bindings();
            if let Some(kind) = associated_method(
                path,
                self.registry,
                self.current_module(),
                &lexical_bindings,
            ) {
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
