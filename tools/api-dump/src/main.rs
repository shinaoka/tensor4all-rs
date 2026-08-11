use clap::Parser;
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use syn::{
    Attribute, Expr, ExprLit, FnArg, ImplItem, Item, ItemFn, ItemImpl, ItemTrait, Lit, Meta, Pat,
    PatIdent, PatType, Receiver, Signature, TraitItem, Visibility,
};
use walkdir::WalkDir;

#[derive(Parser)]
#[command(name = "api-dump")]
#[command(about = "Dump Rust workspace API to Markdown")]
struct Args {
    /// Path to the workspace root (containing Cargo.toml)
    #[arg(default_value = ".")]
    workspace: PathBuf,

    /// Output directory for markdown files
    #[arg(short, long, default_value = "api-docs")]
    output: PathBuf,
}

#[derive(Debug, serde::Deserialize)]
struct CargoToml {
    workspace: Option<WorkspaceConfig>,
    package: Option<PackageConfig>,
}

#[derive(Debug, serde::Deserialize)]
struct WorkspaceConfig {
    members: Option<Vec<String>>,
}

#[derive(Debug, serde::Deserialize)]
struct PackageConfig {
    name: Option<String>,
}

/// Extracted function information
#[derive(Debug)]
struct FuncInfo {
    visibility: String,
    signature: String,
    doc_summary: Option<String>,
    kind: FuncKind,
}

#[derive(Debug)]
enum FuncKind {
    Free,
    Method { impl_for: String },
    TraitMethod { trait_name: String },
    TraitDefaultMethod { trait_name: String },
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    let workspace_path = args.workspace.canonicalize()?;
    let cargo_toml_path = workspace_path.join("Cargo.toml");

    if !cargo_toml_path.exists() {
        eprintln!("Error: Cargo.toml not found at {:?}", cargo_toml_path);
        std::process::exit(1);
    }

    let cargo_content = fs::read_to_string(&cargo_toml_path)?;
    let cargo: CargoToml = toml::from_str(&cargo_content)?;

    // Collect crate paths
    let crate_paths: Vec<PathBuf> = if let Some(ws) = &cargo.workspace {
        if let Some(members) = &ws.members {
            members
                .iter()
                .flat_map(|pattern| expand_glob(&workspace_path, pattern))
                .collect()
        } else {
            vec![workspace_path.clone()]
        }
    } else {
        // Single crate
        vec![workspace_path.clone()]
    };

    // Create output directory
    fs::create_dir_all(&args.output)?;

    for crate_path in crate_paths {
        if let Err(e) = process_crate(&crate_path, &args.output) {
            eprintln!("Warning: Failed to process {:?}: {}", crate_path, e);
        }
    }

    println!("API documentation written to {:?}", args.output);
    Ok(())
}

fn expand_glob(base: &Path, pattern: &str) -> Vec<PathBuf> {
    // Simple glob expansion for workspace members
    if pattern.contains('*') {
        let prefix = pattern.trim_end_matches("/*").trim_end_matches("*");
        let search_dir = base.join(prefix);
        if search_dir.is_dir() {
            fs::read_dir(&search_dir)
                .into_iter()
                .flatten()
                .filter_map(|e| e.ok())
                .map(|e| e.path())
                .filter(|p| p.is_dir() && p.join("Cargo.toml").exists())
                .collect()
        } else {
            vec![]
        }
    } else {
        let path = base.join(pattern);
        if path.join("Cargo.toml").exists() {
            vec![path]
        } else {
            vec![]
        }
    }
}

fn process_crate(crate_path: &Path, output_dir: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let cargo_toml_path = crate_path.join("Cargo.toml");
    let cargo_content = fs::read_to_string(&cargo_toml_path)?;
    let cargo: CargoToml = toml::from_str(&cargo_content)?;

    let crate_name = cargo.package.and_then(|p| p.name).unwrap_or_else(|| {
        crate_path
            .file_name()
            .unwrap()
            .to_string_lossy()
            .to_string()
    });

    let src_dir = crate_path.join("src");
    if !src_dir.exists() {
        return Ok(());
    }

    // Collect all .rs files
    let mut files: BTreeMap<PathBuf, Vec<FuncInfo>> = BTreeMap::new();

    for entry in WalkDir::new(&src_dir)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().is_some_and(|ext| ext == "rs"))
    {
        let file_path = entry.path();
        let relative_path = file_path.strip_prefix(crate_path).unwrap_or(file_path);

        match parse_file(file_path) {
            Ok(funcs) if !funcs.is_empty() => {
                files.insert(relative_path.to_path_buf(), funcs);
            }
            Ok(_) => {}
            Err(e) => {
                eprintln!("Warning: Failed to parse {:?}: {}", file_path, e);
            }
        }
    }

    if files.is_empty() {
        return Ok(());
    }

    // Generate markdown
    let mut md = String::new();
    md.push_str(&format!("# {}\n\n", crate_name));

    for (file_path, funcs) in &files {
        md.push_str(&format!("## {}\n\n", file_path.display()));

        for func in funcs {
            let kind_str = match &func.kind {
                FuncKind::Free => String::new(),
                FuncKind::Method { impl_for } => format!(" (impl {})", impl_for),
                FuncKind::TraitMethod { trait_name } => format!(" (trait {})", trait_name),
                FuncKind::TraitDefaultMethod { trait_name } => {
                    format!(" (trait {} default)", trait_name)
                }
            };

            md.push_str(&format!(
                "### `{} fn {}`{}\n\n",
                func.visibility, func.signature, kind_str
            ));

            if let Some(doc) = &func.doc_summary {
                md.push_str(doc);
                md.push_str("\n\n");
            }
        }
    }

    let output_path = output_dir.join(format!("{}.md", crate_name.replace('-', "_")));
    fs::write(&output_path, md)?;
    println!("  Generated: {:?}", output_path);

    Ok(())
}

fn parse_file(path: &Path) -> Result<Vec<FuncInfo>, Box<dyn std::error::Error>> {
    let content = fs::read_to_string(path)?;
    let syntax = syn::parse_file(&content)?;

    let mut funcs = Vec::new();
    extract_items(&syntax.items, &mut funcs);
    Ok(funcs)
}

/// A public item is one a downstream user can call: `pub` visibility only.
/// `pub(crate)`/`pub(super)`/`pub(self)` (Restricted) and private (Inherited)
/// items are not part of the public API surface and are excluded from the
/// dump.
fn is_public(vis: &Visibility) -> bool {
    matches!(vis, Visibility::Public(_))
}

fn extract_items(items: &[Item], funcs: &mut Vec<FuncInfo>) {
    for item in items {
        match item {
            Item::Fn(item_fn) => {
                if is_public(&item_fn.vis) {
                    funcs.push(extract_fn_info(item_fn));
                }
            }
            Item::Impl(item_impl) => {
                extract_impl_items(item_impl, funcs);
            }
            Item::Trait(item_trait) => {
                extract_trait_items(item_trait, funcs);
            }
            Item::Mod(item_mod) => {
                if let Some((_, items)) = &item_mod.content {
                    extract_items(items, funcs);
                }
            }
            _ => {}
        }
    }
}

fn extract_fn_info(item_fn: &ItemFn) -> FuncInfo {
    FuncInfo {
        visibility: vis_to_string(&item_fn.vis),
        signature: sig_to_string(&item_fn.sig),
        doc_summary: extract_doc_summary(&item_fn.attrs),
        kind: FuncKind::Free,
    }
}

fn extract_impl_items(item_impl: &ItemImpl, funcs: &mut Vec<FuncInfo>) {
    let impl_for = type_to_string(&item_impl.self_ty);

    for impl_item in &item_impl.items {
        if let ImplItem::Fn(method) = impl_item {
            if !is_public(&method.vis) {
                continue;
            }
            funcs.push(FuncInfo {
                visibility: vis_to_string(&method.vis),
                signature: sig_to_string(&method.sig),
                doc_summary: extract_doc_summary(&method.attrs),
                kind: FuncKind::Method {
                    impl_for: impl_for.clone(),
                },
            });
        }
    }
}

fn extract_trait_items(item_trait: &ItemTrait, funcs: &mut Vec<FuncInfo>) {
    let trait_name = item_trait.ident.to_string();

    for trait_item in &item_trait.items {
        if let TraitItem::Fn(method) = trait_item {
            // A trait method is public when its trait is public; non-public
            // traits are excluded from the dump.
            if !is_public(&item_trait.vis) {
                continue;
            }
            let has_default = method.default.is_some();
            funcs.push(FuncInfo {
                visibility: vis_to_string(&item_trait.vis),
                signature: sig_to_string(&method.sig),
                doc_summary: extract_doc_summary(&method.attrs),
                kind: if has_default {
                    FuncKind::TraitDefaultMethod {
                        trait_name: trait_name.clone(),
                    }
                } else {
                    FuncKind::TraitMethod {
                        trait_name: trait_name.clone(),
                    }
                },
            });
        }
    }
}

fn vis_to_string(vis: &Visibility) -> String {
    match vis {
        Visibility::Public(_) => "pub".to_string(),
        Visibility::Restricted(r) => {
            let path = r
                .path
                .segments
                .iter()
                .map(|s| s.ident.to_string())
                .collect::<Vec<_>>()
                .join("::");
            format!("pub({})", path)
        }
        Visibility::Inherited => String::new(),
    }
}

fn sig_to_string(sig: &Signature) -> String {
    let name = &sig.ident;
    let args: Vec<String> = sig.inputs.iter().map(fn_arg_to_string).collect();
    let ret = match &sig.output {
        syn::ReturnType::Default => String::new(),
        syn::ReturnType::Type(_, ty) => format!(" -> {}", type_to_string(ty)),
    };
    format!("{}({}){}", name, args.join(", "), ret)
}

fn fn_arg_to_string(arg: &FnArg) -> String {
    match arg {
        FnArg::Receiver(Receiver {
            reference,
            mutability,
            ..
        }) => {
            let ref_str = if reference.is_some() { "&" } else { "" };
            let mut_str = if mutability.is_some() { "mut " } else { "" };
            format!("{}{}self", ref_str, mut_str)
        }
        FnArg::Typed(PatType { pat, ty, .. }) => {
            let name = if let Pat::Ident(PatIdent { ident, .. }) = pat.as_ref() {
                ident.to_string()
            } else {
                "_".to_string()
            };
            format!("{}: {}", name, type_to_string(ty))
        }
    }
}

fn type_to_string(ty: &syn::Type) -> String {
    quote::quote!(#ty).to_string()
}

fn extract_doc_summary(attrs: &[Attribute]) -> Option<String> {
    let mut lines = Vec::new();

    for attr in attrs {
        if !attr.path().is_ident("doc") {
            continue;
        }
        if let Meta::NameValue(nv) = &attr.meta {
            if let Expr::Lit(ExprLit {
                lit: Lit::Str(s), ..
            }) = &nv.value
            {
                let text = s.value();
                let trimmed = text.trim();

                // Stop at section headers
                if trimmed.starts_with('#') {
                    break;
                }

                if !trimmed.is_empty() {
                    lines.push(trimmed.to_string());
                }

                // Max 3 lines
                if lines.len() >= 3 {
                    break;
                }
            }
        }
    }

    if lines.is_empty() {
        None
    } else {
        Some(lines.join(" "))
    }
}
