//! Development tasks for tensor4all-rs workspace.
//!
//! Usage: `cargo xtask <command>`

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use std::fs;
use std::path::Path;
use std::process::Command;

#[derive(Parser)]
#[command(name = "xtask", about = "Development tasks for tensor4all-rs")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Generate documentation with custom index page
    Doc {
        /// Open documentation in browser after generation
        #[arg(long)]
        open: bool,
    },
    /// Run all CI checks (fmt, clippy, test, doc)
    Ci,
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Doc { open } => cmd_doc(open),
        Commands::Ci => cmd_ci(),
    }
}

fn project_root() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap()
}

fn cmd_doc(open: bool) -> Result<()> {
    let root = project_root();

    // Run cargo doc
    println!("📚 Generating documentation...");
    let status = Command::new("cargo")
        .args(["doc", "--workspace", "--no-deps"])
        .current_dir(root)
        .status()
        .context("Failed to run cargo doc")?;

    if !status.success() {
        anyhow::bail!("cargo doc failed");
    }

    // Generate index.html
    println!("📝 Generating index.html...");
    generate_doc_index(root)?;

    if open {
        let index_path = root.join("target/doc/index.html");
        println!("🌐 Opening {}...", index_path.display());
        #[cfg(target_os = "macos")]
        Command::new("open").arg(&index_path).status().ok();
        #[cfg(target_os = "linux")]
        Command::new("xdg-open").arg(&index_path).status().ok();
        #[cfg(target_os = "windows")]
        Command::new("cmd")
            .args(["/c", "start", "", index_path.to_str().unwrap()])
            .status()
            .ok();
    }

    println!("✅ Documentation generated at target/doc/index.html");
    Ok(())
}

fn cmd_ci() -> Result<()> {
    let root = project_root();

    println!("🔧 Running cargo fmt...");
    run_cargo(root, &["fmt", "--all", "--", "--check"])?;

    println!("📎 Running cargo clippy...");
    run_cargo(root, &["clippy", "--workspace", "--", "-D", "warnings"])?;

    println!("🧪 Running release-mode cargo test...");
    run_cargo(root, &["test", "--release", "--workspace"])?;

    println!("📚 Checking documentation...");
    run_cargo(root, &["doc", "--workspace", "--no-deps"])?;

    println!("✅ All CI checks passed!");
    Ok(())
}

fn run_cargo(dir: &Path, args: &[&str]) -> Result<()> {
    let status = Command::new("cargo")
        .args(args)
        .current_dir(dir)
        .status()
        .with_context(|| format!("Failed to run cargo {}", args.join(" ")))?;

    if !status.success() {
        anyhow::bail!("cargo {} failed", args.join(" "));
    }
    Ok(())
}

fn generate_doc_index(root: &Path) -> Result<()> {
    let crates_dir = root.join("crates");
    let doc_dir = root.join("target/doc");

    // Scan crates directory
    let mut crates = Vec::new();
    for entry in fs::read_dir(&crates_dir).context("Failed to read crates directory")? {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            let cargo_toml = path.join("Cargo.toml");
            if cargo_toml.exists() {
                let content = fs::read_to_string(&cargo_toml)?;
                let toml: toml::Value = content.parse()?;

                let name = toml
                    .get("package")
                    .and_then(|p| p.get("name"))
                    .and_then(|n| n.as_str())
                    .unwrap_or_default();

                let description = toml
                    .get("package")
                    .and_then(|p| p.get("description"))
                    .and_then(|d| d.as_str())
                    .unwrap_or("");

                if !name.is_empty() {
                    // Convert crate name to doc directory name (- to _)
                    let doc_name = name.replace('-', "_");
                    crates.push((name.to_string(), doc_name, description.to_string()));
                }
            }
        }
    }

    crates.sort_by(|a, b| a.0.cmp(&b.0));

    // Generate HTML
    let mut crate_list = String::new();
    for (name, doc_name, desc) in &crates {
        crate_list.push_str(&format!(
            r#"        <div class="crate-card">
            <h3><a href="{}/index.html">{}</a></h3>
            <p>{}</p>
        </div>
"#,
            doc_name,
            name,
            if desc.is_empty() {
                "(no description)"
            } else {
                desc
            }
        ));
    }

    let html = format!(
        r#"<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>tensor4all-rs Documentation</title>
    <style>
        :root {{
            --bg-color: #fff;
            --text-color: #333;
            --link-color: #4a90d9;
            --border-color: #e0e0e0;
            --card-bg: #fafafa;
        }}
        @media (prefers-color-scheme: dark) {{
            :root {{
                --bg-color: #1a1a1a;
                --text-color: #e0e0e0;
                --link-color: #6ab0f3;
                --border-color: #444;
                --card-bg: #252525;
            }}
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            max-width: 900px;
            margin: 0 auto;
            padding: 2rem;
            background: var(--bg-color);
            color: var(--text-color);
            line-height: 1.6;
        }}
        h1 {{ border-bottom: 2px solid var(--border-color); padding-bottom: 0.5rem; }}
        .crate-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
            gap: 1rem;
            margin: 1rem 0;
        }}
        .crate-card {{
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 1rem;
            background: var(--card-bg);
        }}
        .crate-card h3 {{ margin: 0 0 0.5rem 0; }}
        .crate-card a {{
            color: var(--link-color);
            text-decoration: none;
            font-weight: 600;
        }}
        .crate-card a:hover {{ text-decoration: underline; }}
        .crate-card p {{ margin: 0; font-size: 0.9rem; opacity: 0.85; }}
    </style>
</head>
<body>
    <h1>tensor4all-rs</h1>
    <p>Rust implementation of tensor network algorithms for the
       <a href="https://github.com/tensor4all">tensor4all</a> project.</p>

    <h2>Crates ({} total)</h2>
    <div class="crate-grid">
{}    </div>

    <hr style="margin-top: 3rem;">
    <p style="font-size: 0.85rem; opacity: 0.7;">
        Generated by <code>cargo xtask doc</code>
    </p>
</body>
</html>
"#,
        crates.len(),
        crate_list
    );

    fs::write(doc_dir.join("index.html"), html)?;
    Ok(())
}
