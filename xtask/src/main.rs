//! Development tasks for tensor4all-rs workspace.
//!
//! Usage: `cargo xtask <command>`

use anyhow::{bail, Context, Result};
use clap::{Parser, Subcommand};
use std::collections::BTreeSet;
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
    /// Generate and verify the complete public-crate API inventory
    ApiDump,
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Doc { open } => cmd_doc(open),
        Commands::Ci => cmd_ci(),
        Commands::ApiDump => cmd_api_dump(),
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

    require_success(status.success(), &format!("cargo {}", args.join(" ")))
}

fn require_success(success: bool, command: &str) -> Result<()> {
    if !success {
        bail!("{command} failed");
    }
    Ok(())
}

fn cmd_api_dump() -> Result<()> {
    let root = project_root();
    let output = root.join("target/api-dump");

    if output.exists() {
        fs::remove_dir_all(&output).context("failed to clear target/api-dump")?;
    }
    fs::create_dir_all(&output).context("failed to create target/api-dump")?;

    let output_arg = output.to_string_lossy().into_owned();
    run_cargo(
        root,
        &[
            "run",
            "-p",
            "api-dump",
            "--release",
            "--",
            ".",
            "-o",
            &output_arg,
        ],
    )?;

    validate_api_inventory(root, &output)
        .with_context(|| format!("invalid generated API inventory at {}", output.display()))?;
    println!("✅ Complete API inventory verified at {}", output.display());
    Ok(())
}

fn expected_api_files(root: &Path) -> Result<BTreeSet<String>> {
    let output = Command::new("cargo")
        .args(["metadata", "--no-deps", "--format-version", "1"])
        .current_dir(root)
        .output()
        .context("failed to run cargo metadata")?;
    require_success(
        output.status.success(),
        "cargo metadata --no-deps --format-version 1",
    )?;

    let metadata: serde_json::Value =
        serde_json::from_slice(&output.stdout).context("cargo metadata returned invalid JSON")?;
    let packages = metadata
        .get("packages")
        .and_then(serde_json::Value::as_array)
        .context("cargo metadata omitted packages")?;
    let crates_dir = root.join("crates");
    let mut public_crates = Vec::new();

    for package in packages {
        let manifest = package
            .get("manifest_path")
            .and_then(serde_json::Value::as_str)
            .map(Path::new)
            .context("workspace package omitted manifest_path")?;
        if manifest.parent().and_then(Path::parent) != Some(crates_dir.as_path()) {
            continue;
        }
        let name = package
            .get("name")
            .and_then(serde_json::Value::as_str)
            .context("workspace package omitted name")?;
        public_crates.push(name);
    }

    if public_crates.is_empty() {
        bail!("workspace metadata contains no public crates under crates/");
    }
    api_file_names(public_crates)
}

fn api_file_names<'a>(names: impl IntoIterator<Item = &'a str>) -> Result<BTreeSet<String>> {
    let mut expected = BTreeSet::new();
    for name in names {
        let file_name = format!("{}.md", name.replace('-', "_"));
        if !expected.insert(file_name.clone()) {
            bail!("multiple public crates map to API file {file_name}");
        }
    }
    Ok(expected)
}

fn validate_api_inventory(root: &Path, output: &Path) -> Result<()> {
    let expected = expected_api_files(root)?;
    validate_inventory(&expected, output)
}

fn validate_inventory(expected: &BTreeSet<String>, output: &Path) -> Result<()> {
    let actual = fs::read_dir(output)
        .with_context(|| format!("failed to read {}", output.display()))?
        .map(|entry| {
            let path = entry?.path();
            if !path.is_file() {
                bail!("unexpected directory in API inventory: {}", path.display());
            }
            let name = path
                .file_name()
                .and_then(|name| name.to_str())
                .context("API inventory contains a non-UTF-8 filename")?;
            if !name.ends_with(".md") {
                bail!("unexpected API inventory artifact: {name}");
            }
            Ok(name.to_string())
        })
        .collect::<Result<BTreeSet<_>>>()?;

    if &actual != expected {
        let missing = expected.difference(&actual).cloned().collect::<Vec<_>>();
        let extra = actual.difference(expected).cloned().collect::<Vec<_>>();
        bail!("API inventory mismatch; missing: {missing:?}; extra: {extra:?}");
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

#[cfg(test)]
mod tests {
    use super::{api_file_names, require_success, validate_inventory};
    use std::collections::BTreeSet;
    use std::fs;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_inventory() -> PathBuf {
        let suffix = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock is before the Unix epoch")
            .as_nanos();
        let path = std::env::temp_dir().join(format!("tensor4all-api-dump-{suffix}"));
        fs::create_dir(&path).unwrap();
        path
    }

    #[test]
    fn command_failures_are_reported() {
        let error = require_success(false, "test command").unwrap_err();
        assert!(error.to_string().contains("test command failed"));
    }

    #[test]
    fn successful_commands_are_accepted() {
        require_success(true, "test command").unwrap();
    }

    #[test]
    fn inventory_requires_every_expected_file_and_rejects_stale_files() {
        let output = temp_inventory();
        fs::write(output.join("present.md"), "# present").unwrap();
        fs::write(output.join("stale.md"), "# stale").unwrap();
        let expected = BTreeSet::from(["present.md".to_string(), "missing.md".to_string()]);

        let error = validate_inventory(&expected, &output).unwrap_err();
        assert!(error.to_string().contains("missing.md"));
        assert!(error.to_string().contains("stale.md"));
        fs::remove_dir_all(output).unwrap();
    }

    #[test]
    fn inventory_rejects_duplicate_public_crate_filenames() {
        let error = api_file_names(["tensor4all-a", "tensor4all_a"]).unwrap_err();
        assert!(error.to_string().contains("tensor4all_a.md"));
    }
}
