use anyhow::{bail, Result};

mod audit;

use audit::audit;
use std::env;
use std::path::PathBuf;

struct Args {
    root: Option<PathBuf>,
    baseline: Option<PathBuf>,
}

fn parse_args() -> Result<Option<Args>> {
    let mut root = None;
    let mut baseline = None;
    let mut arguments = env::args().skip(1);
    while let Some(argument) = arguments.next() {
        match argument.as_str() {
            "--root" => {
                root = Some(
                    arguments
                        .next()
                        .ok_or_else(|| anyhow::anyhow!("--root requires a path"))?
                        .into(),
                );
            }
            "--baseline" => {
                baseline = Some(
                    arguments
                        .next()
                        .ok_or_else(|| anyhow::anyhow!("--baseline requires a path"))?
                        .into(),
                );
            }
            "--help" | "-h" => return Ok(None),
            other => bail!("unknown argument: {other}"),
        }
    }
    Ok(Some(Args { root, baseline }))
}

fn print_help() {
    println!(
        "Usage: library-panic-audit [--root PATH] [--baseline PATH]\n\n\
         Audit production Rust sources for panic-style paths."
    );
}

fn main() {
    let result = (|| -> Result<i32> {
        let Some(args) = parse_args()? else {
            print_help();
            return Ok(0);
        };
        let root = args.root.unwrap_or(env::current_dir()?);
        let baseline = args
            .baseline
            .unwrap_or_else(|| root.join("scripts/library-panics-baseline.json"));
        let report = audit(&root, &baseline)?;
        Ok(report.exit_code_and_print())
    })();

    match result {
        Ok(code) => std::process::exit(code),
        Err(error) => {
            eprintln!("library panic audit configuration error: {error}");
            std::process::exit(2);
        }
    }
}
