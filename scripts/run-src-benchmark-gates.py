#!/usr/bin/env python3
"""Run reproducible SRC correctness and paired performance gates."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import resource
import signal
import statistics
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class Case:
    name: str
    n_sites: int
    bond: int
    mode: str
    algorithm: str
    max_rank: int = 32
    rank_increment: int = 3


def cases(suite: str) -> list[Case]:
    quick = [
        Case("chain-b32-fixed", 10, 32, "mpo-mps", "src-fixed"),
        Case("chain-b32-adaptive", 10, 32, "mpo-mps", "src-adaptive"),
        Case("tree-n7-b4-fixed", 7, 4, "tree", "src-fixed"),
        Case("tree-n7-b4-adaptive", 7, 4, "tree", "src-adaptive"),
    ]
    if suite == "quick":
        return quick
    full = [
        Case(f"chain-b{bond}-{kind}", 10, bond, "mpo-mps", f"src-{kind}")
        for bond in (4, 8, 16, 32, 64, 128)
        for kind in ("fixed", "adaptive")
    ]
    full.extend(
        Case(f"tree-n{nodes}-b{bond}-{kind}", nodes, bond, "tree", f"src-{kind}")
        for nodes in (3, 7, 10)
        for bond in (2, 4, 8)
        for kind in ("fixed", "adaptive")
    )
    return full


def parse_records(output: str) -> dict[str, list[dict[str, str]]]:
    records: dict[str, list[dict[str, str]]] = {}
    for line in output.splitlines():
        fields = dict(token.split("=", 1) for token in line.split() if "=" in token)
        kind = fields.pop("record", None)
        if kind is not None:
            records.setdefault(kind, []).append(fields)
    return records


def relative_mad(values: list[float]) -> float:
    median = statistics.median(values)
    if median == 0:
        return 0.0 if all(value == 0 for value in values) else float("inf")
    return statistics.median(abs(value - median) for value in values) / median


def bootstrap_median_ci(values: list[float], seed: int, samples: int = 10_000) -> tuple[float, float]:
    rng = random.Random(seed)
    medians = sorted(
        statistics.median(rng.choices(values, k=len(values))) for _ in range(samples)
    )
    return medians[int(0.025 * samples)], medians[int(0.975 * samples)]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_once(
    binary: Path,
    expected_commit: str | None,
    case: Case,
    reps: int,
    timeout: float,
    max_virtual_bytes: int,
    max_rss_kib: int,
    correctness_tol: float,
    expected_backend: str,
    expected_features: str,
) -> dict[str, object]:
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("T4A_BENCH_") and not key.startswith("T4A_PROFILE_")
    }
    env.update(
        {
            "RAYON_NUM_THREADS": "1",
            "BLAS_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "T4A_BENCH_ALGORITHM": case.algorithm,
        }
    )
    command = [
        str(binary),
        str(case.n_sites),
        str(case.bond),
        str(reps),
        case.mode,
        str(case.rank_increment),
        "false",
        str(case.max_rank),
    ]

    def limit_address_space() -> None:
        resource.setrlimit(resource.RLIMIT_AS, (max_virtual_bytes, max_virtual_bytes))

    gnu_time = Path("/usr/bin/time")
    if not gnu_time.is_file():
        raise OSError("GNU /usr/bin/time is required for per-child peak RSS")
    with tempfile.NamedTemporaryFile() as rss_file:
        timed = [str(gnu_time), "-f", "%M", "-o", rss_file.name, *command]
        process = subprocess.Popen(
            timed,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
            preexec_fn=limit_address_space,
        )
        try:
            stdout, stderr = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired as error:
            os.killpg(process.pid, signal.SIGKILL)
            stdout, stderr = process.communicate()
            raise subprocess.TimeoutExpired(
                command,
                timeout,
                output=stdout,
                stderr=stderr,
            ) from error
        rss_file.seek(0)
        rss_text = rss_file.read().decode().strip()
    if process.returncode != 0:
        raise RuntimeError(f"exit={process.returncode}; stderr={stderr[-2000:]}")
    records = parse_records(stdout)
    required_records = ("build", "config", "preflight", "case")
    if any(len(records.get(kind, [])) != 1 for kind in required_records):
        raise RuntimeError(
            "benchmark output lacks exactly one build, config, preflight, and case record"
        )
    build = records["build"][0]
    config = records["config"][0]
    preflight = records["preflight"][0]
    result = records["case"][0]
    if build.get("profile") != "release":
        raise RuntimeError(f"expected release profile, got {build.get('profile')}")
    if expected_commit is not None and build.get("git_commit") != expected_commit:
        raise RuntimeError(
            f"expected commit {expected_commit}, got {build.get('git_commit')}"
        )
    if build.get("backend") != expected_backend:
        raise RuntimeError(
            f"expected backend {expected_backend}, got {build.get('backend')}"
        )
    if build.get("features") != expected_features:
        raise RuntimeError(
            f"expected features {expected_features}, got {build.get('features')}"
        )

    expected_config = {
        "n_sites": str(case.n_sites),
        "physical_dim": "2",
        "mpo_bond": str(case.bond),
        "mps_bond": str(case.bond),
        "requested_max_rank": str(case.max_rank),
        "reps": str(reps),
        "mode": case.mode,
        "algorithm": case.algorithm,
        "rank_increment": str(case.rank_increment),
        "final_svd": "false",
        "src_seed": "1234",
        "adaptive_rtol": "1e-4",
        "adaptive_atol": "0",
        "adaptive_min_rank": "2",
    }
    if any(config.get(key) != value for key, value in expected_config.items()):
        raise RuntimeError(f"benchmark config mismatch: expected {expected_config}, got {config}")
    expected_network_seed = "7" if case.mode == "mpo-mps" else "13"
    if preflight.get("mode") != case.mode or preflight.get("network_seed") != expected_network_seed:
        raise RuntimeError("benchmark preflight mode or network seed mismatch")
    expected_name = (
        f"mpo-mps/{case.algorithm}"
        if case.mode == "mpo-mps"
        else f"tree-mpo-mpo/{case.algorithm}"
    )
    expected_result = {
        "name": expected_name,
        "reps": str(reps),
        "requested_max_rank": str(case.max_rank),
        "src_seed": "1234",
        "center": "S0" if case.mode == "mpo-mps" else "N0000",
    }
    if any(result.get(key) != value for key, value in expected_result.items()):
        raise RuntimeError(f"benchmark case mismatch: expected {expected_result}, got {result}")

    seconds = float(result["per_run_seconds"])
    elapsed = float(result["elapsed_seconds"])
    relative_error = float(result["relative_error"])
    effective_max_bond = int(result["effective_max_bond"])
    if not math.isfinite(seconds) or seconds <= 0 or not math.isfinite(elapsed) or elapsed <= 0:
        raise RuntimeError("benchmark timing must be finite and positive")
    if (
        not math.isfinite(relative_error)
        or relative_error < 0
        or relative_error > correctness_tol
    ):
        raise ValueError(
            f"relative error {relative_error} is outside [0, {correctness_tol}]"
        )
    if not 1 <= effective_max_bond <= case.max_rank:
        raise RuntimeError("effective max bond is outside the declared rank cap")
    rss_kib = int(rss_text)
    if rss_kib > max_rss_kib:
        raise MemoryError(f"peak RSS {rss_kib} KiB exceeds {max_rss_kib} KiB")
    return {
        "seconds": seconds,
        "relative_error": relative_error,
        "peak_rss_kib": rss_kib,
        "stdout": stdout,
        "stderr": stderr,
        "build": build,
        "result": result,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--candidate-commit", required=True)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--baseline-commit")
    parser.add_argument("--suite", choices=("quick", "full"), default="quick")
    parser.add_argument("--pairs", type=int)
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--correctness-tol", type=float, default=1.0e-8)
    parser.add_argument("--expected-backend", default="tenferro")
    parser.add_argument("--expected-features", default="tenferro-cpu-faer")
    parser.add_argument("--required-improvement-percent", type=float, default=0.0)
    parser.add_argument("--allowed-regression-percent", type=float, default=5.0)
    parser.add_argument("--max-dispersion-percent", type=float, default=10.0)
    parser.add_argument("--primary", action="append", default=[])
    parser.add_argument("--timeout-seconds", type=float, default=60.0)
    parser.add_argument("--max-rss-mib", type=int, default=4096)
    parser.add_argument("--max-virtual-mib", type=int, default=8192)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if not args.candidate.is_file():
        parser.error("--candidate must name an existing prebuilt binary")
    if args.baseline is not None and not args.baseline.is_file():
        parser.error("--baseline must name an existing prebuilt binary")
    if args.baseline is None and args.baseline_commit is not None:
        parser.error("--baseline-commit requires --baseline")
    if args.baseline is not None and args.baseline_commit is None:
        parser.error("paired runs require --baseline-commit")
    pairs = (10 if args.suite == "full" else 5) if args.pairs is None else args.pairs
    if pairs < 1 or args.reps < 1:
        parser.error("pairs and reps must be positive")
    minimum_pairs = 10 if args.suite == "full" else 5
    if args.baseline is not None and pairs < minimum_pairs:
        parser.error(
            f"paired {args.suite} runs require at least {minimum_pairs} pairs"
        )
    if not math.isfinite(args.correctness_tol) or args.correctness_tol < 0:
        parser.error("--correctness-tol must be finite and non-negative")
    if (
        not math.isfinite(args.required_improvement_percent)
        or not 0 <= args.required_improvement_percent < 100
    ):
        parser.error("--required-improvement-percent must be finite and in [0, 100)")
    if (
        not math.isfinite(args.allowed_regression_percent)
        or args.allowed_regression_percent < 0
    ):
        parser.error("--allowed-regression-percent must be finite and non-negative")
    if not math.isfinite(args.max_dispersion_percent) or args.max_dispersion_percent <= 0:
        parser.error("--max-dispersion-percent must be finite and positive")
    if (
        not math.isfinite(args.timeout_seconds)
        or args.timeout_seconds <= 0
        or args.max_rss_mib <= 0
        or args.max_virtual_mib <= 0
    ):
        parser.error("timeout and memory limits must be positive")

    selected = cases(args.suite)
    unknown_primary = set(args.primary) - {case.name for case in selected}
    if unknown_primary:
        parser.error(f"unknown primary cases: {sorted(unknown_primary)}")
    report: dict[str, object] = {
        "protocol": vars(args)
        | {
            "candidate": str(args.candidate),
            "baseline": str(args.baseline) if args.baseline else None,
            "output": str(args.output),
            "pairs": pairs,
        },
        "loadavg_before": os.getloadavg(),
        "binaries": {"candidate": {"path": str(args.candidate), "sha256": sha256(args.candidate)}},
        "cases": {},
    }
    if args.baseline:
        report["binaries"]["baseline"] = {"path": str(args.baseline), "sha256": sha256(args.baseline)}

    baseline_errors: list[str] = []
    candidate_errors: list[str] = []
    correctness_errors: list[str] = []
    shared_errors: list[str] = []
    if not Path("/usr/bin/time").is_file():
        shared_errors.append("GNU /usr/bin/time is required for per-child peak RSS")
    for case_index, case in enumerate([] if shared_errors else selected):
        case_report: dict[str, object] = {"config": asdict(case), "baseline": [], "candidate": []}
        report["cases"][case.name] = case_report
        for pair in range(pairs):
            order = ["candidate"] if args.baseline is None else (["baseline", "candidate"] if pair % 2 == 0 else ["candidate", "baseline"])
            for kind in order:
                binary = args.candidate if kind == "candidate" else args.baseline
                commit = args.candidate_commit if kind == "candidate" else args.baseline_commit
                try:
                    result = run_once(
                        binary,
                        commit,
                        case,
                        args.reps,
                        args.timeout_seconds,
                        args.max_virtual_mib << 20,
                        args.max_rss_mib << 10,
                        args.correctness_tol,
                        args.expected_backend,
                        args.expected_features,
                    )
                    case_report[kind].append(result)
                except OSError as error:
                    message = f"{case.name} pair={pair} {kind} launch: {error}"
                    (baseline_errors if kind == "baseline" else candidate_errors).append(message)
                    case_report[kind].append({"error": message})
                except ValueError as error:
                    message = f"{case.name} pair={pair} {kind} correctness: {error}"
                    correctness_errors.append(message)
                    case_report[kind].append({"error": message})
                except (subprocess.TimeoutExpired, RuntimeError, MemoryError) as error:
                    message = f"{case.name} pair={pair} {kind}: {error}"
                    (baseline_errors if kind == "baseline" else candidate_errors).append(message)
                    case_report[kind].append({"error": message})

    inconclusive_reasons = [*shared_errors, *baseline_errors]
    fail_reasons = [*correctness_errors, *candidate_errors]
    if not inconclusive_reasons and not fail_reasons and args.baseline:
        max_dispersion = args.max_dispersion_percent / 100.0
        for case_index, case in enumerate(selected):
            case_report = report["cases"][case.name]
            baseline_times = [run["seconds"] for run in case_report["baseline"]]
            candidate_times = [run["seconds"] for run in case_report["candidate"]]
            ratios = [
                candidate / baseline
                for baseline, candidate in zip(baseline_times, candidate_times)
            ]
            ratio_median = statistics.median(ratios)
            ratio_ci = bootstrap_median_ci(ratios, seed=706_000 + case_index)
            stats = {
                "baseline_median_seconds": statistics.median(baseline_times),
                "candidate_median_seconds": statistics.median(candidate_times),
                "baseline_relative_mad": relative_mad(baseline_times),
                "candidate_relative_mad": relative_mad(candidate_times),
                "paired_ratio_median": ratio_median,
                "paired_ratio_relative_mad": relative_mad(ratios),
                "paired_ratio_ci95": ratio_ci,
            }
            case_report["statistics"] = stats
            if stats["baseline_relative_mad"] > max_dispersion:
                inconclusive_reasons.append(
                    f"{case.name}: baseline dispersion exceeds limit"
                )
                continue
            if (
                stats["candidate_relative_mad"] > max_dispersion
                or stats["paired_ratio_relative_mad"] > max_dispersion
            ):
                fail_reasons.append(
                    f"{case.name}: candidate/paired dispersion exceeds limit"
                )
                continue
            threshold = (
                1.0 - args.required_improvement_percent / 100.0
                if case.name in args.primary
                else 1.0 + args.allowed_regression_percent / 100.0
            )
            if ratio_ci[1] > threshold:
                fail_reasons.append(
                    f"{case.name}: CI upper {ratio_ci[1]:.6f} exceeds {threshold:.6f}"
                )

    if fail_reasons:
        status = "FAIL"
    elif inconclusive_reasons:
        status = "INCONCLUSIVE"
    else:
        status = "PASS"
    reasons = [*inconclusive_reasons, *fail_reasons]
    report["loadavg_after"] = os.getloadavg()
    report["status"] = status
    report["reasons"] = reasons
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"SRC benchmark gate: {status}; report={args.output}")
    for reason in reasons:
        print(reason)
    return {"PASS": 0, "FAIL": 1, "INCONCLUSIVE": 2}[status]


if __name__ == "__main__":
    raise SystemExit(main())
