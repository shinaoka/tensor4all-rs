#!/usr/bin/env python3
"""Paired release experiment and dimension-adjusted node report for issue #732.

Requires numpy for least-squares fits. No timing threshold is implicit: first
run an A/A noise study, then declare a non-regression bound for a new full run.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import statistics
import subprocess
import time

import numpy as np

THREADS = ("RAYON_NUM_THREADS", "BLAS_NUM_THREADS", "OMP_NUM_THREADS",
           "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "TENFERRO_NUM_THREADS")


def cases(full):
    for scalar in ("f64", "c64"):
        for d in ((2, 3) if full else (2,)):
            for profile in range(4):
                for degree in (2, 3, 4):
                    yield (degree, profile, d, scalar, "aci", 1)
                    for batch in ((2, 8, 32, 128) if full else (8, 32)):
                        for mode in ("cold", "warm"):
                            yield (degree, profile, d, scalar, mode, batch)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def relative_mad(values):
    median = statistics.median(values)
    return statistics.median(abs(x - median) for x in values) / median


def interval(values):
    rng = random.Random(732)
    medians = sorted(statistics.median(rng.choices(values, k=len(values))) for _ in range(2000))
    return [medians[50], medians[1949]]


def frequency(cpu):
    path = Path(f"/sys/devices/system/cpu/cpu{cpu}/cpufreq/scaling_cur_freq")
    return int(path.read_text()) if path.exists() else None


def observe(binary, case, repeats, cpu, expected_commit):
    env = os.environ.copy()
    env.update({key: "1" for key in THREADS})
    # Affinity is inherited by the process and provider-created threads.
    load_before = os.getloadavg()
    freq_before = frequency(cpu)
    started = time.monotonic()
    process = subprocess.run([str(binary), *map(str, case), str(repeats)],
                             env=env, capture_output=True, text=True, timeout=120)
    elapsed = time.monotonic() - started
    if process.returncode:
        return {"error": process.stderr, "exit": process.returncode}
    result = json.loads(process.stdout)
    config = result["config"]
    expected = dict(zip(("degree", "profile", "d", "scalar", "mode", "batch"), case))
    if any(config.get(key) != value for key, value in expected.items()):
        raise ValueError(f"case configuration mismatch: {config}")
    if not config["release"] or config["seed"] != 732 or config["tolerance"] != 1e-8:
        raise ValueError(f"build or numerical protocol mismatch: {config}")
    if config["build_commit"] != expected_commit:
        raise ValueError(f"commit mismatch: {config['build_commit']} != {expected_commit}")
    if result["max_relative_error"] > (1e-7 if case[4] == "aci" else 1e-11):
        raise ValueError("correctness gate failed")
    result["host"] = {"load_before": load_before, "load_after": os.getloadavg(),
                      "frequency_before": freq_before, "frequency_after": frequency(cpu),
                      "process_seconds": elapsed}
    return result


def fit_nodes(records):
    # Keep scalar type, d, phase, and requested batch/cache state separate. A
    # degree dummy coefficient is the residual after accounting for actual S.
    # Fit hub rows only: fused leaves intentionally have different physical d.
    groups = {}
    for record in records:
        if "error" in record:
            continue
        config = record["config"]
        for row in record["nodes"]:
            if not row["node"].endswith(":0"):
                continue
            for phase in ("guard", "frame"):
                points = row[f"{phase}_batches"]["points"]
                if not points or row["local_elements"] is None:
                    continue
                key = (config["scalar"], config["d"], config["mode"], config["batch"],
                       phase, row["node"].rsplit(":", 1)[0])
                groups.setdefault(key, []).append((row["local_elements"], row["degree"],
                    row[f"{phase}_ns"] / points, row[f"{phase}_misses"] / points,
                    points / row[f"{phase}_batches"]["calls"]))
    fits = []
    for key, rows in sorted(groups.items()):
        data = np.asarray(rows)
        if set(data[:, 1]) != {2, 3, 4}:
            continue
        # Unit-normalize S for conditioning; report coefficients in ns/element.
        scale = max(data[:, 0])
        design = np.column_stack((np.ones(len(data)), data[:, 0] / scale,
                                  data[:, 1] == 3, data[:, 1] == 4))
        coefficients, _, rank, _ = np.linalg.lstsq(design, data[:, 2], rcond=None)
        if rank != 4:
            continue
        residual = data[:, 2] - design @ coefficients
        # Bootstrap whole experiment records is handled by the paired wall-time
        # summary. This fit is descriptive; repeated node rows are correlated.
        fits.append({"stratum":key,"model":"ns/point = intercept + beta*S + gamma3*[z=3] + gamma4*[z=4]",
                     "intercept_ns":float(coefficients[0]), "beta_ns_per_element":float(coefficients[1]/scale),
                     "degree3_residual_ns":float(coefficients[2]),
                     "degree4_residual_ns":float(coefficients[3]),
                     "rms_residual_ns":float(np.sqrt(np.mean(residual**2))),
                     "rows":len(rows),"miss_fraction_range":[float(min(data[:,3])),float(max(data[:,3]))],
                     "mean_batch_range":[float(min(data[:,4])),float(max(data[:,4]))],
                     "observations":[{"S":int(r[0]),"z":int(r[1]),"ns_per_point":float(r[2]),
                         "residual_ns":float(e)} for r,e in zip(data,residual)]})
    return fits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--baseline-commit", required=True)
    parser.add_argument("--candidate-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--pairs", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--cpu", type=int, default=2)
    parser.add_argument("--max-relative-mad", type=float, default=0.20)
    parser.add_argument("--max-load-per-cpu", type=float, default=1.5)
    parser.add_argument("--max-frequency-ratio", type=float, default=1.5)
    parser.add_argument("--max-regression", type=float)
    args = parser.parse_args()
    if args.pairs < 3 or args.repeats < 1 or args.cpu not in os.sched_getaffinity(0):
        parser.error("need >=3 pairs, positive repetitions and an available CPU")
    if (not 1 <= args.repeats <= 1000 or args.max_relative_mad <= 0
            or args.max_load_per_cpu <= 0 or args.max_frequency_ratio < 1
            or (args.max_regression is not None and args.max_regression < 0)):
        parser.error("invalid repetitions or validity/regression bounds")
    if args.output.exists():
        parser.error("output exists; choose a new path to preserve prior evidence")
    os.sched_setaffinity(0, {args.cpu})
    case_list = list(cases(args.full))
    repo = Path(__file__).resolve().parent.parent
    report = {"protocol": {**vars(args), "case_list":case_list, "threads":dict.fromkeys(THREADS,1),
              "started_utc":datetime.now(timezone.utc).isoformat(),
              "fixture_sha256":digest(repo / "benchmarks/rust/benchmark_treeaci_branch_cost.rs"),
              "lockfile_sha256":digest(repo / "Cargo.lock"),
              "rustc":subprocess.check_output(["rustc", "-Vv"], text=True),
              "affinity":sorted(os.sched_getaffinity(0)), "baseline_sha256":digest(args.baseline),
              "candidate_sha256":digest(args.candidate),
              "hardware":subprocess.check_output(["lscpu"],text=True)},
              "runs":[], "summary":[], "validity_failures":[]}
    # Write protocol before the first sample: its thresholds and case list
    # cannot be changed in response to candidate results.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, default=str))
    for pair in range(args.pairs):
        for case in case_list:
            pair_runs = {}
            for side in (("baseline", "candidate") if pair % 2 == 0 else ("candidate", "baseline")):
                try:
                    pair_runs[side] = observe(getattr(args,side).resolve(), case, args.repeats,
                                             args.cpu, getattr(args,f"{side}_commit"))
                except (ValueError, OSError, subprocess.TimeoutExpired) as error:
                    pair_runs[side] = {"error":str(error)}
            report["runs"].append({"pair":pair,"case":case,**pair_runs})
        print(f"completed pair {pair+1}/{args.pairs} ({len(case_list)} cases)", flush=True)
        args.output.write_text(json.dumps(report, default=str))
    for case in case_list:
        runs = [run for run in report["runs"] if run["case"] == case]
        bad = [run for run in runs if any("error" in run[side] for side in ("baseline","candidate"))]
        if bad:
            report["validity_failures"].append({"case":case,"reason":"failed execution", "runs":bad})
            continue
        medians = {side:[statistics.median(r[side]["times_ns"]) for r in runs] for side in ("baseline","candidate")}
        ratios = [b/a for a,b in zip(medians["baseline"],medians["candidate"])]
        summary = {"case":case,"baseline_ns":statistics.median(medians["baseline"]),
                   "candidate_ns":statistics.median(medians["candidate"]),
                   "ratio":statistics.median(ratios),"ratio_ci95":interval(ratios),
                   "relative_mad":{side:relative_mad(values) for side,values in medians.items()}}
        report["summary"].append(summary)
        if max(summary["relative_mad"].values()) > args.max_relative_mad:
            report["validity_failures"].append({"case":case,"reason":"dispersion"})
        for run in runs:
            for side in ("baseline","candidate"):
                host = run[side]["host"]
                if max(host["load_before"][0],host["load_after"][0]) / os.cpu_count() > args.max_load_per_cpu:
                    report["validity_failures"].append({"case":case,"reason":"host load"})
                frequencies = [host[k] for k in ("frequency_before","frequency_after") if host[k]]
                if frequencies and max(frequencies)/min(frequencies) > args.max_frequency_ratio:
                    report["validity_failures"].append({"case":case,"reason":"frequency drift"})
    report["fits"] = {side:fit_nodes([r[side] for r in report["runs"]]) for side in ("baseline","candidate")}
    report["verdict"] = "INCONCLUSIVE" if report["validity_failures"] else "DESCRIPTIVE"
    if args.max_regression is not None and not report["validity_failures"]:
        report["verdict"] = "PASS" if all(s["ratio_ci95"][1] <= 1+args.max_regression for s in report["summary"]) else "FAIL"
    args.output.write_text(json.dumps(report, default=str))
    print(f"{report['verdict']}: {len(report['summary'])} cases; {len(report['validity_failures'])} validity failures; {args.output}")
    return int(report["verdict"] in ("FAIL","INCONCLUSIVE"))


if __name__ == "__main__":
    raise SystemExit(main())
