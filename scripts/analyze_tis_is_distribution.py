#!/usr/bin/env python3
"""Analyze dumped train/infer IS ratios for TIS upper-bound evidence."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np

QUANTILES = [0, 0.001, 0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 0.999, 1.0]
BINS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 3.0, 5.0, 10.0, math.inf]


def iter_jsonl_files(paths: Iterable[Path]):
    for path in paths:
        if path.is_file() and path.suffix == ".jsonl":
            yield path
        elif path.is_dir():
            yield from sorted(path.rglob("*.jsonl"))


def read_records(paths: list[Path]):
    for file_path in iter_jsonl_files(paths):
        with file_path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line), file_path, line_no
                except json.JSONDecodeError as exc:
                    print(f"[warn] skip invalid json: {file_path}:{line_no}: {exc}")


def safe_array(values):
    values = [v for v in values if v is not None]
    if not values:
        return np.asarray([], dtype=np.float64)
    arr = np.asarray(values, dtype=np.float64)
    return arr[np.isfinite(arr)]


def summarize(values, upper_bound: float):
    arr = safe_array(values)
    row = {"n": int(arr.size)}
    if arr.size == 0:
        for q in QUANTILES:
            row[f"p{q * 100:g}"] = ""
        for i in range(len(BINS) - 1):
            row[f"bin_{BINS[i]}_{BINS[i + 1]}"] = 0
        row.update({"mean": "", "std": "", "frac_le_upper": "", "frac_gt_upper": "", "max_over_upper": ""})
        return row

    row.update(
        {
            "mean": float(arr.mean()),
            "std": float(arr.std(ddof=0)),
            "frac_le_upper": float((arr <= upper_bound).mean()),
            "frac_gt_upper": float((arr > upper_bound).mean()),
            "max_over_upper": float(arr.max() / upper_bound) if upper_bound else "",
        }
    )
    qs = np.quantile(arr, QUANTILES)
    for q, value in zip(QUANTILES, qs):
        row[f"p{q * 100:g}"] = float(value)
    for i in range(len(BINS) - 1):
        lo, hi = BINS[i], BINS[i + 1]
        if math.isinf(hi):
            count = int((arr >= lo).sum())
        else:
            count = int(((arr >= lo) & (arr < hi)).sum())
        row[f"bin_{lo}_{hi}"] = count
    return row


def fmt_num(value):
    return "NA" if value == "" else f"{float(value):.6g}"


def fmt_pct(value):
    return "NA" if value == "" else f"{100 * float(value):.2f}%"


def main():
    parser = argparse.ArgumentParser(description="Analyze TIS train/infer IS ratio dumps.")
    parser.add_argument("inputs", nargs="+", type=Path, help="Dump directories or JSONL files.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Where to write CSV/Markdown summary.")
    parser.add_argument("--model-name", default=None, help="Label used in the report; defaults to input dirname.")
    parser.add_argument("--upper-bound", type=float, default=2.0, help="TIS upper bound to evaluate.")
    args = parser.parse_args()

    output_dir = args.output_dir or (args.inputs[0] if args.inputs[0].is_dir() else args.inputs[0].parent)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_name = args.model_name or args.inputs[0].resolve().name

    by_step = defaultdict(lambda: {"token": [], "seq_mean_token": [], "seq_geo": [], "seq_product": []})
    total_records = 0
    for rec, file_path, line_no in read_records(args.inputs):
        step = int(rec.get("global_step", -1))
        if step < 0:
            print(f"[warn] record without global_step: {file_path}:{line_no}")
            continue
        total_records += 1
        by_step[step]["seq_mean_token"].append(rec.get("seq_token_mean_ratio", rec.get("train_infer_ratio")))
        by_step[step]["seq_geo"].append(rec.get("seq_geo_ratio"))
        by_step[step]["seq_product"].append(rec.get("seq_ratio"))
        by_step[step]["token"].extend(rec.get("all_token_ratios") or [])

    csv_path = output_dir / f"{model_name}.tis_is_distribution.csv"
    md_path = output_dir / f"{model_name}.tis_is_report.md"

    rows = []
    metric_order = ["token", "seq_mean_token", "seq_geo", "seq_product"]
    for step in sorted(by_step):
        for metric in metric_order:
            row = {"model": model_name, "global_step": step, "metric": metric}
            row.update(summarize(by_step[step][metric], args.upper_bound))
            rows.append(row)

    fieldnames = list(rows[0].keys()) if rows else ["model", "global_step", "metric", "n"]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    aggregate = {}
    for metric in metric_order:
        vals = []
        for step in by_step.values():
            vals.extend(step[metric])
        aggregate[metric] = summarize(vals, args.upper_bound)

    with md_path.open("w", encoding="utf-8") as f:
        f.write(f"# TIS IS distribution report: {model_name}\n\n")
        f.write(f"- input records: {total_records}\n")
        f.write(f"- steps: {len(by_step)}\n")
        f.write(f"- upper_bound: {args.upper_bound}\n")
        f.write(f"- CSV: `{csv_path}`\n\n")
        f.write("## Overall upper-bound coverage\n\n")
        f.write("| metric | n | mean | p50 | p90 | p95 | p99 | p99.9 | max | <=UB | >UB |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for metric in metric_order:
            s = aggregate[metric]
            f.write(
                f"| {metric} | {s['n']} | {fmt_num(s['mean'])} | {fmt_num(s['p50'])} | "
                f"{fmt_num(s['p90'])} | {fmt_num(s['p95'])} | {fmt_num(s['p99'])} | "
                f"{fmt_num(s['p99.9'])} | {fmt_num(s['p100'])} | "
                f"{fmt_pct(s['frac_le_upper'])} | {fmt_pct(s['frac_gt_upper'])} |\n"
            )
        f.write("\n## Per-step concise view (token ratio)\n\n")
        f.write("| step | n | mean | p90 | p95 | p99 | max | <=UB | >UB |\n")
        f.write("|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for step in sorted(by_step):
            s = summarize(by_step[step]["token"], args.upper_bound)
            f.write(
                f"| {step} | {s['n']} | {fmt_num(s['mean'])} | {fmt_num(s['p90'])} | "
                f"{fmt_num(s['p95'])} | {fmt_num(s['p99'])} | {fmt_num(s['p100'])} | "
                f"{fmt_pct(s['frac_le_upper'])} | {fmt_pct(s['frac_gt_upper'])} |\n"
            )

    print(f"wrote {csv_path}")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
