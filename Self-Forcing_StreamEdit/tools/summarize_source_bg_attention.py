#!/usr/bin/env python3
"""Summarize 967h source-background attention diagnostics."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean


SCALARS = (
    "source_value_rms",
    "target_value_rms",
    "source_bg_mass_all",
    "source_bg_mass_fg_query",
    "source_bg_mass_bg_query",
    "output_rms_all",
    "output_rms_fg_query",
    "output_rms_bg_query",
    "foreground_fraction",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("jsonl", type=Path)
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Optional destination for block-level CSV statistics.",
    )
    return parser.parse_args()


def load_rows(path):
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def aggregate(rows, group_names):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[name] for name in group_names)].append(row)
    output = []
    for key, items in sorted(groups.items()):
        summary = dict(zip(group_names, key))
        summary["rows"] = len(items)
        for name in SCALARS:
            summary[name] = mean(float(item[name]) for item in items)
        summary["target_to_source_value_rms"] = (
            summary["target_value_rms"]
            / max(summary["source_value_rms"], 1e-12)
        )
        output.append(summary)
    return output


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    rows = load_rows(args.jsonl)
    if not rows:
        raise SystemExit(f"No diagnostics found in {args.jsonl}")

    by_block = aggregate(rows, ("block",))
    csv_path = args.csv or args.jsonl.with_name(
        f"{args.jsonl.stem}_by_block.csv"
    )
    write_csv(csv_path, by_block)
    by_block_layer = aggregate(rows, ("block", "layer"))
    layer_csv_path = csv_path.with_name(
        f"{csv_path.stem}_layer{csv_path.suffix}"
    )
    write_csv(layer_csv_path, by_block_layer)
    by_block_step = aggregate(rows, ("block", "timestep_index"))
    step_csv_path = csv_path.with_name(
        f"{csv_path.stem}_step{csv_path.suffix}"
    )
    write_csv(step_csv_path, by_block_step)

    print(
        "block rows trg/src_v src_bg_mass fg_mass bg_mass out_rms "
        "out_fg out_bg"
    )
    for row in by_block:
        print(
            f"{int(row['block']):>5} {int(row['rows']):>4} "
            f"{row['target_to_source_value_rms']:.4f} "
            f"{row['source_bg_mass_all']:.4f} "
            f"{row['source_bg_mass_fg_query']:.4f} "
            f"{row['source_bg_mass_bg_query']:.4f} "
            f"{row['output_rms_all']:.4f} "
            f"{row['output_rms_fg_query']:.4f} "
            f"{row['output_rms_bg_query']:.4f}"
        )
    print(f"Wrote {csv_path}")
    print(f"Wrote {layer_csv_path}")
    print(f"Wrote {step_csv_path}")


if __name__ == "__main__":
    main()
