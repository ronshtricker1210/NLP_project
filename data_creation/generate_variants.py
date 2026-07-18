#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Generate the controlled real-word-ratio typo dataset matrix
(datasets x ratios; see typo_pipeline.py for the algorithm).

    python generate_variants.py                                        # full matrix
    python generate_variants.py --datasets math500 --subset 40 --ratios 0 0.5 1.0
    python generate_variants.py --push --namespace <hf-username>

Variants are saved under <out>/<dataset>/<variant>/ and optionally pushed as
configs of <namespace>/<dataset>-typos (GPQA always private):

    load_dataset("<namespace>/math500-typos", "real70", split="test")
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional

from typo_pipeline import Config, load_math_subset, process_row

REPO_ROOT = Path(__file__).resolve().parent.parent
# Optional local copies made with save_to_disk (used instead of the Hub when
# present, so generation is fully offline and deterministic).
LOCAL_DATA_ROOT = REPO_ROOT.parent / "datasets"

PRESETS: Dict[str, Dict[str, Any]] = {
    "math500": {
        "dataset_name": "HuggingFaceH4/MATH-500",
        "dataset_config_name": None,
        "dataset_split": "test",
        "text_field": "problem",
        "local_dir": "math500_test",
        "hub_repo": "math500-typos",
        "private": False,
        "keep_columns": None,  # keep everything
    },
    "gsm8k": {
        "dataset_name": "openai/gsm8k",
        "dataset_config_name": "main",
        "dataset_split": "test",
        "text_field": "question",
        "local_dir": "gsm8k_test",
        "hub_repo": "gsm8k-typos",
        "private": False,
        "keep_columns": None,
    },
    "gpqa": {
        "dataset_name": "Idavidrein/gpqa",
        "dataset_config_name": "gpqa_diamond",
        "dataset_split": "train",
        "text_field": "Question",
        "local_dir": "gpqa_diamond",
        "hub_repo": "gpqa-typos",
        "private": True,  # gated source; keep answers off the crawlable web
        "keep_columns": [
            "Question",
            "Correct Answer",
            "Incorrect Answer 1",
            "Incorrect Answer 2",
            "Incorrect Answer 3",
            "Explanation",
            "Subdomain",
            "High-level domain",
            "Record ID",
            "Canary String",
        ],
    },
}

DEFAULT_RATIOS = [round(i / 10, 1) for i in range(8)]  # real0 .. real70


def variant_name(ratio: float) -> str:
    return f"real{int(round(ratio * 100))}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate controlled real-word-ratio typo datasets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--datasets", nargs="+", choices=sorted(PRESETS),
                        default=["math500", "gsm8k", "gpqa"],
                        help="which source datasets to process")
    parser.add_argument("--ratios", nargs="+", type=float, default=DEFAULT_RATIOS,
                        help="target real-word ratios in [0, 1]")
    parser.add_argument("--typo-rate", type=float, default=0.30,
                        help="fraction of eligible words to corrupt")
    parser.add_argument("--two-edit-prob", type=float, default=0.10,
                        help="probability a typo consists of two edits")
    parser.add_argument("--subset", type=int, default=None,
                        help="limit rows per dataset (default: all rows)")
    parser.add_argument("--seed", type=int, default=42,
                        help="base seed (same across variants -> paired rows)")
    parser.add_argument("--num-proc", type=int, default=1,
                        help="parallel workers for datasets.map")
    parser.add_argument("--out", type=Path,
                        default=Path(__file__).resolve().parent / "typo_variants",
                        help="local output root")
    parser.add_argument("--push", action="store_true",
                        help="push each variant to the Hugging Face Hub")
    parser.add_argument("--namespace", type=str, default=None,
                        help="HF username/org to push to (required with --push)")
    parser.add_argument("--private", action="store_true",
                        help="push ALL repos as private (GPQA is always private)")
    args = parser.parse_args()

    for r in args.ratios:
        if not 0.0 <= r <= 1.0:
            parser.error(f"ratio {r} outside [0, 1]")
    if args.push and not args.namespace:
        parser.error("--push requires --namespace <hf-username>")
    return args


def build_config(preset: Dict[str, Any], ratio: float, args: argparse.Namespace) -> Config:
    local_path = LOCAL_DATA_ROOT / preset["local_dir"]
    return Config(
        dataset_name=preset["dataset_name"],
        dataset_config_name=preset["dataset_config_name"],
        dataset_split=preset["dataset_split"],
        text_field=preset["text_field"],
        subset_size=args.subset if args.subset else 10**9,
        local_data_path=local_path if local_path.exists() else None,
        allow_offline_fallback=False,
        typo_rate=args.typo_rate,
        target_real_fraction=ratio,
        two_edit_prob=args.two_edit_prob,
        num_proc=args.num_proc,
        seed=args.seed,
    )


def variant_stats(processed) -> Dict[str, Any]:
    import numpy as np

    ratios = np.array(processed["real_ratio"], dtype=float)
    totals = np.array(processed["num_total"])
    edit_counts = [c for row in processed["typo_edit_counts"] for c in row]
    techniques = [t.split("+")[0] for row in processed["typo_techniques"] for t in row]
    tech_counts: Dict[str, int] = {}
    for t in techniques:
        tech_counts[t] = tech_counts.get(t, 0) + 1
    return {
        "rows": len(processed),
        "mean_ratio": float(np.nanmean(ratios)),
        "mean_typos": float(totals.mean()),
        "min_typos": int(totals.min()),
        "two_edit_pct": 100.0 * sum(1 for c in edit_counts if c == 2) / max(1, len(edit_counts)),
        "tech_counts": tech_counts,
    }


def main() -> None:
    args = parse_args()
    summary: List[Dict[str, Any]] = []

    for preset_key in args.datasets:
        preset = PRESETS[preset_key]
        for ratio in sorted(args.ratios):
            name = variant_name(ratio)
            config = build_config(preset, ratio, args)
            print(f"\n=== {preset_key} / {name} "
                  f"(target={ratio:.2f}, rate={args.typo_rate}) ===")

            dataset = load_math_subset(config)
            keep = preset["keep_columns"]
            if keep:
                dataset = dataset.select_columns(
                    [c for c in keep if c in dataset.column_names]
                )

            map_kwargs = {"num_proc": config.num_proc} if config.num_proc > 1 else {}
            processed = dataset.map(
                process_row,
                with_indices=True,
                fn_kwargs={"config": config},
                desc=f"{preset_key}/{name}",
                **map_kwargs,
            )

            stats = variant_stats(processed)
            print(f"rows={stats['rows']}  target={ratio:.2f}  "
                  f"achieved mean P={stats['mean_ratio']:.3f}  "
                  f"typos/problem={stats['mean_typos']:.1f} (min {stats['min_typos']})  "
                  f"2-edit typos={stats['two_edit_pct']:.1f}%")
            print(f"techniques: {stats['tech_counts']}")

            out_dir = Path(args.out) / preset_key / name
            processed.save_to_disk(str(out_dir))
            print(f"[save] {out_dir}")

            if args.push:
                repo_id = f"{args.namespace}/{preset['hub_repo']}"
                private = preset["private"] or args.private
                processed.push_to_hub(
                    repo_id,
                    config_name=name,
                    split="test",
                    private=private,
                )
                print(f"[push] {repo_id} config={name} private={private}")

            summary.append({"dataset": preset_key, "variant": name,
                            "target": ratio, **stats})

    print("\n" + "=" * 78)
    print(f"{'dataset':<10} {'variant':<9} {'rows':>5} {'target':>7} "
          f"{'mean P':>7} {'typos/row':>10} {'2-edit%':>8}")
    print("-" * 78)
    for s in summary:
        print(f"{s['dataset']:<10} {s['variant']:<9} {s['rows']:>5} "
              f"{s['target']:>7.2f} {s['mean_ratio']:>7.3f} "
              f"{s['mean_typos']:>10.1f} {s['two_edit_pct']:>7.1f}%")
    print("=" * 78)
    print("Done.")


if __name__ == "__main__":
    main()
