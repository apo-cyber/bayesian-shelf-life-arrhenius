"""CLI: python -m paper_a.datagen --n-replicates 1000 [--core | --robustness | --all]
       python -m paper_a.datagen --sample core_041 --n-replicates 10
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import CORE_SCENARIOS, ROBUSTNESS_SCENARIOS
from .generate import generate_case, generate_layer, run_full_generation


def _find_scenario(case_id: str):
    for s in CORE_SCENARIOS + ROBUSTNESS_SCENARIOS:
        if s["case_id"] == case_id:
            return s
    return None


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="論文 A 合成データ生成器")
    p.add_argument("--n-replicates", type=int, default=1000)
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data",
        help="出力ルート (デフォルト: paper_a/data)",
    )
    p.add_argument("--core", action="store_true", help="中核 81 のみ生成")
    p.add_argument("--robustness", action="store_true", help="頑健性 20 のみ生成")
    p.add_argument("--all", action="store_true", help="中核 + 頑健性 (デフォルト)")
    p.add_argument(
        "--sample",
        type=str,
        default=None,
        help="単一 case_id だけ生成して stdout に概要 JSON を出す",
    )
    args = p.parse_args(argv)

    if args.sample:
        scenario = _find_scenario(args.sample)
        if scenario is None:
            print(f"未知の case_id: {args.sample}", file=sys.stderr)
            return 2
        result = generate_case(scenario, n_replicates=args.n_replicates)
        sample_summary = {
            "case_id": scenario["case_id"],
            "scenario": scenario,
            "truth": result["truth"],
            "n_rows": len(result["rows"]),
            "first_5_rows": result["rows"][:5],
            "last_2_rows": result["rows"][-2:],
        }
        print(json.dumps(sample_summary, indent=2, ensure_ascii=False))
        return 0

    do_core = args.core or args.all or (not args.core and not args.robustness)
    do_robust = args.robustness or args.all or (not args.core and not args.robustness)

    if args.core and not args.robustness:
        do_robust = False
    if args.robustness and not args.core:
        do_core = False

    summary = run_full_generation(
        root=args.out_dir,
        n_replicates=args.n_replicates,
        include_core=do_core,
        include_robustness=do_robust,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
