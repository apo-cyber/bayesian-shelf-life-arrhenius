"""Table 3 (温度依存 variant 間の変動) を cell_metrics.json から再集計.

reaggregate.py が cell_metrics.json を書き出したあとに走らせる読み取り専用の
集計。parquet も MCMC 再走行も不要で、robustness_per_case スライスと
robustness 層の truth.json だけを読む。

集計の定義
----------
kinetic model ごとに、robustness 層の 5 case を温度依存 variant
(arrhenius / modified_arrhenius_concave / modified_arrhenius_convex) の 3 群に
分け、群内で per-case 値を **算術平均** して 3 値を得る。variation はその 3 値の

    |max - min| / |3 値の平均| * 100   [%]

として報告する。原稿 Table 3 の 24 値 (3 推定量 × 4 kinetics × 2 指標) は
この定義で完全に再現する (`tests/test_table3_variation.py` で固定)。

★ 再現者向けの注意 — 他の読み方では一致しない
------------------------------------------------
variation の定義は複数の読み方がありうるが、原稿 Table 3 と一致するのは上記
だけである。実測した不一致数 (24 値中):

  - 群内を中央値で代表する            → 22 値が不一致
  - 5 case を 5 variant として扱う     → 24 値が不一致
  - case をまとめず replicate をプール → 22 値が不一致

arrhenius 群だけ 3 case・concave / convex 群は各 1 case という非対称な構成の
ため、「群内平均」と「群内中央値」は arrhenius 群でのみ乖離する。1 群だけ
代表値の定義が変わることで 24 値中 22 値が動く。この検証は
`tests/test_table3_variation.py::test_alternative_readings_do_not_match` に
ある。
"""

from __future__ import annotations

import json
from pathlib import Path

from paper_a.analysis.loaders.synthetic import load_truth

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"

# 温度依存 variant (Section 2.1.2)。順序は出力の再現性のために固定。
K_OF_T_VARIANTS = ("arrhenius", "modified_arrhenius_concave", "modified_arrhenius_convex")

# kinetic model (Section 2.1.1)。
KINETIC_MODELS = ("first_order", "second_order", "autocatalytic", "induction")

# Table 3 の対象は加速データを使う 3 推定量。classical_ich_q1e は 25°C 実測を
# 直接使い加速外挿を経ないため、温度依存 variant への感度を論じる対象ではない。
ESTIMATORS = ("two_stage_conjugate", "mcmc", "classical_ols_multi_temp")

METRICS = ("bias_median", "bias_sd_capped120")


def variation_pct(values: list[float]) -> float:
    """variant 間の |max - min| を平均に対する百分率で返す."""
    mean = sum(values) / len(values)
    return abs(max(values) - min(values)) / abs(mean) * 100.0


def compute_table3(
    per_case_metrics: list[dict],
    truth_by_case: dict[str, dict],
) -> list[dict]:
    """robustness_per_case から Table 3 の行を組み立てる."""
    by_key = {(r["estimator_name"], r["cell_key"]): r for r in per_case_metrics}

    rows: list[dict] = []
    for estimator in ESTIMATORS:
        for kinetics in KINETIC_MODELS:
            row: dict = {"estimator_name": estimator, "kinetics": kinetics}
            for metric in METRICS:
                variant_means: list[float] = []
                for variant in K_OF_T_VARIANTS:
                    case_ids = [
                        c["case_id"]
                        for c in truth_by_case.values()
                        if c["kinetics"] == kinetics and c["k_of_t"] == variant
                    ]
                    values = [
                        by_key[(estimator, cid)][metric]
                        for cid in case_ids
                        if by_key.get((estimator, cid)) is not None
                        and by_key[(estimator, cid)][metric] is not None
                    ]
                    if not values:
                        raise ValueError(
                            f"{estimator} / {kinetics} / {variant}: 有効な {metric} がない"
                        )
                    variant_means.append(sum(values) / len(values))
                row[f"{metric}_variation_pct"] = variation_pct(variant_means)
                row[f"{metric}_by_variant"] = dict(zip(K_OF_T_VARIANTS, variant_means))
            rows.append(row)
    return rows


def main() -> int:
    metrics_path = RESULTS_DIR / "cell_metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"{metrics_path} がない.reaggregate を先に実行.")

    per_case = json.loads(metrics_path.read_text())["robustness_per_case"]
    truth_by_case = load_truth("robustness")

    rows = compute_table3(per_case, truth_by_case)

    out_path = RESULTS_DIR / "table3_variation.json"
    out_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False))
    print(f"→ {out_path}")

    print()
    print("| Estimator | Kinetics | bias_median variation | bias_sd_capped120 variation |")
    print("|---|---|---:|---:|")
    for r in rows:
        print(
            f"| `{r['estimator_name']}` | {r['kinetics'].replace('_', ' ')} "
            f"| {r['bias_median_variation_pct']:.1f}% "
            f"| {r['bias_sd_capped120_variation_pct']:.1f}% |"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
