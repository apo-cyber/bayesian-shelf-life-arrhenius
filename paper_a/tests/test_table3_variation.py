"""table3_variation.py が原稿 Table 3 を再現することを固定するテスト.

Table 3 の 24 値 (3 推定量 × 4 kinetics × 2 指標) を逐語で持ち、コミット済みの
cell_metrics.json から再集計した値と 0.05 %ポイント以内で一致することを要求する
(原稿は小数第 1 位まで表示するため、丸め後に一致すればよい)。

あわせて、variation の他の読み方では Table 3 に一致しないことも固定する。
再現者が「どの読み方が正しいのか」で迷わないようにするためのもの。
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path

import pytest

from paper_a.analysis.loaders.synthetic import load_truth
from paper_a.analysis.table3_variation import (
    ESTIMATORS,
    KINETIC_MODELS,
    K_OF_T_VARIANTS,
    compute_table3,
    variation_pct,
)

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"

# 原稿 Table 3 の逐語値: (estimator, kinetics) -> (bias_median %, bias_sd_capped120 %)
PAPER_TABLE3 = {
    ("two_stage_conjugate", "first_order"): (18.4, 24.4),
    ("two_stage_conjugate", "second_order"): (18.7, 31.2),
    ("two_stage_conjugate", "autocatalytic"): (154.2, 26.5),
    ("two_stage_conjugate", "induction"): (23.4, 38.4),
    ("mcmc", "first_order"): (173.8, 21.4),
    ("mcmc", "second_order"): (245.1, 7.8),
    ("mcmc", "autocatalytic"): (31.9, 73.5),
    ("mcmc", "induction"): (114.6, 40.5),
    ("classical_ols_multi_temp", "first_order"): (205.7, 11.0),
    ("classical_ols_multi_temp", "second_order"): (39.7, 13.5),
    ("classical_ols_multi_temp", "autocatalytic"): (96.8, 38.3),
    ("classical_ols_multi_temp", "induction"): (21.7, 152.7),
}


def _rows() -> dict[tuple[str, str], dict]:
    per_case = json.loads((RESULTS_DIR / "cell_metrics.json").read_text())["robustness_per_case"]
    rows = compute_table3(per_case, load_truth("robustness"))
    return {(r["estimator_name"], r["kinetics"]): r for r in rows}


@pytest.mark.parametrize("key", sorted(PAPER_TABLE3))
def test_matches_paper_table3(key: tuple[str, str]) -> None:
    """採用している読み方が Table 3 の 24 値すべてを再現する."""
    row = _rows()[key]
    expected_bias, expected_sd = PAPER_TABLE3[key]
    assert round(row["bias_median_variation_pct"], 1) == pytest.approx(expected_bias, abs=0.05)
    assert round(row["bias_sd_capped120_variation_pct"], 1) == pytest.approx(expected_sd, abs=0.05)


def test_covers_all_24_values() -> None:
    """行が欠けたまま緑にならないようにする."""
    assert len(_rows()) == len(ESTIMATORS) * len(KINETIC_MODELS) == 12
    assert set(_rows()) == set(PAPER_TABLE3)


def test_alternative_readings_do_not_match() -> None:
    """他の読み方では Table 3 に一致しないことを固定する.

    再現者が variation の定義を取り違えたときに、たまたま近い数字が出て
    「一致した」と誤認するのを防ぐ。実測の不一致数を逐語で持つ。
    """
    per_case = json.loads((RESULTS_DIR / "cell_metrics.json").read_text())["robustness_per_case"]
    by_key = {(r["estimator_name"], r["cell_key"]): r for r in per_case}
    truth_by_case = load_truth("robustness")

    n_mismatch_median = 0  # 群内を中央値で代表
    n_mismatch_five = 0    # 5 case を 5 variant として扱う
    for estimator in ESTIMATORS:
        for kinetics in KINETIC_MODELS:
            cases = [c for c in truth_by_case.values() if c["kinetics"] == kinetics]
            for i, metric in enumerate(("bias_median", "bias_sd_capped120")):
                per_variant_median = [
                    statistics.median(
                        [by_key[(estimator, c["case_id"])][metric] for c in cases if c["k_of_t"] == v]
                    )
                    for v in K_OF_T_VARIANTS
                ]
                all_cases = [by_key[(estimator, c["case_id"])][metric] for c in cases]
                expected = PAPER_TABLE3[(estimator, kinetics)][i]
                if abs(round(variation_pct(per_variant_median), 1) - expected) >= 0.05:
                    n_mismatch_median += 1
                if abs(round(variation_pct(all_cases), 1) - expected) >= 0.05:
                    n_mismatch_five += 1

    assert n_mismatch_median == 22, "群内中央値の読み方の不一致数が変わった"
    assert n_mismatch_five == 24, "5 variant 扱いの読み方の不一致数が変わった"


@pytest.mark.slow
def test_replicate_pooling_does_not_match() -> None:
    """replicate をプールする読み方でも一致しないことを固定する (parquet が要る)."""
    import numpy as np
    import pandas as pd

    parquet_path = RESULTS_DIR / "estimator_results.parquet"
    if not parquet_path.exists():
        pytest.skip("estimator_results.parquet がない")

    df = pd.read_parquet(parquet_path)
    truth_by_case = load_truth("robustness")

    n_mismatch = 0
    for estimator in ESTIMATORS:
        for kinetics in KINETIC_MODELS:
            pooled = {"bias_median": [], "bias_sd_capped120": []}
            for variant in K_OF_T_VARIANTS:
                case_ids = [
                    c["case_id"]
                    for c in truth_by_case.values()
                    if c["kinetics"] == kinetics and c["k_of_t"] == variant
                ]
                sub = df[(df["estimator_name"] == estimator) & (df["case_id"].isin(case_ids))]
                sub = sub[sub["t90_point_estimate_months"].notna()]
                est = sub["t90_point_estimate_months"].astype(float).to_numpy()
                tv = np.array([truth_by_case[c]["t90_true_25c_months"] for c in sub["case_id"]])
                pooled["bias_median"].append(float(np.median(est - tv)))
                pooled["bias_sd_capped120"].append(
                    float(np.std(np.minimum(est, 120.0) - tv, ddof=1))
                )
            for i, metric in enumerate(("bias_median", "bias_sd_capped120")):
                expected = PAPER_TABLE3[(estimator, kinetics)][i]
                if abs(round(variation_pct(pooled[metric]), 1) - expected) >= 0.05:
                    n_mismatch += 1

    assert n_mismatch == 22, "replicate プールの読み方の不一致数が変わった"
