"""Table 1 補強用 bootstrap CI (paper-A Discussion の reviewer 先回り防御).

central core-layer cell (n_T=3, prior=strong) の 4 推定器について、
bias_median と bias_sd_capped120 の 95% BCa bootstrap 信頼区間を計算する。

想定 reviewer 質問への対応:
  1. mcmc の bias_median = +36.4 月は 100 reps/case で安定か → bias_median の CI
  2. two_stage_conjugate と mcmc の "opposite sides" は有意か → 両 CI の重なり判定
  3. bias_sd_capped120 32.46 vs 31.41 (差 1.05) は comparable か → 両 CI の重なり判定

**指標定義は paper_a.analysis.metrics と完全一致させる** (命名は仕様の宣言):
  - 成功 rep = error_code が null かつ t90_point_estimate_months が有限値
    (reaggregate.py が NaN→None 正規化した後の metrics.py 挙動と同値)
  - bias_raw       = est - t90_true_25c_months          (per-case truth、生バイアス)
  - bias_median    = median(bias_raw)                    ← raw (cap しない)
  - bias_sd_capped120 = std(min(est, 120) - truth, ddof=1)
  真値は paper_a/data/core/truth.json の per-case t90_true_25c_months を使う
  (metrics.py と同じ。ハードコードしない。全 core cell で 61.6224 月だが
   将来 truth が変わっても追従するよう case ごとに読む)。

セル定義は aggregation.slice_by を再利用し、論文集計と同一のスライスを保証する。

bootstrap の再標本化単位は「成功 rep」を i.i.d. に取る (pooled median/SD と整合)。
9 case をまたいで pool する点は metrics.py の集計と同じ。
(case クラスタ bootstrap は別エスティマンドであり、本指標は pooled 統計量なので
 i.i.d. rep 再標本化が点推定の定義に直接対応する。)

出力: paper_a/results/bootstrap_ci.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import bootstrap

from paper_a.analysis.loaders.synthetic import load_truth
from paper_a.analysis.metrics import SHELF_LIFE_CAP_MONTHS
from paper_a.analysis.aggregation import slice_by

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
PARQUET_PATH = RESULTS_DIR / "estimator_results.parquet"
OUTPUT_PATH = RESULTS_DIR / "bootstrap_ci.json"

TRUTH_FIELD = "t90_true_25c_months"  # metrics.py の主指標真値 (§B.1)

N_RESAMPLES = 10_000
CONFIDENCE_LEVEL = 0.95
RANDOM_STATE = 42
MIN_SUCCESSFUL = 10  # これ未満は CI を計算しない

# central cell スライス指定 (n_T=3 × prior=strong)。
# slice_by が 3 sampling grid × 3 noise level = 9 case に展開する。
CENTRAL_CELL = {"n_t": 3, "prior_accuracy": "strong"}

ESTIMATORS = [
    "two_stage_conjugate",
    "mcmc",
    "classical_ols_multi_temp",
    "classical_ich_q1e",
]


def _median_stat(data, axis):
    return np.median(data, axis=axis)


def _sd_ddof1_stat(data, axis):
    return np.std(data, axis=axis, ddof=1)


def _bca_ci(values: np.ndarray, statistic) -> tuple[float, float]:
    """1 次元配列に対し BCa 95% CI を返す (low, high)。"""
    res = bootstrap(
        data=(values,),
        statistic=statistic,
        n_resamples=N_RESAMPLES,
        confidence_level=CONFIDENCE_LEVEL,
        method="BCa",
        rng=np.random.default_rng(RANDOM_STATE),
    )
    ci = res.confidence_interval
    return float(ci.low), float(ci.high)


def compute_ci_for_estimator(df_est: pd.DataFrame, truth_by_case: dict[str, dict]) -> dict:
    """1 推定器 (central cell に絞り込み済み) の bias_median / bias_sd_capped120 CI。

    df_est は estimator_name で既にフィルタ済みの DataFrame。
    成功 rep の抽出と真値の引き当ては metrics.py と同一ロジック。
    """
    # 成功 rep = error_code が null かつ t90 が有限値 (metrics.py と同値)
    mask = df_est["error_code"].isna() & np.isfinite(df_est["t90_point_estimate_months"])
    succ = df_est.loc[mask]
    n_succ = int(len(succ))
    n_total = int(len(df_est))

    if n_succ < MIN_SUCCESSFUL:
        note = f"insufficient successful replicates (n={n_succ})"
        return {
            "bias_median": {"point": None, "ci_low": None, "ci_high": None, "note": note},
            "bias_sd_capped120": {"point": None, "ci_low": None, "ci_high": None, "note": note},
            "n_total": n_total,
            "n_successful": n_succ,
        }

    est_vals = succ["t90_point_estimate_months"].to_numpy(dtype=float)
    truth_vals = succ["case_id"].map(lambda c: float(truth_by_case[c][TRUTH_FIELD])).to_numpy()

    bias_raw = est_vals - truth_vals
    bias_capped = np.minimum(est_vals, SHELF_LIFE_CAP_MONTHS) - truth_vals

    # 点推定は metrics.py と完全一致する式で直接計算 (bootstrap 平均ではない)
    bias_median_point = float(np.median(bias_raw))
    bias_sd_point = float(np.std(bias_capped, ddof=1))

    median_lo, median_hi = _bca_ci(bias_raw, _median_stat)
    sd_lo, sd_hi = _bca_ci(bias_capped, _sd_ddof1_stat)

    return {
        "bias_median": {"point": bias_median_point, "ci_low": median_lo, "ci_high": median_hi},
        "bias_sd_capped120": {"point": bias_sd_point, "ci_low": sd_lo, "ci_high": sd_hi},
        "n_total": n_total,
        "n_successful": n_succ,
    }


def main() -> int:
    if not PARQUET_PATH.exists():
        raise FileNotFoundError(
            f"{PARQUET_PATH} がない。python -m paper_a.analysis.run_paper_a を先に実行。"
        )

    truth_by_case = load_truth("core")
    central_cases = slice_by(truth_by_case, layer="core", **CENTRAL_CELL)
    if not central_cases:
        raise ValueError(
            f"central cell {CENTRAL_CELL} に該当する case がない。truth.json のスキーマを確認。"
        )
    central_cases = sorted(central_cases)
    print(f"central cell {CENTRAL_CELL} → {len(central_cases)} cases: {central_cases}")

    # truth は全 core cell で同一 (検証も兼ねて確認)
    truth_values = {float(truth_by_case[c][TRUTH_FIELD]) for c in central_cases}
    truth_value = next(iter(truth_values))
    if len(truth_values) != 1:
        print(f"  WARNING: central cell 内で t90_true が一様でない: {sorted(truth_values)}")

    df = pd.read_parquet(PARQUET_PATH)
    df_central = df[df["case_id"].isin(central_cases)]
    print(f"loaded {len(df)} rows, central cell に {len(df_central)} rows")

    metrics: dict[str, dict] = {}
    n_replicates: dict[str, int] = {}
    for est in ESTIMATORS:
        df_est = df_central[df_central["estimator_name"] == est]
        n_replicates[est] = int(len(df_est))
        print(f"computing CI for {est} (n_total={len(df_est)}) ...")
        metrics[est] = compute_ci_for_estimator(df_est, truth_by_case)
        bm = metrics[est]["bias_median"]
        bs = metrics[est]["bias_sd_capped120"]
        print(
            f"  n_success={metrics[est]['n_successful']}"
            f" | bias_median={bm['point']} CI=[{bm['ci_low']}, {bm['ci_high']}]"
            f" | bias_sd_capped120={bs['point']} CI=[{bs['ci_low']}, {bs['ci_high']}]"
        )

    output = {
        "cell": "n_T=3, prior=strong",
        "cell_filter": CENTRAL_CELL,
        "truth_field": TRUTH_FIELD,
        "true_t90_25C_months": truth_value,
        "case_ids": central_cases,
        "n_replicates_per_estimator": n_replicates,
        "metrics": metrics,
        "bootstrap_config": {
            "n_resamples": N_RESAMPLES,
            "method": "BCa",
            "confidence_level": CONFIDENCE_LEVEL,
            "random_state": RANDOM_STATE,
            "cap_months": SHELF_LIFE_CAP_MONTHS,
            "resampling_unit": "successful replicate (i.i.d., pooled over 9 cases)",
        },
        "metric_definitions": {
            "success": "error_code is null AND t90_point_estimate_months finite (matches metrics.py)",
            "bias_median": "median(est - t90_true_25c_months)  [raw, uncapped]",
            "bias_sd_capped120": "std(min(est, 120) - t90_true_25c_months, ddof=1)",
        },
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n→ {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
