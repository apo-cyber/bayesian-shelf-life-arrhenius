"""3 指標 (仕様書 §4): バイアス・ばらつき / posterior predictive 失敗率 /
カバレッジ確率.補助: CI 上限 > 120 月の発生率 (追加要求 a).

**真値ソース** (仕様書 §B.1、追加要求 b で厳守):
    主指標は **t90_true_25c_months** ベース.target_sl_at_25c_months
    (入力値 30 月) を真値として使ってはならない.両者は kinetics が 1 次以外
    のとき桁違いに異なる (生成器側 kinetics-aware 較正バグと同パターンの
    取り違え防止).test_metrics.py::test_truth_source_is_t90_true で
    必須テスト化.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

# 可視化 cap (前停止点判断 2): cmc-platform `_cap` と整合.
SHELF_LIFE_CAP_MONTHS = 120.0


@dataclass
class CellMetrics:
    """1 集計セル × 1 推定器の指標群.

    bias_sd は 3 種類報告する (D.3 追加要求 b):
    - bias_sd_raw: 全推定成功 rep (error_code is None) で raw 値.外れ値で inf になりうる.
    - bias_sd_capped120: t90 推定値を 120 ヶ月に cap してから bias 計算.主指標.
    - bias_sd_converged_only: converged=True に絞る.MCMC の「収束した時の精度」用補助.
    """

    estimator_name: str
    cell_key: str  # 例: "n_t=2|prior=strong" 等
    n_reps_total: int
    n_reps_success: int          # 推定成功 (error_code is None) の数
    n_reps_failed: int           # 推定失敗 (error_code != None)
    failure_rate: float          # n_failed / n_total

    # 主指標: t90 推定値の真値 t90 に対するバイアス・ばらつき
    bias_median: float | None
    bias_mean: float | None
    bias_sd_raw: float | None
    bias_sd_capped120: float | None      # 主指標 (論文 Methods で報告)
    bias_sd_converged_only: float | None  # 補助 (Discussion: MCMC が収束したときの精度)
    n_reps_converged_only: int            # converged=True の rep 数 (MCMC 用)
    t90_estimate_median: float | None
    t90_estimate_lo_iqr: float | None   # 25 percentile
    t90_estimate_hi_iqr: float | None   # 75 percentile

    # PP 失敗率 (Faya 2018 §4.2.2.2 表記):
    # P(t_label ≤ t90_true) = 「ラベル ≤ 真値」.
    # 解釈 (再開プロンプト B.1 で確認): ここでは "ラベル" = 点推定 t90.
    # 失敗 = 「ラベルが真値を超える」=「t_label > t90_true」=「楽観バイアス」.
    # よって本指標は「P(t_label > t90_true)」を計算する (= 楽観失敗率).
    pp_failure_rate: float | None

    # カバレッジ確率: 真の t90 が 95% CI [lo95, hi95] に含まれる割合
    coverage_probability: float | None

    # 補助指標 (追加要求 a): CI 上限が cap を超える発生率
    ci_hi_above_cap_rate: float | None
    n_reps_with_ci: int  # CI を持つ推定の数 (classical_ich_q1e は hi95=None)

    # 副次診断: MCMC 収束失敗率 (該当推定器のみ非ゼロ)
    mcmc_nonconverged_rate: float


def compute_cell_metrics(
    estimator_name: str,
    cell_key: str,
    results: Iterable[dict],  # EstimatorResult.to_dict() のシーケンス
    truth_by_case: dict[str, dict],
    *,
    truth_field: str = "t90_true_25c_months",
) -> CellMetrics:
    """1 cell の 1 推定器の指標を計算する.

    Parameters
    ----------
    truth_field : str
        真値フィールド名.デフォルトは仕様書 §B.1 主指標.
        誤って "target_sl_at_25c_months" を渡さないよう test_metrics で保証.

    Notes
    -----
    SHELF_LIFE_CAP_MONTHS による cap は cap 発生率の集計にのみ使い、
    bias/variance の生計算には raw 値を保持する (前停止点判断 2: 可視化と
    数値の 2 通り).
    """
    results_list = list(results)
    n_total = len(results_list)
    successes = [r for r in results_list if r["error_code"] is None]
    n_success = len(successes)
    n_failed = n_total - n_success
    failure_rate = n_failed / n_total if n_total else 0.0

    if not successes:
        return CellMetrics(
            estimator_name=estimator_name,
            cell_key=cell_key,
            n_reps_total=n_total,
            n_reps_success=0,
            n_reps_failed=n_failed,
            failure_rate=failure_rate,
            bias_median=None,
            bias_mean=None,
            bias_sd_raw=None,
            bias_sd_capped120=None,
            bias_sd_converged_only=None,
            n_reps_converged_only=0,
            t90_estimate_median=None,
            t90_estimate_lo_iqr=None,
            t90_estimate_hi_iqr=None,
            pp_failure_rate=None,
            coverage_probability=None,
            ci_hi_above_cap_rate=None,
            n_reps_with_ci=0,
            mcmc_nonconverged_rate=0.0,
        )

    biases_raw: list[float] = []
    biases_capped: list[float] = []
    biases_converged: list[float] = []
    t90_estimates: list[float] = []
    pp_failures: list[bool] = []
    covered: list[bool] = []
    ci_hi_above_cap: list[bool] = []
    nonconverged_count = 0
    n_with_ci = 0
    n_converged = 0

    for r in results_list:
        if r["error_code"] == "MCMC_NOT_CONVERGED":
            nonconverged_count += 1
        if r["error_code"] is not None:
            continue
        case_id = r["case_id"]
        if case_id not in truth_by_case:
            continue
        truth_value = float(truth_by_case[case_id][truth_field])
        est = r["t90_point_estimate_months"]
        if est is None:
            continue
        bias_r = float(est) - truth_value
        bias_c = min(float(est), SHELF_LIFE_CAP_MONTHS) - truth_value
        biases_raw.append(bias_r)
        biases_capped.append(bias_c)
        if r.get("converged", True):
            n_converged += 1
            biases_converged.append(bias_r)
        t90_estimates.append(float(est))

        # PP 失敗率 = P(t_label > t90_true) = 楽観バイアスの出現確率
        pp_failures.append(float(est) > truth_value)

        lo = r.get("t90_lo95_months")
        hi = r.get("t90_hi95_months")
        if lo is not None and hi is not None:
            n_with_ci += 1
            covered.append(float(lo) <= truth_value <= float(hi))
            ci_hi_above_cap.append(float(hi) > SHELF_LIFE_CAP_MONTHS)

    bias_raw_arr = np.array(biases_raw)
    bias_cap_arr = np.array(biases_capped)
    bias_conv_arr = np.array(biases_converged)
    est_arr = np.array(t90_estimates)
    return CellMetrics(
        estimator_name=estimator_name,
        cell_key=cell_key,
        n_reps_total=n_total,
        n_reps_success=n_success,
        n_reps_failed=n_failed,
        failure_rate=failure_rate,
        bias_median=float(np.median(bias_raw_arr)) if biases_raw else None,
        bias_mean=float(np.mean(bias_raw_arr)) if biases_raw else None,
        bias_sd_raw=float(np.std(bias_raw_arr, ddof=1)) if len(biases_raw) > 1 else None,
        bias_sd_capped120=float(np.std(bias_cap_arr, ddof=1)) if len(biases_capped) > 1 else None,
        bias_sd_converged_only=(
            float(np.std(bias_conv_arr, ddof=1)) if len(biases_converged) > 1 else None
        ),
        n_reps_converged_only=n_converged,
        t90_estimate_median=float(np.median(est_arr)) if t90_estimates else None,
        t90_estimate_lo_iqr=float(np.percentile(est_arr, 25)) if t90_estimates else None,
        t90_estimate_hi_iqr=float(np.percentile(est_arr, 75)) if t90_estimates else None,
        pp_failure_rate=float(np.mean(pp_failures)) if pp_failures else None,
        coverage_probability=float(np.mean(covered)) if covered else None,
        ci_hi_above_cap_rate=float(np.mean(ci_hi_above_cap)) if ci_hi_above_cap else None,
        n_reps_with_ci=n_with_ci,
        mcmc_nonconverged_rate=nonconverged_count / n_total if n_total else 0.0,
    )
