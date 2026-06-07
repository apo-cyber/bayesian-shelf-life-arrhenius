"""metrics.py の必須テスト.

追加要求 (b):
    bias = t90_estimate − t90_true_25c_months を厳守.
    target_sl_at_25c_months (=入力値 30 月) を間違って参照しないこと.
    生成器側 kinetics-aware 較正バグと同パターンの取り違え防止.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from paper_a.analysis.metrics import compute_cell_metrics

DATA_ROOT = Path(__file__).resolve().parents[1] / "data"


def _truth_sample() -> dict[str, dict]:
    """truth.json の core_001 + robust_14 を使って真値ソースの取り違えを検出する."""
    truth_core = json.loads((DATA_ROOT / "core" / "truth.json").read_text())
    truth_robust = json.loads((DATA_ROOT / "robustness" / "truth.json").read_text())
    by_case = {c["case_id"]: c for c in truth_core["cases"]}
    by_case.update({c["case_id"]: c for c in truth_robust["cases"]})
    return by_case


def test_truth_source_is_t90_true_not_target_sl():
    """bias 計算は truth["t90_true_25c_months"] を使い、
    truth["target_sl_at_25c_months"] (=入力 30) を使わないこと.

    1 次速度論の core_001 では t90_true_25c=61.6224、target_sl=30.0 で
    両者は明確に異なる.推定値=61.6224 のとき bias_median は 0 となるべき
    (正解).target_sl を真値に取ると bias_median が ~31.62 と桁違いに大きく
    なってしまうため、これで誤参照を検出できる.

    自触媒 robust_14 では t90_true=55.84、target_sl=30.0 でさらに乖離が大きく、
    取り違えがあれば必ず検出される (生成器側バグと同パターンの予防).
    """
    truth_by_case = _truth_sample()
    t90_true_core = truth_by_case["core_001"]["t90_true_25c_months"]
    t90_true_robust = truth_by_case["robust_14"]["t90_true_25c_months"]
    target_sl_core = truth_by_case["core_001"]["target_sl_at_25c_months"]
    target_sl_robust = truth_by_case["robust_14"]["target_sl_at_25c_months"]

    # 設計確認: 真値 t90 と target_sl は別物
    assert abs(t90_true_core - target_sl_core) > 1.0, (
        "1 次速度論で t90 と target_sl が同じになる場合、誤参照検出力が落ちる"
    )
    assert abs(t90_true_robust - target_sl_robust) > 1.0, (
        "頑健性 case で t90 と target_sl が同じになる場合、誤参照検出力が落ちる"
    )

    # 推定値 = 真 t90 ぴったり → bias_median = 0 になるべき
    results = [
        {
            "estimator_name": "classical_ols_multi_temp",
            "case_id": "core_001",
            "replicate_id": 0,
            "t90_point_estimate_months": t90_true_core,
            "t90_lo95_months": t90_true_core - 5.0,
            "t90_hi95_months": t90_true_core + 5.0,
            "error_code": None,
            "converged": True,
        }
    ]
    cell = compute_cell_metrics(
        estimator_name="classical_ols_multi_temp",
        cell_key="test",
        results=results,
        truth_by_case=truth_by_case,
    )
    assert cell.bias_median == pytest.approx(0.0, abs=1e-9), (
        f"bias_median が 0 でない (={cell.bias_median}).真値ソースが間違っている可能性 "
        f"(target_sl_at_25c=30 を参照していないか確認)."
    )


def test_metrics_default_truth_field_is_t90():
    """compute_cell_metrics のデフォルト truth_field は 't90_true_25c_months'.

    シグネチャ確認.今後の安易な変更を防ぐ.
    """
    import inspect
    sig = inspect.signature(compute_cell_metrics)
    default = sig.parameters["truth_field"].default
    assert default == "t90_true_25c_months", (
        f"truth_field の default が変わった (={default}).論文主指標は t90."
    )


def test_pp_failure_rate_direction():
    """PP 失敗率 = P(t_label > t90_true) の方向性を確認."""
    truth_by_case = _truth_sample()
    truth = truth_by_case["core_001"]["t90_true_25c_months"]
    # 推定値 = truth + 10 (楽観バイアス)
    results = [
        {
            "estimator_name": "classical_ols_multi_temp",
            "case_id": "core_001",
            "replicate_id": i,
            "t90_point_estimate_months": truth + 10.0,
            "t90_lo95_months": None,
            "t90_hi95_months": None,
            "error_code": None,
            "converged": True,
        }
        for i in range(10)
    ]
    cell = compute_cell_metrics(
        estimator_name="classical_ols_multi_temp",
        cell_key="optimistic",
        results=results,
        truth_by_case=truth_by_case,
    )
    assert cell.pp_failure_rate == 1.0, (
        "全 rep で楽観バイアス → pp_failure_rate=1.0 (失敗率 100%) になるべき"
    )

    # 推定値 = truth - 10 (保守バイアス) → 失敗率 0
    results2 = [
        {**r, "t90_point_estimate_months": truth - 10.0}
        for r in results
    ]
    cell2 = compute_cell_metrics(
        estimator_name="classical_ols_multi_temp",
        cell_key="conservative",
        results=results2,
        truth_by_case=truth_by_case,
    )
    assert cell2.pp_failure_rate == 0.0


def test_ci_above_cap_rate():
    """補助指標 ci_hi_above_cap_rate が計算される (追加要求 a).

    CI 上限 = 200 月 (>120 月 cap) が 1 件、80 月が 1 件 → 0.5 になるべき.
    """
    truth_by_case = _truth_sample()
    results = [
        {
            "estimator_name": "classical_ols_multi_temp",
            "case_id": "core_001",
            "replicate_id": 0,
            "t90_point_estimate_months": 60.0,
            "t90_lo95_months": 40.0,
            "t90_hi95_months": 200.0,  # > 120 cap
            "error_code": None,
            "converged": True,
        },
        {
            "estimator_name": "classical_ols_multi_temp",
            "case_id": "core_001",
            "replicate_id": 1,
            "t90_point_estimate_months": 60.0,
            "t90_lo95_months": 40.0,
            "t90_hi95_months": 80.0,   # < 120 cap
            "error_code": None,
            "converged": True,
        },
    ]
    cell = compute_cell_metrics(
        estimator_name="classical_ols_multi_temp",
        cell_key="cap_test",
        results=results,
        truth_by_case=truth_by_case,
    )
    assert cell.ci_hi_above_cap_rate == pytest.approx(0.5, abs=1e-9)
    assert cell.n_reps_with_ci == 2
