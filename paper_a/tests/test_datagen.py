"""生成器のサニティテスト: 行数、列スキーマ、決定論性、ノイズスケール."""
from __future__ import annotations

import numpy as np

from paper_a.datagen.config import COLUMN_NAMES, CORE_SCENARIOS, ROBUSTNESS_SCENARIOS
from paper_a.datagen.generate import generate_case
from paper_a.datagen.temperature import arrhenius


def test_first_core_case_row_count_matches_n_t_times_n_points():
    scenario = next(s for s in CORE_SCENARIOS if s["case_id"] == "core_001")
    result = generate_case(scenario, n_replicates=3)
    expected = 3 * scenario["n_t"] * scenario["n_points"]
    assert len(result["rows"]) == expected


def test_data_columns_follow_datarow_convention():
    scenario = CORE_SCENARIOS[40]  # 任意セル
    result = generate_case(scenario, n_replicates=1)
    row = result["rows"][0]
    assert COLUMN_NAMES["temperature"] in row
    assert COLUMN_NAMES["time"] in row
    assert COLUMN_NAMES["response"] in row
    assert COLUMN_NAMES["case_id"] in row
    assert COLUMN_NAMES["replicate_id"] in row


def test_seed_determinism():
    scenario = CORE_SCENARIOS[0]
    a = generate_case(scenario, n_replicates=2)
    b = generate_case(scenario, n_replicates=2)
    for ra, rb in zip(a["rows"], b["rows"]):
        assert ra == rb, "同一 case_id × replicate_id では同一の値が再現されるべき"


def test_t0_has_noise_with_correct_scale():
    """t=0 にも測定誤差がのる.
    分布は log 空間ガウスで、初期含量近傍に集中するべき."""
    scenario = next(s for s in CORE_SCENARIOS if s["case_id"] == "core_001")
    result = generate_case(scenario, n_replicates=200)
    t0_responses = [
        r[COLUMN_NAMES["response"]] for r in result["rows"]
        if r[COLUMN_NAMES["time"]] == 0.0
    ]
    assert len(t0_responses) > 0
    initial = scenario["initial_content"]
    arr = np.array(t0_responses)
    # 全点が initial_content 近傍 (絶対誤差 < 5%)
    assert np.all(np.abs(arr - initial) < 5.0), "t=0 が initial 近傍に分布するべき"
    # ノイズが効いていれば SD > 0
    assert float(np.std(arr)) > 0.0, "t=0 にもノイズが乗っているはず"


def test_truth_metadata_has_data_class_tag():
    scenario = CORE_SCENARIOS[0]
    result = generate_case(scenario, n_replicates=1)
    assert result["truth"]["data_class"] == "synthetic"


def test_robustness_cases_use_baseline_axes():
    """頑健性層は n_T=3 / n_points=4 / noise=medium / prior=accurate を固定."""
    for s in ROBUSTNESS_SCENARIOS:
        assert s["n_t"] == 3
        assert s["n_points"] == 4
        assert s["noise_level"] == "medium"
        assert s["prior_accuracy"] == "accurate"


def test_calibration_reproduces_target_sl_first_order():
    """target_sl と Ea から逆算した ln A で生成した k_25 が target_sl(25°C) と整合する (1次)."""
    from paper_a.datagen.temperature import calibrate_lna_for_true_sl

    target = 30.0
    ea = 80.0
    ln_a, k_25 = calibrate_lna_for_true_sl(target, ea, kinetics="first_order")
    k_25_check = arrhenius(25.0, ln_a, ea)
    assert abs(float(k_25_check) - k_25) < 1e-9
    sl_calc = -np.log(95.0 / 100.0) / k_25
    assert abs(float(sl_calc) - target) < 1e-6


def test_calibration_reproduces_target_sl_all_kinetics():
    """全 kinetics で真値 SL(25°C) = target が成立する (バイアス評価基準の保証)."""
    from paper_a.datagen.temperature import calibrate_lna_for_true_sl, solve_true_sl

    target = 30.0
    ea = 80.0
    for kin in ["first_order", "second_order", "induction", "autocatalytic"]:
        ln_a, k_25 = calibrate_lna_for_true_sl(target, ea, kinetics=kin)
        k_25_from_arrh = arrhenius(25.0, ln_a, ea)
        sl_true = solve_true_sl(k_at_temp=float(k_25_from_arrh), kinetics=kin)
        # 自触媒は数値解のため誤差緩め、それ以外は厳密
        tol = 0.05 if kin == "autocatalytic" else 1e-6
        assert abs(sl_true - target) < tol, f"{kin}: sl_true={sl_true} != target={target}"


def test_truth_includes_t90_and_method_documentation():
    """新しい truth スキーマで t90_true_25c_months と true_sl_method が記録される."""
    scenario = CORE_SCENARIOS[0]
    result = generate_case(scenario, n_replicates=1)
    truth = result["truth"]
    assert "t90_true_25c_months" in truth
    assert "sl_at_spec_true_25c_months" in truth
    assert "true_sl_method" in truth
    method = truth["true_sl_method"]
    for k in ["calibration_chain", "k_25_formula", "lna_formula",
              "k_at_temp_formula", "true_sl_formula", "true_t90_formula",
              "computed_k_25", "computed_ln_a"]:
        assert k in method, f"true_sl_method に {k} が記録されていない"
