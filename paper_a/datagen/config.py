"""中核 81 + 頑健性 20 シナリオ定義および列名規約.

仕様: README.md「Data」セクション / paper_a/docs/confirmed_parameters.md
中核 = n_T {2,3,4} × n_points {3,4,6} × ノイズ {小,中,大} × Prior 正確性 {正確,中庸乖離,強乖離}
     = 3×3×3×3 = 81
頑健性 = 分解速度論 {一次,二次,自触媒,誘導期} × 温度依存性
       {Arrhenius 低 Ea, Arrhenius 中 Ea, Arrhenius 高 Ea, 非 Arrh 凹, 非 Arrh 凸}
     = 4×5 = 20

中核代表点 (頑健性の固定軸): n_T=3, n_points=4, ノイズ中, Prior 正確.

----------------------------------------------------------------------
頑健性層の 4×5 = 20 分割の根拠 (仕様書 §2.1 の素直な読み 4×4=16 からの逸脱)
----------------------------------------------------------------------

仕様書 §2.1 は「温度依存性 {Arrhenius 低/中/高 Ea, 非 Arrhenius}」と書いており、
素直に読むと 4 (kinetics) × 4 (temp_dep) = 16 になる.本実装は仕様書記載の
20 シナリオに到達するため、非 Arrhenius を「凹 (concave: 高温側で k 加速)」と
「凸 (convex: 高温側で k 緩和)」の 2 種に分け、4×5 = 20 とした.

選定理由:
- Arrhenius からの逸脱は方向(凹/凸)で意味が異なる.凹 (k(T) が高温で
  予測より速く増える) は MgSt 触媒・水分加水分解の典型、凸 (k(T) が高温で
  予測より遅く増える) は固体相転移・拡散律速の典型.
- 1 種の「非 Arrhenius」では推定器の頑健性を片側でしか試せず、対称性の
  検証が成立しない.
- 数学的実装は modified Arrhenius (T^n 因子) で n=±4 を用いる
  (`temperature.py::modified_arrhenius_concave/convex`).

この設計判断は paper_a/docs/confirmed_parameters.md にも記録.後続で
仕様書本体に追記する根拠とする.
"""
from __future__ import annotations

from itertools import product
from typing import Any, TypedDict

# 列名 (DataRow 規約に準拠.外部設定で差し替え可能なように dict として公開)
COLUMN_NAMES: dict[str, str] = {
    "temperature": "temperature",
    "time": "time_months",
    "response": "content_percent",
    "case_id": "case_id",
    "replicate_id": "replicate_id",
}

# 真の Ea (頑健性層と Faya 包含のため 16/25 kcal/mol も内包)
EA_TRUE_KJ_LOW = 67.0   # ≈ 16 kcal/mol (Faya 低 Ea)
EA_TRUE_KJ_MID = 80.0   # 典型医薬品 Ea (cmc-platform デフォルトと整合)
EA_TRUE_KJ_HIGH = 104.6  # ≈ 25 kcal/mol (Faya 高 Ea)

# Prior 正確性: 真値からの乖離 (kJ/mol)
PRIOR_DEVIATION_KJ = {
    "accurate": 0.0,
    "moderate": 10.0,
    "strong": 25.0,
}
# Prior SD (cmc-platform 既定 30 kJ/mol と整合.弱情報事前)
PRIOR_SD_KJ_DEFAULT = 30.0

# ノイズレベル (log 空間 ln(C/C0) ガウス SD).
# 中=0.02 は audit/mcmc_benchmark.py SIGMA_OBS と整合.
NOISE_SIGMA = {
    "small": 0.01,
    "medium": 0.02,
    "large": 0.05,
}

# 温度設計 (n_T 別).昇順.
TEMPERATURE_SETS = {
    2: [40.0, 60.0],
    3: [40.0, 50.0, 60.0],
    4: [40.0, 50.0, 60.0, 70.0],
}
DEFAULT_TEMP_GRID = TEMPERATURE_SETS[3]

# 時間グリッド (n_points 別).t=0 を必ず含む.
TIME_GRIDS_MONTHS = {
    3: [0.0, 3.0, 6.0],
    4: [0.0, 2.0, 4.0, 6.0],
    6: [0.0, 1.0, 2.0, 3.0, 4.0, 6.0],
}
DEFAULT_TIME_GRID = TIME_GRIDS_MONTHS[4]

# 25°C 長期試験データ (classical_ich_q1e 用、ICH Q1A 長期 36 ヶ月).
# n_points は case の加速試験と一致させ、時間グリッドだけ 25°C 用に置き換える.
# 全 case で温度は 25.0°C 単一.
LONG_TERM_25C_TEMP_C = 25.0
LONG_TERM_25C_DURATION_MONTHS = 36.0
LONG_TERM_25C_TIME_GRIDS = {
    3: [0.0, 18.0, 36.0],
    4: [0.0, 12.0, 24.0, 36.0],
    6: [0.0, 6.0, 12.0, 18.0, 24.0, 36.0],
}

# 中核 81 ベースライン (頑健性層が固定する代表点でもある)
CORE_BASELINE: dict[str, Any] = {
    "kinetics": "first_order",
    "k_of_t": "arrhenius",
    "ea_true_kj": EA_TRUE_KJ_MID,
    "target_sl_at_25c_months": 30.0,  # 真の t90 (25°C) ≈ 30 ヶ月.外挿評価の基準値
    "spec_lower": 95.0,
    "initial_content": 100.0,
    "prior_ea_sd_kj": PRIOR_SD_KJ_DEFAULT,
}


class ScenarioSpec(TypedDict, total=False):
    case_id: str
    layer: str
    n_t: int
    n_points: int
    noise_level: str
    prior_accuracy: str
    temperatures_c: list[float]
    time_points_months: list[float]
    kinetics: str
    k_of_t: str
    ea_true_kj: float
    target_sl_at_25c_months: float
    spec_lower: float
    initial_content: float
    sigma_obs: float
    prior_ea_kj: float
    prior_ea_sd_kj: float
    notes: str


def _build_core_scenarios() -> list[ScenarioSpec]:
    out: list[ScenarioSpec] = []
    n_t_levels = [2, 3, 4]
    n_points_levels = [3, 4, 6]
    noise_levels = ["small", "medium", "large"]
    prior_levels = ["accurate", "moderate", "strong"]
    idx = 0
    for n_t, n_points, noise, prior_acc in product(
        n_t_levels, n_points_levels, noise_levels, prior_levels
    ):
        idx += 1
        ea_true = CORE_BASELINE["ea_true_kj"]
        prior_ea = ea_true + PRIOR_DEVIATION_KJ[prior_acc]
        out.append({
            "case_id": f"core_{idx:03d}",
            "layer": "core",
            "n_t": n_t,
            "n_points": n_points,
            "noise_level": noise,
            "prior_accuracy": prior_acc,
            "temperatures_c": list(TEMPERATURE_SETS[n_t]),
            "time_points_months": list(TIME_GRIDS_MONTHS[n_points]),
            "kinetics": CORE_BASELINE["kinetics"],
            "k_of_t": CORE_BASELINE["k_of_t"],
            "ea_true_kj": ea_true,
            "target_sl_at_25c_months": CORE_BASELINE["target_sl_at_25c_months"],
            "spec_lower": CORE_BASELINE["spec_lower"],
            "initial_content": CORE_BASELINE["initial_content"],
            "sigma_obs": NOISE_SIGMA[noise],
            "prior_ea_kj": prior_ea,
            "prior_ea_sd_kj": CORE_BASELINE["prior_ea_sd_kj"],
        })
    return out


def _build_robustness_scenarios() -> list[ScenarioSpec]:
    out: list[ScenarioSpec] = []
    kinetics_list = ["first_order", "second_order", "autocatalytic", "induction"]
    temp_deps = [
        ("arrhenius_low_ea", "arrhenius", EA_TRUE_KJ_LOW),
        ("arrhenius_mid_ea", "arrhenius", EA_TRUE_KJ_MID),
        ("arrhenius_high_ea", "arrhenius", EA_TRUE_KJ_HIGH),
        ("non_arrhenius_concave", "modified_arrhenius_concave", EA_TRUE_KJ_MID),
        ("non_arrhenius_convex", "modified_arrhenius_convex", EA_TRUE_KJ_MID),
    ]
    n_t = 3
    n_points = 4
    noise = "medium"
    prior_acc = "accurate"
    idx = 0
    for kin in kinetics_list:
        for temp_tag, k_of_t, ea_true in temp_deps:
            idx += 1
            prior_ea = ea_true + PRIOR_DEVIATION_KJ[prior_acc]
            out.append({
                "case_id": f"robust_{idx:02d}",
                "layer": "robustness",
                "n_t": n_t,
                "n_points": n_points,
                "noise_level": noise,
                "prior_accuracy": prior_acc,
                "temperatures_c": list(TEMPERATURE_SETS[n_t]),
                "time_points_months": list(TIME_GRIDS_MONTHS[n_points]),
                "kinetics": kin,
                "k_of_t": k_of_t,
                "ea_true_kj": ea_true,
                "target_sl_at_25c_months": CORE_BASELINE["target_sl_at_25c_months"],
                "spec_lower": CORE_BASELINE["spec_lower"],
                "initial_content": CORE_BASELINE["initial_content"],
                "sigma_obs": NOISE_SIGMA[noise],
                "prior_ea_kj": prior_ea,
                "prior_ea_sd_kj": CORE_BASELINE["prior_ea_sd_kj"],
                "notes": f"kinetics={kin}; temp_dep={temp_tag}",
            })
    return out


CORE_SCENARIOS: list[ScenarioSpec] = _build_core_scenarios()
ROBUSTNESS_SCENARIOS: list[ScenarioSpec] = _build_robustness_scenarios()


# Faya 2018 (博士論文 Table 4.2) の 4 design point.
# E は kcal/mol → kJ/mol (×4.184) 換算.σ は t90 推定量の SD で本設計の
# 「ノイズ」と直接同型ではないが、Ea×ノイズの 2×2 設計を本中核 81 が
# 真部分集合として包含することを Faya 文脈に翻訳して保持する.
# 詳細: README.md「Data」セクション / paper_a/docs/confirmed_parameters.md.
FAYA_DESIGN_POINTS = [
    {"E_kcal": 16, "sigma_label": "low",  "t90_months": 37.48, "ea_kj": EA_TRUE_KJ_LOW},
    {"E_kcal": 16, "sigma_label": "high", "t90_months": 37.48, "ea_kj": EA_TRUE_KJ_LOW},
    {"E_kcal": 25, "sigma_label": "low",  "t90_months": 20.56, "ea_kj": EA_TRUE_KJ_HIGH},
    {"E_kcal": 25, "sigma_label": "high", "t90_months": 20.56, "ea_kj": EA_TRUE_KJ_HIGH},
]
