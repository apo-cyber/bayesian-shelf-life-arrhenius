"""シナリオ → 反復データ生成のオーケストレーション.

各反復で log 空間ガウスノイズを付加.出力は 2 ファイル:
- 観測データ (DataRow 規約準拠列 + case_id/replicate_id)
- 真値メタデータ (case_id 単位、ノイズなし真値・k 真・Ea 真等)
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np

from .config import (
    COLUMN_NAMES,
    CORE_SCENARIOS,
    LONG_TERM_25C_DURATION_MONTHS,
    LONG_TERM_25C_TEMP_C,
    LONG_TERM_25C_TIME_GRIDS,
    ROBUSTNESS_SCENARIOS,
    ScenarioSpec,
)
from .kinetics import KINETICS_REGISTRY
from .temperature import (
    K_OF_T_REGISTRY,
    calibrate_lna_for_true_sl,
    solve_true_sl,
)

DATA_CLASS_TAG = "synthetic"  # 研究公正性 §10.3: 合成データ明示ラベル


def _make_rng(case_id: str, replicate_id: int) -> np.random.Generator:
    """case_id + replicate_id から決定的に rng を作る.

    Python の hash() は文字列に対してプロセス毎にランダム化されるため、
    hashlib で実行間でも再現可能なシード派生に揃える.
    """
    key = f"{case_id}|{replicate_id}".encode("utf-8")
    digest = hashlib.sha256(key).digest()
    seed = int.from_bytes(digest[:4], "big")
    return np.random.default_rng(seed)


def _make_rng_25c(case_id: str, replicate_id: int) -> np.random.Generator:
    """25°C 長期試験用の独立シード (加速試験の RNG 順序を乱さない).

    "_25c" 接尾を付けた key で SHA-256 派生.加速データの bit-identity を保つ.
    """
    key = f"{case_id}|{replicate_id}|_25c".encode("utf-8")
    digest = hashlib.sha256(key).digest()
    seed = int.from_bytes(digest[:4], "big")
    return np.random.default_rng(seed)


def derive_mcmc_seed(case_id: str, replicate_id: int) -> int:
    """解析側 MCMC の seed (生成器と同じ key 派生を流用).

    paper_a/analysis/estimators/mcmc.py から呼び出して、同 case × replicate で
    実行間 bit-identical な MCMC 結果を保証する.
    """
    key = f"{case_id}|{replicate_id}|mcmc".encode("utf-8")
    digest = hashlib.sha256(key).digest()
    return int.from_bytes(digest[:4], "big")


def _true_response(
    kinetics: str,
    k_value: float,
    times: np.ndarray,
    initial_content: float,
    extra_params: dict | None = None,
) -> np.ndarray:
    fn = KINETICS_REGISTRY[kinetics]
    kwargs = dict(extra_params or {})
    return fn(times, k_value, c0=initial_content, **kwargs)


def _add_noise(
    true_content: np.ndarray,
    sigma_obs: float,
    initial_content: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """log 空間ガウスノイズ: ln(C_obs/C0) = ln(C_true/C0) + N(0, sigma).

    t=0 を含む全点に独立に乗せる(`docs/audit/mcmc_benchmark.py` の合成データ
    生成パターンと整合.実測の t=0 もアッセイ誤差を含むため).
    C_true ≤ 0 は 0 にクリップ.
    """
    true_content = np.asarray(true_content, dtype=float)
    out = np.zeros_like(true_content)
    pos = true_content > 0
    noise = rng.normal(0.0, sigma_obs, size=int(pos.sum()))
    ln_ratio_true = np.log(true_content[pos] / initial_content)
    out[pos] = initial_content * np.exp(ln_ratio_true + noise)
    return out


def _true_sl_method_description(
    kinetics: str,
    k_of_t: str,
    target_sl: float,
    spec_lower: float,
    initial_content: float,
    ea_kj: float,
    ln_a: float,
) -> dict:
    """真 SL/真 t90 算出の全経路を式・手続きとして記述する (査読耐性用).

    全 4 kinetics で同じ詳細度の記述を返す.経路:
      target_sl_at_25c_months → k_25 (kinetics 個別の式) → ln A (Arrhenius)
      → k_T (各加速温度 = Arrhenius または修正 Arrhenius) → 真 SL/真 t90
      (kinetics 個別の式を t について解く)
    """
    R = 8.314
    t_25k = 298.15
    if k_of_t == "arrhenius":
        arrh_to_lna = f"ln A = ln(k_25) + Ea*1000/(R*T_25K) = ln(k_25) + {ea_kj}*1000/({R}*{t_25k})"
        kT_formula = "k(T) = exp(ln A - Ea*1000/(R*T_K))"
    elif k_of_t == "modified_arrhenius_concave":
        arrh_to_lna = f"ln A = ln(k_25) + Ea*1000/(R*T_25K) - 4*ln(T_25K/T_ref)  (T_ref=313.15K, n=+4)"
        kT_formula = "k(T) = (T/T_ref)^4 * exp(ln A - Ea*1000/(R*T_K))"
    elif k_of_t == "modified_arrhenius_convex":
        arrh_to_lna = f"ln A = ln(k_25) + Ea*1000/(R*T_25K) + 4*ln(T_25K/T_ref)  (T_ref=313.15K, n=-4)"
        kT_formula = "k(T) = (T/T_ref)^-4 * exp(ln A - Ea*1000/(R*T_K))"
    else:
        arrh_to_lna = f"(unknown k_of_t: {k_of_t})"
        kT_formula = "(unknown)"

    if kinetics == "first_order":
        k25_formula = f"k_25 = -ln(spec_lower/initial)/target_sl = -ln({spec_lower}/{initial_content})/{target_sl}"
        sl_formula = "t_SL(T) = -ln(spec_lower/initial) / k(T)"
        t90_formula = "t_90(T) = -ln(0.9) / k(T)   # content = 90% (10% degradation)"
    elif kinetics == "second_order":
        k25_formula = f"k_25 = (initial/spec_lower - 1) / (initial*target_sl) = ({initial_content}/{spec_lower} - 1)/({initial_content}*{target_sl})"
        sl_formula = "t_SL(T) = (initial/spec_lower - 1) / (initial * k(T))"
        t90_formula = "t_90(T) = (initial/90 - 1) / (initial * k(T))"
    elif kinetics == "induction":
        k25_formula = f"k_25 = (-ln(spec_lower/initial))^(1/n_avrami) / target_sl  (n_avrami=2)"
        sl_formula = "t_SL(T) = (-ln(spec_lower/initial))^(1/n_avrami) / k(T)"
        t90_formula = "t_90(T) = (-ln(0.9))^(1/n_avrami) / k(T)"
    elif kinetics == "autocatalytic":
        k25_formula = (
            "k_25 solved numerically (brentq, bracket=[1e-8, 10]) such that "
            "C(target_sl; k_25, autocatalytic with alpha=0.05) = spec_lower. "
            "ODE: dC/dt = -k C (1 + alpha (C0 - C))  (Euler ~200 steps/unit time)."
        )
        sl_formula = (
            "t_SL(T) solved numerically (brentq) on the integrated trajectory at temperature T, "
            "crossing content=spec_lower."
        )
        t90_formula = (
            "t_90(T) solved numerically (brentq) on the integrated trajectory at temperature T, "
            "crossing content=90% of initial."
        )
    else:
        k25_formula = sl_formula = t90_formula = f"(unknown kinetics: {kinetics})"

    return {
        "calibration_chain": "target_sl_at_25c_months -> k_25 -> ln A (Arrhenius) -> k(T) at each temperature -> t_SL(T) / t_90(T)",
        "k_25_formula": k25_formula,
        "lna_formula": arrh_to_lna,
        "k_at_temp_formula": kT_formula,
        "true_sl_formula": sl_formula,
        "true_t90_formula": t90_formula,
        "computed_k_25": None,  # 後で埋める
        "computed_ln_a": ln_a,
    }


def generate_case(scenario: ScenarioSpec, n_replicates: int) -> dict:
    """単一シナリオの観測データと真値メタを生成する.

    速度論種別ごとに k_25 を `target_sl_at_25c_months` から逆算 → Arrhenius
    で ln A 確定 → 各加速温度の k_T → 各温度の真 SL/真 t90 を kinetics モデル
    に従って算出.全 kinetics で真 SL = target が成立する(頑健性層のバイアス
    評価が崩れない).真 t90 (content=90%) は仕様書 §4 主指標 (Faya Fig 4.5)
    のための別量として並列に記録する.

    Returns
    -------
    dict
        keys:
          - "rows": list[dict] DataRow 規約準拠の観測行 (×反復)
          - "truth": dict ケース単位の真値メタ
    """
    times = np.array(scenario["time_points_months"], dtype=float)
    temps_c = np.array(scenario["temperatures_c"], dtype=float)
    ea_true = float(scenario["ea_true_kj"])
    target_sl = float(scenario["target_sl_at_25c_months"])
    spec_lower = float(scenario["spec_lower"])
    initial_content = float(scenario["initial_content"])
    sigma_obs = float(scenario["sigma_obs"])
    kinetics = scenario["kinetics"]
    k_of_t_name = scenario["k_of_t"]

    # kinetics 個別に k_25 を逆算 → Arrhenius で ln A 確定
    ln_a_true, k_25_true = calibrate_lna_for_true_sl(
        target_sl_at_25c_months=target_sl,
        ea_kj=ea_true,
        kinetics=kinetics,
        spec_lower=spec_lower,
        initial_content=initial_content,
        k_of_t=k_of_t_name,
    )
    # k(T) at each accelerated temp
    k_of_t_fn = K_OF_T_REGISTRY[k_of_t_name]
    k_true_by_temp = {float(T): float(k_of_t_fn(T, ln_a_true, ea_true)) for T in temps_c}

    # 真 SL / 真 t90 を各温度で数値解 (kinetics モデルに従う)
    sl_at_spec_true_by_temp = {
        str(float(T)): solve_true_sl(
            k_at_temp=k_true_by_temp[float(T)],
            kinetics=kinetics,
            spec_lower=spec_lower,
            initial_content=initial_content,
        )
        for T in temps_c
    }
    k_25 = float(k_of_t_fn(25.0, ln_a_true, ea_true))
    sl_at_spec_true_25c = solve_true_sl(
        k_at_temp=k_25,
        kinetics=kinetics,
        spec_lower=spec_lower,
        initial_content=initial_content,
    )
    # 標準 pharma t90 = content 90% (10% 分解) 到達時間
    t90_true_25c = solve_true_sl(
        k_at_temp=k_25,
        kinetics=kinetics,
        spec_lower=90.0,
        initial_content=initial_content,
    )
    t90_true_by_temp = {
        str(float(T)): solve_true_sl(
            k_at_temp=k_true_by_temp[float(T)],
            kinetics=kinetics,
            spec_lower=90.0,
            initial_content=initial_content,
        )
        for T in temps_c
    }

    method_doc = _true_sl_method_description(
        kinetics=kinetics,
        k_of_t=k_of_t_name,
        target_sl=target_sl,
        spec_lower=spec_lower,
        initial_content=initial_content,
        ea_kj=ea_true,
        ln_a=ln_a_true,
    )
    method_doc["computed_k_25"] = k_25_true

    rows: list[dict] = []
    case_id = scenario["case_id"]

    for rep in range(n_replicates):
        rng = _make_rng(case_id, rep)
        for T_c in temps_c:
            k_val = k_true_by_temp[float(T_c)]
            true_content = _true_response(
                kinetics=kinetics,
                k_value=k_val,
                times=times,
                initial_content=initial_content,
            )
            obs_content = _add_noise(true_content, sigma_obs, initial_content, rng)
            for t_i, c_i in zip(times, obs_content):
                rows.append({
                    COLUMN_NAMES["case_id"]: case_id,
                    COLUMN_NAMES["replicate_id"]: rep,
                    COLUMN_NAMES["temperature"]: float(T_c),
                    COLUMN_NAMES["time"]: float(t_i),
                    COLUMN_NAMES["response"]: float(c_i),
                })

    # 25°C 長期試験データ (classical_ich_q1e 用、Faya Fig 4.5 物理量整合)
    n_points = int(scenario["n_points"])
    long_term_times = np.array(LONG_TERM_25C_TIME_GRIDS[n_points], dtype=float)
    long_term_rows: list[dict] = []
    for rep in range(n_replicates):
        rng_25c = _make_rng_25c(case_id, rep)
        true_content_25c = _true_response(
            kinetics=kinetics,
            k_value=k_25,
            times=long_term_times,
            initial_content=initial_content,
        )
        obs_content_25c = _add_noise(true_content_25c, sigma_obs, initial_content, rng_25c)
        for t_i, c_i in zip(long_term_times, obs_content_25c):
            long_term_rows.append({
                COLUMN_NAMES["case_id"]: case_id,
                COLUMN_NAMES["replicate_id"]: rep,
                COLUMN_NAMES["temperature"]: float(LONG_TERM_25C_TEMP_C),
                COLUMN_NAMES["time"]: float(t_i),
                COLUMN_NAMES["response"]: float(c_i),
            })

    truth = {
        "case_id": case_id,
        "layer": scenario["layer"],
        "data_class": DATA_CLASS_TAG,
        "kinetics": kinetics,
        "k_of_t": k_of_t_name,
        "ea_true_kj_mol": ea_true,
        "ln_a_true": ln_a_true,
        "target_sl_at_25c_months": target_sl,
        # 仕様書 §4 主指標: 真の t90 (content=90% 到達時間) を 25°C と各加速温度で算出.
        # 推定器バイアス評価はこの値を基準に行う (Faya 2018 Fig 4.5 と整合).
        "t90_true_25c_months": t90_true_25c,
        "t90_true_by_temp_months": t90_true_by_temp,
        # 実務指標 (補助): spec_lower 到達時間.target との一致確認用.
        "sl_at_spec_true_25c_months": sl_at_spec_true_25c,
        "sl_at_spec_true_by_temp_months": sl_at_spec_true_by_temp,
        "spec_lower": spec_lower,
        "initial_content": initial_content,
        "n_t": int(scenario["n_t"]),
        "n_points": int(scenario["n_points"]),
        "noise_level": scenario["noise_level"],
        "sigma_obs": sigma_obs,
        "prior_accuracy": scenario["prior_accuracy"],
        "prior_ea_kj_mol": float(scenario["prior_ea_kj"]),
        "prior_ea_sd_kj_mol": float(scenario["prior_ea_sd_kj"]),
        "temperatures_c": [float(T) for T in temps_c],
        "time_points_months": [float(t) for t in times],
        "k_true_by_temp": {str(float(T)): k for T, k in k_true_by_temp.items()},
        "true_sl_method": method_doc,
        # 25°C 長期試験データ仕様 (classical_ich_q1e 用)
        "long_term_25c": {
            "temperature_c": float(LONG_TERM_25C_TEMP_C),
            "duration_months": float(LONG_TERM_25C_DURATION_MONTHS),
            "time_points_months": [float(t) for t in long_term_times],
            "k_25_true": float(k_25),
        },
        "n_replicates": n_replicates,
        "notes": scenario.get("notes", ""),
    }

    return {"rows": rows, "long_term_25c_rows": long_term_rows, "truth": truth}


def generate_layer(
    scenarios: Iterable[ScenarioSpec],
    n_replicates: int,
    out_dir: Path,
) -> dict:
    """シナリオリスト → data.csv + truth.json を書き出す.

    Returns 概要 (rows 数、case 数、出力先).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    data_path = out_dir / "data.csv"
    long_term_path = out_dir / "long_term_25c.csv"
    truth_path = out_dir / "truth.json"

    truth_records: list[dict] = []
    n_rows_total = 0
    n_rows_25c_total = 0

    header_cols = [
        COLUMN_NAMES["case_id"],
        COLUMN_NAMES["replicate_id"],
        COLUMN_NAMES["temperature"],
        COLUMN_NAMES["time"],
        COLUMN_NAMES["response"],
    ]

    def _write_row(f, row: dict) -> None:
        f.write(
            f"{row[COLUMN_NAMES['case_id']]},"
            f"{row[COLUMN_NAMES['replicate_id']]},"
            f"{row[COLUMN_NAMES['temperature']]:.4f},"
            f"{row[COLUMN_NAMES['time']]:.4f},"
            f"{row[COLUMN_NAMES['response']]:.6f}\n"
        )

    with data_path.open("w") as f_acc, long_term_path.open("w") as f_25c:
        f_acc.write(",".join(header_cols) + "\n")
        f_25c.write(",".join(header_cols) + "\n")
        for scenario in scenarios:
            result = generate_case(scenario, n_replicates=n_replicates)
            truth_records.append(result["truth"])
            for row in result["rows"]:
                _write_row(f_acc, row)
                n_rows_total += 1
            for row in result["long_term_25c_rows"]:
                _write_row(f_25c, row)
                n_rows_25c_total += 1

    with truth_path.open("w") as f:
        json.dump(
            {"data_class": DATA_CLASS_TAG, "cases": truth_records},
            f,
            indent=2,
            ensure_ascii=False,
        )

    return {
        "n_cases": len(truth_records),
        "n_rows": n_rows_total,
        "n_rows_25c": n_rows_25c_total,
        "data_path": str(data_path),
        "long_term_25c_path": str(long_term_path),
        "truth_path": str(truth_path),
    }


def run_full_generation(
    root: Path,
    n_replicates: int = 1000,
    include_core: bool = True,
    include_robustness: bool = True,
) -> dict:
    """中核 + 頑健性を一括生成して root 配下に書き出す."""
    out: dict = {}
    if include_core:
        out["core"] = generate_layer(CORE_SCENARIOS, n_replicates, root / "core")
    if include_robustness:
        out["robustness"] = generate_layer(
            ROBUSTNESS_SCENARIOS, n_replicates, root / "robustness"
        )
    return out
