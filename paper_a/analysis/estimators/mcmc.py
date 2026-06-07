"""推定器 #2: Full Bayesian MCMC.

このモデルは著者の private cmc-platform プロジェクトの MCMC ベンチマーク
(run_mcmc) に由来し、MIT デュアルライセンスで本パッケージに収録する.

実装方針:
    由来元の run_mcmc を analysis 配下に移植し、prior を
    関数引数で露出させ、spec_lower を露出させ、seed を生成器整合で受ける.
    モデル構造 (ln_A ~ N(20, 100), Ea ~ N(prior, sd), sigma_obs ~ HalfNormal(0.1))
    は変更しない (cmc-platform の Bayesian 安定性監査結果を継承).

収束判定:
    R-hat < 1.01 かつ ESS > 400 で converged=True、それ以外で converged=False
    + error_code="MCMC_NOT_CONVERGED".

case 別 prior (判断 3) + seed (追加要求):
    prior_ea_kj/sd は truth.json から case 別に受ける.
    seed は `paper_a.datagen.derive_mcmc_seed(case_id, replicate_id)` で
    生成器整合.同 seed で 2 回走らせて bit-identical な事後標本を保証する
    単体テストを追加 (test_mcmc.py::test_mcmc_seed_reproducibility).

================================================================
Methods 節素材 (論文で参照)
================================================================

サンプラー詳細:
    PyMC 5.28.5 / arviz 0.23.4 / NUTS (No-U-Turn Sampler).
    nuts_sampler="numpyro" (numpyro 0.21.0 + JAX 0.10.0 backend) を採用.
    chains=4, draws=2000, tune=1000, target_accept=0.95.

numpyro backend 同等性検証 (2026-05-20、core_041 rep=0 / random_seed=42):
    | 量             | PyMC default   | numpyro       | Δ        |
    | t90_mean       | 156.10 mo      | 152.55 mo     | 2.27%    |
    | t90_lo95       | 48.25 mo       | 48.12 mo      | 0.28%    |
    | t90_hi95       | 425.04 mo      | 427.88 mo     | 0.67%    |
    | rhat_max       | 1.0000         | 1.0000        | -        |
    | ess_min        | 1753           | 1649          | -        |
    | elapsed (s)    | 13.1           | 2.9           | 4.5x     |

    判定: 全 t90 指標 Δ < 5%、R-hat/ESS とも閾値 (1.01 / 400) を充分に
    満たしクリア.numpyro は MCMC 結果として等価とみなせる.

計算コスト (案 B = MCMC 100 reps/case × 101 cases = 10,100 走行):
    PyMC default: 13s × 10,100 ≈ 36.5 時間 (連続実行リスク大).
    numpyro:       2.9s × 10,100 ≈ 8.1 時間  (夜間 1 回完走可).

設計上の非対称性 (案 B):
    他 3 推定器は 1000 reps/case で走らせるのに対し、MCMC のみ 100 reps/case.
    Faya 2018 の 3000 reps × 4 design points = 12,000 MCMC 走行と同等規模.
    Methods に明記: "MCMC を計算コスト上 100 reps、他 3 推定器を 1000 reps とした.
    全推定器の SE が同等オーダーとなる設計であり、Faya 2018 (12,000 MCMC 走行)
    と整合する規模である." 集計層は rep 数差を箱ひげの幅で自然に表現する.
"""
from __future__ import annotations

import warnings as _warnings
from typing import Any

import numpy as np

from paper_a.datagen import derive_mcmc_seed

from .base import EstimatorResult

R_GAS = 8.314
T_25K = 298.15

ESTIMATOR_NAME = "mcmc"

# MCMC NUTS 設定 (cmc-platform ベンチマーク既存値と整合)
DEFAULT_DRAWS = 2000
DEFAULT_TUNE = 1000
DEFAULT_CHAINS = 4
DEFAULT_TARGET_ACCEPT = 0.95
# nuts_sampler="numpyro" で 4.5x speedup (core_041 で検証).R-hat/ESS/t90 ±2.3% 内で
# PyMC default と同等 (本番 10,100 走行: 36h → 8h).
# fallback で "default" (PyMC C backend) も指定可能.
DEFAULT_NUTS_SAMPLER = "numpyro"

# 収束閾値
RHAT_THRESHOLD = 1.01
ESS_THRESHOLD = 400.0


def estimate(
    data_rows: list[dict],
    case_id: str,
    replicate_id: int,
    prior_ea_kj: float,
    prior_ea_sd_kj: float,
    spec_lower: float = 90.0,
    initial_content: float = 100.0,
    target_temp_c: float = 25.0,
    column_names: dict | None = None,
    draws: int = DEFAULT_DRAWS,
    tune: int = DEFAULT_TUNE,
    chains: int = DEFAULT_CHAINS,
    target_accept: float = DEFAULT_TARGET_ACCEPT,
    seed: int | None = None,
    nuts_sampler: str = DEFAULT_NUTS_SAMPLER,
) -> EstimatorResult:
    """加速試験データに Full Bayesian MCMC (PyMC NUTS) を適用して t90(25°C) を推定.

    Parameters
    ----------
    seed : int, optional
        None なら `derive_mcmc_seed(case_id, replicate_id)` から自動派生 (推奨).
        明示指定は seed 再現性テスト用.
    """
    # 遅延インポート (PyMC は requirements-dev のみ).import の重さも遅延.
    with _warnings.catch_warnings():
        _warnings.filterwarnings("ignore", category=FutureWarning)
        import arviz as az
        import pymc as pm

    cols = column_names or {
        "temperature": "temperature",
        "time": "time_months",
        "response": "content_percent",
    }

    # 温度別グループ化
    groups: dict[float, list] = {}
    for row in data_rows:
        T_c = float(row[cols["temperature"]])
        t = float(row[cols["time"]])
        C = float(row[cols["response"]])
        groups.setdefault(T_c, []).append((t, C))

    temps_unique = sorted(groups.keys())
    n_t = len(temps_unique)
    T_K_arr = np.array([T + 273.15 for T in temps_unique], dtype=float)

    # c0 は各温度の t=0 値 (なければ initial_content)
    c0_by_temp: dict[float, float] = {}
    for T_c, pts in groups.items():
        t0 = min(pts, key=lambda x: x[0])
        c0_by_temp[T_c] = float(t0[1]) if t0[0] == 0 else initial_content

    times_obs: list[float] = []
    ln_ratio_obs: list[float] = []
    temp_idx_obs: list[int] = []
    temp_to_idx = {T_c: i for i, T_c in enumerate(temps_unique)}
    for T_c, pts in groups.items():
        for t, C in pts:
            if t == 0 or C <= 0:
                continue
            ln_ratio_obs.append(float(np.log(C / c0_by_temp[T_c])))
            times_obs.append(float(t))
            temp_idx_obs.append(temp_to_idx[T_c])

    if not times_obs or n_t < 2:
        return EstimatorResult(
            estimator_name=ESTIMATOR_NAME,
            case_id=case_id,
            replicate_id=replicate_id,
            t90_point_estimate_months=None,
            t90_lo95_months=None,
            t90_hi95_months=None,
            converged=False,
            error_code="INSUFFICIENT_TEMPERATURES",
            diagnostics={"n_t_observed": n_t},
            spec_lower_used=spec_lower,
        )

    times_obs_arr = np.array(times_obs)
    ln_ratio_obs_arr = np.array(ln_ratio_obs)
    temp_idx_obs_arr = np.array(temp_idx_obs)

    seed_used = seed if seed is not None else derive_mcmc_seed(case_id, replicate_id)

    try:
        with _warnings.catch_warnings():
            _warnings.filterwarnings("ignore")
            with pm.Model():
                ln_A = pm.Normal("ln_A", mu=20.0, sigma=100.0)
                Ea_kj = pm.Normal("Ea_kj", mu=prior_ea_kj, sigma=prior_ea_sd_kj)
                sigma_obs = pm.HalfNormal("sigma_obs", sigma=0.1)

                ln_k_temp = ln_A - Ea_kj * 1000.0 / (R_GAS * T_K_arr)
                k_temp = pm.math.exp(ln_k_temp)
                mu_pred = -k_temp[temp_idx_obs_arr] * times_obs_arr
                pm.Normal("y_obs", mu=mu_pred, sigma=sigma_obs, observed=ln_ratio_obs_arr)

                sample_kwargs = dict(
                    draws=draws, tune=tune, chains=chains,
                    target_accept=target_accept, random_seed=seed_used,
                    progressbar=False,
                )
                if nuts_sampler == "default":
                    # PyMC C backend: multiprocess cores
                    sample_kwargs["cores"] = chains
                else:
                    sample_kwargs["nuts_sampler"] = nuts_sampler
                idata = pm.sample(**sample_kwargs)
    except Exception as e:
        return EstimatorResult(
            estimator_name=ESTIMATOR_NAME,
            case_id=case_id,
            replicate_id=replicate_id,
            t90_point_estimate_months=None,
            t90_lo95_months=None,
            t90_hi95_months=None,
            converged=False,
            error_code="OTHER",
            diagnostics={"exception_class": type(e).__name__, "message": str(e)},
            spec_lower_used=spec_lower,
        )

    posterior = idata.posterior
    ln_A_samples = posterior["ln_A"].values.flatten()
    Ea_samples = posterior["Ea_kj"].values.flatten() * 1000.0  # J/mol

    T_target_K = target_temp_c + 273.15
    k_target = np.exp(ln_A_samples - Ea_samples / (R_GAS * T_target_K))
    log_ratio = -np.log(spec_lower / initial_content)
    t90_samples = log_ratio / k_target

    # cmc-platform 整合: 120 月 cap は raw posterior には適用しない.
    t90_point = float(np.mean(t90_samples))
    t90_lo = float(np.percentile(t90_samples, 2.5))
    t90_hi = float(np.percentile(t90_samples, 97.5))

    summary = az.summary(idata, var_names=["ln_A", "Ea_kj", "sigma_obs"])
    rhat_max = float(summary["r_hat"].max())
    ess_min = float(summary["ess_bulk"].min())

    converged = rhat_max < RHAT_THRESHOLD and ess_min > ESS_THRESHOLD
    error_code = None if converged else "MCMC_NOT_CONVERGED"

    return EstimatorResult(
        estimator_name=ESTIMATOR_NAME,
        case_id=case_id,
        replicate_id=replicate_id,
        t90_point_estimate_months=t90_point,
        t90_lo95_months=t90_lo,
        t90_hi95_months=t90_hi,
        converged=converged,
        error_code=error_code,
        diagnostics={
            "rhat_max": rhat_max,
            "ess_min": ess_min,
            "ea_post_mean": float(np.mean(Ea_samples) / 1000.0),
            "ea_post_sd": float(np.std(Ea_samples) / 1000.0),
            "n_obs": int(len(times_obs_arr)),
            "prior_ea_kj_used": prior_ea_kj,
            "prior_ea_sd_kj_used": prior_ea_sd_kj,
            "seed_used": int(seed_used),
            "draws": draws,
            "tune": tune,
            "chains": chains,
        },
        spec_lower_used=spec_lower,
    )
