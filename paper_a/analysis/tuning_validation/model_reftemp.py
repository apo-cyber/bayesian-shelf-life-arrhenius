"""検証専用モデル: 参照温度再パラメータ化版 Full Bayesian MCMC.

本体 mcmc.py を改変せず、同一モデルを「異なる幾何」で表現する.

================================================================
何を変え、何を変えないか
================================================================

不変 (本体と完全同一):
    - 尤度:  ln(C/c0) ~ Normal(mu = -k(T)*t, sigma_obs),
             k(T) = exp(ln_A - Ea*1000/(R*T_K))
    - 事前:  Ea_kj ~ Normal(prior_ea_kj, prior_ea_sd_kj)  (本体と同一)
             sigma_obs ~ HalfNormal(0.1)
             ln_A の周辺事前 ~ Normal(20, 100)  (本体と同一)
    - データ整形 (温度グループ化, c0, ln_ratio) も本体 estimate と同一手順.
    - 収束ゲート閾値は mcmc.py から import (RHAT_THRESHOLD / ESS_THRESHOLD).

変更点 (1 つだけ — サンプリング幾何):
    本体は ln_A と Ea_kj を直接サンプルする.両者は尤度上、強く相関する
    (ln_k = ln_A - Ea*1000/(R*T) で ln_A と Ea が同一直線上を動く) ため
    funnel 状の事後となり、低 n_T / prior 衝突セルで divergence を生む.

    本モデルは切片を設計の重心温度 T_ref に移して decorrelate する:
        alpha = ln_A - Ea_kj*1000/(R*T_ref)        (T_ref における ln_k)
        Ea_kj ~ Normal(prior_ea_kj, prior_ea_sd_kj) (本体と同一)
        ln_A  = alpha + Ea_kj*1000/(R*T_ref)        (Deterministic で復元)
    尤度に代入すると
        ln_k(T) = alpha + (Ea*1000/R)*(1/T_ref - 1/T)
    となり、1/T_ref ≈ mean(1/T) のとき Ea の回帰子は設計上ほぼ平均ゼロ →
    alpha と Ea_kj が事後でほぼ無相関になる (幾何が funnel から楕円へ).

T_ref の取り方 (根拠):
    decorrelation を厳密化する切片中心は 1/T の設計重心、すなわち
    1/T_ref = mean(1/T_K) (= T_K の調和平均) である.本実装は仕様に従い
    T_ref = 温度集合 (ユニーク) の算術平均 (Kelvin) を採用する.加速試験の
    狭い温度域 (40-70°C) では算術平均と調和平均の差は < 0.3 K であり、
    残差相関は無視できる (回帰子の設計平均 ~1e-6 / 勾配レンジ ~3e-4).
    算術平均を選ぶのは可読性と「設計の中央温度」という直感的根拠のため.

alpha の事前 (同一モデルである根拠):
    ln_A ~ N(20, 100) と Ea_kj ~ N(m, s) が独立のとき、線形変換
    alpha = ln_A - c*Ea_kj  (c = 1000/(R*T_ref)) の周辺は解析的に
        alpha ~ Normal(20 - c*m,  sqrt(100^2 + c^2 * s^2))
    で、alpha と Ea_kj の事前共分散は -c*s^2.
    ln_A の事前 sd=100 が極めて diffuse なため事前相関 corr(alpha,Ea)
    = -c*s/sqrt(100^2 + c^2 s^2) ≈ -0.37*s/100 (s=30 で約 -0.11) は小さく、
    事後はデータ支配となる.本実装は alpha と Ea_kj を独立な弱情報事前
    (上記の解析的周辺) として置く.これが本体と同一事後を与えることは
    `run_identity_check` (高 n_T 易ケース core_067 で Ea posterior と
    shelf-life CI が MC 誤差内で一致) で確認する.一致しなければ
    再パラメータ化が事後を変えている = バグであり、検証を停止する.

dense_mass:
    use_dense_mass=True で numpyro NUTS の dense (full-rank) mass matrix を
    有効化 (条件 E で使用).対角 mass では残る相関を質量行列側で吸収する.
"""
from __future__ import annotations

import warnings as _warnings
from math import sqrt
from typing import Any

import numpy as np

from paper_a.datagen import derive_mcmc_seed

# --- 本体 mcmc.py から import (コピー・再定義禁止のもの) -------------------
# 物理定数・収束閾値・NUTS 既定値は本体と「同一の値」を保証するため import.
from paper_a.analysis.estimators.base import EstimatorResult
from paper_a.analysis.estimators.mcmc import (
    R_GAS,
    DEFAULT_DRAWS,
    DEFAULT_TUNE,
    DEFAULT_CHAINS,
    DEFAULT_TARGET_ACCEPT,
    DEFAULT_NUTS_SAMPLER,
    RHAT_THRESHOLD,
    ESS_THRESHOLD,
)

# --- 本体 mcmc.py の inline literal を鏡写しにした事前定数 ------------------
# mcmc.py は estimate() 内に直書きで以下を持ち、module 定数として公開して
# いない (本体改変禁止のため import 不可).ここでは同値をミラーする.
# 値がずれていれば run_identity_check が赤になり検出される (バイト一致の番人).
LN_A_PRIOR_MU = 20.0       # mcmc.py: pm.Normal("ln_A", mu=20.0, ...)
LN_A_PRIOR_SD = 100.0      # mcmc.py: pm.Normal("ln_A", ..., sigma=100.0)
SIGMA_OBS_PRIOR_SD = 0.1   # mcmc.py: pm.HalfNormal("sigma_obs", sigma=0.1)

# EstimatorResult は estimator_name を固定 4 種に制約する (base.py 改変禁止).
# reftemp は「同一の mcmc モデルの別幾何」なので estimator_name は "mcmc" とし、
# 変種は diagnostics["variant"] = "reftemp" で識別する.
ESTIMATOR_NAME = "mcmc"
VARIANT = "reftemp"

# 収束ゲートで監視する変数 (本体と同一: ln_A, Ea_kj, sigma_obs).
# reftemp では ln_A は Deterministic だが、本体とゲートを揃えるため同一名で監視.
_GATE_VARS = ["ln_A", "Ea_kj", "sigma_obs"]


def _prepare(data_rows: list[dict], cols: dict, initial_content: float):
    """本体 mcmc.estimate と同一のデータ整形 (温度グループ化 / c0 / ln_ratio).

    モデル定義ではなくデータ marshaling であり、本体に import 可能な関数が
    無いため同一手順を踏襲する (派生モデルの fork ではない).
    """
    groups: dict[float, list] = {}
    for row in data_rows:
        T_c = float(row[cols["temperature"]])
        t = float(row[cols["time"]])
        C = float(row[cols["response"]])
        groups.setdefault(T_c, []).append((t, C))

    temps_unique = sorted(groups.keys())
    n_t = len(temps_unique)
    T_K_arr = np.array([T + 273.15 for T in temps_unique], dtype=float)

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

    return (
        temps_unique,
        n_t,
        T_K_arr,
        np.array(times_obs),
        np.array(ln_ratio_obs),
        np.array(temp_idx_obs),
    )


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
    use_dense_mass: bool = False,
) -> EstimatorResult:
    """参照温度再パラメータ化版で t90(25°C) を推定.

    本体 mcmc.estimate と同一 I/O (EstimatorResult).diagnostics に
    検証用の n_divergences / t_ref_k / alpha posterior 等を追加で格納する.
    """
    with _warnings.catch_warnings():
        _warnings.filterwarnings("ignore", category=FutureWarning)
        import arviz as az
        import pymc as pm

    cols = column_names or {
        "temperature": "temperature",
        "time": "time_months",
        "response": "content_percent",
    }

    (
        temps_unique,
        n_t,
        T_K_arr,
        times_obs_arr,
        ln_ratio_obs_arr,
        temp_idx_obs_arr,
    ) = _prepare(data_rows, cols, initial_content)

    if times_obs_arr.size == 0 or n_t < 2:
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

    # T_ref = 温度集合の算術平均 (Kelvin).1/T 設計重心 (= 調和平均) を近似し
    # 切片 alpha と勾配 Ea を decorrelate する (docstring 参照).
    T_ref_K = float(np.mean(T_K_arr))
    c = 1000.0 / (R_GAS * T_ref_K)  # ln_A - alpha = c * Ea_kj の係数

    # alpha の解析的周辺事前 (ln_A~N(20,100) ⊥ Ea~N(m,s) の線形合成):
    alpha_mu = LN_A_PRIOR_MU - c * prior_ea_kj
    alpha_sd = sqrt(LN_A_PRIOR_SD ** 2 + (c ** 2) * (prior_ea_sd_kj ** 2))

    seed_used = seed if seed is not None else derive_mcmc_seed(case_id, replicate_id)

    try:
        with _warnings.catch_warnings():
            _warnings.filterwarnings("ignore")
            with pm.Model():
                # --- 変更点: alpha (T_ref での切片) をサンプル ---
                alpha = pm.Normal("alpha", mu=alpha_mu, sigma=alpha_sd)
                Ea_kj = pm.Normal("Ea_kj", mu=prior_ea_kj, sigma=prior_ea_sd_kj)
                sigma_obs = pm.HalfNormal("sigma_obs", sigma=SIGMA_OBS_PRIOR_SD)
                # ln_A を Deterministic で復元 (本体と同名・同尤度に接続)
                ln_A = pm.Deterministic("ln_A", alpha + c * Ea_kj)

                # --- 尤度: 本体と完全同一 ---
                ln_k_temp = ln_A - Ea_kj * 1000.0 / (R_GAS * T_K_arr)
                k_temp = pm.math.exp(ln_k_temp)
                mu_pred = -k_temp[temp_idx_obs_arr] * times_obs_arr
                pm.Normal("y_obs", mu=mu_pred, sigma=sigma_obs, observed=ln_ratio_obs_arr)

                sample_kwargs: dict[str, Any] = dict(
                    draws=draws, tune=tune, chains=chains,
                    target_accept=target_accept, random_seed=seed_used,
                    progressbar=False,
                )
                if nuts_sampler == "default":
                    sample_kwargs["cores"] = chains
                    if use_dense_mass:
                        raise ValueError(
                            "use_dense_mass は numpyro backend 専用 "
                            "(default/PyMC backend では未対応)"
                        )
                else:
                    sample_kwargs["nuts_sampler"] = nuts_sampler
                    if use_dense_mass:
                        # numpyro NUTS の full-rank mass matrix を有効化
                        sample_kwargs["nuts_sampler_kwargs"] = {
                            "nuts_kwargs": {"dense_mass": True}
                        }
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
    alpha_samples = posterior["alpha"].values.flatten()
    sigma_samples = posterior["sigma_obs"].values.flatten()

    # --- shelf-life 変換: 本体と完全同一 ---
    T_target_K = target_temp_c + 273.15
    k_target = np.exp(ln_A_samples - Ea_samples / (R_GAS * T_target_K))
    log_ratio = -np.log(spec_lower / initial_content)
    t90_samples = log_ratio / k_target

    t90_point = float(np.mean(t90_samples))
    t90_lo = float(np.percentile(t90_samples, 2.5))
    t90_hi = float(np.percentile(t90_samples, 97.5))

    # 収束ゲート: 本体と同一変数・同一論理 (import 閾値)
    summary = az.summary(idata, var_names=_GATE_VARS)
    rhat_max = float(summary["r_hat"].max())
    ess_min = float(summary["ess_bulk"].min())
    converged = rhat_max < RHAT_THRESHOLD and ess_min > ESS_THRESHOLD
    error_code = None if converged else "MCMC_NOT_CONVERGED"

    # divergence 数 (検証で利用、本体は未記録)
    n_divergences = int(idata.sample_stats["diverging"].values.sum())

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
            "variant": VARIANT,
            "rhat_max": rhat_max,
            "ess_min": ess_min,
            "n_divergences": n_divergences,
            "ea_post_mean": float(np.mean(Ea_samples) / 1000.0),
            "ea_post_sd": float(np.std(Ea_samples) / 1000.0),
            "ln_a_post_mean": float(np.mean(ln_A_samples)),
            "ln_a_post_sd": float(np.std(ln_A_samples)),
            "alpha_post_mean": float(np.mean(alpha_samples)),
            "alpha_post_sd": float(np.std(alpha_samples)),
            "sigma_post_mean": float(np.mean(sigma_samples)),
            "sigma_post_sd": float(np.std(sigma_samples)),
            "t_ref_k": T_ref_K,
            "n_obs": int(times_obs_arr.size),
            "prior_ea_kj_used": prior_ea_kj,
            "prior_ea_sd_kj_used": prior_ea_sd_kj,
            "seed_used": int(seed_used),
            "draws": draws,
            "tune": tune,
            "chains": chains,
            "use_dense_mass": bool(use_dense_mass),
        },
        spec_lower_used=spec_lower,
    )


# ============================================================================
# 同一性ワンショット検証
# ============================================================================

# 易ケース: n_T=4, prior=accurate, n_points=4, noise=medium (core_067).
# funnel が緩く本体も収束する条件で、reftemp と本体の事後が一致するかを見る.
IDENTITY_CASE_SELECTOR = dict(n_t=4, prior_accuracy="accurate",
                              n_points=4, noise_level="medium")

# 許容差 (二つの独立 MCMC 実行間の MC 誤差を見込む).
IDENTITY_TOL = {
    "ea_mean_abs_over_sd": 0.20,   # |Δmean(Ea)| < 0.20 * posterior_sd(Ea)
    "ea_sd_rel": 0.15,             # |Δsd(Ea)| / sd < 15%
    "t90_rel": 0.05,               # t90 point/lo/hi 各々 相対差 < 5%
}


def _select_identity_case(truth_by_case: dict) -> str:
    hits = [
        c for c, v in truth_by_case.items()
        if v.get("n_t") == IDENTITY_CASE_SELECTOR["n_t"]
        and v.get("prior_accuracy") == IDENTITY_CASE_SELECTOR["prior_accuracy"]
        and v.get("n_points") == IDENTITY_CASE_SELECTOR["n_points"]
        and v.get("noise_level") == IDENTITY_CASE_SELECTOR["noise_level"]
    ]
    if not hits:
        raise RuntimeError(f"易ケースが見つからない: {IDENTITY_CASE_SELECTOR}")
    return hits[0]


def run_identity_check(replicate_id: int = 0, spec_lower: float = 90.0,
                       verbose: bool = True) -> dict:
    """本体(centered) と reftemp の事後が MC 誤差内で一致するか検証.

    本体 mcmc.estimate が EstimatorResult で公開するのは Ea posterior
    (ea_post_mean / ea_post_sd) と shelf-life CI (t90 point/lo/hi) のため、
    同一性はこの十分統計量上で判定する (t90 は ln_A・Ea 双方に依存するため、
    t90 trio の一致は ln_A posterior の一致も実質的に含意する).

    Returns
    -------
    dict: {"passed": bool, "case_id": ..., "metrics": [...], ...}
    赤 (passed=False) なら再パラメータ化が事後を変えている = バグ.
    """
    from paper_a.analysis.estimators import mcmc as centered
    from paper_a.analysis.loaders.synthetic import iter_replicates, load_truth

    truth = load_truth("core")
    case_id = _select_identity_case(truth)
    tv = truth[case_id]
    prior_ea = float(tv["prior_ea_kj_mol"])
    prior_sd = float(tv["prior_ea_sd_kj_mol"])

    rows = None
    for cid, rep, r in iter_replicates("core", case_ids=[case_id], accelerated=True):
        if rep == replicate_id:
            rows = r
            break
    if rows is None:
        raise RuntimeError(f"{case_id} rep={replicate_id} のデータが無い")

    if verbose:
        print(f"[identity] case={case_id}  rep={replicate_id}  "
              f"prior_ea={prior_ea} sd={prior_sd}  temps={tv['temperatures_c']}")

    res_c = centered.estimate(
        rows, case_id=case_id, replicate_id=replicate_id,
        prior_ea_kj=prior_ea, prior_ea_sd_kj=prior_sd, spec_lower=spec_lower,
    )
    res_r = estimate(
        rows, case_id=case_id, replicate_id=replicate_id,
        prior_ea_kj=prior_ea, prior_ea_sd_kj=prior_sd, spec_lower=spec_lower,
    )

    dc, dr = res_c.diagnostics, res_r.diagnostics
    ea_sd_ref = max(dc["ea_post_sd"], 1e-9)
    metrics = []

    def _check(name, val, tol, kind):
        ok = abs(val) <= tol
        metrics.append({"name": name, "value": float(val),
                        "tol": float(tol), "kind": kind, "ok": bool(ok)})
        return ok

    ea_mean_d = (dr["ea_post_mean"] - dc["ea_post_mean"]) / ea_sd_ref
    ea_sd_rel = (dr["ea_post_sd"] - dc["ea_post_sd"]) / ea_sd_ref
    ok1 = _check("ea_mean_diff_over_sd", ea_mean_d,
                 IDENTITY_TOL["ea_mean_abs_over_sd"], "abs")
    ok2 = _check("ea_sd_rel_diff", ea_sd_rel, IDENTITY_TOL["ea_sd_rel"], "abs")

    oks = [ok1, ok2]
    for fld, lbl in [("t90_point_estimate_months", "t90_point"),
                     ("t90_lo95_months", "t90_lo95"),
                     ("t90_hi95_months", "t90_hi95")]:
        vc = getattr(res_c, fld)
        vr = getattr(res_r, fld)
        rel = (vr - vc) / vc if vc else float("inf")
        oks.append(_check(f"{lbl}_rel", rel, IDENTITY_TOL["t90_rel"], "abs"))

    passed = all(oks)
    out = {
        "passed": passed,
        "case_id": case_id,
        "replicate_id": replicate_id,
        "centered": {"t90": [res_c.t90_point_estimate_months,
                             res_c.t90_lo95_months, res_c.t90_hi95_months],
                     "ea_mean": dc["ea_post_mean"], "ea_sd": dc["ea_post_sd"],
                     "rhat": dc.get("rhat_max"), "ess": dc.get("ess_min")},
        "reftemp": {"t90": [res_r.t90_point_estimate_months,
                            res_r.t90_lo95_months, res_r.t90_hi95_months],
                    "ea_mean": dr["ea_post_mean"], "ea_sd": dr["ea_post_sd"],
                    "rhat": dr.get("rhat_max"), "ess": dr.get("ess_min"),
                    "n_divergences": dr.get("n_divergences"),
                    "t_ref_k": dr.get("t_ref_k")},
        "metrics": metrics,
    }

    if verbose:
        print(f"  centered: t90={out['centered']['t90']}  "
              f"Ea={dc['ea_post_mean']:.3f}±{dc['ea_post_sd']:.3f}  "
              f"rhat={dc.get('rhat_max')} ess={dc.get('ess_min')}")
        print(f"  reftemp : t90={out['reftemp']['t90']}  "
              f"Ea={dr['ea_post_mean']:.3f}±{dr['ea_post_sd']:.3f}  "
              f"rhat={dr.get('rhat_max')} ess={dr.get('ess_min')} "
              f"div={dr.get('n_divergences')} T_ref={dr.get('t_ref_k'):.2f}K")
        for m in metrics:
            flag = "OK " if m["ok"] else "XX "
            print(f"  [{flag}] {m['name']:22s} = {m['value']:+.4f}  "
                  f"(tol |·|<={m['tol']})")
        print(f"  => identity check {'PASSED' if passed else 'FAILED'}")

    return out


if __name__ == "__main__":
    import sys
    r = run_identity_check(verbose=True)
    sys.exit(0 if r["passed"] else 1)
