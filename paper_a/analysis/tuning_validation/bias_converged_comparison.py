"""読み取り専用: two_stage_conjugate vs mcmc を「全reps」と「収束ケースのみ」で比較.

論点 (荒井さん経由の確認依頼):
    Paper A 本文の「二段階推定 vs MCMC」の優位性が、MCMC の非収束ケースを
    除外した収束ケースのみの比較でも保たれるか.

重要な前提確認 (metrics.py の挙動):
    metrics.py は error_code != None を「失敗」として bias_median から除外する.
    MCMC の非収束 rep は error_code="MCMC_NOT_CONVERGED" を持つため、
    **本文の bias_median は既に収束ケースのみ** で計算されている
    (n_reps_success == n_reps_converged_only).したがって本スクリプトの
    「全reps」は、非収束 rep の点推定も含めた (本文では未報告の) より楽観的な
    視点であり、「収束ケースのみ」が本文値に対応する.

定義:
    - bias = t90_point_estimate_months - t90_true_25c_months  (生値、月)
      正のバイアス = 真の shelf-life を過大評価 = 反保守的 (危険側).
    - 真値フィールドは metrics.py と同一 (t90_true_25c_months).
    - dispersion は外れ値で mean/SD が発散するため (mcmc は exp 変換で t90 が
      爆発しうる)、cap あり版を報告:
        est_capped = min(est, SHELF_LIFE_CAP_MONTHS)   (metrics.py から import)
        IQR_cap / SD_cap は bias_capped 上で算出.bias_med は生値の中央値 (頑健).
    - 全reps     : t90 点推定が非 null の全 rep (converged 不問).
    - 収束ケースのみ: converged==True の rep (= 本文の対象集合).

本スクリプトは estimator_results.parquet を後付け集計するのみ.
estimators/ ・ metrics.py ・ vendor/ は不改変、cap 定数は import 再利用.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

# cap 定数は metrics.py から import 再利用 (再定義禁止)
from paper_a.analysis.metrics import SHELF_LIFE_CAP_MONTHS

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results" / "estimator_results.parquet"
DATA = ROOT / "data"
TRUTH_FIELD = "t90_true_25c_months"  # metrics.py の主指標と同一
ESTIMATORS = ["two_stage_conjugate", "mcmc"]


def load() -> pd.DataFrame:
    df = pd.read_parquet(RESULTS)
    truth: dict[str, float] = {}
    n_t_map: dict[str, int] = {}
    layer_map: dict[str, str] = {}
    for layer in ("core", "robustness"):
        for c in json.load(open(DATA / layer / "truth.json"))["cases"]:
            truth[c["case_id"]] = c[TRUTH_FIELD]
            n_t_map[c["case_id"]] = c.get("n_t")
            layer_map[c["case_id"]] = layer
    df = df[df["estimator_name"].isin(ESTIMATORS)].copy()
    df["t90_true"] = df["case_id"].map(truth)
    df["n_t"] = df["case_id"].map(n_t_map)
    df["layer"] = df["case_id"].map(layer_map)
    return df


def _stats(sub: pd.DataFrame, n_total: int) -> dict:
    """点推定を持つ rep 集合 sub から bias 統計を算出.n_total は分母 (適用率用)."""
    have = sub[sub["t90_point_estimate_months"].notna() & sub["t90_true"].notna()]
    n = len(have)
    if n == 0:
        return {"n": 0, "bias_med": None, "iqr_cap": None, "sd_cap": None,
                "applic": 0.0}
    est = have["t90_point_estimate_months"].to_numpy(dtype=float)
    truth = have["t90_true"].to_numpy(dtype=float)
    bias = est - truth
    est_cap = np.minimum(est, SHELF_LIFE_CAP_MONTHS)
    bias_cap = est_cap - truth
    q25, q75 = np.percentile(bias_cap, [25, 75])
    return {
        "n": n,
        "bias_med": float(np.median(bias)),
        "iqr_cap": float(q75 - q25),
        "sd_cap": float(np.std(bias_cap, ddof=1)) if n > 1 else None,
        "applic": n / n_total if n_total else 0.0,
    }


def aggregate_view(df: pd.DataFrame, label: str) -> list[dict]:
    rows = []
    for est in ESTIMATORS:
        s = df[df["estimator_name"] == est]
        n_total = len(s)
        n_conv = int((s["converged"] == True).sum())  # noqa: E712
        allv = _stats(s, n_total)
        cvv = _stats(s[s["converged"] == True], n_total)  # noqa: E712
        rows.append({
            "scope": label, "estimator": est, "n_total": n_total,
            "n_converged": n_conv,
            "nonconv_rate": (n_total - n_conv) / n_total if n_total else 0.0,
            "all": allv, "conv": cvv,
        })
    return rows


def print_aggregate_table(rows: list[dict]):
    print("\n## 表1. 全シナリオ集約: two_stage vs mcmc — 全reps と 収束ケースのみ")
    print("(bias_med: 月、正=過大評価=反保守的 / IQR_cap・SD_cap: cap120後の散布 / applic: 点推定が得られた率)\n")
    hdr = (f"| {'scope':16s} | estimator | view | n | bias_med | IQR_cap | SD_cap | applic% |")
    print(hdr)
    print("|" + "---|" * 8)
    for r in rows:
        for view in ("all", "conv"):
            v = r[view]
            vlabel = "全reps" if view == "all" else "収束のみ"
            sd = "—" if v["sd_cap"] is None else f"{v['sd_cap']:.1f}"
            bm = "—" if v["bias_med"] is None else f"{v['bias_med']:+.2f}"
            iqr = "—" if v["iqr_cap"] is None else f"{v['iqr_cap']:.1f}"
            print(f"| {r['scope']:16s} | {r['estimator']:18s} | {vlabel:6s} "
                  f"| {v['n']:6d} | {bm:>8} | {iqr:>7} | {sd:>6} | {v['applic']*100:5.1f} |")
        if r["estimator"] == "mcmc":
            print(f"|   └ mcmc 非収束率: {r['nonconv_rate']*100:.1f}% "
                  f"(n_total={r['n_total']}, n_converged={r['n_converged']})"
                  + " " * 20 + "|")


def print_nt_table(df: pd.DataFrame):
    """n_T 別 (core 層、n_T が設計軸として変動する). two_stage vs mcmc(収束のみ)."""
    print("\n## 表2. n_T 別層別 (core 層): two_stage(全=収束) vs mcmc(収束ケースのみ)")
    print("(低 n_T で mcmc の収束サンプルが枯れていないかを確認)\n")
    print(f"| n_T | estimator | view | n (=収束数) | bias_med | IQR_cap | mcmc非収束率 |")
    print("|" + "---|" * 6)
    core = df[df["layer"] == "core"]
    for n_t in (2, 3, 4):
        sub = core[core["n_t"] == n_t]
        # two_stage: 収束概念なし → 全rep (n_T<3 では N_CONDS_TOO_LOW で適用不可)
        ts = sub[sub["estimator_name"] == "two_stage_conjugate"]
        ts_v = _stats(ts, len(ts))
        ts_bm = "適用不可" if ts_v["bias_med"] is None else f"{ts_v['bias_med']:+.2f}"
        ts_iqr = "—" if ts_v["iqr_cap"] is None else f"{ts_v['iqr_cap']:.1f}"
        print(f"| {n_t} | two_stage_conjugate | 全rep | {ts_v['n']:5d} | "
              f"{ts_bm:>7} | {ts_iqr:>6} | — |")
        # mcmc: 収束ケースのみ
        mc = sub[sub["estimator_name"] == "mcmc"]
        mc_total = len(mc)
        mc_conv = mc[mc["converged"] == True]  # noqa: E712
        mc_v = _stats(mc_conv, mc_total)
        ncr = (mc_total - len(mc_conv)) / mc_total if mc_total else 0.0
        bm = "—" if mc_v["bias_med"] is None else f"{mc_v['bias_med']:+.2f}"
        iqr = "—" if mc_v["iqr_cap"] is None else f"{mc_v['iqr_cap']:.1f}"
        print(f"| {n_t} | mcmc                | 収束のみ | {mc_v['n']:5d} | "
              f"{bm:>7} | {iqr:>6} | {ncr*100:.1f}% (conv {len(mc_conv)}/{mc_total}) |")


def main() -> int:
    df = load()
    rows = []
    rows += aggregate_view(df, "pooled (core+robust)")
    rows += aggregate_view(df[df["layer"] == "core"], "core のみ")
    rows += aggregate_view(df[df["layer"] == "robustness"], "robustness のみ")
    print_aggregate_table(rows)
    print_nt_table(df)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
