"""Supplementary Table S1 (MCMC 非収束率の σ 別層別) を parquet から再集計.

S1 は cell_metrics.json のどのスライスにも対応せず (観測ノイズ σ で層別した
スライスが無い)、estimator_results.parquet と core 層の truth.json から直接
組み立てる必要がある。reaggregate.py と同じ流儀 (results/ を読んで results/ へ
JSON 出力・FileNotFoundError で依存順序を強制)。

集計の定義
----------
core 層の MCMC 推定行を (n_T, prior_accuracy, sigma_obs) で層別し、各セルで

    非収束率 = converged が真でない replicate 数 / replicate 総数 * 100  [%]

を報告する。サンプリング時点数 n_pts ∈ {3, 4, 6} は周辺化する。各セルは
3 case × 100 replicate = 300 runs。非収束の判定 (R-hat >= 1.01 または
bulk ESS < 400) は推定実行時に `converged` 列へ確定済みで、ここでは再判定しない。

`consistency_with_figure3` は各 (n_T, prior) 行を σ で周辺化した値で、原稿
S1 の脚注が Figure 3 の 9 セルと一致すると述べているもの。図と表が同じ母集団を
見ていることの検算として一緒に出力する。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from paper_a.analysis.loaders.synthetic import load_truth

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"

N_T_LEVELS = (2, 3, 4)
PRIOR_LEVELS = ("accurate", "moderate", "strong")
SIGMA_LEVELS = (0.01, 0.02, 0.05)


def _core_mcmc_frame(parquet_path: Path) -> pd.DataFrame:
    """core 層の MCMC 行に n_t / prior / sigma を付与して返す."""
    df = pd.read_parquet(parquet_path)
    truth = load_truth("core")

    mcmc = df[df["estimator_name"] == "mcmc"].copy()
    mcmc = mcmc[mcmc["case_id"].isin(truth)]
    mcmc["n_t"] = mcmc["case_id"].map(lambda c: truth[c]["n_t"])
    mcmc["prior_accuracy"] = mcmc["case_id"].map(lambda c: truth[c]["prior_accuracy"])
    mcmc["sigma_obs"] = mcmc["case_id"].map(lambda c: float(truth[c]["sigma_obs"]))
    return mcmc


def _nonconvergence_pct(frame: pd.DataFrame) -> tuple[float, int]:
    """非収束率 [%] と replicate 総数を返す."""
    n = len(frame)
    if n == 0:
        raise ValueError("該当 replicate が無い")
    n_nonconv = int((~frame["converged"].fillna(False).astype(bool)).sum())
    return 100.0 * n_nonconv / n, n


def compute_s1(parquet_path: Path | None = None) -> dict:
    path = parquet_path or (RESULTS_DIR / "estimator_results.parquet")
    if not path.exists():
        raise FileNotFoundError(f"{path} がない.run_paper_a を先に実行.")

    mcmc = _core_mcmc_frame(path)

    rows: list[dict] = []
    for n_t in N_T_LEVELS:
        for prior in PRIOR_LEVELS:
            row: dict = {"n_t": n_t, "prior_accuracy": prior, "by_sigma": {}}
            for sigma in SIGMA_LEVELS:
                sub = mcmc[
                    (mcmc["n_t"] == n_t)
                    & (mcmc["prior_accuracy"] == prior)
                    & (np.isclose(mcmc["sigma_obs"], sigma))
                ]
                pct, n = _nonconvergence_pct(sub)
                row["by_sigma"][f"{sigma}"] = {"nonconvergence_pct": pct, "n_replicates": n}
            marginal = mcmc[(mcmc["n_t"] == n_t) & (mcmc["prior_accuracy"] == prior)]
            pct, n = _nonconvergence_pct(marginal)
            row["marginalised_over_sigma"] = {"nonconvergence_pct": pct, "n_replicates": n}
            rows.append(row)

    return {
        "description": (
            "MCMC non-convergence rate (%) by n_T x prior accuracy x observation noise sigma, "
            "marginalised over the sampling-time grid n_pts in {3, 4, 6}."
        ),
        "rows": rows,
    }


def main() -> int:
    payload = compute_s1()

    out_path = RESULTS_DIR / "supp_table_s1.json"
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"→ {out_path}")

    print()
    print("| $n_T$ | Prior accuracy | $\\sigma = 0.01$ | $\\sigma = 0.02$ | $\\sigma = 0.05$ |")
    print("|:-----:|:---------------|----------------:|----------------:|----------------:|")
    for r in payload["rows"]:
        cells = " | ".join(
            f"{r['by_sigma'][f'{s}']['nonconvergence_pct']:.1f}" for s in SIGMA_LEVELS
        )
        print(f"| {r['n_t']} | {r['prior_accuracy']:<8} | {cells} |")

    print()
    print("Consistency with Figure 3 (σ で周辺化):")
    for n_t in N_T_LEVELS:
        vals = [
            r["marginalised_over_sigma"]["nonconvergence_pct"]
            for r in payload["rows"]
            if r["n_t"] == n_t
        ]
        print(f"  n_T = {n_t}: " + " / ".join(f"{v:.1f}" for v in vals) + "  (accurate / moderate / strong)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
