"""エンドツーエンド解析 orchestration.

設計上の非対称性 (前停止点判断、案 B):
    - classical_ols_multi_temp / classical_ich_q1e / two_stage_conjugate:
      1000 reps/case (合成データ全反復).
    - mcmc: 100 reps/case (計算コスト上、Faya 2018 並み).

集計層は rep 数差を箱ひげの幅で自然に表現する.数値表では rep 数を明記.

出力:
    paper_a/results/estimator_results.parquet (4 推定器 × cell × rep)
    paper_a/results/cell_metrics.json         (集計 3 指標 + 補助)
    paper_a/figures/fig_t90_estimates_by_cell.png         (Faya 形式)
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from paper_a.analysis.aggregation import compute_all_slices
from paper_a.analysis.estimators import (
    classical_ich_q1e,
    classical_ols_multi_temp,
    mcmc as mcmc_estimator,
    two_stage_conjugate,
)
from paper_a.analysis.figures import (
    fig_t90_estimates_by_cell,
    fig_mcmc_nonconvergence_heatmap,
    fig_zoom_core_cell,
)
from paper_a.analysis.loaders.synthetic import iter_replicates, load_truth

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
FIGURES_DIR = Path(__file__).resolve().parents[1] / "figures"

# 推定器別 rep 数 (案 B): MCMC のみ 100 reps.他 3 推定器は 1000 reps.
REPS_PER_ESTIMATOR = {
    "classical_ols_multi_temp": 1000,
    "classical_ich_q1e": 1000,
    "two_stage_conjugate": 1000,
    "mcmc": 100,
}


def _run_acc(est_module, *, layer: str, truth_by_case: dict, max_reps: int, label: str) -> list[dict]:
    """加速試験データを使う 3 推定器の共通実行ループ."""
    out: list[dict] = []
    print(f"  [{label}] iter_replicates(layer={layer})  max_reps={max_reps}")
    t0 = time.time()
    n_done = 0
    for case_id, rep, rows in iter_replicates(layer, accelerated=True):
        if rep >= max_reps:
            continue
        truth = truth_by_case[case_id]
        if label == "two_stage_conjugate":
            result = est_module.estimate(
                rows, case_id=case_id, replicate_id=rep,
                prior_ea_kj=float(truth["prior_ea_kj_mol"]),
                prior_ea_sd_kj=float(truth["prior_ea_sd_kj_mol"]),
                spec_lower=90.0,
            )
        else:
            result = est_module.estimate(
                rows, case_id=case_id, replicate_id=rep, spec_lower=90.0,
            )
        out.append(result.to_dict())
        n_done += 1
    print(f"  [{label}] {n_done} 推定完了  経過 {time.time() - t0:.1f}s")
    return out


def _run_ich(est_module, *, layer: str, max_reps: int) -> list[dict]:
    """classical_ich_q1e は 25°C 長期試験データを使う."""
    out: list[dict] = []
    print(f"  [classical_ich_q1e] iter_replicates(layer={layer}, accelerated=False)")
    t0 = time.time()
    n_done = 0
    for case_id, rep, rows in iter_replicates(layer, accelerated=False):
        if rep >= max_reps:
            continue
        result = est_module.estimate(
            rows, case_id=case_id, replicate_id=rep, spec_lower=90.0,
        )
        out.append(result.to_dict())
        n_done += 1
    print(f"  [classical_ich_q1e] {n_done} 推定完了  経過 {time.time() - t0:.1f}s")
    return out


def _run_mcmc(*, layer: str, truth_by_case: dict, max_reps: int, nuts_sampler: str) -> list[dict]:
    """MCMC は 100 reps/case 上限.numpyro backend を採用."""
    out: list[dict] = []
    print(f"  [mcmc] iter_replicates(layer={layer})  max_reps={max_reps}  sampler={nuts_sampler}")
    t0 = time.time()
    n_done = 0
    for case_id, rep, rows in iter_replicates(layer, accelerated=True):
        if rep >= max_reps:
            continue
        truth = truth_by_case[case_id]
        result = mcmc_estimator.estimate(
            rows, case_id=case_id, replicate_id=rep,
            prior_ea_kj=float(truth["prior_ea_kj_mol"]),
            prior_ea_sd_kj=float(truth["prior_ea_sd_kj_mol"]),
            spec_lower=90.0,
            nuts_sampler=nuts_sampler,
        )
        out.append(result.to_dict())
        n_done += 1
        if n_done % 50 == 0:
            elapsed = time.time() - t0
            rate = n_done / elapsed
            print(f"    [mcmc] {n_done} 推定済 ({rate:.2f}/s, 経過 {elapsed:.0f}s)")
    print(f"  [mcmc] {n_done} 推定完了  経過 {time.time() - t0:.1f}s")
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--layer", choices=["core", "robustness", "both"], default="both")
    p.add_argument(
        "--max-reps-mcmc",
        type=int,
        default=REPS_PER_ESTIMATOR["mcmc"],
        help="MCMC の rep 上限 (default 100、案 B)",
    )
    p.add_argument(
        "--max-reps-classical",
        type=int,
        default=REPS_PER_ESTIMATOR["classical_ols_multi_temp"],
        help="他 3 推定器の rep 上限 (default 1000)",
    )
    p.add_argument(
        "--skip-mcmc",
        action="store_true",
        help="MCMC をスキップ (他 3 推定器のみ走らせる).図の動作確認時に使う.",
    )
    p.add_argument("--nuts-sampler", default="numpyro", choices=["numpyro", "default"])
    args = p.parse_args(argv)

    layers = ["core", "robustness"] if args.layer == "both" else [args.layer]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    all_results: dict[str, list[dict]] = {
        "classical_ols_multi_temp": [],
        "classical_ich_q1e": [],
        "two_stage_conjugate": [],
        "mcmc": [],
    }
    all_truth: dict[str, dict] = {}

    for layer in layers:
        print(f"\n=== layer={layer} ===")
        truth_by_case = load_truth(layer)
        all_truth.update(truth_by_case)

        all_results["classical_ols_multi_temp"].extend(_run_acc(
            classical_ols_multi_temp, layer=layer,
            truth_by_case=truth_by_case, max_reps=args.max_reps_classical,
            label="classical_ols_multi_temp",
        ))
        all_results["two_stage_conjugate"].extend(_run_acc(
            two_stage_conjugate, layer=layer,
            truth_by_case=truth_by_case, max_reps=args.max_reps_classical,
            label="two_stage_conjugate",
        ))
        all_results["classical_ich_q1e"].extend(_run_ich(
            classical_ich_q1e, layer=layer, max_reps=args.max_reps_classical,
        ))
        if not args.skip_mcmc:
            all_results["mcmc"].extend(_run_mcmc(
                layer=layer, truth_by_case=truth_by_case,
                max_reps=args.max_reps_mcmc, nuts_sampler=args.nuts_sampler,
            ))

    # parquet 出力 (4 推定器 × case × rep)
    rows: list[dict] = []
    for est_name, results in all_results.items():
        for r in results:
            row = dict(r)
            # diagnostics は JSON 文字列にして parquet 互換に
            row["diagnostics"] = json.dumps(row.get("diagnostics", {}), ensure_ascii=False)
            rows.append(row)
    df = pd.DataFrame(rows)
    parquet_path = RESULTS_DIR / "estimator_results.parquet"
    df.to_parquet(parquet_path, index=False)
    print(f"\n→ {parquet_path}  ({len(df)} 推定行)")

    # 集計
    metrics_by_slice = compute_all_slices(all_results, all_truth)
    metrics_json_path = RESULTS_DIR / "cell_metrics.json"
    metrics_json_path.write_text(json.dumps(
        {k: [asdict(c) for c in cells] for k, cells in metrics_by_slice.items()},
        indent=2, ensure_ascii=False,
    ))
    print(f"→ {metrics_json_path}")

    # t90 推定値分布図 (全 9 cell、Faya 2018 Fig 8 と同形式)
    fig_path = fig_t90_estimates_by_cell(
        results_by_estimator=all_results,
        truth_by_case=all_truth,
        output_path=FIGURES_DIR / "fig_t90_estimates_by_cell.png",
    )
    print(f"→ {fig_path}")

    # 拡大図: n_T=3 × Prior 3 水準 (D.3 追加要求 a、核心 cell)
    zoom_path = fig_zoom_core_cell(
        results_by_estimator=all_results,
        truth_by_case=all_truth,
        n_t=3,
        output_path=FIGURES_DIR / "fig_zoom_n_t_3.png",
    )
    print(f"→ {zoom_path}")

    # MCMC 非収束率ヒートマップ (D.3 追加要求 a、論文 Figure 3 候補)
    heatmap_path = fig_mcmc_nonconvergence_heatmap(
        metrics_by_slice,
        output_path=FIGURES_DIR / "fig_mcmc_nonconvergence.png",
    )
    print(f"→ {heatmap_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
