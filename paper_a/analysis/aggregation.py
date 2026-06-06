"""§B.3 必須スライス + cell ごとの指標集計.

スライス:
1. 低 n_T × Prior 強乖離 cell (核心命題立証).
2. n_T 別 (n_T=2/3/4).
3. Prior 別 (accurate/moderate/strong).
4. MCMC 収束失敗率を n_T × Prior cell ごと.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Iterable

from .metrics import CellMetrics, compute_cell_metrics


def slice_by(
    truth_by_case: dict[str, dict],
    *,
    layer: str = "core",
    n_t: int | None = None,
    prior_accuracy: str | None = None,
    noise_level: str | None = None,
    n_points: int | None = None,
    kinetics: str | None = None,
    k_of_t: str | None = None,
) -> set[str]:
    """指定条件にマッチする case_id のセットを返す."""
    out: set[str] = set()
    for case_id, t in truth_by_case.items():
        if t.get("layer") != layer:
            continue
        if n_t is not None and int(t.get("n_t", -1)) != n_t:
            continue
        if prior_accuracy is not None and t.get("prior_accuracy") != prior_accuracy:
            continue
        if noise_level is not None and t.get("noise_level") != noise_level:
            continue
        if n_points is not None and int(t.get("n_points", -1)) != n_points:
            continue
        if kinetics is not None and t.get("kinetics") != kinetics:
            continue
        if k_of_t is not None and t.get("k_of_t") != k_of_t:
            continue
        out.add(case_id)
    return out


def compute_all_slices(
    results_by_estimator: dict[str, list[dict]],
    truth_by_case: dict[str, dict],
) -> dict[str, list[CellMetrics]]:
    """全推定器・全スライスの CellMetrics を計算して dict で返す.

    Returns
    -------
    {
      "core_all":          [CellMetrics, ...],   # 4 推定器
      "core_low_n_t_strong_prior": [...],         # §B.3 核心
      "core_by_n_t":       [...],                # n_T 別 (各 3 cell × 4 推定器)
      "core_by_prior":     [...],                # Prior 別
      "robustness_all":    [...],
    }
    """
    out: dict[str, list[CellMetrics]] = defaultdict(list)

    # core: 全体
    core_cases = slice_by(truth_by_case, layer="core")
    for est_name, results in results_by_estimator.items():
        cell_results = [r for r in results if r["case_id"] in core_cases]
        out["core_all"].append(
            compute_cell_metrics(est_name, "core_all", cell_results, truth_by_case)
        )

    # core: 核心スライス (n_T=2 × Prior=strong)
    cell_cases = slice_by(truth_by_case, layer="core", n_t=2, prior_accuracy="strong")
    for est_name, results in results_by_estimator.items():
        cell_results = [r for r in results if r["case_id"] in cell_cases]
        out["core_low_n_t_strong_prior"].append(
            compute_cell_metrics(est_name, "n_t=2|prior=strong", cell_results, truth_by_case)
        )

    # core: n_T 別 × Prior 別 (3 × 3 = 9 cell × 4 推定器)
    for n_t in (2, 3, 4):
        for prior in ("accurate", "moderate", "strong"):
            cell_cases = slice_by(truth_by_case, layer="core", n_t=n_t, prior_accuracy=prior)
            for est_name, results in results_by_estimator.items():
                cell_results = [r for r in results if r["case_id"] in cell_cases]
                out["core_n_t_x_prior"].append(
                    compute_cell_metrics(
                        est_name,
                        f"n_t={n_t}|prior={prior}",
                        cell_results,
                        truth_by_case,
                    )
                )

    # robustness: all
    robust_cases = slice_by(truth_by_case, layer="robustness")
    for est_name, results in results_by_estimator.items():
        cell_results = [r for r in results if r["case_id"] in robust_cases]
        out["robustness_all"].append(
            compute_cell_metrics(est_name, "robustness_all", cell_results, truth_by_case)
        )

    # robustness: kinetics 別 (4 種)
    for kin in ("first_order", "second_order", "autocatalytic", "induction"):
        cell_cases = slice_by(truth_by_case, layer="robustness", kinetics=kin)
        for est_name, results in results_by_estimator.items():
            cell_results = [r for r in results if r["case_id"] in cell_cases]
            out["robustness_by_kinetics"].append(
                compute_cell_metrics(est_name, f"kinetics={kin}", cell_results, truth_by_case)
            )

    # robustness: 温度依存性別 (5 種)
    for kt in (
        "arrhenius",
        "modified_arrhenius_concave",
        "modified_arrhenius_convex",
    ):
        cell_cases = slice_by(truth_by_case, layer="robustness", k_of_t=kt)
        for est_name, results in results_by_estimator.items():
            cell_results = [r for r in results if r["case_id"] in cell_cases]
            out["robustness_by_k_of_t"].append(
                compute_cell_metrics(est_name, f"k_of_t={kt}", cell_results, truth_by_case)
            )

    # robustness: 個別 case (20 case × 4 推定器)
    for case_id in sorted(robust_cases):
        for est_name, results in results_by_estimator.items():
            cell_results = [r for r in results if r["case_id"] == case_id]
            out["robustness_per_case"].append(
                compute_cell_metrics(est_name, case_id, cell_results, truth_by_case)
            )

    return dict(out)
