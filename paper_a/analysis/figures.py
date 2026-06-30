"""t90 推定値分布図 (主指標、Faya 2018 Fig 8 と同形式).

横軸: cell 群 (例: n_T 別 × Prior 別).
縦軸: t90 推定値の 95% 区間 (CI cap=120 月で打ち切り表示).
4 推定器を色分け.真 t90 を水平線で表示.

cap: 可視化のみ raw → 120 月で打ち切り、数値は raw 保持.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .metrics import SHELF_LIFE_CAP_MONTHS

ESTIMATOR_COLORS = {
    "two_stage_conjugate": "#2563eb",       # blue (フィードバックウィジェットと整合)
    "mcmc": "#ef4444",                       # red
    "classical_ols_multi_temp": "#16a34a",   # green
    "classical_ich_q1e": "#a855f7",          # purple
}


def _save_png_and_tiff(fig, output_path: Path) -> None:
    """査読用 PNG (dpi=150, 従来どおり) と、ポータル図アップロード用の
    300dpi LZW-TIFF (同一 stem の .tiff) を併産する.

    PNG は docx 埋め込みの正本 (内容・dpi 不変). TIFF はカラー図の IFA 要件
    (300dpi) を満たす production 解像度の姉妹ファイル. 描画内容は両者同一で、
    cell_metrics.json からの再描画のみ (MCMC 再実行不要)."""
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    tiff_path = output_path.with_suffix(".tiff")
    fig.savefig(
        tiff_path,
        dpi=300,
        bbox_inches="tight",
        format="tiff",
        pil_kwargs={"compression": "tiff_lzw"},
    )


def fig_t90_estimates_by_cell(
    results_by_estimator: dict[str, list[dict]],
    truth_by_case: dict[str, dict],
    *,
    cells: list[tuple[int, str]] | None = None,
    cap_months: float = SHELF_LIFE_CAP_MONTHS,
    truth_field: str = "t90_true_25c_months",
    output_path: Path | None = None,
    ylim: tuple[float, float] | None = None,
    ylabel: str | None = None,
) -> Path:
    """t90 推定値分布図を出力 (Faya 2018 Fig 8 と同形式).

    図内タイトルは付さない (journal 慣習: 説明は manuscript の caption に委譲).

    Parameters
    ----------
    cells : list of (n_t, prior_accuracy) tuples
        横軸の cell.None で全 n_T (2,3,4) × prior (accurate/moderate/strong)
        の 9 cell.
    cap_months : float
        可視化 cap (raw 値ではなく表示のみ).
    """
    if cells is None:
        cells = [(n_t, p) for n_t in (2, 3, 4) for p in ("accurate", "moderate", "strong")]

    # 真値 (代表値): core 81 は target_sl=30 × first_order で全 case 同じ
    # t90_true_25c=61.6224 月になる設計.uniformity を確認して単一値として扱う.
    # 頑健性 20 case は kinetics 別に異なるが、本図は core 中心の評価.
    core_truth_values = [
        float(truth_by_case[c][truth_field])
        for c in truth_by_case if truth_by_case[c].get("layer") == "core"
    ]
    if not core_truth_values:
        truth_value = 0.0
        truth_uniform = True
    else:
        vmin, vmax = min(core_truth_values), max(core_truth_values)
        truth_uniform = (vmax - vmin) < 1e-6
        truth_value = core_truth_values[0] if truth_uniform else float(np.median(core_truth_values))

    estimators = list(ESTIMATOR_COLORS.keys())
    n_cells = len(cells)
    n_est = len(estimators)
    fig_width = max(8.0, n_cells * 1.2)
    fig, ax = plt.subplots(figsize=(fig_width, 5.5))

    # cell × estimator → 推定値の分布 (rep 集合)
    cell_x_positions: list[float] = []
    cell_labels: list[str] = []
    for ci, (n_t, prior) in enumerate(cells):
        cell_center = float(ci)
        cell_x_positions.append(cell_center)
        cell_labels.append(f"n_T={n_t}\nprior={prior[:3]}")

        for ei, est_name in enumerate(estimators):
            # cell に該当する case_id を抽出
            case_ids = {
                cid for cid, t in truth_by_case.items()
                if t.get("layer") == "core"
                and int(t.get("n_t", -1)) == n_t
                and t.get("prior_accuracy") == prior
            }
            est_results = [
                r for r in results_by_estimator.get(est_name, [])
                if r["case_id"] in case_ids and r["error_code"] is None
                and r["t90_point_estimate_months"] is not None
            ]
            if not est_results:
                continue
            ests = np.clip(
                np.array([r["t90_point_estimate_months"] for r in est_results]),
                0.0,
                cap_months,
            )
            offset = (ei - (n_est - 1) / 2) * 0.18
            x_pos = cell_center + offset
            # 箱ひげ風: median + IQR + min/max
            med = float(np.median(ests))
            q25 = float(np.percentile(ests, 25))
            q75 = float(np.percentile(ests, 75))
            mn = float(np.min(ests))
            mx = float(np.max(ests))
            color = ESTIMATOR_COLORS[est_name]
            # whisker
            ax.plot([x_pos, x_pos], [mn, mx], color=color, alpha=0.4, linewidth=1)
            # IQR box
            ax.bar(
                x_pos,
                q75 - q25,
                bottom=q25,
                width=0.14,
                color=color,
                alpha=0.35,
                edgecolor=color,
                linewidth=1,
            )
            # median
            ax.plot([x_pos - 0.07, x_pos + 0.07], [med, med], color=color, linewidth=2)

    # 真値の水平線 (core 81 は uniform=True、頑健性混在の場合のみ "median" 表記)
    truth_label_prefix = "true t90 (uniform across core)" if truth_uniform else "true t90 (median)"
    ax.axhline(
        truth_value,
        color="black",
        linestyle="--",
        linewidth=1,
        label=f"{truth_label_prefix} = {truth_value:.2f} mo",
    )
    ax.axhline(
        cap_months,
        color="gray",
        linestyle=":",
        linewidth=0.8,
        label=f"display cap = {cap_months:.0f} mo",
    )

    ax.set_xticks(cell_x_positions)
    ax.set_xticklabels(cell_labels, fontsize=9)
    ax.set_ylabel(
        ylabel if ylabel is not None
        else "t90 estimate at 25°C (months, capped at 120 for display)"
    )
    if ylim is not None:
        ax.set_ylim(*ylim)
    else:
        ax.set_ylim(0, cap_months * 1.05)
    ax.grid(True, alpha=0.3)

    # 凡例: 推定器の色 + 真値線
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=c, alpha=0.5, label=name)
        for name, c in ESTIMATOR_COLORS.items()
    ]
    handles.append(
        plt.Line2D([0], [0], color="black", linestyle="--", label=f"true t90 = {truth_value:.2f}")
    )
    ax.legend(handles=handles, loc="upper right", fontsize=8, ncol=1)

    if output_path is None:
        output_path = Path("paper_a/figures") / "fig_t90_estimates_by_cell.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    _save_png_and_tiff(fig, output_path)
    plt.close(fig)
    return output_path


def fig_zoom_core_cell(
    results_by_estimator: dict[str, list[dict]],
    truth_by_case: dict[str, dict],
    *,
    n_t: int = 3,
    cap_months: float = SHELF_LIFE_CAP_MONTHS,
    truth_field: str = "t90_true_25c_months",
    output_path: Path | None = None,
    ylim: tuple[float, float] | None = (30, 110),
    ylabel: str | None = "t90 estimate at 25°C (months)",
) -> Path:
    """核心 cell (n_T 固定 × Prior 3 水準) の拡大図.追加要求 (a).

    zoom 図は中央コア層 (n_T=3) の差分を見るため y 軸を (30, 110) に拡大表示する.
    値自体は cap_months で clip 済みであり、mcmc 箱が上端でクリップされるのは
    zoom の意図どおり (full 図 Figure 1 は cap 全域を表示する).
    """
    cells = [(n_t, p) for p in ("accurate", "moderate", "strong")]
    if output_path is None:
        output_path = Path("paper_a/figures") / f"fig_zoom_n_t_{n_t}.png"
    return fig_t90_estimates_by_cell(
        results_by_estimator=results_by_estimator,
        truth_by_case=truth_by_case,
        cells=cells,
        cap_months=cap_months,
        truth_field=truth_field,
        output_path=output_path,
        ylim=ylim,
        ylabel=ylabel,
    )


def fig_mcmc_nonconvergence_heatmap(
    metrics_by_slice: dict,
    output_path: Path | None = None,
) -> Path:
    """MCMC 非収束率の n_T × Prior ヒートマップ (追加要求 a、論文 Figure 3 候補).

    Faya 2018 が報告していない独立した新規貢献の可視化.
    """
    # core_n_t_x_prior から MCMC エントリだけ抜き出す
    cells = metrics_by_slice.get("core_n_t_x_prior", [])
    n_t_values = [2, 3, 4]
    prior_values = ["accurate", "moderate", "strong"]
    heatmap = np.full((len(n_t_values), len(prior_values)), np.nan)

    for c in cells:
        # CellMetrics dataclass / dict 両対応
        est = c.estimator_name if hasattr(c, "estimator_name") else c["estimator_name"]
        if est != "mcmc":
            continue
        key = c.cell_key if hasattr(c, "cell_key") else c["cell_key"]
        nc = c.mcmc_nonconverged_rate if hasattr(c, "mcmc_nonconverged_rate") else c["mcmc_nonconverged_rate"]
        try:
            n_t_part, prior_part = key.split("|")
            n_t_val = int(n_t_part.split("=")[1])
            prior_val = prior_part.split("=")[1]
            i = n_t_values.index(n_t_val)
            j = prior_values.index(prior_val)
            heatmap[i, j] = nc * 100.0
        except (ValueError, KeyError):
            continue

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(heatmap, cmap="Reds", vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(range(len(prior_values)))
    ax.set_xticklabels([f"prior={p}" for p in prior_values])
    ax.set_yticks(range(len(n_t_values)))
    ax.set_yticklabels([f"n_T={n}" for n in n_t_values])

    # セル内に数値表示
    for i in range(len(n_t_values)):
        for j in range(len(prior_values)):
            val = heatmap[i, j]
            txt = "—" if np.isnan(val) else f"{val:.1f}%"
            color = "white" if (not np.isnan(val) and val > 50) else "black"
            ax.text(j, i, txt, ha="center", va="center", color=color, fontsize=11)

    fig.colorbar(im, ax=ax, label="non-convergence rate (%)")

    if output_path is None:
        output_path = Path("paper_a/figures") / "fig_mcmc_nonconvergence.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    _save_png_and_tiff(fig, output_path)
    plt.close(fig)
    return output_path
