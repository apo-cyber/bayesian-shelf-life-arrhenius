"""調律検証マトリクスの集計・事前登録判定.

入力: paper_a/results/tuning_validation/[<subdir>/]raw_{cond}_{cell}.jsonl
出力 (図は出さない — figures は paper-A 側の領分):
    - tuning_summary.csv       : (condition, cell) 毎の全統計
    - tuning_summary.md        : 条件×セルの nonconv_rate[95%CI] ピボット + 内訳
    - tuning_verdicts.md       : 事前登録判定を機械適用した verdict 文字列

各 (condition, cell) で算出:
    nonconv_rate, 内訳 (rhat 起因% / ess 起因% / both% / error%),
    median divergences (reftemp 条件のみ — centered は本体未公開で null),
    二項 Wilson 95% CI.

事前登録判定 (本文の主張を機械的に守る番人):
    中心セル nt3_strong  E: <10%→rescued_by_tuning / 10-20%→partial / >20%→structurally_fragile
    最悪セル nt2_strong  E: >40%→identification_failure_irreducible
    反証      nt3_strong  B 単独で baseline(A) 比 大幅低下 → WARN artifact_of_parameterization
    prior 衝突 nt2_strong vs nt2_accurate の差分で寄与を定量
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results" / "tuning_validation"

Z95 = 1.959963984540054  # 標準正規 97.5% 分位

CONDITION_ORDER = ["A", "B", "C", "D", "E"]
CELL_ORDER = ["nt3_strong", "nt2_strong", "nt2_accurate"]

# --- 事前登録した閾値 (本文主張に直結、ここで一元管理) ---
CENTER_CELL = "nt3_strong"
WORST_CELL = "nt2_strong"
DIAG_CELL = "nt2_accurate"
V_RESCUED_MAX = 0.10       # nt3_strong E < 10% → rescued
V_PARTIAL_MAX = 0.20       # 10-20% → partial、>20% → structurally_fragile
V_IRREDUCIBLE_MIN = 0.40   # nt2_strong E > 40% → identification_failure
# 反証 (parameterization artifact): B が A 比で大幅に下げ、かつ B 自体が低い
V_ARTIFACT_B_MAX = 0.15
V_ARTIFACT_DROP_MIN = 0.20


def wilson_ci(k: int, n: int, z: float = Z95) -> tuple[float, float]:
    """二項割合 k/n の Wilson score 95% CI."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def _median(xs: list[float]):
    if not xs:
        return None
    s = sorted(xs)
    m = len(s)
    mid = m // 2
    return float(s[mid]) if m % 2 else float((s[mid - 1] + s[mid]) / 2)


def load_cells(in_dir: Path) -> dict[tuple[str, str], list[dict]]:
    """raw_*.jsonl を (condition, cell) -> records にロード."""
    cells: dict[tuple[str, str], list[dict]] = {}
    for path in sorted(in_dir.glob("raw_*.jsonl")):
        recs = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
        if not recs:
            continue
        cond = recs[0].get("condition")
        cell = recs[0].get("cell")
        cells.setdefault((cond, cell), []).extend(recs)
    return cells


def summarize(recs: list[dict]) -> dict:
    n = len(recs)
    nonconv = [r for r in recs if not r["converged"]]
    k = len(nonconv)
    lo, hi = wilson_ci(k, n)

    # 内訳 (非収束のうち)
    reasons = {"rhat": 0, "ess": 0, "both": 0, "error": 0}
    for r in nonconv:
        fr = r.get("fail_reason")
        if fr in reasons:
            reasons[fr] += 1
        else:
            reasons["error"] += 1
    breakdown = {key: (reasons[key] / k if k else 0.0) for key in reasons}

    divs = [r["n_divergences"] for r in recs if r.get("n_divergences") is not None]
    med_div = _median(divs)

    return {
        "n": n,
        "n_nonconv": k,
        "nonconv_rate": k / n if n else 0.0,
        "ci_lo": lo,
        "ci_hi": hi,
        "frac_rhat": breakdown["rhat"],
        "frac_ess": breakdown["ess"],
        "frac_both": breakdown["both"],
        "frac_error": breakdown["error"],
        "median_divergences": med_div,
        "divergences_available": len(divs) > 0,
        "model": recs[0].get("model"),
    }


def build_verdicts(stats: dict[tuple[str, str], dict]) -> dict:
    """事前登録判定を機械適用."""
    v: dict = {}

    def rate(cond, cell):
        s = stats.get((cond, cell))
        return s["nonconv_rate"] if s else None

    # 1) 中心セル nt3_strong, 条件 E
    e_center = rate("E", CENTER_CELL)
    if e_center is None:
        v["center_E"] = "N/A (nt3_strong×E のデータ無し)"
    elif e_center < V_RESCUED_MAX:
        v["center_E"] = f"rescued_by_tuning (nonconv={e_center:.1%} < {V_RESCUED_MAX:.0%})"
    elif e_center <= V_PARTIAL_MAX:
        v["center_E"] = f"partial (nonconv={e_center:.1%} in [{V_RESCUED_MAX:.0%},{V_PARTIAL_MAX:.0%}])"
    else:
        v["center_E"] = f"structurally_fragile (nonconv={e_center:.1%} > {V_PARTIAL_MAX:.0%})"

    # 2) 最悪セル nt2_strong, 条件 E
    e_worst = rate("E", WORST_CELL)
    if e_worst is None:
        v["worst_E"] = "N/A (nt2_strong×E のデータ無し)"
    elif e_worst > V_IRREDUCIBLE_MIN:
        v["worst_E"] = f"identification_failure_irreducible (nonconv={e_worst:.1%} > {V_IRREDUCIBLE_MIN:.0%})"
    else:
        v["worst_E"] = f"not_irreducible (nonconv={e_worst:.1%} <= {V_IRREDUCIBLE_MIN:.0%})"

    # 3) 反証チェック: B (reftemp 単独) が baseline A 比で中心セルを大幅低下
    a_center = rate("A", CENTER_CELL)
    b_center = rate("B", CENTER_CELL)
    if a_center is not None and b_center is not None:
        drop = a_center - b_center
        if b_center < V_ARTIFACT_B_MAX and drop >= V_ARTIFACT_DROP_MIN:
            v["artifact_check"] = (
                f"WARN: artifact_of_parameterization — A={a_center:.1%} → B={b_center:.1%} "
                f"(drop={drop:.1%}).中心セルの非収束が幾何だけで解消 → "
                f"本文の『構造的失敗』主張を要緩和 (tuning で救えるなら構造的でない)"
            )
        else:
            v["artifact_check"] = (
                f"no_artifact (A={a_center:.1%} → B={b_center:.1%}, drop={drop:.1%}); "
                f"幾何だけでは中心セルを基準まで救えない → 構造的主張は維持可"
            )
    else:
        v["artifact_check"] = "N/A (A または B の nt3_strong データ無し)"

    # 4) prior 衝突寄与: nt2_strong vs nt2_accurate (条件 A, E で算出可)
    contrib = {}
    for cond in ("A", "E"):
        rs = rate(cond, WORST_CELL)       # n_T=2, prior strong
        ra = rate(cond, DIAG_CELL)        # n_T=2, prior accurate
        if rs is not None and ra is not None:
            diff = rs - ra
            if ra >= 0.20 and diff < 0.10:
                interp = "identification_dominant (accurate でも高非収束 → n_T 不足が主因)"
            elif diff >= 0.15:
                interp = "prior_conflict_dominant (strong のみ高い → 事前衝突が主因)"
            else:
                interp = "mixed"
            contrib[cond] = {
                "nt2_strong": rs, "nt2_accurate": ra,
                "prior_conflict_contribution": diff, "interpretation": interp,
            }
        else:
            contrib[cond] = "N/A"
    v["prior_conflict"] = contrib

    return v


def write_csv(stats: dict[tuple[str, str], dict], path: Path):
    cols = ["condition", "cell", "model", "n", "n_nonconv", "nonconv_rate",
            "ci_lo", "ci_hi", "frac_rhat", "frac_ess", "frac_both", "frac_error",
            "median_divergences", "divergences_available"]
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        keys = sorted(stats, key=lambda ck: (CONDITION_ORDER.index(ck[0]) if ck[0] in CONDITION_ORDER else 99,
                                             CELL_ORDER.index(ck[1]) if ck[1] in CELL_ORDER else 99))
        for (cond, cell) in keys:
            s = stats[(cond, cell)]
            w.writerow([cond, cell, s["model"], s["n"], s["n_nonconv"],
                        f"{s['nonconv_rate']:.4f}", f"{s['ci_lo']:.4f}", f"{s['ci_hi']:.4f}",
                        f"{s['frac_rhat']:.3f}", f"{s['frac_ess']:.3f}",
                        f"{s['frac_both']:.3f}", f"{s['frac_error']:.3f}",
                        "" if s["median_divergences"] is None else f"{s['median_divergences']:.1f}",
                        s["divergences_available"]])


def _fmt_rate(s: dict | None) -> str:
    if s is None:
        return "—"
    div = "" if s["median_divergences"] is None else f", div~{s['median_divergences']:.0f}"
    return f"{s['nonconv_rate']:.0%} [{s['ci_lo']:.0%}–{s['ci_hi']:.0%}]{div}"


def write_markdown(stats: dict[tuple[str, str], dict], verdicts: dict, path: Path):
    present_cells = [c for c in CELL_ORDER if any(ck[1] == c for ck in stats)]
    present_conds = [c for c in CONDITION_ORDER if any(ck[0] == c for ck in stats)]
    lines = ["# MCMC 調律検証 — 集計", "",
             "非収束率 nonconv_rate [Wilson 95% CI] (div~ = median divergences, reftemp のみ)", ""]
    header = "| 条件 | " + " | ".join(present_cells) + " |"
    sep = "|" + "---|" * (len(present_cells) + 1)
    lines += [header, sep]
    cond_desc = {"A": "centered base", "B": "reftemp", "C": "centered t=.99",
                 "D": "centered tune2k", "E": "reftemp dense .99/2k"}
    for cond in present_conds:
        cells_str = " | ".join(_fmt_rate(stats.get((cond, cell))) for cell in present_cells)
        lines.append(f"| **{cond}** {cond_desc.get(cond,'')} | {cells_str} |")
    lines += ["", "## 非収束内訳 (rhat / ess / both / error, 非収束分の構成比)", "",
              "| 条件×セル | rhat | ess | both | error |", "|---|---|---|---|---|"]
    keys = sorted(stats, key=lambda ck: (CONDITION_ORDER.index(ck[0]) if ck[0] in CONDITION_ORDER else 99,
                                         CELL_ORDER.index(ck[1]) if ck[1] in CELL_ORDER else 99))
    for (cond, cell) in keys:
        s = stats[(cond, cell)]
        lines.append(f"| {cond}×{cell} | {s['frac_rhat']:.0%} | {s['frac_ess']:.0%} "
                     f"| {s['frac_both']:.0%} | {s['frac_error']:.0%} |")
    path.write_text("\n".join(lines) + "\n")


def write_verdicts(verdicts: dict, path: Path):
    lines = ["# 事前登録判定 (verdict)", ""]
    lines += [f"- **中心セル {CENTER_CELL} × E**: {verdicts['center_E']}",
              f"- **最悪セル {WORST_CELL} × E**: {verdicts['worst_E']}",
              f"- **反証チェック (artifact)**: {verdicts['artifact_check']}", "",
              "## prior 衝突寄与 (nt2_strong − nt2_accurate)"]
    for cond, c in verdicts["prior_conflict"].items():
        if isinstance(c, dict):
            lines.append(f"- 条件 {cond}: strong={c['nt2_strong']:.0%}, accurate={c['nt2_accurate']:.0%}, "
                         f"寄与差={c['prior_conflict_contribution']:+.0%} → {c['interpretation']}")
        else:
            lines.append(f"- 条件 {cond}: {c}")
    path.write_text("\n".join(lines) + "\n")


def main(argv: list[str] | None = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="調律検証マトリクス集計")
    ap.add_argument("--subdir", default=None, help="results/tuning_validation 配下のサブディレクトリ")
    args = ap.parse_args(argv)

    in_dir = RESULTS_DIR / args.subdir if args.subdir else RESULTS_DIR
    cells = load_cells(in_dir)
    if not cells:
        print(f"raw_*.jsonl が {in_dir} に無い")
        return 1

    stats = {key: summarize(recs) for key, recs in cells.items()}
    verdicts = build_verdicts(stats)

    write_csv(stats, in_dir / "tuning_summary.csv")
    write_markdown(stats, verdicts, in_dir / "tuning_summary.md")
    write_verdicts(verdicts, in_dir / "tuning_verdicts.md")

    print(f"=== aggregated {len(cells)} (cond×cell) cells from {in_dir} ===")
    print((in_dir / "tuning_summary.md").read_text())
    print((in_dir / "tuning_verdicts.md").read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
