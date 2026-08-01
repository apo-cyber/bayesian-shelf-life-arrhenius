"""5 条件 × セル のペア設計ランナー (MCMC 調律検証).

ペア設計の生命線:
    データは config.py / datagen の既存生成器が生んだ data.csv をそのまま使い、
    同一 (case_id, replicate_id) を全条件で共有する.条件間で動かすのは
    サンプラー引数 (model 幾何 / target_accept / tune / dense_mass) のみ.
    データ再生成は一切しない.

5 条件 (draws=2000, chains=4 は全条件固定 — ESS 差を混合効率に帰属させるため):
    A: centered(本体)  target=0.95 tune=1000               baseline
    B: reftemp         target=0.95 tune=1000               幾何のみ変更
    C: centered(本体)  target=0.99 tune=1000               tuning (accept↑)
    D: centered(本体)  target=0.95 tune=2000               tuning (warmup↑)
    E: reftemp dense   target=0.99 tune=2000               全部入り

セル (baseline 軸 n_points=4 / noise=medium を固定し n_T × prior を動かす):
    主   : (n_T=3, prior=strong), (n_T=2, prior=strong)   ← 全条件
    診断 : (n_T=2, prior=accurate)                         ← 条件 A, E のみ

記録 (各 fit): rhat_max, ess_bulk_min, n_divergences, converged, fail_reason.
    converged / fail_reason は mcmc.py から import した閾値で算出 (再定義禁止).
    n_divergences は本体 estimator が未公開のため centered 条件では null
    (reftemp 条件 B/E のみ実数).主判定指標は nonconv_rate (全条件で算出可).

出力: paper_a/results/tuning_validation/[<subdir>/]raw_{condition}_{cell}.jsonl
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from paper_a.analysis.estimators import mcmc as centered_estimator
from paper_a.analysis.estimators.mcmc import RHAT_THRESHOLD, ESS_THRESHOLD
from paper_a.analysis.loaders.synthetic import iter_replicates, load_truth
from paper_a.analysis.tuning_validation import model_reftemp

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results" / "tuning_validation"

DRAWS = 2000   # 全条件固定 (変更禁止)
CHAINS = 4     # 全条件固定 (変更禁止)
SPEC_LOWER = 90.0  # 本番 run_paper_a と同一 (収束判定には不変、t90 変換のみに影響)

# --- 5 条件定義 ---
CONDITIONS: dict[str, dict] = {
    "A": {"model": "centered", "reparam": False, "dense": False, "target": 0.95, "tune": 1000},
    "B": {"model": "reftemp",  "reparam": True,  "dense": False, "target": 0.95, "tune": 1000},
    "C": {"model": "centered", "reparam": False, "dense": False, "target": 0.99, "tune": 1000},
    "D": {"model": "centered", "reparam": False, "dense": False, "target": 0.95, "tune": 2000},
    "E": {"model": "reftemp",  "reparam": True,  "dense": True,  "target": 0.99, "tune": 2000},
}

# --- セル定義 (n_points=4 / noise=medium baseline を固定) ---
# applicable: そのセルを回す条件.診断セルは A,E のみ.
CELLS: list[dict] = [
    {"label": "nt3_strong",   "n_t": 3, "prior": "strong",   "applicable": ["A", "B", "C", "D", "E"]},
    {"label": "nt2_strong",   "n_t": 2, "prior": "strong",   "applicable": ["A", "B", "C", "D", "E"]},
    {"label": "nt2_accurate", "n_t": 2, "prior": "accurate", "applicable": ["A", "E"]},
]
BASELINE_NPOINTS = 4
BASELINE_NOISE = "medium"


def _resolve_case(truth: dict, n_t: int, prior: str) -> dict:
    hits = [
        v for v in truth.values()
        if v.get("n_t") == n_t and v.get("prior_accuracy") == prior
        and v.get("n_points") == BASELINE_NPOINTS and v.get("noise_level") == BASELINE_NOISE
    ]
    if len(hits) != 1:
        raise RuntimeError(
            f"セル (n_t={n_t}, prior={prior}, n_points={BASELINE_NPOINTS}, "
            f"noise={BASELINE_NOISE}) が一意でない: {len(hits)} 件"
        )
    return hits[0]


def _load_reps(case_id: str, n_reps: int, rep_start: int) -> dict[int, list[dict]]:
    """case の rep_start..rep_start+n_reps-1 のデータを dict にロード."""
    want = set(range(rep_start, rep_start + n_reps))
    out: dict[int, list[dict]] = {}
    for cid, rep, rows in iter_replicates("core", case_ids=[case_id], accelerated=True):
        if rep in want:
            out[rep] = rows
        if len(out) == len(want):
            break
    missing = want - set(out)
    if missing:
        raise RuntimeError(f"{case_id}: rep {sorted(missing)[:5]}... が data.csv に無い")
    return out


def _fail_reason(error_code: str | None, rhat, ess, converged: bool) -> str | None:
    """非収束内訳分離.閾値は import した RHAT_THRESHOLD / ESS_THRESHOLD."""
    if rhat is None or ess is None:
        return "error"
    if converged:
        return None
    rhat_fail = rhat >= RHAT_THRESHOLD
    ess_fail = ess <= ESS_THRESHOLD
    if rhat_fail and ess_fail:
        return "both"
    if rhat_fail:
        return "rhat"
    if ess_fail:
        return "ess"
    return "error"  # converged=False なのにどちらも閾内 → 想定外


def _run_one_fit(cond: dict, case: dict, rep: int, rows: list[dict]) -> dict:
    prior_ea = float(case["prior_ea_kj_mol"])
    prior_sd = float(case["prior_ea_sd_kj_mol"])
    common = dict(
        case_id=case["case_id"], replicate_id=rep,
        prior_ea_kj=prior_ea, prior_ea_sd_kj=prior_sd,
        spec_lower=SPEC_LOWER, draws=DRAWS, tune=cond["tune"],
        chains=CHAINS, target_accept=cond["target"],
    )
    if cond["model"] == "reftemp":
        res = model_reftemp.estimate(rows, use_dense_mass=cond["dense"], **common)
    else:
        # 本体 estimator をそのまま使用 (production-faithful baseline).
        res = centered_estimator.estimate(rows, **common)

    d = res.diagnostics or {}
    rhat = d.get("rhat_max")
    ess = d.get("ess_min")
    n_div = d.get("n_divergences")  # centered では None (本体未公開)
    # converged を import 閾値で再算出 (mcmc.py と同一論理).error 時は False.
    if rhat is not None and ess is not None:
        converged = (rhat < RHAT_THRESHOLD) and (ess > ESS_THRESHOLD)
    else:
        converged = False
    fail_reason = _fail_reason(res.error_code, rhat, ess, converged)

    return {
        "case_id": case["case_id"],
        "n_t": case["n_t"],
        "prior_accuracy": case["prior_accuracy"],
        "replicate_id": rep,
        "model": cond["model"],
        "target_accept": cond["target"],
        "tune": cond["tune"],
        "dense_mass": cond["dense"],
        "rhat_max": rhat,
        "ess_min": ess,
        "n_divergences": n_div,
        "converged": bool(converged),
        "error_code": res.error_code,
        "fail_reason": fail_reason,
        "t90_point": res.t90_point_estimate_months,
        "t90_lo95": res.t90_lo95_months,
        "t90_hi95": res.t90_hi95_months,
        "seed_used": d.get("seed_used"),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="MCMC 調律検証マトリクスランナー")
    ap.add_argument("--smoke", action="store_true",
                    help="条件 A,B × 主2セル × N=20 の早期確認")
    ap.add_argument("--conditions", default=None,
                    help="カンマ区切り (例 A,B).未指定で全条件 A-E")
    ap.add_argument("--cells", default=None,
                    help="カンマ区切り label (例 nt3_strong,nt2_strong).未指定で全セル")
    ap.add_argument("--n-reps", type=int, default=120)
    ap.add_argument("--rep-start", type=int, default=0)
    ap.add_argument("--subdir", default=None,
                    help="results/tuning_validation 配下のサブディレクトリ")
    args = ap.parse_args(argv)

    if args.smoke:
        conditions = ["A", "B"]
        cell_labels = ["nt3_strong", "nt2_strong"]
        n_reps = 20 if args.n_reps == 120 else args.n_reps
        subdir = args.subdir or "smoke"
    else:
        conditions = args.conditions.split(",") if args.conditions else list(CONDITIONS)
        cell_labels = args.cells.split(",") if args.cells else [c["label"] for c in CELLS]
        n_reps = args.n_reps
        subdir = args.subdir or ""

    out_dir = RESULTS_DIR / subdir if subdir else RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    truth = load_truth("core")
    cells = [c for c in CELLS if c["label"] in cell_labels]
    # 実行する (condition, cell) ペアを applicable で絞る
    jobs: list[tuple[str, dict, dict]] = []
    for cell in cells:
        case = _resolve_case(truth, cell["n_t"], cell["prior"])
        for cond_name in conditions:
            if cond_name not in cell["applicable"]:
                continue
            jobs.append((cond_name, cell, case))

    total_fits = len(jobs) * n_reps
    print(f"=== tuning matrix: {len(jobs)} (cond×cell) × {n_reps} reps "
          f"= {total_fits} fits  (draws={DRAWS}, chains={CHAINS}) ===")
    print(f"    out: {out_dir}")
    for cond_name, cell, case in jobs:
        c = CONDITIONS[cond_name]
        print(f"    {cond_name} × {cell['label']:13s} -> {case['case_id']} "
              f"[{c['model']:8s} target={c['target']} tune={c['tune']} "
              f"dense={c['dense']}]")

    # --- JIT warmup (計測から除外) ---
    print("--- JIT warmup (除外) ---")
    w_cond, w_cell, w_case = jobs[0]
    w_rows = _load_reps(w_case["case_id"], 1, args.rep_start)[args.rep_start]
    t_w = time.time()
    _ = _run_one_fit(CONDITIONS[w_cond], w_case, args.rep_start, w_rows)
    # reftemp/centered で別 JIT 経路のため両系統を暖機
    other_model = "reftemp" if CONDITIONS[w_cond]["model"] == "centered" else "centered"
    other = next((cn for cn in conditions if CONDITIONS[cn]["model"] == other_model), None)
    if other is not None:
        _ = _run_one_fit(CONDITIONS[other], w_case, args.rep_start, w_rows)
    print(f"    warmup done ({time.time() - t_w:.1f}s)")

    # --- 本番ループ ---
    done = 0
    elapsed_fit = 0.0
    t_start = time.time()
    for cond_name, cell, case in jobs:
        cond = CONDITIONS[cond_name]
        reps = _load_reps(case["case_id"], n_reps, args.rep_start)
        out_path = out_dir / f"raw_{cond_name}_{cell['label']}.jsonl"
        meta = {"condition": cond_name, "cell": cell["label"], **cond}
        with out_path.open("w") as fh:  # run 毎に新規 (run 内では fit 毎に追記)
            for rep in range(args.rep_start, args.rep_start + n_reps):
                t0 = time.time()
                rec = _run_one_fit(cond, case, rep, reps[rep])
                dt = time.time() - t0
                rec.update(meta)
                rec["elapsed_s"] = round(dt, 3)
                fh.write(json.dumps(rec) + "\n")
                fh.flush()
                done += 1
                elapsed_fit += dt
                if done % 10 == 0 or done == total_fits:
                    mean_dt = elapsed_fit / done
                    eta = mean_dt * (total_fits - done)
                    print(f"  [{done:4d}/{total_fits}] {cond_name}×{cell['label']} "
                          f"rep={rep}  {dt:.2f}s  mean={mean_dt:.2f}s  "
                          f"ETA={eta/60:.1f}min")
        # セル単位の即時サマリ
        recs = [json.loads(l) for l in out_path.read_text().splitlines()]
        ncv = sum(1 for r in recs if not r["converged"])
        print(f"  -> {out_path.name}: nonconv {ncv}/{len(recs)} "
              f"({100*ncv/len(recs):.0f}%)  written")

    print(f"=== done: {total_fits} fits in {(time.time()-t_start)/60:.1f}min ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
