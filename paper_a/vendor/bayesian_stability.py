# MIT License — Copyright (c) 2026 Yasushi Arai
#
# Provenance (author's own code, dual-licensed):
#   source:  apo-cyber/cmc-platform (private)
#   path:    apps/navigator/backend/app/calculations/phase5/bayesian_stability.py
#   commit:  34012d0
#   copied:  2026-06-06
#   scope:   "scientific core only (MIT dual-license of author's own code)"
#
# Vendored verbatim EXCEPT the FastAPI serialization helper `_sanitize_response`,
# which is product glue not reached by this package's scientific path and has
# been removed. All scientific functions, constants, and the
# `StabilityHardFailWarning` exception are unchanged to preserve exact numerical
# reproducibility. `run_bayesian_stability` is the entry point used by
# paper_a.analysis.estimators.two_stage_conjugate.
"""
ベイズアレニウス安定性予測（MCMC不要・解析的近似）

監査ステータス: Layer 1-5 検証済(2026-05-12 完遂)
詳細は docs/audit/audit_final_report.md(監査総括レポート)を参照

主要参照ドキュメント:
- ICH Q1E (2003) §B.2.2.2 "Other methods" framework
- MCMC ベンチマーク: docs/audit/l3_benchmark_results.md
- 古典 ICH §B.1 比較: docs/audit/l4_8_classical_vs_bayesian_comparison.md
- ICH Q1E 整合: docs/audit/bayesian-multi-temp-icq1e-alignment.md

モデル:
    C(t, T) = C₀ × exp(-k(T) × t)      （一次分解）
    ln k(T) = β₀ + β₁ × (1/T)          （アレニウス方程式、log線形）
    β₀ = ln(A),  β₁ = -Ea / R

ベイズ更新（正規共役）:
    事前: β ~ N(μ₀, Σ₀)
    尤度: ln k̂ᵢ ~ N(β₀ + β₁×(1/Tᵢ), σᵢ²)   （σᵢ: delta法で k̂ᵢ の SE から換算）
    事後: β | data ~ N(μₙ, Σₙ)
         Σₙ⁻¹ = Σ₀⁻¹ + X^T W X
         μₙ   = Σₙ (Σ₀⁻¹ μ₀ + X^T W y)

有効期間予測:
    τ = -ln(spec_lower / C₀) / k_pred
    不確かさ: ln(k) の予測分散から delta 法で τ の CI を計算
"""

import math

import numpy as np
from scipy import stats
from typing import List, Dict, Optional

R_GAS = 8.314  # J/(mol·K)

# ── 警告閾値定数(Layer 5 監査由来)──────────────────────────────────────
# 詳細は docs/audit/layer5_handover_list.md 参照
N_CONDS_RECOMMENDED = 5       # 信頼区間精度の推奨下限(L3.4a)
N_CONDS_MINIMUM = 3            # 計算実行の最低基準(L3.5.1、< 3 で ValueError)
SE_K_RATIO_WARN = 0.15         # delta 法警告閾値(L3.4b、誤差 ~5%)
R_SQUARED_ARRHENIUS_WARN = 0.9  # Arrhenius 直線性警告閾値(L3.5.3 silent failure)
EA_PRIOR_RANGE_MIN_KJ = 60.0   # 典型医薬品の Ea 下限(L3.5.3 / L5 引き継ぎ #6)
EA_PRIOR_RANGE_MAX_KJ = 120.0  # 同上限
SHELF_LIFE_CAP_MONTHS = 120.0  # 120 ヶ月キャップ(_cap デフォルトと整合)

# ── B4 警告閾値(Phase B4 由来)─────────────────────────────────────────
# 詳細は docs/audit/audit_final_report.md §13 参照
PQ_N_POINTS_HARD_FAIL_MAX = 2   # この値以下で OLS SE が縮退し sigma_acc=0 → 422
PQ_N_POINTS_WARN_MAX = 3        # この値以下(かつ Hard fail 域外)で警告
PRIOR_INCONSISTENCY_HARD_FAIL_P = 0.01  # 両側 Bayesian p-value がこれ未満で 422
PRIOR_INCONSISTENCY_WARN_P = 0.05       # 両側 Bayesian p-value がこれ未満で警告


class StabilityHardFailWarning(Exception):
    """安定性計算が警告ベースの hard fail を必要とする場合に送出される例外.

    ルーター層で HTTPException(status_code=422) に変換され、
    レスポンス detail は {"code": ..., "message": ..., "detail": ...} の形となる.
    """

    def __init__(self, code: str, message: str, detail: Optional[Dict] = None):
        self.code = code
        self.message = message
        self.detail = detail or {}
        super().__init__(message)


def _warning(code: str, level: str, message: str, **context) -> Dict:
    """warnings 配列に追加する 1 件分を構築する."""
    return {"code": code, "level": level, "message": message, "context": context}


def _prior_data_inconsistency_pvalue(
    prior_mean: float,
    prior_sd: float,
    data_mean: float,
    data_se: float,
) -> float:
    """Prior と data の Ea 推定値の不整合を両側 p-value として定量化する.

    p = 2 * min(P(theta_prior ≤ theta_data), P(theta_prior ≥ theta_data))
      = 2 * min(P(D ≤ 0), 1 - P(D ≤ 0))
    ここで D = theta_prior - theta_data ~ N(prior_mean - data_mean, prior_sd² + data_se²).
    """
    var_total = prior_sd ** 2 + data_se ** 2
    if var_total <= 0.0:
        return 1.0
    z = (prior_mean - data_mean) / float(np.sqrt(var_total))
    p_left = float(stats.norm.cdf(z))
    return 2.0 * min(p_left, 1.0 - p_left)


def _cap(v: float, max_val: float = 120.0) -> float:
    """
    shelf_life 値の上限クリッピング。

    背景:
        非有意な分解 (k ≈ 0) の場合、shelf_life の予測値が異常に
        大きくなる (数百ヶ月、数千ヶ月)。これは数値的妥当性は
        あるが、実務的には意味がない (120 ヶ月 = 10 年で打ち切り)。

    制約:
        本関数の使用は run_bayesian_stability のみ。単一温度モード
        (calculate_single_temp_arrhenius_extrapolation) は別ロジックで
        上限管理しているため、本関数を呼ばない。

    実装状況:
        値がキャップに到達した場合は結果 dict の `capped: True` フラグ + warnings
        配列の SHELF_LIFE_CAPPED で UI に明示する(Step 5a A7、Layer 5 監査由来)。
    """
    return float(min(max(v, 0.0), max_val))


def run_bayesian_stability(
    data_rows: List[Dict],
    storage_temps: Optional[List[float]] = None,
    spec_lower: float = 95.0,
    initial_content: float = 100.0,
    prior_ea_kj: float = 80.0,
    prior_ea_sd_kj: float = 30.0,
) -> dict:
    """
    ベイズアレニウス安定性予測 (多温度モード、真の正規共役ベイズ更新)。

    Parameters
    ----------
    data_rows : list of dict
        keys: temperature (°C, float), time_months (float), content_percent (float)
    storage_temps : list of float, optional
        長期保存温度 [°C]（デフォルト [25.0, 30.0]）
    spec_lower : float
        規格下限 (%)（デフォルト 95%）
    initial_content : float
        初期含量 (%)（デフォルト 100%）
    prior_ea_kj : float
        事前活性化エネルギー [kJ/mol]
    prior_ea_sd_kj : float
        事前 Ea 標準偏差 [kJ/mol]

    Raises
    ------
    ValueError
        - 温度水準数 < 3、または k_hat > 0 を満たす温度が < 3 水準
        - その他の入力データの数学的妥当性違反
    StabilityHardFailWarning (Phase B4 由来)
        - 温度条件別の観測点数が ≤ 2 (PQ_N_POINTS_TOO_LOW)
        - Prior Ea と OLS-Ea の両側 Bayesian p-value が < 0.01
          (PRIOR_DATA_INCONSISTENCY)
        ルーター層で HTTPException(status_code=422) に変換される.

    設計判断と監査メモ (Layer 1-6 監査クローズ / 本体 2026-05-12, B3/B4/B5 2026-05-16):

    本関数は ICH Q1E §B.2.2.2 "Other methods" の枠組を借りた代替手法として
    位置づけられている (§B.2.2.2 の literal 文脈は batch poolability 代替手段
    だが、本実装は §B.2.2.2 末尾「Statistical procedures other than those
    described above can be used」を Bayesian Arrhenius 拡張の根拠としている)。
    Layer 1 (数学モデル) 〜 Layer 6 (監査クローズ) の系統的監査を完了している。
    監査透明性のため、関連ドキュメントを以下に列挙する:

    - 監査計画 / チェックリスト:
      docs/audit/bayesian-multi-temp-audit-plan.md
    - ICH Q1E 整合性メモ:
      docs/audit/bayesian-multi-temp-icq1e-alignment.md
    - Layer 3 中間まとめ (MCMC ベンチマーク結果 + 古典 ICH §B 検討の起点):
      docs/audit/layer3_interim_summary.md
    - L3.4 ベンチマーク (正規近似 / delta 法 / MCMC vs 2 段階):
      docs/audit/l3_benchmark_results.md
    - L4.8 古典 ICH §B.1 比較 (先行検証、本手法は古典より約 15% 保守的):
      docs/audit/l4_8_classical_vs_bayesian_comparison.md
    - B3 4 者比較 (Z/t/MCMC/古典、§B.1/§2.6 原文確認済、古典との差 +17.9%):
      docs/audit/layer4_classical_ich_comparison.md
    - Layer 5 引き継ぎリスト (UI 警告 8 項目、Step 5a で実装済):
      docs/audit/layer5_handover_list.md
    - 監査総括 (Layer 1-6):
      docs/audit/audit_final_report.md §13 (B4), §14 (B3), §15 (Layer 1-6 総括)

    既知の制約と Layer 5/B4 で実装した対処:

        (a) n_conds < 3 はエラー化 (Step 5a A2)
            df=0 + フロアで residual 不確実性が機能せず楽観バイアス +1.7 ヶ月。
        (b) n_conds < 5 は LOW_N_CONDS 警告 (Step 5a A1)
            t/z 比 = 6.48 (n_conds=3) で厳密 Bayesian 実用不能、正規近似で代用。
        (c) SE_k/k > 0.15 は HIGH_SE_K_RATIO 警告 (Step 5a A3)
            delta 法相対誤差 > 5% の境界。
        (d) k_hat ≤ 0 silent 置換を K_HAT_ZERO_FALLBACK フラグ化 (Step 5a A4)
        (e) R²_arrhenius < 0.9 で LOW_R_SQUARED_ARRHENIUS 警告 (Step 5a A5)
        (f) Ea 物理的妥当範囲 (60-120 kJ/mol) 外で UNUSUAL_PRIOR_EA 警告 (A6)
        (g) 120 ヶ月キャップ到達で SHELF_LIFE_CAPPED フラグ + 警告 (A7)
        (h) 古典 ICH §B.1 単温度比較を補助サブビューとして提供 (Step 5b)。
            B3 で単温度モードへの古典 §B.1 適用は構造的に不可 (N/A) と確定。
        (i) PQ_N_POINTS_TOO_LOW: 温度別観測点数 ≤ 2 で 422、== 3 で警告 (Phase B4)
        (j) PRIOR_DATA_INCONSISTENCY: prior と OLS-Ea の両側 Bayesian p-value
            < 0.01 で 422、< 0.05 で警告 (Phase B4)

    その他の設計判断 (本関数ロジック自体):

        1. log-normal 補正の非対称性 (論点 c, L4.6 で詳細評価予定):
           - 点推定 k_mean = exp(lnk_mean + lnk_var_param/2) は
             parametric 分散 (lnk_var_param) のみで補正
           - CI 計算 (k_lo95, k_hi95) は parametric + residual の合計分散
             (lnk_var_total) を使用
           - 点推定と CI で分散の取り扱いが非対称
           - 意図的か偶発的かは未検証

        2. 残差分散のフロア構造は 2 段階 (論点 e, L2.6 で検証済み):
           - 式: sigma2_resid = max(SS_res / max(n_conds - 2, 1), 1e-6)
           - 第 1 段: 自由度フロア `max(n_conds - 2, 1)` (0 除算回避)
             * n_conds=2 のとき df=0 → フロア 1 が効いて分母 1
             * n_conds=3 のとき df=1 → フロアと等しく分母 1
             * n_conds≥4 のとき df>1 → 自由度フロア不活性
           - 第 2 段: 分散フロア `max(..., 1e-6)` (sigma2_resid 値の下限)
             * SS_res ≈ 0 のときのみ活性(n_conds=2 で完全フィット時など)
             * 実データではほぼ不活性 (L2.9 感度分析: 1e-7〜1e-4 で結果不変)
           - 注意: n_conds=2 では SS_res ≈ 0 となり residual 由来の不確実性が
             機能しない。UI で「最低 3 条件推奨」表示が必要 (L5.1)。

        3. shelf_life 120 ヶ月キャップ (論点 f, L4.7 で詳細評価予定):
           - _cap(τ, 120.0) で 10 年で打ち切り
           - 非有意な k → 0 ケースで予測が暴れないための実用的処置
           - キャップ到達時は capped: True フラグ + SHELF_LIFE_CAPPED 警告で
             UI に明示する (Step 5a A7 で実装済)

        4. β₀ (= ln A) 事前分布は uninformative (論点 j, L2.5 で検証済み):
           - 実装: mu0[0] = 20.0, Sigma0_inv[0,0] = 1/100²
           - β₀ は ln A に対応。SD=100 で実質 uninformative。
           - uninformative にする根拠:
             (a) ln A は物質・反応経路・単位系に強く依存し、informative 事前を
                設定する科学的根拠が希薄。
             (b) CMC 申請の関心は 25°C 外挿点 k であり、β₀ 自体は中間パラメータ。
             (c) β₁ がデータで強く制約されれば、β₀ も Arrhenius プロットの
                形状から自動的に決まる。
           - β₀ と β₁ の事前共相関は 0 (Sigma0_inv が対角行列)。
             Kinetic Compensation Effect (KCE: ln A と Ea の線形相関) は
             同系反応で知られるが、CMC の多様な分解経路に汎用適用できないため
             採用していない。
           - 感度分析 (L2.8): β₀_SD ≥ 100 で結果実質不変、SD=10 でも変動 < 2%。

        5. デフォルト Ea ~ N(80, 30) kJ/mol の根拠 (L2.4 で検証済み):
           - 実用的医薬品分解で観測される Ea 範囲(50-150 kJ/mol)を
             ±2σ で覆う弱情報事前。
           - 単温度モード (calculate_single_temp_arrhenius_extrapolation) の
             router デフォルト N(92, 5) との差:
             - 多温度モードは複数温度の OLS から十分な情報が得られるため、
               prior は弱情報 (SD=30) で良い。
             - 単温度モードは 1 加速温度のみで Arrhenius 外挿に必要な Ea を
               prior でほぼ決め打ち (SD=5) する必要がある。
           - Aposartan 想定データでは両 prior とも同一の Ea≈92.5、
             SL_lo95≈42-43 ヶ月に収束 (L2.10 感度分析)。

        6. σ² plug-in による Empirical Bayes 構造 (L2.6 / L2.7 で検証済み):
           - 厳密 Bayesian なら σ² ~ Inverse-Gamma の事前を入れ、β の周辺
             事後を t 分布として扱うべきだが、本実装は σ² を SS_res/df の
             plug-in 値として正規近似で打ち切る。
           - 結果として事後信頼区間は (特に n_conds が小さい場合) 楽観的
             (過小評価) 傾向。L3.4 で定量化済 (差 < 5%、l3_benchmark_results.md
             §4.2 / B2b L52, L64)。Z 値 ≈ MCMC が確定 (B2b 多温度、B2a 単温度)。
           - 規制的位置づけは ICH Q1E §B.2.2.2 "Other methods" 枠で扱い、
             詳細は docs/audit/bayesian-multi-temp-icq1e-alignment.md 参照。

    参照:
        - 単一温度モード (calculate_single_temp_arrhenius_extrapolation)
          は Faya 2018 / Chau 2023 の概念フレームと整合 (本文未取得につき
          抄録レベル)、ICH Q1E §B.2.2.2 "Other methods" 枠で扱う
        - 多温度モード Layer 1-2 監査:
          docs/audit/bayesian-multi-temp-audit-plan.md
        - ICH Q1E 整合性メモ:
          docs/audit/bayesian-multi-temp-icq1e-alignment.md
        - Layer 3 (誤差伝播の定量化), Layer 4 (古典 §B.1 比較), Layer 5 (UI
          統合), Layer 6 (監査クローズ) はすべて 2026-05-16 までに完了
    """
    if storage_temps is None:
        storage_temps = [25.0, 30.0]

    warnings_list: List[Dict] = []

    # ── Step 1: 温度条件別にデータを集約 ──────────────────────────────
    temp_groups: dict[float, list] = {}
    for row in data_rows:
        T = float(row["temperature"])
        t = float(row["time_months"])
        C = float(row["content_percent"])
        temp_groups.setdefault(T, []).append((t, C))

    if len(temp_groups) < N_CONDS_MINIMUM:
        # L3.5.1: n_conds=2 で SS_res 異常縮小 → SL_lo95 +1.7 ヶ月楽観バイアス
        raise ValueError(
            f"多温度ベイジアン解析には最低 {N_CONDS_MINIMUM} 温度のデータが必要です。"
            f"現在 {len(temp_groups)} 温度。"
            "温度水準が少ないと信頼区間が過度に楽観化されます(参照: docs/audit/layer3_interim_summary.md)。"
        )

    # ── B4: PQ_N_POINTS_TOO_LOW(温度条件別、Phase B4)────────────────
    # n ≤ 2: OLS SE が縮退(完全フィット)し sigma_acc=0 の偽の確実性が出るため 422
    # n == 3: 警告のみ(計算続行、信頼区間が楽観化されうる)
    pq_hard_fail = [
        (T_C, len(pts)) for T_C, pts in sorted(temp_groups.items())
        if len(pts) <= PQ_N_POINTS_HARD_FAIL_MAX
    ]
    if pq_hard_fail:
        T_C, n_pts = pq_hard_fail[0]
        raise StabilityHardFailWarning(
            code="PQ_N_POINTS_TOO_LOW",
            message=(
                f"温度 {T_C}°C のデータ点数が {n_pts} 点と少なすぎます"
                f"(最低 {PQ_N_POINTS_HARD_FAIL_MAX + 1} 点必要)。"
                "OLS の標準誤差が縮退し、偽の確実性につながります。"
            ),
            detail={
                "temperature_c": T_C,
                "n_points": n_pts,
                "min_required": PQ_N_POINTS_HARD_FAIL_MAX + 1,
                "offending_conditions": [
                    {"temperature_c": tc, "n_points": n}
                    for tc, n in pq_hard_fail
                ],
            },
        )
    pq_warn = [
        (T_C, len(pts)) for T_C, pts in sorted(temp_groups.items())
        if PQ_N_POINTS_HARD_FAIL_MAX < len(pts) <= PQ_N_POINTS_WARN_MAX
    ]
    for T_C, n_pts in pq_warn:
        warnings_list.append(_warning(
            code="PQ_N_POINTS_TOO_LOW",
            level="warning",
            message=(
                f"温度 {T_C}°C のデータ点数が {n_pts} 点です。"
                "傾きの推定誤差が大きい可能性があり(信頼区間が広がります)、"
                "可能であれば 4 点以上を推奨します。"
            ),
            temperature_c=T_C,
            n_points=n_pts,
            recommended_min=PQ_N_POINTS_WARN_MAX + 1,
        ))

    # A6: Ea 事前範囲チェック(物理的妥当範囲、L3.5.3)
    if prior_ea_kj < EA_PRIOR_RANGE_MIN_KJ or prior_ea_kj > EA_PRIOR_RANGE_MAX_KJ:
        warnings_list.append(_warning(
            code="UNUSUAL_PRIOR_EA",
            level="warning",
            message=(
                f"指定された事前 Ea = {prior_ea_kj} kJ/mol が典型医薬品の範囲"
                f"({EA_PRIOR_RANGE_MIN_KJ:.0f}-{EA_PRIOR_RANGE_MAX_KJ:.0f} kJ/mol)から"
                "外れています。設定意図をご確認ください。"
            ),
            prior_ea_kj=prior_ea_kj,
            range_min=EA_PRIOR_RANGE_MIN_KJ,
            range_max=EA_PRIOR_RANGE_MAX_KJ,
        ))

    # A1: n_conds < 5 警告(L3.4a t/z 比 6.48 → 厳密 Bayesian 実用不能)
    n_conds_observed = len(temp_groups)
    if n_conds_observed < N_CONDS_RECOMMENDED:
        warnings_list.append(_warning(
            code="LOW_N_CONDS",
            level="warning",
            message=(
                f"温度条件数が {n_conds_observed} です。"
                f"信頼区間の精度を確保するため、{N_CONDS_RECOMMENDED} 温度以上を推奨します"
                "(t 補正版での再確認推奨、参照: L3.4a)。"
            ),
            n_conds=n_conds_observed,
            recommended=N_CONDS_RECOMMENDED,
        ))

    # ── Step 2: 各温度で一次反応速度定数 k̂ を推定 ───────────────────
    k_estimates: list[float] = []
    k_se_estimates: list[float] = []
    temperatures_K: list[float] = []
    condition_fits: list[dict] = []

    for T_C, points in sorted(temp_groups.items()):
        T_K = T_C + 273.15
        times = np.array([p[0] for p in points])
        contents = np.array([p[1] for p in points])

        # 初期含量: t=0 があれば実測値、なければ initial_content
        c0_idx = int(np.argmin(times))
        c0 = float(contents[c0_idx]) if times[c0_idx] == 0 else initial_content

        valid = (contents > 0) & (times >= 0)
        t_v = times[valid]
        c_v = contents[valid]

        if len(t_v) < 2:
            continue

        y = np.log(c_v / c0)  # ln(C/C₀) = -k × t

        slope, intercept, r_val, _, se = stats.linregress(t_v, y)
        k_hat = float(-slope)

        k_zero_fallback = False
        if k_hat <= 0:
            # A4: silent 置換 → フラグ化(Layer 1 / L3.5.3)
            k_hat = 1e-7
            k_se = 1e-7
            k_zero_fallback = True
            warnings_list.append(_warning(
                code="K_HAT_ZERO_FALLBACK",
                level="warning",
                message=(
                    f"温度 {T_C}°C で有意な分解が観察されませんでした(k ≤ 0)。"
                    "Significant degradation 不在のためベイジアン解析の前提を満たしていません。"
                    "ICH Q1E §B.1 古典手法をご検討ください。"
                ),
                temperature_c=T_C,
            ))
        else:
            k_se = max(float(se), k_hat * 0.05)

        # A3: SE_k/k > 0.15 警告(L3.4b、delta 法破綻境界)
        se_k_ratio = k_se / k_hat if k_hat > 0 else 0.0
        if se_k_ratio > SE_K_RATIO_WARN and not k_zero_fallback:
            warnings_list.append(_warning(
                code="HIGH_SE_K_RATIO",
                level="warning",
                message=(
                    f"温度 {T_C}°C で SE_k/k = {se_k_ratio:.3f} です。"
                    f"delta 法近似誤差が大きくなる可能性があります(閾値 {SE_K_RATIO_WARN:.2f})。"
                    "観測点数の追加または異なる温度範囲をご検討ください。"
                ),
                temperature_c=T_C,
                se_k_ratio=round(se_k_ratio, 4),
                threshold=SE_K_RATIO_WARN,
            ))

        k_estimates.append(k_hat)
        k_se_estimates.append(k_se)
        temperatures_K.append(T_K)

        t_fit = np.linspace(0, float(np.max(t_v)) * 1.1, 50)
        condition_fits.append({
            "temperature_c": T_C,
            "k_hat_per_month": round(k_hat, 8),
            "k_se": round(k_se, 8),
            "se_k_ratio": round(se_k_ratio, 4),
            "k_zero_fallback": k_zero_fallback,
            "r_squared": round(float(r_val ** 2), 4),
            "observed": [
                {"time_months": round(float(tt), 3), "content_percent": round(float(cc), 4)}
                for tt, cc in zip(t_v, c_v)
            ],
            "fitted": [
                {"time_months": round(float(tt), 3), "content_percent": round(float(c0 * np.exp(-k_hat * tt)), 4)}
                for tt in t_fit
                if c0 * np.exp(-k_hat * tt) > 0
            ],
        })

    n_conds = len(k_estimates)
    if n_conds < N_CONDS_MINIMUM:
        # 各温度の有効データ点数フィルタ後に 3 未満になった場合
        raise ValueError(
            f"k_hat > 0 を満たす有効な温度条件が {N_CONDS_MINIMUM} 水準未満です(現在 {n_conds} 水準)。"
            "観測点数不足または分解非有意のためベイジアン解析を実行できません。"
        )

    # ── B4: PRIOR_DATA_INCONSISTENCY(Phase B4)────────────────────────
    # 「prior 無し」OLS で Ea を推定し、prior と統計的に整合するか両側 p-value で評価.
    # 共役 Normal-Normal の枠で差の分布 D ~ N(prior_mean - data_mean, prior_sd² + data_se²)
    # の尾部確率を取る(共通の Bayesian prior-data inconsistency 検出指標).
    T_arr_check = np.array(temperatures_K)
    inv_T_check = 1.0 / T_arr_check
    ln_k_check = np.log(np.array(k_estimates))
    if n_conds >= 3 and float(np.std(inv_T_check)) > 0:
        slope_ols, _, _, _, se_slope_ols = stats.linregress(inv_T_check, ln_k_check)
        ea_ols_kj = float(-slope_ols * R_GAS / 1000.0)
        ea_ols_se_kj = float(abs(se_slope_ols) * R_GAS / 1000.0)
        p_inconsistency = _prior_data_inconsistency_pvalue(
            prior_mean=prior_ea_kj,
            prior_sd=prior_ea_sd_kj,
            data_mean=ea_ols_kj,
            data_se=ea_ols_se_kj,
        )
        if p_inconsistency < PRIOR_INCONSISTENCY_HARD_FAIL_P:
            raise StabilityHardFailWarning(
                code="PRIOR_DATA_INCONSISTENCY",
                message=(
                    "事前 Ea と観測データに統計的な強い不整合が検出されました"
                    f"(Bayesian p = {p_inconsistency:.4f} < {PRIOR_INCONSISTENCY_HARD_FAIL_P:.2f})。"
                    "事前分布の根拠と観測データの妥当性を見直してください。"
                ),
                detail={
                    "p_value": round(p_inconsistency, 6),
                    "prior_ea_kj": prior_ea_kj,
                    "prior_ea_sd_kj": prior_ea_sd_kj,
                    "data_ea_kj": round(ea_ols_kj, 4),
                    "data_ea_se_kj": round(ea_ols_se_kj, 4),
                    "threshold": PRIOR_INCONSISTENCY_HARD_FAIL_P,
                },
            )
        if p_inconsistency < PRIOR_INCONSISTENCY_WARN_P:
            warnings_list.append(_warning(
                code="PRIOR_DATA_INCONSISTENCY",
                level="warning",
                message=(
                    "事前 Ea と観測データに統計的な乖離が見られます"
                    f"(Bayesian p = {p_inconsistency:.3f})。事前分布設定の妥当性をご確認ください。"
                ),
                p_value=round(p_inconsistency, 6),
                prior_ea_kj=prior_ea_kj,
                prior_ea_sd_kj=prior_ea_sd_kj,
                data_ea_kj=round(ea_ols_kj, 4),
                data_ea_se_kj=round(ea_ols_se_kj, 4),
            ))

    # ── Step 3: ベイズアレニウス回帰 ────────────────────────────────
    T_arr = np.array(temperatures_K)
    k_arr = np.array(k_estimates)
    k_se_arr = np.array(k_se_estimates)

    x_vals = 1.0 / T_arr          # 1/T [K⁻¹]
    y_vals = np.log(k_arr)        # ln(k̂)

    # delta 法: Var(ln k̂) ≈ (SE_k / k)²
    obs_var = np.maximum((k_se_arr / k_arr) ** 2, 1e-8)

    X = np.column_stack([np.ones(n_conds), x_vals])    # デザイン行列 (n, 2)
    W = np.diag(1.0 / obs_var)                          # 精度行列

    # 事前分布
    mu0 = np.array([
        20.0,                                    # β₀ = ln(A): 緩い事前
        -prior_ea_kj * 1000.0 / R_GAS,          # β₁ = -Ea/R
    ])
    Sigma0_inv = np.diag([
        1.0 / 100.0 ** 2,                        # ln(A): 非常に分散させた事前
        1.0 / (prior_ea_sd_kj * 1000.0 / R_GAS) ** 2,
    ])

    # 事後更新
    Sigma_n_inv = Sigma0_inv + X.T @ W @ X
    Sigma_n = np.linalg.inv(Sigma_n_inv)
    mu_n = Sigma_n @ (Sigma0_inv @ mu0 + X.T @ W @ y_vals)

    ea_post_kj = float(-mu_n[1] * R_GAS / 1000.0)
    ea_post_sd_kj = float(np.sqrt(Sigma_n[1, 1]) * R_GAS / 1000.0)
    ln_a_post = float(mu_n[0])

    # 残差分散（予測不確かさのフロア）
    y_fitted = X @ mu_n
    ss_res = float(np.sum((y_vals - y_fitted) ** 2))
    ss_tot = float(np.sum((y_vals - np.mean(y_vals)) ** 2))
    r_squared_arr = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0
    sigma2_resid = max(ss_res / max(n_conds - 2, 1), 1e-6)

    # A5: R²_arrhenius < 0.9 警告(L3.5.3 silent failure 防止)
    if r_squared_arr < R_SQUARED_ARRHENIUS_WARN:
        warnings_list.append(_warning(
            code="LOW_R_SQUARED_ARRHENIUS",
            level="warning",
            message=(
                f"Arrhenius プロットの直線性が低い(R² = {r_squared_arr:.3f})。"
                "多温度ベイジアン解析の適用範囲外の可能性があります。"
                "古典 ICH §B.1 単温度解析を推奨します。"
            ),
            r_squared=round(r_squared_arr, 4),
            threshold=R_SQUARED_ARRHENIUS_WARN,
        ))

    # ── Step 4: 各保存条件での予測 ──────────────────────────────────
    log_ratio = float(-np.log(spec_lower / initial_content))  # -ln(L/C₀)

    storage_results = []
    for T_store_C in storage_temps:
        T_store_K = T_store_C + 273.15
        x_pred = np.array([1.0, 1.0 / T_store_K])

        lnk_mean = float(x_pred @ mu_n)
        lnk_var_param = float(x_pred @ Sigma_n @ x_pred)
        lnk_var_total = lnk_var_param + sigma2_resid

        # k の点推定（対数正規補正）
        k_mean = float(np.exp(lnk_mean + lnk_var_param / 2))
        k_lo90 = float(np.exp(lnk_mean - 1.645 * np.sqrt(lnk_var_total)))
        k_hi90 = float(np.exp(lnk_mean + 1.645 * np.sqrt(lnk_var_total)))
        k_lo95 = float(np.exp(lnk_mean - 1.960 * np.sqrt(lnk_var_total)))
        k_hi95 = float(np.exp(lnk_mean + 1.960 * np.sqrt(lnk_var_total)))

        def shelf_life(k: float) -> float:
            return _cap(log_ratio / k, 120.0) if k > 1e-10 else 120.0

        sl_mean = shelf_life(k_mean)
        sl_lo90 = shelf_life(k_hi90)   # k が大きい → 有効期間が短い（悲観的）
        sl_hi90 = shelf_life(k_lo90)
        sl_lo95 = shelf_life(k_hi95)
        sl_hi95 = shelf_life(k_lo95)

        # コンテンツプロファイル（t=0 〜 sl_hi95 × 1.3 or 60ヶ月）
        t_max = min(sl_hi95 * 1.3, 60.0)
        t_points = np.linspace(0, t_max, 60)

        def content(k: float, t: float) -> float:
            return round(float(np.clip(initial_content * np.exp(-k * t), 0, initial_content)), 3)

        profile = [
            {
                "time_months": round(float(t), 3),
                "content_mean": content(k_mean, t),
                "content_lo90": content(k_hi90, t),   # hi k → lo content
                "content_hi90": content(k_lo90, t),
                "content_lo95": content(k_hi95, t),
                "content_hi95": content(k_lo95, t),
            }
            for t in t_points
        ]

        # A7: 120 ヶ月キャップ到達フラグ(L3.5.3 / L4.7)
        capped = sl_lo95 >= SHELF_LIFE_CAP_MONTHS
        if capped:
            warnings_list.append(_warning(
                code="SHELF_LIFE_CAPPED",
                level="info",
                message=(
                    f"{T_store_C:.0f}°C の外挿が {SHELF_LIFE_CAP_MONTHS:.0f} ヶ月キャップに到達しました。"
                    "実用上は『有効期間 120 ヶ月以上』と解釈してください。"
                    "データの分解傾向が弱く、Significant degradation の前提を満たさない可能性があります。"
                ),
                storage_temp_c=T_store_C,
                cap_months=SHELF_LIFE_CAP_MONTHS,
            ))

        storage_results.append({
            "storage_temp_c": T_store_C,
            "label": f"{T_store_C:.0f}°C",
            "k_mean_per_month": round(k_mean, 8),
            "lnk_mean": round(lnk_mean, 6),
            "lnk_sd_param": round(float(np.sqrt(lnk_var_param)), 6),
            "lnk_sd_total": round(float(np.sqrt(lnk_var_total)), 6),
            "shelf_life_mean_months": round(sl_mean, 1),
            "shelf_life_lo90_months": round(sl_lo90, 1),
            "shelf_life_hi90_months": round(sl_hi90, 1),
            "shelf_life_lo95_months": round(sl_lo95, 1),
            "shelf_life_hi95_months": round(sl_hi95, 1),
            "recommended_shelf_life_months": round(sl_lo95, 1),  # 95%CI 下限(保守的、ICH Q1E §2.6 lower one-sided CI 規定と整合)
            "capped": capped,
            "profile": profile,
        })

    # ── Step 5: アレニウスプロット用データ ───────────────────────────
    x_range = float(np.max(x_vals) - np.min(x_vals))
    margin = max(x_range * 0.2, 1e-4)
    x_plot = np.linspace(float(np.min(x_vals)) - margin, float(np.max(x_vals)) + margin, 80)
    X_plot = np.column_stack([np.ones(80), x_plot])
    y_plot = X_plot @ mu_n
    arr_sd = np.array([float(np.sqrt(xr @ Sigma_n @ xr + sigma2_resid)) for xr in X_plot])

    arrhenius_plot = [
        {
            "inv_T_1000": round(float(xi * 1000), 5),
            "lnk_mean": round(float(yi), 4),
            "lnk_lo95": round(float(yi - 1.96 * si), 4),
            "lnk_hi95": round(float(yi + 1.96 * si), 4),
        }
        for xi, yi, si in zip(x_plot, y_plot, arr_sd)
    ]

    arrhenius_observed = [
        {
            "inv_T_1000": round(float(1000.0 / T_arr[i]), 5),
            "lnk_obs": round(float(y_vals[i]), 4),
            "temperature_c": round(float(T_arr[i] - 273.15), 1),
        }
        for i in range(n_conds)
    ]

    return {
        "ea_posterior_kj_mol": round(ea_post_kj, 2),
        "ea_posterior_sd_kj_mol": round(ea_post_sd_kj, 2),
        "ea_prior_kj_mol": prior_ea_kj,
        "ea_ci_90": [
            round(ea_post_kj - 1.645 * ea_post_sd_kj, 2),
            round(ea_post_kj + 1.645 * ea_post_sd_kj, 2),
        ],
        "ea_ci_95": [
            round(ea_post_kj - 1.96 * ea_post_sd_kj, 2),
            round(ea_post_kj + 1.96 * ea_post_sd_kj, 2),
        ],
        "ln_a_posterior": round(ln_a_post, 4),
        "r_squared_arrhenius": round(r_squared_arr, 4),
        "spec_lower": spec_lower,
        "initial_content": initial_content,
        "n_conditions": n_conds,
        "storage_results": storage_results,
        "condition_fits": condition_fits,
        "arrhenius_plot": arrhenius_plot,
        "arrhenius_observed": arrhenius_observed,
        "warnings": warnings_list,
    }


def calculate_single_temp_arrhenius_extrapolation(
    temp_c: float,
    times_days: list,
    contents: list,
    prior_ea_kj: float,
    prior_ea_sd_kj: float,
    pred_temp_c: float = 25.0,
    spec_lower: float = 95.0,
    c0: float = 100.0,
) -> dict:
    """
    単一加速温度のデータと prior Ea を用いて長期保存条件での有効期間を外挿する
    （ICH Q1A 加速試験対応）。

    手法の本質:
        加速温度での k_acc をデータから OLS で推定し、prior の Ea
        （アレニウス解析から引き継ぎ可）を用いて予測保存温度での k に
        Arrhenius 式で外挿する。delta 法で不確実性を伝播させて CI を計算する。

    注意:
        これは prior + delta 法による不確実性伝播であり、Ea のデータによる
        事後更新は行わない。厳密な Bayesian 事後更新ではなく、Arrhenius 外挿
        に prior Ea の不確実性を反映させる手法である。多温度モード
        (run_bayesian_stability) では真の正規共役ベイズ更新が行われるが、
        本関数では prior Ea を「既知パラメータ＋既知不確実性」として扱う。

    構造的制限 (Phase B3/B4 で確認、Layer 6 監査クローズ):
        単温度モードでは Arrhenius 関係式を 1 点の (T, k) で解けないため
        Ea_data の独立推定が不可能。結果として:
        - PRIOR_DATA_INCONSISTENCY 警告 (B4) は適用範囲外
        - 古典 ICH §B.1 アルゴリズム (25°C 回帰データ要) も適用範囲外
        詳細: docs/audit/paper_theory_draft.md §8.3、
              docs/audit/layer4_classical_ich_comparison.md §4

    Raises
    ------
    ValueError
        - times_days と contents の長さ不一致
    StabilityHardFailWarning (Phase B4 由来)
        - PQ データ点数 ≤ 2 (PQ_N_POINTS_TOO_LOW)
        ルーター層で HTTPException(status_code=422) に変換される.

    推奨保守値:
        recommended_shelf_life_months は 95% CI 下限 (sl_lo95)。
        ICH Q1E §2.6 が「lower one-sided 95 percent confidence limit」を
        含量低下属性の保守的推定として規定するのと整合する (ICH Q1E は
        頻度論的回帰を主に想定するが、本実装は §B.2.2.2 "Other methods" の
        枠組で扱う Bayesian 拡張)。

    Prior 設計について:
        Prior 設計は Faya et al. (Stat Med. 2018; 37(17):2599-2615) および
        Chau et al. (AAPS PharmSciTech. 2023; 24(8):250) の概念フレームワーク
        ─ 加速試験・歴史的データの結果を長期予測の Prior に組み込む
        経験ベイズ的アプローチ ─ に整合する (本文未取得、抄録レベルの整合)。

        引数 `prior_ea_sd_kj` は Ea の真の標準偏差 (= アレニウス回帰の OLS で
        推定された Ea の標準誤差 SE そのもの) として受け取り、L637 の
        delta 法計算で σ_Ea として直接消費する:

            σ_Ea_contrib = |Δ(1/T)| × 1000 / R × prior_ea_sd_kj
                         = |∂(ln k_pred)/∂Ea| × σ_Ea

        この設計は B1.5 検証 (`docs/audit/b1_5_split_determination.md`,
        commit 0f29271) で独立確認済。出力 95% CI は SL の対数空間で
        ±1.960 × σ_lnk_total として構築され (L646-649)、ICH Q1E §2.6
        "lower one-sided 95 percent confidence limit" と整合する。

        感度分析結果は tests/test_bayesian_stability.py::test_prior_sd_sensitivity
        を参照 (SE × {1.0, 1.96, 2.5} の SL_lo95 単調性検証)。
    """
    warnings_list: List[Dict] = []

    # ── B4: PQ_N_POINTS_TOO_LOW(単温度モード、Phase B4)──────────────
    if len(times_days) != len(contents):
        raise ValueError("times_days と contents の長さが一致しません")
    n_pts_single = len(times_days)
    if n_pts_single <= PQ_N_POINTS_HARD_FAIL_MAX:
        raise StabilityHardFailWarning(
            code="PQ_N_POINTS_TOO_LOW",
            message=(
                f"PQ データ点数が {n_pts_single} 点と少なすぎます"
                f"(最低 {PQ_N_POINTS_HARD_FAIL_MAX + 1} 点必要)。"
                "OLS の標準誤差が縮退し、偽の確実性につながります。"
            ),
            detail={
                "n_points": n_pts_single,
                "min_required": PQ_N_POINTS_HARD_FAIL_MAX + 1,
            },
        )
    if n_pts_single <= PQ_N_POINTS_WARN_MAX:
        warnings_list.append(_warning(
            code="PQ_N_POINTS_TOO_LOW",
            level="warning",
            message=(
                f"PQ データ点数が {n_pts_single} 点です。"
                "傾きの推定誤差が大きい可能性があり(信頼区間が広がります)、"
                "可能であれば 4 点以上を推奨します。"
            ),
            n_points=n_pts_single,
            recommended_min=PQ_N_POINTS_WARN_MAX + 1,
        ))

    # A8: 単温度モード常時警告(L4.5、prior 依存性が高い)
    warnings_list.append(_warning(
        code="SINGLE_TEMP_PRIOR_DEPENDENCY",
        level="info",
        message=(
            "単温度モードは事前分布(prior)の選択に強く依存します"
            "(L4.5 検証: 同データで prior SD=5 → SL=43.4、SD=30 → SL=13.6)。"
            "事前分布の根拠を慎重に設定してください。"
            "可能であれば多温度モードの使用を推奨します。"
        ),
        prior_ea_kj=prior_ea_kj,
        prior_ea_sd_kj=prior_ea_sd_kj,
    ))

    # A6: Ea 事前範囲チェック(物理的妥当範囲、L3.5.3)
    if prior_ea_kj < EA_PRIOR_RANGE_MIN_KJ or prior_ea_kj > EA_PRIOR_RANGE_MAX_KJ:
        warnings_list.append(_warning(
            code="UNUSUAL_PRIOR_EA",
            level="warning",
            message=(
                f"指定された事前 Ea = {prior_ea_kj} kJ/mol が典型医薬品の範囲"
                f"({EA_PRIOR_RANGE_MIN_KJ:.0f}-{EA_PRIOR_RANGE_MAX_KJ:.0f} kJ/mol)から"
                "外れています。設定意図をご確認ください。"
            ),
            prior_ea_kj=prior_ea_kj,
            range_min=EA_PRIOR_RANGE_MIN_KJ,
            range_max=EA_PRIOR_RANGE_MAX_KJ,
        ))

    t = np.array(times_days, dtype=float)
    C = np.array(contents, dtype=float)

    c0_eff = float(C[int(np.argmin(t))]) if float(t[int(np.argmin(t))]) == 0.0 else c0
    y = np.log(C / c0_eff)

    slope, _, r_val, _, se_slope = stats.linregress(t, y)
    k_acc = float(-slope)
    se_k_acc = float(se_slope)
    r2 = float(r_val ** 2)

    # ── A4: k_acc ≤ 0 の silent 置換をフラグ化(多温度 K_HAT_ZERO_FALLBACK と対称)──
    # 単温度では加速条件の ln(C) vs t 一次回帰の傾きから k_acc を推定する。
    # Significant degradation 不在(平坦・増加データ)では k_acc ≤ 0 となり、
    # 従来は np.log(k_acc) が nan/-inf を生み、レスポンス描画時(allow_nan=False)
    # に 500 化していた(2026-05-25 障害)。多温度モードと同じく 1e-7 へ
    # フォールバックして警告化する(有効期間は 120 ヶ月キャップに収束)。
    k_zero_fallback = False
    if (not math.isfinite(k_acc)) or k_acc <= 0:
        k_acc = 1e-7
        se_k_acc = 1e-7
        k_zero_fallback = True
        warnings_list.append(_warning(
            code="K_HAT_ZERO_FALLBACK",
            level="warning",
            message=(
                f"加速温度 {temp_c}°C で有意な分解が観察されませんでした"
                "(k ≤ 0 または推定不能)。Significant degradation 不在のため"
                "単温度ベイジアン外挿の前提を満たしていません。"
                "ICH Q1E §B.1 古典手法をご検討ください。"
            ),
            acc_temp_c=temp_c,
        ))

    # ── A5: 一次反応速度フィットの直線性が低い場合の警告 ─────────────────
    # 単温度モードに Arrhenius プロットは存在しない(温度水準が 1 点のため)。
    # ここで評価するのは加速条件の ln(C) vs t 一次回帰の R²(= 一次分解モデルの
    # 妥当性)であり、アレニウス直線性とは別物。CLAUDE.md「命名は仕様」ルールに
    # 従い LOW_R_SQUARED_ARRHENIUS ではなく LOW_R_SQUARED_KINETIC_FIT とする。
    # 閾値は多温度の直線性閾値(R_SQUARED_ARRHENIUS_WARN = 0.9)を流用。
    if math.isfinite(r2) and not k_zero_fallback and r2 < R_SQUARED_ARRHENIUS_WARN:
        warnings_list.append(_warning(
            code="LOW_R_SQUARED_KINETIC_FIT",
            level="warning",
            message=(
                f"加速条件の一次反応フィットの直線性が低い(R² = {r2:.3f})。"
                "一次分解モデルの妥当性が低く、外挿の信頼性が下がる可能性があります。"
                "観測点の追加や反応次数の再検討をご検討ください。"
            ),
            r_squared=round(r2, 4),
            threshold=R_SQUARED_ARRHENIUS_WARN,
        ))

    R_gas = R_GAS
    T_acc = temp_c + 273.15
    T_pred = pred_temp_c + 273.15
    delta_inv_T = 1.0 / T_pred - 1.0 / T_acc

    ln_k_pred = np.log(k_acc) - prior_ea_kj * 1000.0 / R_gas * delta_inv_T
    k_pred = float(np.exp(ln_k_pred))

    sigma_lnk_acc = abs(se_k_acc / k_acc) if k_acc > 0 else 0.0
    sigma_ea_contrib = abs(delta_inv_T) * 1000.0 / R_gas * prior_ea_sd_kj
    sigma_lnk_total = float(np.sqrt(sigma_lnk_acc ** 2 + sigma_ea_contrib ** 2))

    # 加速係数: k_acc / k_pred = exp(Ea/R * delta_inv_T)  (T_acc > T_pred のとき > 1)
    acc_factor = float(np.exp(prior_ea_kj * 1000.0 / R_gas * delta_inv_T))

    def shelf_life_months(k: float) -> float:
        return -np.log(spec_lower / c0_eff) / k / 30.44

    z95 = 1.960
    sl_mean = shelf_life_months(k_pred)
    sl_lo95 = sl_mean * np.exp(-z95 * sigma_lnk_total)
    sl_hi95 = sl_mean * np.exp(+z95 * sigma_lnk_total)

    # A7: 120 ヶ月キャップ到達フラグ(L3.5.3 / L4.7)
    # bool()/float() で numpy 型を Python 組込み型へ確定変換する。
    # np.bool_ は FastAPI の JSONResponse(allow_nan=False)でシリアライズ
    # 不可のため、これを怠ると全リクエストが 500 化する(2026-05-25 障害)。
    capped = bool(sl_lo95 >= SHELF_LIFE_CAP_MONTHS)
    sl_lo95_reported = float(min(sl_lo95, SHELF_LIFE_CAP_MONTHS))
    if capped:
        warnings_list.append(_warning(
            code="SHELF_LIFE_CAPPED",
            level="info",
            message=(
                f"外挿が {SHELF_LIFE_CAP_MONTHS:.0f} ヶ月キャップに到達しました。"
                "実用上は『有効期間 120 ヶ月以上』と解釈してください。"
                "データの分解傾向が弱く、Significant degradation の前提を満たさない可能性があります。"
            ),
            cap_months=SHELF_LIFE_CAP_MONTHS,
        ))

    # 返却前に numpy 型を Python float/bool へ確定変換する(原因A の再発防止)。
    # (cmc-platform 本番ではルーター層の _sanitize_response が nan/inf 最終防御を
    #  担うが、その serialization glue は本 vendored コピーには含めない。)
    return {
        "shelf_life_mean_months": float(round(sl_mean, 1)),
        "shelf_life_lo95_months": float(round(sl_lo95_reported, 1)),
        "shelf_life_hi95_months": float(round(min(sl_hi95, SHELF_LIFE_CAP_MONTHS), 1)),
        "recommended_shelf_life_months": float(round(sl_lo95_reported, 1)),
        "capped": capped,
        "k_zero_fallback": k_zero_fallback,
        "k_acc": float(round(k_acc, 6)),
        "k_pred": float(round(k_pred, 8)),
        "r2_acc": float(round(r2, 4)),
        "acc_factor": float(round(acc_factor, 2)),
        "acc_equiv_months": float(round(float(max(t)) / 30.44 * acc_factor, 1)),
        "sigma_lnk_acc": float(round(sigma_lnk_acc, 4)),
        "sigma_ea_contrib": float(round(sigma_ea_contrib, 4)),
        "sigma_lnk_total": float(round(sigma_lnk_total, 4)),
        "pred_temp_c": pred_temp_c,
        "acc_temp_c": temp_c,
        "spec_lower": spec_lower,
        "warnings": warnings_list,
    }
