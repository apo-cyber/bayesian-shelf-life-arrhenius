# 論文 A 確定パラメータ・設計判断一覧 (2026-05-20)

仕様: `docs/strategy/paper-A-dataset-spec-DRAFT.md`
判断元: 元プロンプト §3 + apo-cyber 判断確定 (Q1〜Q3) + 本セッションでの追加判断.

本ファイルは論文 Methods 節の素材として自己完結にまとめる.数値リテラル
直書き禁止 (§10.2) 原則の補助索引であり、権威ソースは
`paper_a/datagen/config.py` 等のコードに常駐する.

---

## (1) 確定数値パラメータ一覧

### 1.1 速度論定数

| パラメータ | 記号 | 値 | 出所・備考 |
|------------|------|----|-----------|
| 気体定数 | R | 8.314 J/(mol·K) | SI |
| 参照温度 (25°C) | T_25K | 298.15 K | ICH 長期保存 |
| 修正 Arrhenius 参照温度 | T_ref | 313.15 K (=40°C) | n=±4 因子の中心点 |

### 1.2 真 Ea (kJ/mol)

| 名称 | 記号 | 値 | 出所 |
|------|------|----|------|
| Ea 低 (Faya 低) | EA_TRUE_KJ_LOW | 67.0 | 16 kcal/mol を kJ 換算 |
| Ea 中 (代表薬物) | EA_TRUE_KJ_MID | 80.0 | cmc-platform 既定と整合 |
| Ea 高 (Faya 高) | EA_TRUE_KJ_HIGH | 104.6 | 25 kcal/mol を kJ 換算 |

### 1.3 ノイズ (log 空間ガウス SD)

| 水準 | 値 |
|------|-----|
| small | 0.01 |
| medium | 0.02 (= `docs/audit/mcmc_benchmark.py` SIGMA_OBS) |
| large | 0.05 |

### 1.4 Prior 正確性 (真値からの kJ/mol 乖離)

| 水準 | オフセット |
|------|-----------|
| accurate | +0 |
| moderate | +10 |
| strong | +25 |

Prior SD は全 case 共通: **30 kJ/mol** (cmc-platform `run_bayesian_stability` 既定).

### 1.5 計算定数

| パラメータ | 値 | 備考 |
|------------|-----|------|
| 目標 SL at 25°C | 30.0 月 | spec_lower=95% 到達時間.全シナリオ共通 |
| spec_lower | 95.0% | 実務指標.t90 評価とは別軸 |
| initial_content | 100.0% | t=0 公称 |
| Avrami 指数 (induction) | n_avrami = 2.0 | S 字形誘導期の典型 |
| autocatalytic 加速係数 | alpha = 0.05 | dC/dt = -k C (1 + alpha (C0-C)) |
| autocatalytic 数値積分刻み | ~200 steps/unit time | Euler |
| 修正 Arrhenius 指数 (凹) | n = +4 (concave) | 高温で k 加速 |
| 修正 Arrhenius 指数 (凸) | n = -4 (convex) | 高温で k 緩和 |

### 1.6 n_T 温度設計 (°C)

| n_T | 温度群 |
|-----|--------|
| 2 | [40, 60] |
| 3 | [40, 50, 60] |
| 4 | [40, 50, 60, 70] |

### 1.7 n_points 時間グリッド (月)

| n_points | 時間群 |
|----------|--------|
| 3 | [0, 3, 6] |
| 4 | [0, 2, 4, 6] |
| 6 | [0, 1, 2, 3, 4, 6] |

### 1.8 反復数

| 層 | シナリオ数 | 反復/シナリオ | 総行数 |
|----|-----------|--------------|--------|
| 中核 | 81 | 1000 | 1,053,000 |
| 頑健性 | 20 | 1000 | 240,000 |

---

## (2) 仕様書 §2.1 からの逸脱: 頑健性 20 を 4×5 とした凹/凸 split

**仕様書 §2.1 表記**:
> 分解速度論 {一次,二次,自触媒,誘導期} × 温度依存性 {Arrhenius 低/中/高 Ea, 非 Arrhenius}

素直に読むと 4 (kinetics) × 4 (temp_dep) = **16** だが、仕様書本文は **20** と
明記している.「20」確定値と整合させる解釈として実装は 4×5 = 20 を採用:

| 軸 | 水準 (×5) |
|----|-----------|
| 速度論 (×4) | first_order / second_order / autocatalytic / induction (Avrami n=2) |
| 温度依存性 (×5) | arrhenius_low_ea / arrhenius_mid_ea / arrhenius_high_ea / **non_arrhenius_concave** / **non_arrhenius_convex** |

**凹/凸 split の物理的根拠**:

| 区分 | k(T) 振る舞い | 想定機構の典型例 |
|------|--------------|----------------|
| non_arrhenius_concave (T^+4) | 高温で予測より速く k 増加 | MgSt 触媒分解、固体水分加水分解、相転移加速、結晶水脱離前後 |
| non_arrhenius_convex (T^-4) | 高温で予測より遅く k 増加 | 拡散律速 (固体相内拡散の遅延)、ガラス転移近傍の運動性凍結 |

**設計上の意義**:
- 「非 Arrhenius」を 1 種にまとめると、推定器の頑健性を片方向 (例えば加速側のみ)
  でしか試せず、対称性検証が成立しない.両方向を別水準として持つことで
  「2 段 OLS/共役回帰が方向に依存せず Arrhenius 逸脱を吸収できるか」を試せる.
- 数学的実装は modified Arrhenius (T^n 因子) で n=±4 を採用
  (`paper_a/datagen/temperature.py::modified_arrhenius_concave/convex`).
  n=±4 は経験的に「k(T_max) が標準 Arrhenius 予測の 2 倍/半分程度」をもたらす
  実験的検出可能領域に対応する.

**後続作業**: 本判断を仕様書 §2.1 に追記する.20 確定値と 4 軸 × 5 軸の対応を明示.

---

## (3) kinetics-aware 較正の設計判断: 案 A(目標 t90 先決め → k_25 逆算)を採った理由

**選択肢の比較**:

| 案 | 概要 | 採否 |
|----|------|------|
| **A** | 目標 SL/t90 を 25°C で先に決め、kinetics 個別に k_25 を逆算する | **採用** |
| B | 真の k_25 を固定し、kinetics 個別の真 SL/t90 を成り行きで決める | 不採用 |

**採用理由 (案 A)**:

1. **仕様書 §2.1「中核代表点を固定して副因子を振る」思想と整合**.頑健性層の
   役割は「速度論誤特定下でも推定器が崩れないか」を試すことで、SL 真値が
   シナリオ間で大きく異なると比較が成立しない (k 固定だと 1 次と 2 次で
   SL が桁違いになる).
2. **Faya 2018 Fig 4.5 形式の比較がフェア**.Faya は t90 を真値として固定し
   推定器のバイアス・ばらつきを比較する.本実装も同じく真 t90 を全 case で
   揃えることで Fig 4.5 と同じ評価軸が立つ.
3. **真値メタデータの解釈が一貫**.target_sl_at_25c_months = 30 が「全 case
   共通の真値」として論文 Methods に書ける.案 B では「kinetics に依存して
   真値が変わる」と書く必要があり、推定器バイアス評価の基準点として弱い.

**案 B 不採用の理由**:
- k_25 を全 kinetics で 0.001710 /月 (1 次計算値) に固定すると、2 次反応では
  真 SL ≈ 0.3 月、誘導期では真 SL ≈ 11 月 など、桁が揃わず推定器ごとの
  バイアス比較が解釈困難.
- 加速温度の k_T も大きく変動し、合成データの観測 content が現実的範囲を
  外れる (例: 60°C で content が数日で 0% に達する).

**実装**: `paper_a/datagen/temperature.py::calibrate_lna_for_true_sl`.
速度論ごとに k_25 を解析解または brentq 数値解で逆算 → Arrhenius で ln A 確定.

---

## (4) target_sl / t90_true / sl_at_spec_true の使い分け方針

3 つの SL 量を truth.json に並列記録する.役割が異なる.

| 量 | 定義 | 役割 |
|----|------|------|
| `target_sl_at_25c_months` | 較正入力 (=30 月、全 case 共通) | パラメータ入力の記録.次節 §5.3 の整合確認用 |
| `sl_at_spec_true_25c_months` | content=spec_lower (95%) 到達時間 (真の速度論で数値解) | **実務指標**.target との一致確認 (1 次は厳密一致、自触媒は brentq 数値誤差 < 1e-9) |
| `t90_true_25c_months` | content=90% 到達時間 (真の速度論で数値解) | **論文主指標**.Faya 2018 Fig 4.5 と同じ評価軸 |

**論文での使い分け** (apo-cyber 指示、2026-05-20):
- **主指標 = `t90_true_25c_months`**.推定器バイアス・ばらつき・カバレッジ・
  posterior predictive 失敗率 (仕様書 §4 三指標) の評価は全て t90_true に対して行う.
- **補助指標 = `sl_at_spec_true_25c_months`**.実務 SL の文脈で言及するが、
  推定器評価の主軸には置かない.target との完全一致を確認することで
  較正アルゴリズムの正しさを示す.
- 加速温度ごとの真 t90 (`t90_true_by_temp_months`) は、推定器が「中間温度の
  k_T を真値からどれだけ外しているか」の副次的診断に使う.

**例 (robust_14: 自触媒 + 非 Arrhenius_concave)**:
- target_sl_at_25c = 30 月 (入力)
- sl_at_spec_true_25c = 29.999... 月 (target と一致、brentq 精度)
- t90_true_25c = 55.84 月 (1 次の 61.62 月より短い = 自触媒加速の物理)
- 主指標は 55.84 を真値として推定器の t90 推定値と比較する.

---

## (5) `true_sl_method` 記述フォーマット標準 (5 経路を式で記述)

全 truth.json の全 case で同じ詳細度・同じ 5 キーを持つ dict を記録する
(査読耐性).記述は実装と同期する自動生成
(`paper_a/datagen/generate.py::_true_sl_method_description`).

### フォーマット (固定 8 キー)

```json
{
  "calibration_chain": "target_sl_at_25c_months -> k_25 -> ln A (Arrhenius) -> k(T) at each temperature -> t_SL(T) / t_90(T)",
  "k_25_formula":      "<kinetics 個別の k_25 逆算式>",
  "lna_formula":       "<k_of_t 別の ln A 算出式 (Arrhenius/修正 Arrhenius)>",
  "k_at_temp_formula": "<k(T) の温度依存式>",
  "true_sl_formula":   "<spec_lower 到達時間の算出式>",
  "true_t90_formula":  "<content=90% 到達時間の算出式>",
  "computed_k_25":     <数値、ケース固有>,
  "computed_ln_a":     <数値、ケース固有>
}
```

### 各経路のサンプル (kinetics × k_of_t 組み合わせ別)

| 経路 | first_order × arrhenius | autocatalytic × modified_arrhenius_concave |
|------|------------------------|-------------------------------------------|
| k_25_formula | `k_25 = -ln(spec/initial)/target_sl` | `brentq, C(target_sl; k_25, autocat alpha=0.05) = spec; ODE: dC/dt = -k C (1 + alpha(C0-C))` |
| lna_formula | `ln A = ln(k_25) + Ea*1000/(R*T_25K)` | `ln A = ln(k_25) + Ea*1000/(R*T_25K) - 4*ln(T_25K/T_ref)  (T_ref=313.15K, n=+4)` |
| k_at_temp_formula | `k(T) = exp(ln A - Ea*1000/(R*T_K))` | `k(T) = (T/T_ref)^4 * exp(ln A - Ea*1000/(R*T_K))` |
| true_sl_formula | `t_SL(T) = -ln(spec/initial)/k(T)` | `brentq on integrated trajectory crossing spec` |
| true_t90_formula | `t_90(T) = -ln(0.9)/k(T)` | `brentq on integrated trajectory crossing 90%` |

全 4 kinetics × 3 k_of_t 組み合わせの完全文字列は実装
(`_true_sl_method_description`) を参照.組み合わせは 12 通りで、全 case が
そのいずれかに該当する.

### 査読耐性上の意味

- 数値だけでなく算出経路を truth に同梱することで、第三者が単純電卓・
  scipy で truth 値を再計算できる.
- 「真値が定まっている」という前提を論文 Methods に書く際、計算手順を
  truth.json 同梱で示すことで、レビュアーが計算過程を確認可能.
- 既知の落とし穴 (§7 段階 5 教訓「関数名と実装の乖離」型のエラー) を
  truth レベルで予防する (記述と数値が齟齬を起こすと CI で検出).

---

## (6) 補足: シード・実行間再現性

シード生成: `numpy.random.default_rng(int.from_bytes(sha256("{case_id}|{replicate_id}").digest()[:4], "big"))`

Python の `hash()` は文字列に対して PYTHONHASHSEED で randomize されるため
hashlib を採用.これにより 101 シナリオ × 1000 反復が PYTHONHASHSEED に依存せず
**実行間で bit-identical** に再現される.

### 中核 81 再生成の bit-identity 検証 (2026-05-20)

kinetics-aware 較正改修の影響範囲確認:
- 中核 81 (全 first_order) → 較正経路が 1 次の場合に同等のため data.csv bit-identical
- 頑健性 20 → kinetics ごとに k_25 が変わるため再生成必要

検証結果:
| 項目 | 値 |
|------|-----|
| backup MD5 | `fbf96172cab77352f3e46b21d897c3f3` |
| post-refactor MD5 | `fbf96172cab77352f3e46b21d897c3f3` |
| diff exit_code | 0 (差分なし) |

頑健性 20 は本セッションで再生成完了 (新較正で真 SL=target 保証、各 case の
真 t90 が kinetics 物理に従って算出される).データの値は旧版と変わるが、
シードは同一なのでノイズ実現値は決定的に再現可能.

---

## (7) 列名規約 (DataRow 互換、外部設定差替可)

```
case_id, replicate_id, temperature, time_months, content_percent
```

cmc-platform `DataRow` (`temperature` / `time_months` / `content_percent`) と互換.
`COLUMN_NAMES` 辞書 (`paper_a/datagen/config.py`) で差替可能.メタ部分は別ファイル
(truth.json).

---

## (8) 4 推定器とファイル所在 (仕様 §B 確定、未実装は計画のみ)

| # | 呼称 (論文中・コード) | 所在 | 実装状況 |
|---|-----------------------|------|---------|
| 1 | `two_stage_conjugate` | `paper_a/analysis/estimators/two_stage_conjugate.py` | 計画のみ |
| 2 | `mcmc` | `paper_a/analysis/estimators/mcmc.py` | 計画のみ |
| 3 | `classical_ols_multi_temp` | `paper_a/analysis/estimators/classical_ols_multi_temp.py` | 計画のみ (analysis 配下新規最小実装) |
| 4 | `classical_ich_q1e` | `paper_a/analysis/estimators/classical_ich_q1e.py` | 計画のみ |

`bayesian_full.py` は作らない (元プロンプト記載分を案 C で破棄).
t 補正版 2 段 OLS は感度解析専用に `paper_a/analysis/sensitivity/t_correction.py` で保持予定.

---

## (9) 暫定状態の注記

**Faya 2018 包含 (仕様 §2.2) の運用**: 中核 81 自体は Ea=80 (mid) 固定で Ea 軸を
持たない.E ∈ {16, 25} kcal/mol = {67, 104.6} kJ/mol は頑健性 20 に
arrhenius_low_ea / arrhenius_high_ea として包含.仕様書本文「中核 81 の真部分集合」
と厳密整合させるには中核設計に Ea 軸を 1 段追加する必要があるが、本セッションは
apo-cyber 指示 (中核 81 は影響範囲外) に従い再設計を保留.次セッションで仕様書側を
「Ea は頑健性層で網羅、中核は Ea 固定で他 4 軸を体系化」と明示するか、中核を拡張するかを判断する.

`paper_a/tests/test_faya_subset.py` は現状の運用 (Faya の Ea 値が中核+頑健性の
いずれかに存在) を保証するメタテストとして配置.

---

## 出力ファイル (生成器)

```
paper_a/data/core/data.csv               1,053,000 行 (38 MB、加速試験)
paper_a/data/core/long_term_25c.csv        351,000 行 (12 MB、25°C 長期)
paper_a/data/core/truth.json             81 cases (新スキーマ)
paper_a/data/robustness/data.csv           240,000 行 (8.9 MB、加速試験)
paper_a/data/robustness/long_term_25c.csv   80,000 行 (3 MB、25°C 長期)
paper_a/data/robustness/truth.json       20 cases (新スキーマ)
```

全 case で `data_class: "synthetic"` ラベルが truth に明示される (研究公正性 §10.3).

---

# Part II — 解析パイプライン (paper_a/analysis、2026-05-21 確定)

## (10) 4 推定器・実行設定 (案 B + numpyro)

実行構成 (再開プロンプト判断・前停止点 D.16 / D.3 確定):

| # | 推定器 (固定呼称) | 入力 | 1 case あたり rep | 計算コスト/rep |
|---|------------------|------|-----------------|--------------|
| 1 | `two_stage_conjugate` | 加速試験 (n_T 別) | **1000** | ~0.7 ms |
| 2 | `mcmc` (numpyro NUTS) | 加速試験 | **100** (案 B) | ~2.9 s |
| 3 | `classical_ols_multi_temp` | 加速試験 | 1000 | ~0.3 ms |
| 4 | `classical_ich_q1e` | **25°C 長期 (36 ヶ月)** | 1000 | ~0.1 ms |

総推定: **313,100** (101 cases × 3 acc × 1000 + 101 × 100 mcmc + 101 × 1000 ich).
所要: 非 MCMC 約 3 分 + MCMC numpyro 約 6.6 時間 = 約 6.7 時間 (本番一括夜間).

numpyro 同等性検証 (前停止点 D.16): PyMC default vs numpyro で t90 主指標 Δ<5%
(2.27/0.28/0.67%)、R-hat 一致、ESS 同等、4.5x speedup.詳細は §3 と
`paper_a/analysis/estimators/mcmc.py` docstring.

---

## (11) Faya Fig 4.5 整合と truth_value 単一性

中核 81 は全 case で `target_sl_at_25c_months=30 × first_order × Arrhenius` 設計
のため、`t90_true_25c_months` は **61.6224 月の単一値**.
`paper_a/analysis/figures.py::faya_fig_4_5_draft` は `truth_uniform=True` を
検出して水平線 1 本で表示 (median 計算は冗長な fallback).

頑健性 20 は kinetics 別に t90_true が分散 (例: 自触媒で 55.84、誘導期で 42.98)
するため、case 別に truth_by_case[case_id]['t90_true_25c_months'] を bias 評価
に使う (`paper_a/analysis/metrics.py::compute_cell_metrics`).

---

## (12) 主指標と 3-flavor bias_sd (前停止点 D.3 追加要求 b 反映)

**主指標 = `bias_sd_capped120`**.t90 推定値を 120 ヶ月 cap してから bias_sd
計算 (cmc-platform `_cap` と整合).論文 Results 主節で報告.

3 flavor 全種:
- **bias_sd_raw**: 全推定成功 rep の bias_sd.外れ値で inf になりうる.Methods 注釈用.
- **bias_sd_capped120 (主)**: cap 後.外れ値耐性ありで実務的解釈可能.Table 1 値.
- **bias_sd_converged_only**: converged=True の rep のみ (MCMC 用、収束時の精度).
  Discussion の補助分析「MCMC は収束した時だけなら精度同等」を裏付け.

数値発見 (n_T=3 × prior=strong 核心 cell):
- raw: 2-stage 32.46, MCMC **inf**, classical_ols 1.47×10⁶
- **cap120: 2-stage 32.46, MCMC 31.41** (両者ほぼ同等), classical_ols 36.87
- conv_only: 2-stage 32.46, MCMC **inf** (cap せず収束のみで raw と同型)

「raw で MCMC 劣位」と「cap 後で MCMC ≈ 2-stage」の差は **少数の posterior 長尾
extreme outlier** に由来.Methods / Discussion 両方で記述する素材.

---

## (13) 論文 narrative 3 層構造 (D.3 確定)

apo-cyber 判断 (2026-05-21、選択肢 4): classical_ich_q1e 主軸化は採らない
(apples-to-apples でない、§3.1 文脈引用確定、CMC 実務読者の常識で新規性なし).

### Layer 1: Results 主節 — 中核 81 (主戦場、§3.3.1 新規性そのまま)

> **「2-stage OLS/共役回帰が低 n_T で MCMC に精度匹敵 vs MCMC より実務的にロバスト」**

数値根拠 (n_T=3 × prior=strong):
- **精度同等**: bias_sd_capped120 で 2-stage 32.5 ≈ MCMC 31.4
- **ロバスト性で 50 倍差**: 失敗率 2-stage **0.9%** vs MCMC **45.2%**
- n_T=4 でも保持: 2-stage 0.1% vs MCMC 18.4% (180 倍差)

Figures (論文 Results 主節):
- **Figure 1**: Fig 4.5 全体 (9 cell × 4 推定器、`fig_4_5_draft.png`)
- **Figure 2**: n_T=3 拡大 (核心 cell 拡大、`fig_zoom_n_t_3.png`)
- **Figure 3**: MCMC 非収束ヒートマップ (中核 81 限定、Faya 未報告独立貢献、
  `fig_mcmc_nonconvergence.png`)
- **Table 1**: cap 後 bias_sd と失敗率 (4 推定器 × 9 cell マトリクス)

### Layer 2: Results 副節 + Discussion — 頑健性 20 (適用範囲の誠実開示)

> 「1 次仮定が成立する範囲では 2-stage 圧倒、誘導期等の極端な速度論誤特定では
> 2-stage 失敗率が増大 (53%、Avrami n=2 で 1 次仮定が大きく崩れるため)」

これは論文の弱点ではなく **査読耐性を上げる誠実性**.Faya 2018 が単純 OLS 比較で
止めたところを、本論文は 2-stage/共役の適用範囲を kinetics 多様性に対して
体系的に検証する.

Table 2 (Results 副節): 速度論種別ごとの 4 推定器挙動 (`robustness_by_kinetics`).
特異 case の挙動 (robust_14 等) は Discussion で個別議論.

### Layer 3: Discussion — 推定器棲み分け

| 推定器 | 適用シーン | 限界 |
|--------|----------|------|
| 2-stage_conjugate (本論文主軸) | 加速試験、n_T≥3、1 次反応近似が妥当 | 誘導期等の極端速度論誤特定で失敗率上昇 |
| mcmc | 加速試験、計算資源が許す | Prior 強乖離で 45% 非収束、極端 case で長尾事後 |
| classical_ols_multi_temp | 加速試験、簡易ベンチマーク | 低 n_T で CI 発散、Faya 2018 と同知見 |
| classical_ich_q1e | **25°C 長期データ専用、規制 baseline** | 加速試験には適用不可、3 年待つ前提 |

**本研究の貢献位置づけ**: 加速試験データの 2-stage 信頼性予測.
classical_ich_q1e は加速試験データを扱えないため棲み分け対象 (Methods で
apples-to-apples でない点を明示).

---

## (14) 残課題 (future work、Discussion で誠実記述)

1. **MCMC 非収束率の n_T × prior 別構造**: 中核で 45-71% (n_T=2-3, prior=strong)、
   頑健性で 25.1% (n_T=3 固定).Prior 乖離だけでは説明不可、kinetics 種別の
   影響も介在.target_accept 緩和等の補正は cmc-platform 監査結果継承から外れる
   ため不採用 (前停止点 D.3 判断 c).

2. **誘導期 (Avrami n=2) での 2-stage 失敗率 53%**: 1 次仮定崩壊が定量的に
   どのレベルから致命的になるかは未解明.次セッションの Discussion 執筆時に
   原因究明を試みる.

3. **MCMC を 100 → 1000 rep に拡張するかは投稿後リビジョン対応で判断**.
   現 100 rep でも箱ひげ位置・幅は安定 (前停止点 D.3 判断 b).

4. **実測公表データ (aspirin / resveratrol / Tamura 2020) 組込**: 仕様書 §5.6
   確定済 3 件.本セッションでは空シグネチャ + TODO のみ
   (`paper_a/analysis/loaders/published.py`).仕様書 §10.4 ステップ 6 で実装.

---

## (15) 出力ファイル (解析)

```
paper_a/results/estimator_results.parquet  313,100 推定行 (~35 MB、.gitignore 除外)
paper_a/results/cell_metrics.json          全スライス指標 (~70 KB、同梱)
paper_a/figures/fig_4_5_draft.png          Figure 1 全体図
paper_a/figures/fig_zoom_n_t_3.png         Figure 2 核心拡大
paper_a/figures/fig_mcmc_nonconvergence.png Figure 3 ヒートマップ (Faya 未報告)
```

集計スライス (`cell_metrics.json` の keys):
- `core_all`, `core_low_n_t_strong_prior`, `core_n_t_x_prior`
- `robustness_all`, `robustness_by_kinetics`, `robustness_by_k_of_t`,
  `robustness_per_case`
