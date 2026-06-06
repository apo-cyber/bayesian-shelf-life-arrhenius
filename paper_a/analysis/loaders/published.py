"""実測公表データの空シグネチャ + TODO (仕様書 §5.6、再開プロンプト §E).

3 候補 (2026-05-19 §5.6 で確定):
1. **aspirin (Ozyilmaz, EMU J Pharm Sci 2025;8(1):1-7)**: 主軸ケース.
   4 温度 (21/37/45/60℃) × 5 時点 (0/30/60/90/120 分) の残存濃度%.
   1 次・Arrhenius 成立 (r=0.97-0.99、Ea=7.48 kcal).dergipark オープンアクセス.
2. **resveratrol (Biagini, Symmetry 2024;16:493)**: 頑健性層・非 Arrhenius.
   25/30/40℃、CC BY.著者結論は non-Arrhenius/super-Arrhenius.
3. **Tamura 博士論文 2020 第二章**: 最有力候補 (apo-cyber 提供).
   アセチルサリチル酸混合末、60 条件 (3 温度×4 湿度×5 MgSt).
   0 次・拡張 Arrhenius、長期外挿検証込み.研究公正性: 他者公表データ.

本セッションでは実装しない (仕様書 §10.4 ステップ 6).空シグネチャと TODO
のみを配置し、将来セッションで data_class="published_literature" タグ付き
truth を作成する.
"""
from __future__ import annotations


def load_aspirin_ozyilmaz_2025() -> tuple[list[dict], dict]:
    """aspirin (Ozyilmaz 2025、4 温度 × 5 時点、1 次・Arrhenius 成立).

    Returns
    -------
    (data_rows, truth)
        data_rows: DataRow 規約準拠の観測行.
        truth: data_class="published_literature", citation=出典書誌、ea_true=
               文献記載値、t90_true=文献から再計算した真値 (or 文献記載値).

    TODO (将来セッション、仕様書 §10.4 ステップ 6):
        Ozyilmaz et al. EMU J Pharm Sci 2025;8(1):1-7 の Table 1 から
        4 温度 × 5 時点の残存濃度%、ln C を転記.列名は cmc-platform DataRow
        規約に合わせる (temperature/time_months/content_percent).時間は分単位
        → 月単位に換算注意.研究公正性 §10.3: data_class="published_literature",
        citation を truth に明示.apo-cyber 自身の実測と書かない.
    """
    raise NotImplementedError(
        "実装未着手 (仕様書 §10.4 ステップ 6、クリティカルパス外).将来セッションで "
        "Ozyilmaz 2025 Table 1 の数値を転記."
    )


def load_resveratrol_biagini_2024() -> tuple[list[dict], dict]:
    """resveratrol (Biagini, Symmetry 2024;16:493、頑健性層・非 Arrhenius).

    TODO: Symmetry 2024 Table 2 から 25/30/40°C の各時点濃度を転記.
    data_class="published_literature", non_arrhenius=True を truth に明示.
    """
    raise NotImplementedError(
        "実装未着手.将来セッションで Biagini 2024 Table 2 の数値を転記."
    )


def load_tamura_2020_chapter2() -> tuple[list[dict], dict]:
    """Tamura 博士論文 2020 第二章 (アセチルサリチル酸混合末、60 条件).

    最有力実測候補 (apo-cyber アップロード).Table 12-14 の 3 温度 × 4 湿度 ×
    5 MgSt = 60 条件、時点別の類縁物質総量% と k.最小二乗 Arrhenius が
    論文 A 方法論と同型.0 次・拡張 Arrhenius.長期外挿検証 (Table 14) 込み.

    TODO: 湿度・MgSt 含量を固定して温度のみ Arrhenius 部分を抽出した
    n_T=3 主軸データに整形.研究公正性 §10.3: "文献から取得した公表データ
    (Tamura 博士論文 2020 / Tamura et al. Chem Pharm Bull 2020)" と記述、
    apo-cyber 自身の実測と書かない.
    """
    raise NotImplementedError(
        "実装未着手.将来セッションで Tamura 2020 第二章 Table 12-14 の数値を転記、"
        "湿度・MgSt 固定で温度 Arrhenius 抽出."
    )
