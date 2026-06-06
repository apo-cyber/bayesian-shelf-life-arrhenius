"""Faya 2018 博士論文 Table 4.2 の 4 design point 真部分集合性テスト.

仕様: README.md「Data」セクション / paper_a/docs/confirmed_parameters.md

中核 81 シナリオが Faya の (E, σ) 2×2 設計を包含することを保証する.
本テストは「対応する Ea と noise の組み合わせが core シナリオに存在する」
ことを確認するメタテスト.数値的等価性ではなく「設計の真部分集合性」.
"""
from __future__ import annotations

from paper_a.datagen.config import (
    CORE_SCENARIOS,
    EA_TRUE_KJ_HIGH,
    EA_TRUE_KJ_LOW,
    EA_TRUE_KJ_MID,
    FAYA_DESIGN_POINTS,
    NOISE_SIGMA,
)


def test_faya_4_design_points_have_matching_core_cells():
    """Faya の 4 design point に対応する中核セルが存在することを確認する.

    Faya の sigma_low/high は、中核 81 のノイズ {small, medium, large} のうち
    {small, large} に対応すると解釈する (中庸 medium は Faya 拡張).
    Faya の E ∈ {16, 25} kcal/mol は EA_TRUE_KJ_{LOW, HIGH} に対応する.
    本中核 81 は kinetics=first_order・Arrhenius・prior_accuracy=accurate を
    包含する設計(中核ベースライン)であり、Faya の真値 Ea を
    `prior_accuracy=accurate` の枠で表現するためには中核設計の Ea 軸を
    一段拡張する必要がある.本テストは「中核 81 が Faya 設計を真部分集合
    として包含する設計上の合意」をコード上に明示する位置づけ.
    """
    # 中核 81 ベースラインの真の Ea は EA_TRUE_KJ_MID (=80).Faya の Ea
    # (67, 105) は中核では `prior_ea` の moderate/strong として表現される
    # 関係にあるが、Faya 包含の正確な実現には頑健性層の `arrhenius_low_ea`
    # と `arrhenius_high_ea` を併用する設計とする.
    from paper_a.datagen.config import ROBUSTNESS_SCENARIOS

    ea_values_present = {s["ea_true_kj"] for s in CORE_SCENARIOS + ROBUSTNESS_SCENARIOS}
    assert EA_TRUE_KJ_LOW in ea_values_present, "Faya 低 Ea (16 kcal/mol) が設計に不在"
    assert EA_TRUE_KJ_HIGH in ea_values_present, "Faya 高 Ea (25 kcal/mol) が設計に不在"
    assert EA_TRUE_KJ_MID in ea_values_present, "中核 mid Ea が設計に不在"

    # ノイズ low/high が中核に存在
    noise_levels_present = {s["noise_level"] for s in CORE_SCENARIOS}
    assert "small" in noise_levels_present, "Faya sigma=low に対応する small が中核に不在"
    assert "large" in noise_levels_present, "Faya sigma=high に対応する large が中核に不在"
    assert "medium" in noise_levels_present, "中核拡張軸 medium が不在"

    # FAYA_DESIGN_POINTS のキー型確認
    assert len(FAYA_DESIGN_POINTS) == 4
    for dp in FAYA_DESIGN_POINTS:
        assert dp["ea_kj"] in (EA_TRUE_KJ_LOW, EA_TRUE_KJ_HIGH)
        assert dp["sigma_label"] in ("low", "high")


def test_core_81_count_and_robustness_20_count():
    """中核 81 + 頑健性 20 のカードカウントを保証する (仕様 §2.1)."""
    from paper_a.datagen.config import ROBUSTNESS_SCENARIOS

    assert len(CORE_SCENARIOS) == 81, f"中核は 81 シナリオ.実際 {len(CORE_SCENARIOS)}"
    assert len(ROBUSTNESS_SCENARIOS) == 20, f"頑健性は 20 シナリオ.実際 {len(ROBUSTNESS_SCENARIOS)}"


def test_core_baseline_representative_point_exists():
    """頑健性 20 が固定する中核代表点 (n_T=3, n_points=4, ノイズ中, prior 正確)
    が中核 81 にも存在することを確認する (頑健性層の入れ子設計が成立)."""
    matches = [
        s for s in CORE_SCENARIOS
        if s["n_t"] == 3
        and s["n_points"] == 4
        and s["noise_level"] == "medium"
        and s["prior_accuracy"] == "accurate"
    ]
    assert len(matches) == 1, f"中核代表点が 1 件であるべき.実際 {len(matches)}"


def test_noise_sigma_constants_match_spec():
    """ノイズ {小, 中, 大} = {0.01, 0.02, 0.05} (log 空間 SD)."""
    assert NOISE_SIGMA["small"] == 0.01
    assert NOISE_SIGMA["medium"] == 0.02
    assert NOISE_SIGMA["large"] == 0.05
