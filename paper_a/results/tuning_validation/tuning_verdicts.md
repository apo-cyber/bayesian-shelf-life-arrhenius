# 事前登録判定 (verdict)

- **中心セル nt3_strong × E**: rescued_by_tuning (nonconv=2.5% < 10%)
- **最悪セル nt2_strong × E**: not_irreducible (nonconv=19.2% <= 40%)
- **反証チェック (artifact)**: no_artifact (A=30.0% → B=12.5%, drop=17.5%); 幾何だけでは中心セルを基準まで救えない → 構造的主張は維持可

## prior 衝突寄与 (nt2_strong − nt2_accurate)
- 条件 A: strong=71%, accurate=65%, 寄与差=+6% → identification_dominant (accurate でも高非収束 → n_T 不足が主因)
- 条件 E: strong=19%, accurate=13%, 寄与差=+6% → mixed
