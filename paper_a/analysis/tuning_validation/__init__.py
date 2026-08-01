"""MCMC 調律検証 (tuning validation).

本体 estimator (paper_a/analysis/estimators/mcmc.py) を一切改変せず、
参照温度再パラメータ化が「同一モデル・異なる幾何」であることを検証し、
5 条件 × セルのペア設計で非収束ゲートの内訳を定量する検証専用パッケージ.

本パッケージのコードは論文本文の数値には寄与しない (検証のみ).
非収束ゲートの閾値は mcmc.py から import して再利用する (バイト一致保証).
"""
