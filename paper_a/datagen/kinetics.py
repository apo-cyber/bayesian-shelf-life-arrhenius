"""分解速度論モデル: 一次 / 二次 / 自触媒 / 誘導期.

各モデルは content_percent(t, k, **params) を返す.
真値生成時はノイズなし.ノイズ付加は generate.py 側で log 空間ガウスを別途加算.
"""
from __future__ import annotations

import numpy as np


def first_order(t: np.ndarray, k: float, c0: float = 100.0, **_) -> np.ndarray:
    """C(t) = C0 * exp(-k t).  k の単位は 1/month."""
    return c0 * np.exp(-k * t)


def second_order(t: np.ndarray, k: float, c0: float = 100.0, **_) -> np.ndarray:
    """C(t) = C0 / (1 + k C0 t).  k の単位は 1/(% · month)."""
    return c0 / (1.0 + k * c0 * t)


def autocatalytic(t: np.ndarray, k: float, c0: float = 100.0, alpha: float = 0.05, **_) -> np.ndarray:
    """単純化した自触媒一次: dC/dt = -k C (1 + alpha (C0 - C)) を数値積分.

    alpha=0 で純粋一次に縮退.alpha>0 で分解が時間とともに加速.
    """
    t_arr = np.atleast_1d(t).astype(float)
    n_steps_per_unit = 200
    t_max = float(np.max(t_arr)) if t_arr.size > 0 else 0.0
    n_total = max(2, int(np.ceil(t_max * n_steps_per_unit)) + 1)
    t_dense = np.linspace(0.0, max(t_max, 1e-9), n_total)
    c = np.empty_like(t_dense)
    c[0] = c0
    for i in range(1, n_total):
        dt = t_dense[i] - t_dense[i - 1]
        rate = k * c[i - 1] * (1.0 + alpha * (c0 - c[i - 1]))
        c[i] = max(c[i - 1] - rate * dt, 1e-6)
    return np.interp(t_arr, t_dense, c)


def induction(t: np.ndarray, k: float, c0: float = 100.0, n_avrami: float = 2.0, **_) -> np.ndarray:
    """Avrami 型: C(t) = C0 * exp(-(k t)^n).  n>1 で誘導期 (S 字).

    一次は n=1 に縮退する関係に近いが、本パッケージでは別カテゴリとして扱う.
    """
    t_arr = np.atleast_1d(t).astype(float)
    return c0 * np.exp(-((k * t_arr) ** n_avrami))


KINETICS_REGISTRY = {
    "first_order": first_order,
    "second_order": second_order,
    "autocatalytic": autocatalytic,
    "induction": induction,
}
