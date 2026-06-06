"""合成データ (paper_a/data/{core,robustness}) のローダ.

case_id × replicate_id でグループ化して推定器に渡す形式に整形する.
truth は別途 truth.json から読む (推定器に渡さない、評価時のみ参照).
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterator

DATA_ROOT = Path(__file__).resolve().parents[2] / "data"


def load_truth(layer: str) -> dict[str, dict]:
    """truth.json を case_id → truth dict にして返す."""
    path = DATA_ROOT / layer / "truth.json"
    with path.open() as f:
        payload = json.load(f)
    return {c["case_id"]: c for c in payload["cases"]}


def iter_replicates(
    layer: str,
    *,
    case_ids: list[str] | None = None,
    accelerated: bool = True,
) -> Iterator[tuple[str, int, list[dict]]]:
    """data.csv または long_term_25c.csv を case × replicate でストリーム.

    Yields
    ------
    (case_id, replicate_id, rows) ※ rows は DataRow dict のリスト
    """
    fname = "data.csv" if accelerated else "long_term_25c.csv"
    path = DATA_ROOT / layer / fname
    if not path.exists():
        raise FileNotFoundError(
            f"{path} が存在しません.python -m paper_a.datagen --all で再生成してください."
        )

    case_filter = set(case_ids) if case_ids is not None else None
    current_key: tuple[str, int] | None = None
    buffer: list[dict] = []
    with path.open() as f:
        for r in csv.DictReader(f):
            cid = r["case_id"]
            if case_filter is not None and cid not in case_filter:
                continue
            rep = int(r["replicate_id"])
            key = (cid, rep)
            if current_key is not None and key != current_key:
                yield current_key[0], current_key[1], buffer
                buffer = []
            current_key = key
            buffer.append({
                "temperature": float(r["temperature"]),
                "time_months": float(r["time_months"]),
                "content_percent": float(r["content_percent"]),
            })
        if current_key is not None and buffer:
            yield current_key[0], current_key[1], buffer
