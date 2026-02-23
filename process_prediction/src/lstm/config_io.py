from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict


def load_config(path: str | Path) -> Dict[str, Any]:
    """Load config.json and fill simple defaults (kept intentionally minimal)."""
    p = Path(path)
    cfg = json.loads(p.read_text(encoding="utf-8"))

    d = cfg["data"]
    if d.get("small_overlap_seconds", None) is None:
        d["small_overlap_seconds"] = float(d["small_window_seconds"]) / 2.0
    d.setdefault("big_overlap_minutes", 0)
    d.setdefault("cover_all_samples", True)
    return cfg
