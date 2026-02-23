from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

# PAD_TOKEN = "<PAD>"
# UNK_TOKEN = "<UNK>"
# UNK_LABEL = "<UNK_LABEL>"


# --------------------------- I/O ---------------------------

def read_sequences(
    data_dir: str | Path,
    csv_glob: str,
    input_col: str,
    output_col: str,
) -> Dict[str, Dict[str, List[str]]]:
    """Read S*.csv -> {subject: {'x': [...], 'y': [...]}}."""
    data_dir = Path(data_dir)
    files = sorted(data_dir.glob(csv_glob))
    if not files:
        raise FileNotFoundError(f"No '{csv_glob}' in {data_dir}")

    out: Dict[str, Dict[str, List[str]]] = {}
    for p in files:
        df = pd.read_csv(p).dropna(subset=[input_col, output_col]).reset_index(drop=True)
        x = df[input_col].astype(str).tolist()
        y = df[output_col].astype(str).tolist()
        if len(x) != len(y):
            raise RuntimeError(f"[{p.stem}] input/output length mismatch")
        out[p.stem] = {"x": x, "y": y}
    return out


def split_subjects(subjects: List[str], train_ratio: float, val_ratio: float, seed: int):
    rng = np.random.default_rng(seed)
    s = subjects[:]
    rng.shuffle(s)
    n = len(s)
    n_tr = int(n * train_ratio)
    n_va = int(n * val_ratio)
    return s[:n_tr], s[n_tr:n_tr + n_va], s[n_tr + n_va:]


def build_maps(raw: Dict[str, Dict[str, List[str]]], train_subjects: List[str]):
    """Build token/label maps from TRAIN only (fast via set-union)."""
    # tok = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    # lab = {UNK_LABEL: 0}

    x_uniq, y_uniq = set(), set()
    for s in train_subjects:
        x_uniq.update(raw[s]["x"])
        y_uniq.update(raw[s]["y"])

    token2id = {t: i for i, t in enumerate(sorted(x_uniq))}
    label2id = {y: i for i, y in enumerate(sorted(y_uniq))}
    return token2id, label2id
    # for t in sorted(x_uniq):
    #     if t not in tok:
    #         tok[t] = len(tok)
    # for y in sorted(y_uniq):
    #     if y not in lab:
    #         lab[y] = len(lab)
    #
    # return tok, lab


# --------------------------- helpers ---------------------------

def _to_samples_minutes(m: float, hz: int) -> int:
    return int(round(m * 60.0 * hz))


def _to_samples_seconds(s: float, hz: int) -> int:
    return int(round(s * hz))


def _stride(size: int, overlap: int, name: str) -> int:
    if overlap < 0 or overlap >= size:
        raise ValueError(f"{name}: overlap must be in [0, size), got {overlap} vs size {size}")
    return size - overlap


def _full_window_starts(length: int, win: int, stride: int, cover_all: bool) -> np.ndarray:
    """
    Start indices for FULL windows only (no padding):
      - all starts where start <= length-win
      - if cover_all=True, ensure end-aligned start (length-win) is included
    If length < win -> empty.
    """
    last = length - win
    if last < 0:
        return np.zeros((0,), dtype=np.int64)

    starts = np.arange(0, last + 1, stride, dtype=np.int64)
    if cover_all and starts.size and starts[-1] != last:
        starts = np.append(starts, last)
    if cover_all and starts.size == 0:
        starts = np.array([last], dtype=np.int64)
    return starts


def _gather_windows_1d(x: np.ndarray, starts: np.ndarray, win: int) -> np.ndarray:
    """Vectorized gather of FULL windows: returns [K, win]."""
    idx = starts[:, None] + np.arange(win, dtype=np.int64)[None, :]
    return x[idx]


def _min_small_windows_for_big_time(big_size: int, small_size: int, small_stride: int) -> int:
    """
    Minimal number of SMALL windows K so that covered samples >= big_size.

      coverage(K) = (K-1)*small_stride + small_size
      K = ceil((big_size - small_size)/small_stride) + 1
    """
    if big_size <= 0:
        return 0
    if big_size <= small_size:
        return 1
    num = big_size - small_size
    return int((num + small_stride - 1) // small_stride + 1)


def _big_start_samples(total_samples: int, big_size: int, big_stride: int, cover_all: bool) -> np.ndarray:
    """Big-window starts in sample space; end-aligned last window if cover_all=True."""
    last = total_samples - big_size
    if last < 0:
        return np.array([0], dtype=np.int64) if cover_all and total_samples > 0 else np.zeros((0,), np.int64)

    starts = np.arange(0, last + 1, big_stride, dtype=np.int64)
    if cover_all and starts.size and starts[-1] != last:
        starts = np.append(starts, last)
    if cover_all and starts.size == 0:
        starts = np.array([last], dtype=np.int64)
    return starts


# --------------------------- core (subject-wise) ---------------------------
def subject_to_big_items_over_small_windows(
    x_ids: np.ndarray,
    y_ids: np.ndarray,
    hz: int,
    big_minutes: float,
    big_overlap_minutes: float,
    small_seconds: float,
    small_overlap_seconds: Optional[float],
    cover_all: bool,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Small windows first (FULL only, no padding), then BIG windows over the small-window axis.
    BIG windows behave like batching:
      - target length = K small windows (K chosen to cover ~big_minutes)
      - last BIG window can be shorter (no padding) if cover_all=True
    """
    n = int(x_ids.shape[0])
    if n <= 0:
        return []

    # ---- SMALL windows (full only) ----
    small_size = _to_samples_seconds(float(small_seconds), hz)
    if small_overlap_seconds is None:
        small_overlap_seconds = float(small_seconds) / 2.0
    small_stride = _stride(small_size, _to_samples_seconds(float(small_overlap_seconds), hz), "small")

    small_starts = _full_window_starts(n, small_size, small_stride, cover_all=cover_all)
    if small_starts.size == 0:
        return []

    X_small = _gather_windows_1d(x_ids, small_starts, small_size)  # [Ns, W]
    Y_small = _gather_windows_1d(y_ids, small_starts, small_size)  # [Ns, W]
    Ns = int(X_small.shape[0])

    # ---- BIG windows over small windows (batch-like, variable last) ----
    big_size = _to_samples_minutes(float(big_minutes), hz)
    big_ov_samp = _to_samples_minutes(float(big_overlap_minutes), hz)
    K = _min_small_windows_for_big_time(big_size, small_size, small_stride)  # target #small windows per big window
    if K <= 0:
        return []

    # overlap in "small-window steps" (approx)
    ov_k = int(big_ov_samp // small_stride) if big_ov_samp > 0 else 0
    ov_k = int(np.clip(ov_k, 0, max(K - 1, 0)))
    step = max(1, K - ov_k)

    items: List[Tuple[np.ndarray, np.ndarray]] = []
    for st in range(0, Ns, step):
        en = min(st + K, Ns)
        if not cover_all and (en - st) < K:
            break  # drop last partial big window
        items.append((X_small[st:en], Y_small[st:en]))  # no padding; last can be shorter
    return items


# --------------------------- API used by main ---------------------------

def build_items(
    raw: Dict[str, Dict[str, List[str]]],
    subjects: List[str],
    token2id: Dict[str, int],
    label2id: Dict[str, int],
    hz: int,
    big_min: float,
    big_ov_min: float,
    small_sec: float,
    small_ov_sec: Optional[float],
    cover_all: bool,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Build a flat list of BIG-window items (X_big, Y_big) for given subjects.
    """
    # unk_x = token2id[UNK_TOKEN]
    # unk_y = label2id[UNK_LABEL]
    # pad_id = token2id[PAD_TOKEN]

    items: List[Tuple[np.ndarray, np.ndarray]] = []
    for s in subjects:
        # x_ids = np.fromiter((token2id.get(t, unk_x) for t in raw[s]["x"]), dtype=np.int64)
        # y_ids = np.fromiter((label2id.get(v, unk_y) for v in raw[s]["y"]), dtype=np.int64)
        x_ids = np.fromiter((token2id[t] for t in raw[s]["x"]), dtype=np.int64)
        y_ids = np.fromiter((label2id[v] for v in raw[s]["y"]), dtype=np.int64)

        items.extend(
            subject_to_big_items_over_small_windows(
                x_ids=x_ids,
                y_ids=y_ids,
                hz=hz,
                big_minutes=big_min,
                big_overlap_minutes=big_ov_min,
                small_seconds=small_sec,
                small_overlap_seconds=small_ov_sec,
                cover_all=cover_all,
            )
        )
    return items
