from __future__ import annotations
from typing import List, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset

IGNORE_INDEX = -100


class BigWindowSeqDataset(Dataset):
    """
    One item = one BIG window:
      X: [T,W] token ids
      Y: [T,W] label ids (padded positions = IGNORE_INDEX)
    """
    def __init__(self, items: List[Tuple[np.ndarray, np.ndarray]]) -> None:
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int):
        X, Y = self.items[i]
        return torch.from_numpy(X.astype(np.int64)), torch.from_numpy(Y.astype(np.int64))


def pad_big_windows(batch, pad_token_id: int):
    """
    Pad BIG-window length T to Tmax. W is fixed.
    Returns:
      Xp: [B,Tmax,W]
      Yp: [B,Tmax,W] (padded = IGNORE_INDEX)
    """
    Xs, Ys = zip(*batch)
    B = len(Xs)
    Tmax = max(x.shape[0] for x in Xs)
    W = Xs[0].shape[1]

    Xp = torch.full((B, Tmax, W), pad_token_id, dtype=torch.long)
    Yp = torch.full((B, Tmax, W), IGNORE_INDEX, dtype=torch.long)

    for i, (X, Y) in enumerate(zip(Xs, Ys)):
        t = X.shape[0]
        Xp[i, :t] = X
        Yp[i, :t] = Y

    return Xp, Yp


def prepare_data(cfg: dict):
    d, sp = cfg["data"], cfg["split"]

    raw = read_sequences(d["data_dir"], d["csv_glob"], d["input_col"], d["output_col"])
    subjects = sorted(raw.keys())

    tr_sub, va_sub, te_sub = split_subjects(
        subjects, float(sp["train_ratio"]), float(sp["val_ratio"]), int(sp["seed"])
    )
    print(f"[INFO] subjects: train={len(tr_sub)} val={len(va_sub)} test={len(te_sub)}")

    token2id, label2id = build_maps(raw, tr_sub)
    return raw, tr_sub, va_sub, te_sub, token2id, label2id


def build_loaders(cfg: dict, raw, tr_sub, va_sub, te_sub, token2id, label2id):
    d, tr = cfg["data"], cfg["train"]

    hz = int(d["hz"])
    big_min = float(d["big_window_minutes"])
    big_ov = float(d.get("big_overlap_minutes", 0))
    small_sec = float(d["small_window_seconds"])
    small_ov = float(d["small_overlap_seconds"])
    cover_all = bool(d.get("cover_all_samples", True))

    train_items = build_items(raw, tr_sub, token2id, label2id, hz, big_min, big_ov, small_sec, small_ov, cover_all)
    val_items   = build_items(raw, va_sub, token2id, label2id, hz, big_min, big_ov, small_sec, small_ov, cover_all)
    test_items  = build_items(raw, te_sub, token2id, label2id, hz, big_min, big_ov, small_sec, small_ov, cover_all)

    print(f"[INFO] big windows: train={len(train_items)} val={len(val_items)} test={len(test_items)}")

    pad_id = token2id[PAD_TOKEN]
    collate = lambda b: pad_big_windows(b, pad_token_id=pad_id)

    bs = int(tr["batch_size"])
    dl_tr = DataLoader(BigWindowSeqDataset(train_items), batch_size=bs, shuffle=True,  collate_fn=collate)
    dl_va = DataLoader(BigWindowSeqDataset(val_items),   batch_size=bs, shuffle=False, collate_fn=collate)
    dl_te = DataLoader(BigWindowSeqDataset(test_items),  batch_size=bs, shuffle=False, collate_fn=collate)

    return dl_tr, dl_va, dl_te, pad_id
