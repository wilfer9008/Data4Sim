from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple, List

import torch
from torch.utils.data import DataLoader

from config_io import load_config
from windowing import read_sequences, split_subjects, build_maps, build_items#, PAD_TOKEN
from dataset import BigWindowSeqDataset, pad_big_windows
from model import LSTM
from train_eval import set_seed, train_loop


# ------------------------  saving ------------------------

def save_learning_results(out_dir: Path, learning_results: dict) -> Path:
    """Step 9: save the learning results dict to JSON."""
    p = out_dir / "metrics.json"
    p.write_text(json.dumps(learning_results, indent=2), encoding="utf-8")
    print(f"[INFO] saved metrics json: {p}")
    return p



# ------------------------ main (steps only) ------------------------

def main(cfg_path: str = "config.json"):
    #####################################################################
    # 1) Load config + seed + output dir
    #####################################################################
    cfg = load_config(cfg_path)
    d, sp, tr, mo = cfg["data"], cfg["split"], cfg["train"], cfg["model"]

    set_seed(int(sp["seed"]))
    out_dir = Path(tr["out_dir"]); out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config_used.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    #####################################################################
    # 2) Select device
    #####################################################################
    device = torch.device(tr["device"] if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device={device}")

    #####################################################################
    # 3) Read data + split subjects
    #####################################################################
    raw = read_sequences(d["data_dir"], d["csv_glob"], d["input_col"], d["output_col"])
    subjects = sorted(raw.keys())
    tr_sub, va_sub, te_sub = split_subjects(
        subjects, float(sp["train_ratio"]), float(sp["val_ratio"]), int(sp["seed"])
    )
    print(f"[INFO] subjects: train={len(tr_sub)} val={len(va_sub)} test={len(te_sub)}")

    #####################################################################
    # 4) Build vocab / label maps (train only) + save them
    #####################################################################
    token2id, label2id = build_maps(raw, tr_sub)
    (out_dir / "token2id.json").write_text(json.dumps(token2id, indent=2), encoding="utf-8")
    (out_dir / "label2id.json").write_text(json.dumps(label2id, indent=2), encoding="utf-8")

    #####################################################################
    # 5) Windowing: build big-window items from subjects
    #####################################################################
    hz = int(d["hz"])
    big_min = float(d["big_window_minutes"])
    big_ov = float(d.get("big_overlap_minutes", 0))
    small_sec = float(d["small_window_seconds"])
    small_ov = float(d["small_overlap_seconds"])
    cover_all = bool(d.get("cover_all_samples", True))

    train_items = build_items(raw, tr_sub, token2id, label2id, hz, big_min, big_ov, small_sec, small_ov, cover_all)
    val_items   = build_items(raw, va_sub, token2id, label2id, hz, big_min, 0, small_sec, 0, cover_all)
    test_items  = build_items(raw, te_sub, token2id, label2id, hz, big_min, 0, small_sec, 0, cover_all)

    print(f"[INFO] big windows: train={len(train_items)} val={len(val_items)} test={len(test_items)}")

    #####################################################################
    # 6) Build DataLoaders (shuffle only train)
    #####################################################################
    # pad_id = token2id[PAD_TOKEN]
    # collate = lambda b: pad_big_windows(b, pad_token_id=pad_id)
    #
    # dl_tr = DataLoader(BigWindowSeqDataset(train_items), batch_size=int(tr["batch_size"]), shuffle=True,  collate_fn=collate)
    # dl_va = DataLoader(BigWindowSeqDataset(val_items),   batch_size=int(tr["batch_size"]), shuffle=False, collate_fn=collate)
    # dl_te = DataLoader(BigWindowSeqDataset(test_items),  batch_size=int(tr["batch_size"]), shuffle=False, collate_fn=collate)

    dl_tr = DataLoader(BigWindowSeqDataset(train_items), batch_size=int(tr["batch_size"]), shuffle=True)
    dl_va = DataLoader(BigWindowSeqDataset(val_items), batch_size=int(tr["batch_size"]), shuffle=False)
    dl_te = DataLoader(BigWindowSeqDataset(test_items), batch_size=int(tr["batch_size"]), shuffle=False)

    #####################################################################
    # 7) Build model + optimizer
    #####################################################################
    model = LSTM(
        vocab_size=len(token2id),
        num_classes=len(label2id),
        emb_dim=int(mo["emb_dim"]),
        hid=int(mo["hidden_dim"]),
        layers=int(mo["layers"]),
        dropout=float(mo["dropout"]),
        # pad_id=pad_id,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=float(tr["lr"]), weight_decay=float(tr["weight_decay"]))

    #####################################################################
    # 8) Train loop + track metrics + save best/last
    #####################################################################
    learning_results = train_loop(cfg, model, opt, device, dl_tr, dl_va, dl_te, out_dir)

    #####################################################################
    # 9) Save metrics to JSON
    #####################################################################
    save_learning_results(out_dir, learning_results)


if __name__ == "__main__":
    main("config.json")
