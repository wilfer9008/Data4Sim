from __future__ import annotations
from typing import Dict
import numpy as np
import torch
import torch.nn as nn
from dataset import IGNORE_INDEX
from sklearn.metrics import accuracy_score, f1_score


def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def step(model, batch, opt, device, clip: float, train: bool):
    X, Y = batch
    X = X.to(device, non_blocking=True)
    Y = Y.to(device, non_blocking=True)

    model.train(train)
    if train:
        opt.zero_grad(set_to_none=True)

    logits = model(X)  # [B,T,W,C]
    B, T, W, C = logits.shape

    # mask padded windows: padded labels are IGNORE_INDEX already (from collate)
    loss_fn = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    loss = loss_fn(logits.reshape(-1, C), Y.reshape(-1))

    if train:
        loss.backward()
        if clip and clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        opt.step()

    with torch.no_grad():
        pred = logits.argmax(-1)               # [B,T,W]
        mask = (Y != IGNORE_INDEX)

        if mask.sum().item() == 0:
            acc, f1_w = 0.0, 0.0
        else:
            y_true = Y[mask].detach().cpu().numpy().astype(np.int64)
            y_pred = pred[mask].detach().cpu().numpy().astype(np.int64)

            acc = float(accuracy_score(y_true, y_pred))
            f1_w = float(
                f1_score(
                    y_true, y_pred, average="weighted",
                    labels=np.arange(logits.size(-1)),  # keep label space stable
                    zero_division=0
                )
            )
    return {"loss": float(loss.item()),
            "acc": acc,
            "f1_w": f1_w}

def _pack(m: dict) -> dict:
    return {"loss": float(m["loss"]), "acc": float(m["acc"]), "f1_w": float(m["f1_w"])}

@torch.no_grad()
def eval_checkpoint(model, opt, device, clip: float, dl_tr, dl_va, dl_te, ckpt: Path) -> Dict[str, dict]:
    model.load_state_dict(torch.load(ckpt, map_location=device))
    trm = run_epoch(model, dl_tr, opt, device, clip, train=False)
    vam = run_epoch(model, dl_va, opt, device, clip, train=False)
    tem = run_epoch(model, dl_te, opt, device, clip, train=False)
    return {"train": _pack(trm), "val": _pack(vam), "test": _pack(tem)}

def run_epoch(model, loader, opt, device, clip: float, train: bool) -> Dict[str, float]:
    loss_sum = acc_sum = f1_sum = 0.0
    n = 0
    for batch in loader:
        m = step(model, batch, opt, device, clip, train=train)
        loss_sum += m["loss"]
        acc_sum += m["acc"]
        f1_sum += m["f1_w"]
        n += 1
    n = max(n, 1)
    return {"loss": loss_sum / n, "acc": acc_sum / n, "f1_w": f1_sum / n}



def train_loop(cfg: dict, model, opt, device, dl_tr, dl_va, dl_te, out_dir: Path) -> dict:
    """
    Step 8:
      - train for all epochs
      - save last.pt every epoch
      - save best.pt based on val f1_w
      - return a single results dict:
          {
            "all_epochs": [...],
            "best": {...},
            "last": {...}
          }
    """
    tr_cfg = cfg["train"]
    epochs = int(tr_cfg["epochs"])
    clip = float(tr_cfg["grad_clip"])

    best_path = out_dir / "best.pt"
    last_path = out_dir / "last.pt"

    def pack(m: dict) -> dict:
        return {"loss": float(m["loss"]), "acc": float(m["acc"]), "f1_w": float(m["f1_w"])}

    history = []
    best_val_f1 = -1.0
    best_epoch = -1
    best_epoch_metrics = None

    for ep in range(1, epochs + 1):
        trm = run_epoch(model, dl_tr, opt, device, clip, train=True)
        vam = run_epoch(model, dl_va, opt, device, clip, train=False)
        tem = run_epoch(model, dl_te, opt, device, clip, train=False)

        row = {"epoch": ep, "train": pack(trm), "val": pack(vam), "test": pack(tem)}
        history.append(row)

        print(
            f"Epoch {ep:03d} | "
            f"TR loss={row['train']['loss']:.4f} acc={row['train']['acc']:.4f} f1_w={row['train']['f1_w']:.4f} | "
            f"VA loss={row['val']['loss']:.4f} acc={row['val']['acc']:.4f} f1_w={row['val']['f1_w']:.4f} | "
            f"TE loss={row['test']['loss']:.4f} acc={row['test']['acc']:.4f} f1_w={row['test']['f1_w']:.4f}"
        )

        torch.save(model.state_dict(), last_path)

        if row["val"]["f1_w"] > best_val_f1:
            best_val_f1 = row["val"]["f1_w"]
            best_epoch = ep
            best_epoch_metrics = row
            torch.save(model.state_dict(), best_path)

    # evaluate checkpoints after training (so "best" and "last" are final, not buffered)
    last_eval = eval_checkpoint(model, opt, device, clip, dl_tr, dl_va, dl_te, last_path)
    best_eval = eval_checkpoint(model, opt, device, clip, dl_tr, dl_va, dl_te, best_path) if best_path.exists() else None

    return {
        "all_epochs": history,
        "best": {
            "epoch": best_epoch,
            "selected_by": "val_f1_w",
            "epoch_metrics": best_epoch_metrics,   # metrics at the time it was selected
            "eval_metrics": best_eval,             # metrics after loading best.pt
            "checkpoint": str(best_path),
            "best_val_f1_w": float(best_val_f1),
        },
        "last": {
            "epoch": epochs,
            "eval_metrics": last_eval,
            "checkpoint": str(last_path),
        },
    }

