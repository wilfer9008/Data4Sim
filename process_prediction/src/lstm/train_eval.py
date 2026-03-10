from __future__ import annotations
from typing import Dict
import numpy as np
import torch
import torch.nn as nn
from dataset import IGNORE_INDEX
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import wandb
import torch
import torch.nn.functional as F


def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def step(model, batch, opt, device, clip: float, train: bool, l1_lambda: float = 0.0):
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
    # Optional L1 regularization (safe: no effect if l1_lambda=0.0)
    if train and l1_lambda and l1_lambda > 0:
        l1 = 0.0
        for p in model.parameters():
            if p.requires_grad:
                l1 = l1 + p.abs().sum()
        loss = loss + (l1_lambda * l1)

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


def eval_detailed_sklearn(model, loader, device) -> dict:
    """
    Detailed eval using sklearn (only call for last/best).
    Returns:
      - loss, acc, f1_w (optional if you already compute elsewhere)
      - confusion_matrix
      - per_class precision/recall/f1/support (from classification_report)
    """
    model.eval()

    y_true_all = []
    y_pred_all = []

    for X, Y in loader:
        X = X.to(device, non_blocking=True)
        Y = Y.to(device, non_blocking=True)

        logits = model(X)
        pred = logits.argmax(-1)

        mask = (Y != IGNORE_INDEX)
        if mask.any():
            y_true_all.append(Y[mask].detach().cpu())
            y_pred_all.append(pred[mask].detach().cpu())

    if len(y_true_all) == 0:
        return {"confusion_matrix": None, "per_class": None}

    y_true = torch.cat(y_true_all).numpy().astype(int)
    y_pred = torch.cat(y_pred_all).numpy().astype(int)

    num_classes = int(max(y_true.max(), y_pred.max()) + 1)

    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))

    # dict with per-class precision/recall/f1/support
    rep = classification_report(
        y_true, y_pred,
        labels=list(range(num_classes)),
        output_dict=True,
        zero_division=0
    )

    # Extract per-class arrays in class-index order
    precision = [rep[str(i)]["precision"] for i in range(num_classes)]
    recall    = [rep[str(i)]["recall"]    for i in range(num_classes)]
    f1        = [rep[str(i)]["f1-score"]  for i in range(num_classes)]
    support   = [rep[str(i)]["support"]   for i in range(num_classes)]

    return {
        "confusion_matrix": cm.tolist(),
        "per_class": {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        },
        "macro_avg": rep["macro avg"],
        "weighted_avg": rep["weighted avg"],
        "accuracy": rep.get("accuracy", accuracy_score(y_true, y_pred)),
    }


def eval_checkpoint_sklearn(model, opt, device, clip: float, dl_tr, dl_va, dl_te, ckpt: Path) -> Dict[str, dict]:
    """
    Same call signature as eval_checkpoint(...).
    Uses sklearn-based detailed metrics (per-class precision/recall + confusion matrix).
    Note: opt and clip are intentionally unused (kept for compatibility).
    """
    _ = opt, clip  # keep signature identical; avoid "unused" lint noise

    model.load_state_dict(torch.load(ckpt, map_location=device))
    return {
        "train": eval_detailed_sklearn(model, dl_tr, device),
        "val":   eval_detailed_sklearn(model, dl_va, device),
        "test":  eval_detailed_sklearn(model, dl_te, device),
    }

def run_epoch(model, loader, opt, device, clip: float, train: bool, l1_lambda: float = 0.0) -> Dict[str, float]:
    loss_sum = acc_sum = f1_sum = 0.0
    n = 0
    for batch in loader:
        m = step(model, batch, opt, device, clip, train=train, l1_lambda=l1_lambda if train else 0.0)
        loss_sum += m["loss"]
        acc_sum += m["acc"]
        f1_sum += m["f1_w"]
        n += 1
    n = max(n, 1)
    return {"loss": loss_sum / n, "acc": acc_sum / n, "f1_w": f1_sum / n}



def train_loop(cfg: dict, model, opt, device, dl_tr, dl_va, dl_te, out_dir: Path, input_col) -> dict:
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
    l1_lambda = float(cfg["train"].get("l1_lambda", 0.0))

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
        trm = run_epoch(model, dl_tr, opt, device, clip, train=True, l1_lambda=l1_lambda)
        vam = run_epoch(model, dl_va, opt, device, clip, train=False)
        tem = run_epoch(model, dl_te, opt, device, clip, train=False)

        if wandb.run is not None:
            wandb.log({"val/f1_w": vam['f1_w'], "epoch": ep})

        row = {"epoch": ep, "train": pack(trm), "val": pack(vam), "test": pack(tem)}
        history.append(row)

        print(
            f"Epoch {ep:03d} | "
            f"TR loss={row['train']['loss']:.4f} acc={row['train']['acc']:.4f} f1_w={row['train']['f1_w']:.4f} | "
            f"VA loss={row['val']['loss']:.4f} acc={row['val']['acc']:.4f} f1_w={row['val']['f1_w']:.4f} | "
            f"TE loss={row['test']['loss']:.4f} acc={row['test']['acc']:.4f} f1_w={row['test']['f1_w']:.4f}"
        )

        if row["val"]["f1_w"] > best_val_f1:
            best_val_f1 = row["val"]["f1_w"]
            best_epoch = ep
            best_epoch_metrics = row
            torch.save(model.state_dict(), best_path)

    torch.save(model.state_dict(), last_path)

    # evaluate checkpoints after training (so "best" and "last" are final, not buffered)
    last_eval = eval_checkpoint_sklearn(model, opt, device, clip, dl_tr, dl_va, dl_te, last_path)
    best_eval = eval_checkpoint_sklearn(model, opt, device, clip, dl_tr, dl_va, dl_te, best_path) if best_path.exists() else None
    # last_eval = eval_checkpoint(model, opt, device, clip, dl_tr, dl_va, dl_te, last_path)
    # best_eval = eval_checkpoint(model, opt, device, clip, dl_tr, dl_va, dl_te, best_path) if best_path.exists() else None

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

