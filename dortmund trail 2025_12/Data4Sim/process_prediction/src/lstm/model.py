# model.py
from __future__ import annotations

import torch
import torch.nn as nn


class LSTM(nn.Module):
    """
    Stateful LSTM over SMALL windows.

    Input X: [B, T, W]
      B: big-window batch size (independent sequences)
      T: number of small windows in the big window (variable, padded in collate)
      W: samples per small window (e.g., 200)

    Behavior:
      - Run LSTM over each window (sequence length W)
      - Carry (h,c) from window t to window t+1 (within each big window)
      - Output per-sample logits: [B, T, W, C]
    """

    def __init__(self, vocab_size: int, num_classes: int, emb_dim: int, hid: int, layers: int, dropout: float):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim)
        self.lstm = nn.LSTM(
            input_size=emb_dim,
            hidden_size=hid,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.head = nn.Linear(hid, num_classes)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Args:
          X: [B,T,W] int64 token ids

        Returns:
          logits: [B,T,W,C]
        """
        B, T, W = X.shape
        h = c = None
        logits_steps = []

        for t in range(T):
            x_t = X[:, t, :]                    # [B,W]
            e_t = self.emb(x_t)                 # [B,W,E]

            out_t, (h, c) = self.lstm(e_t, (h, c) if h is not None else None)  # [B,W,H]
            logits_steps.append(self.head(out_t))                              # [B,W,C]

        return torch.stack(logits_steps, dim=1)  # [B,T,W,C]
