# model.py
from __future__ import annotations

import torch
import torch.nn as nn

class EncoderDecoderFeatureLSTM(nn.Module):
    """
    Feature-based encoder-decoder LSTM.

    Input:
      X: [B, T, F]
        B = batch size
        T = number of small windows in the big window
        F = feature dimension (e.g. 128, but not hardcoded)

    Output:
      logits: [B, T, W, C]
        W = output_steps (label length, e.g. 500, but not hardcoded)
        C = number of classes
    """

    def __init__(
        self,
        num_classes: int,
        emb_dim: int,
        hid: int,
        layers: int,
        dropout: float,
        output_steps: int,
    ):
        super().__init__()

        self.output_steps = int(output_steps)

        # project each scalar feature value to emb_dim
        self.in_proj = nn.Linear(1, emb_dim)

        self.encoder = nn.LSTM(
            input_size=emb_dim,
            hidden_size=hid,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
        )

        self.decoder = nn.LSTM(
            input_size=emb_dim,
            hidden_size=hid,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
        )

        # learned decoder input token
        self.decoder_token = nn.Parameter(torch.zeros(1, 1, emb_dim))

        self.head = nn.Linear(hid, num_classes)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Args:
          X: [B, T, F] float tensor

        Returns:
          logits: [B, T, W, C]
        """
        X = X.float()

        B, T, F = X.shape
        logits_steps = []

        h_enc = c_enc = None

        for t in range(T):
            x_t = X[:, t, :]            # [B,F]
            x_t = x_t.unsqueeze(-1)     # [B,F,1]
            enc_in = self.in_proj(x_t)  # [B,F,E]

            # encoder over feature dimension
            enc_out, (h_enc, c_enc) = self.encoder(
                enc_in,
                (h_enc, c_enc) if h_enc is not None else None
            )

            # decoder generates label sequence of length output_steps
            dec_in = self.decoder_token.expand(B, self.output_steps, -1)  # [B,W,E]
            dec_out, _ = self.decoder(dec_in, (h_enc, c_enc))             # [B,W,H]

            logits_t = self.head(dec_out)   # [B,W,C]
            logits_steps.append(logits_t)

        return torch.stack(logits_steps, dim=1)  # [B,T,W,C]