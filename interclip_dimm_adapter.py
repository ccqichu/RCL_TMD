# interclip_dimm_adapter.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class ConditionalConcatAdapter(nn.Module):
    """
    InterCLIP-style conditional self-attention adapter via token concatenation.

    Given:
        x:    [B, Lx, D]   (sequence to be updated, e.g., query tokens)
        cond: [B, Lc, D]   (conditioning sequence, e.g., another modality tokens)

    Steps:
        1) cat = [x; cond] along token dimension
        2) 1-layer Transformer-like block on cat (self-attn + FFN + LN)
        3) take first Lx tokens -> x_adapt
        4) gated residual: x_new = x + tanh(beta) * (x_adapt - x)

    beta is initialized to 0 for stability (x_new == x at init).
    """

    def __init__(
        self,
        d_model: int = 768,
        nhead: int = 8,
        dropout: float = 0.1,
        ff_mult: int = 4,
        init_beta: float = 0.0,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=nhead, dropout=dropout, batch_first=True
        )

        hidden_ff = d_model * ff_mult
        self.ffn = nn.Sequential(
            nn.Linear(d_model, hidden_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_ff, d_model),
        )

        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

        # gating scalar (trainable), initialized to 0
        self.beta = nn.Parameter(torch.tensor(float(init_beta)))

    @staticmethod
    def _ensure_bool_mask(mask: Optional[torch.Tensor], B: int, L: int, device) -> torch.Tensor:
        if mask is None:
            return torch.zeros(B, L, dtype=torch.bool, device=device)
        return mask.to(device=device).bool()

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        x_pad_mask: Optional[torch.Tensor] = None,
        cond_pad_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Returns:
            x_new: [B, Lx, D]
        """
        B, Lx, D = x.shape
        Lc = cond.size(1)
        device = x.device

        x_mask = self._ensure_bool_mask(x_pad_mask, B, Lx, device)
        c_mask = self._ensure_bool_mask(cond_pad_mask, B, Lc, device)
        cat_mask = torch.cat([x_mask, c_mask], dim=1)  # [B, Lx+Lc]

        cat = torch.cat([x, cond], dim=1)  # [B, Lx+Lc, D]

        # self-attention on concatenated sequence
        attn_out, _ = self.self_attn(
            query=cat, key=cat, value=cat,
            key_padding_mask=cat_mask
        )
        cat = self.ln1(cat + self.drop(attn_out))

        # FFN
        ffn_out = self.ffn(cat)
        cat = self.ln2(cat + self.drop(ffn_out))

        x_adapt = cat[:, :Lx, :]  # [B, Lx, D]
        gate = torch.tanh(self.beta)  # scalar
        x_new = x + gate * (x_adapt - x)
        return x_new
