# counterfactual.py
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Literal, Tuple

import torch
import torch.nn.functional as F


ReplaceMode = Literal["zero", "mean_con", "noise"]
PCLoss = Literal["kl", "mse"]


def infer_patch_grid(lv: int) -> int | None:
    """
    Infer CLIP-ViT patch grid size from token length.
    CLIP ViT usually: Lv = 1 + G*G (CLS + patches)
    """
    if lv <= 1:
        return None
    n = lv - 1
    g = int(round(math.sqrt(n)))
    return g if g * g == n else None


def build_border_prior(
    lv: int,
    border_width: int = 1,
    device=None,
    dtype=torch.float32
) -> torch.Tensor:
    """
    Build a [Lv] prior mask where border patches = 1, others = 0.
    Token 0 is CLS -> always 0.
    border_width: number of rings from the boundary.
    """
    prior = torch.zeros(lv, device=device, dtype=dtype)
    g = infer_patch_grid(lv)
    if g is None:
        warnings.warn(
            "Counterfactual border prior disabled: cannot infer square patch grid from Lv.",
            RuntimeWarning
        )
        return prior

    bw = max(1, int(border_width))
    # patches are 1..Lv-1
    for idx in range(1, lv):
        k = idx - 1
        r = k // g
        c = k % g
        if (r < bw) or (r >= g - bw) or (c < bw) or (c >= g - bw):
            prior[idx] = 1.0
    return prior


def apply_border_counterfactual(
    v_ref: torch.Tensor,
    m_v: torch.Tensor,
    border_width: int = 1,
    replace_mode: ReplaceMode = "mean_con",
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply counterfactual intervention ONLY on *border & inconsistent* visual tokens.

    Args:
        v_ref: [B, Lv, D] vision tokens after your pre-CID cross-attn (V_ref)
        m_v:   [B, Lv]    consistency soft mask from CID (higher=more consistent)
        border_width: rings considered as border
        replace_mode:
            - "zero": remove those tokens (set to 0)
            - "mean_con": replace them with pooled consistent token (recommended)
            - "noise": replace with random noise (stronger perturbation)

    Returns:
        v_inc_cf: [B, Lv, D] counterfactual inconsistent tokens (for DIMM)
        cf_strength: [B] mean intervention weight per sample
    """
    assert v_ref.dim() == 3, f"v_ref must be [B,Lv,D], got {v_ref.shape}"
    assert m_v.dim() == 2, f"m_v must be [B,Lv], got {m_v.shape}"
    b, lv, d = v_ref.shape
    assert m_v.shape[0] == b and m_v.shape[1] == lv

    prior = build_border_prior(lv, border_width, device=v_ref.device, dtype=v_ref.dtype)  # [Lv]
    prior = prior.unsqueeze(0).expand(b, lv)  # [B, Lv]

    # bias weight = "inconsistent" * "border prior"
    bias_w = (1.0 - m_v).clamp(0.0, 1.0) * prior  # [B, Lv]
    cf_strength = bias_w.mean(dim=1)  # [B]

    if replace_mode == "zero":
        v_cf = v_ref * (1.0 - bias_w.unsqueeze(-1))

    elif replace_mode == "mean_con":
        # pooled consistent token (teacher content): weighted by m_v
        w = m_v.clamp_min(0.0)
        denom = w.sum(dim=1, keepdim=True).clamp_min(eps)  # [B,1]
        v_pool = (v_ref * w.unsqueeze(-1)).sum(dim=1, keepdim=True) / denom.unsqueeze(-1)  # [B,1,D]
        v_cf = v_ref * (1.0 - bias_w.unsqueeze(-1)) + v_pool * bias_w.unsqueeze(-1)

    elif replace_mode == "noise":
        noise = torch.randn_like(v_ref)
        # normalize noise scale roughly to v_ref
        scale = v_ref.detach().std().clamp_min(eps)
        v_cf = v_ref * (1.0 - bias_w.unsqueeze(-1)) + noise * (bias_w.unsqueeze(-1) * scale)

    else:
        raise ValueError(f"Unknown replace_mode: {replace_mode}")

    # only use CF on the inconsistent branch; keep original m_v split
    v_inc_cf = (1.0 - m_v).unsqueeze(-1) * v_cf  # [B,Lv,D]
    return v_inc_cf, cf_strength


def prediction_consistency_loss(
    logits: torch.Tensor,
    logits_cf: torch.Tensor,
    kind: PCLoss = "kl",
    tau: float = 1.0,
    detach_teacher: bool = True,
    eps: float = 1e-8
) -> torch.Tensor:
    """
    Prediction consistency between original and counterfactual predictions.

    - KL: match soft distributions (optionally stopgrad on teacher/original)
    - MSE: match logits directly

    Returns: scalar loss
    """
    if kind == "mse":
        return F.mse_loss(logits, logits_cf)

    if kind == "kl":
        t = max(1e-6, float(tau))
        p = F.softmax(logits / t, dim=-1)
        q = F.softmax(logits_cf / t, dim=-1)

        if detach_teacher:
            p = p.detach()

        log_q = torch.log(q.clamp_min(eps))
        # KL(p || q): F.kl_div expects input=log-probs (q), target=probs (p)
        return F.kl_div(log_q, p, reduction="batchmean")

    raise ValueError(f"Unknown pc loss kind: {kind}")
