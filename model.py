from typing import Optional

from torch import Tensor
from transformers import CLIPModel
# from transformers import BertConfig
# from transformers.models.bert.modeling_bert import BertLayer
import torch.nn as nn
import torch
import torch.nn.functional as F
import copy


# from backbone import build_backbone

# from transformers import BertTokenizer, BertModel


# ============================================================================
# Helper functions for CID-DIMM
# ============================================================================

def masked_mean(x: torch.Tensor, pad_mask: Optional[torch.Tensor] = None):
    if pad_mask is None:
        return x.mean(dim=1)

    # 对齐 pad_mask 长度到 x 的序列长度
    if pad_mask.size(1) != x.size(1):
        if pad_mask.size(1) > x.size(1):
            pad_mask = pad_mask[:, :x.size(1)]
        else:
            extra = torch.ones(
                pad_mask.size(0),
                x.size(1) - pad_mask.size(1),
                dtype=pad_mask.dtype,
                device=pad_mask.device
            )
            pad_mask = torch.cat([pad_mask, extra], dim=1)

    pad_mask = pad_mask.bool()
    keep = (~pad_mask).float().unsqueeze(-1)
    denom = keep.sum(dim=1).clamp_min(1.0)
    return (x * keep).sum(dim=1) / denom


def make_neg_index(batch_size: int, device, labels: Optional[torch.Tensor] = None):
    """
    Generate negative sample indices for contrastive learning.
    Ensures idx[i] != i for all i.
    Args:
        batch_size: batch size
        labels: optional labels tensor [B] to avoid same-label negatives
        device: torch device
    Returns:
        [B] tensor of shuffled indices
    """
    if labels is None:
        idx = torch.randperm(batch_size, device=device)
        # Avoid idx[i]==i, use circular shift if collision exists
        if torch.any(idx == torch.arange(batch_size, device=device)):
            idx = torch.roll(idx, shifts=1, dims=0)
        return idx

    labels = labels.view(-1)
    idx = torch.empty(batch_size, dtype=torch.long, device=device)
    all_indices = torch.arange(batch_size, device=device)
    for i in range(batch_size):
        candidates = all_indices[labels != labels[i]]
        if candidates.numel() == 0:
            candidates = all_indices[all_indices != i]
        if candidates.numel() == 0:
            idx[i] = i
        else:
            j = torch.randint(0, candidates.numel(), (1,), device=device)
            idx[i] = candidates[j]
    return idx


def align_attention_mask(attn_mask: torch.Tensor, seq_len: int):
    """
    Align attention mask length to sequence length.
    Pads with 0 (mask) or truncates as needed.
    """
    if attn_mask.size(1) > seq_len:
        return attn_mask[:, :seq_len]
    if attn_mask.size(1) < seq_len:
        batch_size = attn_mask.size(0)
        extra_len = seq_len - attn_mask.size(1)
        extra_pad = torch.zeros(batch_size, extra_len,
                                dtype=attn_mask.dtype,
                                device=attn_mask.device)
        return torch.cat([attn_mask, extra_pad], dim=1)
    return attn_mask


class CrossAttention(nn.Module):
    def __init__(self, feature_dim, dropout_prob=0.1):
        super(CrossAttention, self).__init__()
        self.text_linear = nn.Linear(feature_dim, feature_dim)
        self.extra_linear = nn.Linear(feature_dim, feature_dim)
        self.query_proj = nn.Linear(feature_dim, feature_dim)
        self.key_proj = nn.Linear(feature_dim, feature_dim)
        self.value_proj = nn.Linear(feature_dim, feature_dim)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, query, key, value, key_padding_mask=None):
        # Avoid cross-sample attention when inputs are [B, D]
        squeezed = False
        if query.dim() == 2:
            query = query.unsqueeze(1)
            key = key.unsqueeze(1)
            value = value.unsqueeze(1)
            squeezed = True
        if query.shape[-1] != 768:
            query = self.text_linear(query)
        if key.shape[-1] != 768:
            key = self.extra_linear(key)
            value = self.extra_linear(value)
        query = self.query_proj(query)
        key = self.key_proj(key)
        value = self.value_proj(value)
        attention_scores = torch.matmul(query, key.transpose(-1, -2))
        attention_scores = attention_scores / (key.size(-1) ** 0.5)
        if key_padding_mask is not None:
            attention_scores = attention_scores.masked_fill(
                key_padding_mask.unsqueeze(1),
                -1e4
            )
        attention_weights = F.softmax(attention_scores, dim=-1)
        attended_values = torch.matmul(attention_weights, value)
        attended_values = self.dropout(attended_values)
        if squeezed:
            attended_values = attended_values.squeeze(1)
        return attended_values


# ============================================================================
# CID Module: Consistent/Inconsistent Decomposition with soft mask
# ============================================================================

class CIDModule(nn.Module):
    """
    CID Module with bilateral (text + vision) soft mask decomposition.

    Inputs:
        - T_tok: [B, Lt, 512] CLIP text last_hidden_state
        - V_tok: [B, Lv, 768] CLIP vision last_hidden_state
        - attn_mask: [B, Lt] attention mask (1=valid, 0=pad)

    Outputs:
        - T_con: [B, Lt, 768] consistent text tokens
        - T_inc: [B, Lt, 768] inconsistent text tokens
        - V_con: [B, Lv, 768] consistent vision patches
        - V_inc: [B, Lv, 768] inconsistent vision patches
        - m_t: [B, Lt] text consistency mask
        - m_v: [B, Lv] vision consistency mask
        - loss_ratio: scalar, ratio loss (bilateral)
        - loss_itm: scalar, image-text matching mask loss (bilateral)
    """

    def __init__(self, text_dim=512, vision_dim=768, hidden_dim=768,
                 rho=0.3, rho_t=0.5, delta=0.1, tau0=1.0, tau_min=0.4, decay=0.9995,
                 neg_sampling="label_aware", tau_schedule_mode='step'):
        super(CIDModule, self).__init__()

        # Project text from 512 to 768
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.hidden_dim = hidden_dim

        # Hyperparameters
        self.rho = rho  # target ratio for consistent patches/tokens
        self.rho_t = rho_t
        self.delta = delta  # margin for ITM loss
        self.tau0 = tau0  # initial temperature
        self.tau_min = tau_min  # minimum temperature
        self.decay = decay  # temperature decay rate
        self.neg_sampling = neg_sampling
        self.tau_schedule_mode = tau_schedule_mode  # 'step' or 'epoch'

        # Temperature annealing: maintain global step/epoch as buffer
        self.register_buffer('global_step', torch.tensor(0, dtype=torch.long))
        self.register_buffer('current_epoch', torch.tensor(0, dtype=torch.long))
        self.register_buffer('current_tau', torch.tensor(tau0, dtype=torch.float))

    def set_epoch(self, epoch):
        """
        Set current epoch for epoch-based temperature scheduling.
        Call this at the beginning of each training epoch for better control.

        Args:
            epoch: current epoch number (0-indexed)
        """
        self.current_epoch.fill_(epoch)
        # Update temperature based on epoch
        if self.tau_schedule_mode == 'epoch':
            # Epoch-based decay: tau = tau0 * (decay ** epoch)
            # With decay=0.95, tau reaches tau_min in ~10-12 epochs
            epoch_decay = 0.95  # Slower decay than step-based
            tau = max(self.tau_min, self.tau0 * (epoch_decay ** epoch))
            self.current_tau.fill_(tau)

    def set_tau(self, tau):
        """
        Manually set temperature value.

        Args:
            tau: temperature value to set
        """
        self.current_tau.fill_(max(self.tau_min, tau))

    def forward(self, T_tok, V_tok, attn_mask, labels: Optional[torch.Tensor] = None):
        """
        Args:
            T_tok: [B, Lt, 512] text features from CLIP
            V_tok: [B, Lv, 768] vision features from CLIP
            attn_mask: [B, Lt] attention mask (1=valid, 0=pad)

        Returns:
            T_con, T_inc, V_con, V_inc, m_t, m_v, loss_ratio, loss_itm
        """
        B, Lt, _ = T_tok.shape
        Lv = V_tok.size(1)
        device = T_tok.device

        # 1. Project text to 768 dimensions (if needed)
        if T_tok.size(-1) == self.hidden_dim:
            T = T_tok
        else:
            T = self.text_proj(T_tok)  # [B, Lt, 768]
        V = V_tok  # [B, Lv, 768]

        # 2. Create padding mask (True = padding)
        pad_mask = (attn_mask == 0)  # [B, Lt]
        valid = (~pad_mask).float()
        valid_den = valid.sum().clamp_min(1.0)  # scalar

        # 3. Temperature annealing (supports 'step' or 'epoch' mode)
        if self.tau_schedule_mode == 'step':
            # Step-based decay (original implementation)
            if self.training:
                self.global_step += 1
            tau = max(self.tau_min, self.tau0 * (self.decay ** self.global_step.item()))
        elif self.tau_schedule_mode == 'epoch':
            # Epoch-based decay (updated via set_epoch())
            # Use pre-computed current_tau from set_epoch()
            tau = self.current_tau.item()
        else:
            # Fallback to initial temperature
            tau = self.tau0

        # ====================================================================
        # Bilateral Decomposition: Text and Vision
        # ====================================================================

        # 4a. Compute alignment scores: T -> V
        # A_tv = T @ V^T / tau -> [B, Lt, Lv]
        A_tv = torch.matmul(T, V.transpose(1, 2)) / tau  # [B, Lt, Lv]
        A_tv = A_tv.masked_fill(pad_mask.unsqueeze(-1), -1e4)

        # s_v = max over text tokens -> [B, Lv]
        s_v = A_tv.max(dim=1).values  # [B, Lv]



        # token-side score: max over patches -> [B, Lt]
        # (No need to compute A_vt separately; it's A_tv.transpose(1,2))
        s_t = A_tv.max(dim=2).values  # [B, Lt]
        s_t = s_t.masked_fill(pad_mask, -1e4)
        # 5. Soft mask using softmax with scaling
        # Vision side mask
        p_v = F.softmax(s_v, dim=-1)
        m_v = torch.clamp(self.rho * Lv * p_v, 0.0, 1.0)  # [B, Lv]

        # Text side mask (only for non-padding positions)
        # Count valid tokens per sample
        valid_counts = valid.sum(dim=1, keepdim=True).clamp_min(1.0)  # [B, 1]        p_t = F.softmax(s_t, dim=-1)  # [B, Lt]
 
        p_t = F.softmax(s_t, dim=-1)  # [B, Lt]

        m_t = torch.clamp(self.rho_t * valid_counts * p_t, 0.0, 1.0)  # [B, Lt]
        m_t = m_t * valid  # pad -> 0.0
        # 6. Split into consistent and inconsistent parts
        # Vision side
        V_con = m_v.unsqueeze(-1) * V  # [B, Lv, 768]
        V_inc = (1 - m_v.unsqueeze(-1)) * V  # [B, Lv, 768]

        T_con = m_t.unsqueeze(-1) * T  # [B, Lt, 768]
        m_t_inc = (1.0 - m_t) * valid  # pad -> 0.0
        T_inc = m_t_inc.unsqueeze(-1) * T  # [B, Lt, 768]
        # 7. L_ratio: encourage mask mean to be close to rho (bilateral)
        loss_ratio_v = (m_v.mean() - self.rho) ** 2
        # # For text, only consider non-padding positions
        m_t_mean = (m_t * valid).sum() / valid_den
        loss_ratio_t = (m_t_mean - self.rho_t) ** 2
        loss_ratio = loss_ratio_v + loss_ratio_t
        # 8. L_itm: image-text matching mask loss (bilateral)
        # Create negative samples by shuffling batch
        if B <= 1:
            idx_neg = torch.zeros(B, dtype=torch.long, device=device)
        elif self.neg_sampling == "shuffle" or labels is None:
            idx_neg = make_neg_index(B, device, labels=None)
        elif self.neg_sampling == "hard_negative":
            # Hard Negative Mining: select most similar but different-label samples
            # This forces the model to learn fine-grained semantic conflicts
            T_pool = masked_mean(T, pad_mask)  # [B, 768]
            V_pool = V.mean(dim=1)  # [B, 768]
            T_norm = F.normalize(T_pool, dim=-1)
            V_norm = F.normalize(V_pool, dim=-1)
            # Compute cross-modal similarity matrix
            sim = torch.matmul(T_norm, V_norm.transpose(0, 1))  # [B, B]
            # Use symmetric similarity score
            score = sim + sim.transpose(0, 1)  # [B, B]

            # Mask out invalid candidates:
            # 1) Self (diagonal)
            # 2) Same-label samples (if labels provided)
            mask = torch.eye(B, device=device, dtype=torch.bool)
            if labels is not None:
                labels_view = labels.view(-1)
                same_label = (labels_view.view(B, 1) == labels_view.view(1, B))
                mask = mask | same_label
            has_neg = (~mask).any(dim=1)

            # Set masked positions to very low similarity
            score = score.masked_fill(mask, -1e9)
            # Select hardest negative (highest similarity among valid candidates)
            idx_neg = score.argmax(dim=1)  # [B]
            if (~has_neg).any():
                idx_fallback = make_neg_index(B, device, labels=labels)
                idx_neg = torch.where(has_neg, idx_neg, idx_fallback)
        elif self.neg_sampling == "low_sim":
            T_pool = masked_mean(T, pad_mask)  # [B, 768]
            V_pool = V.mean(dim=1)  # [B, 768]
            T_norm = F.normalize(T_pool, dim=-1)
            V_norm = F.normalize(V_pool, dim=-1)
            sim = torch.matmul(T_norm, V_norm.transpose(0, 1))  # [B, B]
            score = sim + sim.transpose(0, 1)
            mask = torch.eye(B, device=device, dtype=torch.bool)
            if labels is not None:
                labels = labels.view(-1)
                same = labels.view(B, 1) == labels.view(1, B)
                diff_exists = (~same).any(dim=1)
                mask = mask | (same & diff_exists.unsqueeze(1))
            score = score.masked_fill(mask, float("inf"))
            idx_neg = score.argmin(dim=1)
        else:
            idx_neg = make_neg_index(B, device, labels=labels)
        V_neg = V[idx_neg]  # [B, Lv, 768]
        T_neg = T[idx_neg]  # [B, Lt, 768]

        # 8a. Vision side ITM loss
        A_neg_v = torch.matmul(T, V_neg.transpose(1, 2)) / tau  # [B, Lt, Lv]
        A_neg_v = A_neg_v.masked_fill(pad_mask.unsqueeze(-1), -1e4)
        s_v_neg = A_neg_v.max(dim=1).values  # [B, Lv]
        p_v_neg = F.softmax(s_v_neg, dim=-1)
        m_v_neg = torch.clamp(self.rho * Lv * p_v_neg, 0.0, 1.0)
        loss_itm_v = F.relu(m_v_neg.mean() - m_v.mean() + self.delta)

        # 8b. Text side ITM loss
        A_neg_t = torch.matmul(T_neg, V.transpose(1, 2)) / tau  # [B, Lt, Lv]
        A_neg_t = A_neg_t.masked_fill(pad_mask.unsqueeze(-1), -1e4)
        s_t_neg = A_neg_t.max(dim=2).values  # [B, Lt]
        s_t_neg = s_t_neg.masked_fill(pad_mask, -1e4)        
        p_t_neg = F.softmax(s_t_neg, dim=-1)
        m_t_neg = torch.clamp(self.rho_t * valid_counts * p_t_neg, 0.0, 1.0)
        m_t_neg = m_t_neg * valid
        m_t_neg_mean = (m_t_neg * valid).sum() / valid_den
        loss_itm_t = F.relu(m_t_neg_mean - m_t_mean + self.delta)
        loss_itm = loss_itm_v + loss_itm_t

        return T_con, T_inc, V_con, V_inc, m_t, m_v, loss_ratio, loss_itm


# ============================================================================
# DIMM Module: Disentangled Interaction and Multimodal Modeling
# ============================================================================

class DIMMModule(nn.Module):
    """
    DIMM Module with 4 evidence channels:
        1) Inter-Match (T_con -> V_con)
        2) Inter-Mismatch (T_con -> V_inc, T_inc -> V_con)
        3) Intra-Text Conflict (T_con -> T_inc)
        4) Intra-Vision Conflict (V_con -> V_inc)
    """

    def __init__(self, hidden_dim=768, num_heads=8, dropout=0.1,
                 vision_conf_threshold=0.1):
        super(DIMMModule, self).__init__()
        self.vision_conf_threshold = vision_conf_threshold

        # Inter-Match
        self.mha_match = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # Inter-Mismatch (two directions)
        self.mha_mis_tc_vi = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.mha_mis_ti_vc = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # Intra-Text Conflict
        self.mha_tconf = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # Intra-Vision Conflict
        self.mha_vconf = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.ln_match = nn.LayerNorm(hidden_dim)
        self.ln_mis = nn.LayerNorm(hidden_dim)  # Shared LN for both mismatch directions
        self.ln_tconf = nn.LayerNorm(hidden_dim)
        self.ln_vconf = nn.LayerNorm(hidden_dim)

        # MLP for merging two mismatch directions
        self.mis_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim)
        )

        # Final fusion MLP (3 text channels + 1 vision channel = 4 * 768 = 3072)
        self.final_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )

    @staticmethod
    def _weighted_pool(x: torch.Tensor, weights: torch.Tensor):
        weights = weights.clamp_min(0.0)
        denom = weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
        return (x * weights.unsqueeze(-1)).sum(dim=1) / denom

    def forward(self, T_con, T_inc, V_con, V_inc, pad_mask, m_v=None):
        # Ensure pad_mask matches sequence length
        if pad_mask.size(1) != T_con.size(1):
            if pad_mask.size(1) > T_con.size(1):
                pad_mask = pad_mask[:, :T_con.size(1)]
            else:
                batch_size = pad_mask.size(0)
                extra_len = T_con.size(1) - pad_mask.size(1)
                extra_padding = torch.ones(batch_size, extra_len,
                                          dtype=pad_mask.dtype,
                                          device=pad_mask.device).bool()
                pad_mask = torch.cat([pad_mask, extra_padding], dim=1)

        # ====================================================================
        # Text Evidence Channels (channel-level pooling for better separation)
        # ====================================================================

        # Channel 1: Inter-Match (T_con -> V_con)
        T_match, _ = self.mha_match(
            query=T_con,
            key=V_con,
            value=V_con,
            key_padding_mask=None
        )
        T_match = self.ln_match(T_con + T_match)  # [B, Lt, 768]
        z_match = masked_mean(T_match, pad_mask)  # [B, 768] - Pool immediately

        # Channel 2a: Inter-Mismatch (T_con -> V_inc)
        T_mis1, _ = self.mha_mis_tc_vi(
            query=T_con,
            key=V_inc,
            value=V_inc,
            key_padding_mask=None
        )
        T_mis1 = self.ln_mis(T_con + T_mis1)  # [B, Lt, 768]

        # Channel 2b: Inter-Mismatch (T_inc -> V_con)
        T_mis2, _ = self.mha_mis_ti_vc(
            query=T_inc,
            key=V_con,
            value=V_con,
            key_padding_mask=None
        )
        T_mis2 = self.ln_mis(T_inc + T_mis2)  # [B, Lt, 768]

        # Pool each mismatch direction separately, then merge at feature level
        z_mis1 = masked_mean(T_mis1, pad_mask)  # [B, 768]
        z_mis2 = masked_mean(T_mis2, pad_mask)  # [B, 768]
        # Merge two mismatch directions at feature level
        z_mis = self.mis_mlp(torch.cat([z_mis1, z_mis2], dim=-1))  # [B, 768]

        # Channel 3: Intra-Text Conflict (T_con -> T_inc)
        T_conf, _ = self.mha_tconf(
            query=T_con,
            key=T_inc,
            value=T_inc,
            key_padding_mask=pad_mask
        )
        T_conf = self.ln_tconf(T_con + T_conf)  # [B, Lt, 768]
        z_conf = masked_mean(T_conf, pad_mask)  # [B, 768] - Pool immediately

        # ====================================================================
        # Channel-level Fusion: Concatenate pooled features
        # ====================================================================

        # Concatenate 3 text channel features at feature dimension
        # Shape: [B, 768*3]
        z_text_channels = torch.cat([z_match, z_mis, z_conf], dim=-1)  # [B, 2304]

        # ====================================================================
        # Vision Channel 4: Intra-Vision Conflict (V_con -> V_inc)
        # ====================================================================

        if m_v is None:
            m_v = torch.ones(V_con.size(0), V_con.size(1),
                             dtype=V_con.dtype, device=V_con.device)
        if m_v.size(1) != V_con.size(1):
            if m_v.size(1) > V_con.size(1):
                m_v = m_v[:, :V_con.size(1)]
            else:
                extra_len = V_con.size(1) - m_v.size(1)
                extra = torch.ones(m_v.size(0), extra_len,
                                   dtype=m_v.dtype, device=m_v.device)
                m_v = torch.cat([m_v, extra], dim=1)
        v_mask = (m_v < self.vision_conf_threshold)
        V_conf, _ = self.mha_vconf(
            query=V_con,
            key=V_inc,
            value=V_inc,
            key_padding_mask=v_mask
        )
        V_conf = self.ln_vconf(V_con + V_conf)
        z_vision = self._weighted_pool(V_conf, m_v)  # [B, 768]

        # ====================================================================
        # Final Fusion: Combine text and vision channels
        # ====================================================================

        # Concatenate 3 text channels + 1 vision channel
        z_all_channels = torch.cat([z_text_channels, z_vision], dim=-1)  # [B, 2304+768=3072]
        z_cid = self.final_mlp(z_all_channels)  # [B, 768]
        return z_cid


class RCLMuFN(nn.Module):
    def __init__(self, args):
        super(RCLMuFN, self).__init__()
        self.model = CLIPModel.from_pretrained("/home/user/chengtaiyu/models/clip-vit-base-patch32")
        if args.simple_linear:
            self.text_linear =  nn.Linear(args.text_size, args.image_size)
            self.image_linear =  nn.Linear(args.image_size, args.image_size)
        else:
            self.text_linear =  nn.Sequential(
                nn.Linear(args.text_size, args.image_size),
                nn.Dropout(args.dropout_rate),
                nn.GELU()
            )
            self.image_linear =  nn.Sequential(
                nn.Linear(args.image_size, args.image_size),
                nn.Dropout(args.dropout_rate),
                nn.GELU()
            )
        self.classifier_fuse = nn.Linear(args.image_size , args.label_number)
        self.cross_att = CrossAttention(feature_dim=768, dropout_prob=0.1)
        self.loss_fct = nn.CrossEntropyLoss()
        # Learnable fusion weights (softmax-normalized) for image/text mixes.
        self.res_weight = nn.Parameter(torch.log(torch.tensor([0.6, 0.4], dtype=torch.float)))
        # self.fuse_weight = nn.Parameter(torch.log(torch.tensor([0.7, 0.3], dtype=torch.float)))

        self.d_model = 768
        self.nheads = 8
        self.dim_feedforward = 2048
        # ==========================
        # Post-DIMM: Simple LayerNorm for final output
        # ==========================
        self.post_ln = nn.LayerNorm(768)



        # ========================================================================
        # CID-DIMM Integration
        # ========================================================================
        # Instantiate CID and DIMM modules
        self.cid = CIDModule(
            text_dim=512,
            vision_dim=768,
            hidden_dim=768,
            rho=0.3,
            rho_t=0.5,
            delta=0.1,
            tau0=1.0,
            tau_min=0.4,
            decay=0.9995,
            neg_sampling=args.neg_sampling,
            tau_schedule_mode=getattr(args, 'tau_schedule_mode', 'step')  # 'step' or 'epoch'
        )
        self.dimm = DIMMModule(
            hidden_dim=768,
            num_heads=8,
            dropout=0.1
        )

        # Alignment and connection layers
        self.cid_proj = nn.Linear(768, 768, bias=True)
        self.cid_ln = nn.LayerNorm(768)
        self.alpha_pre = 0.1
        self.pre_ln_t = nn.LayerNorm(768)
        self.pre_ln_v = nn.LayerNorm(768)

        # Loss weights (dynamically scheduled by train.py, these are initial values)
        # NOTE: train.py will override these values at the start of each epoch
        # using _schedule_lambda() based on args.lambda_*_start/end
        self.lambda_ratio = getattr(args, 'lambda_ratio_start', 0.0)
        self.lambda_itm = getattr(args, 'lambda_itm_start', 0.0)

        # ========================================================================
        # Multi-layer Feature Fusion (LAFF-style)
        # ========================================================================
        # Learnable weights for fusing last 4 layers of CLIP features
        # Initialized to uniform weights (1/4 each)
        self.layer_weights_text = nn.Parameter(torch.ones(4) / 4)
        self.layer_weights_vision = nn.Parameter(torch.ones(4) / 4)

    def set_epoch(self, epoch):
        """
        Set current epoch for temperature scheduling in CID module.
        Call this at the beginning of each training epoch.

        Args:
            epoch: current epoch number (0-indexed)
        """
        if hasattr(self, 'cid'):
            self.cid.set_epoch(epoch)

    def forward(self, inputs, batch, labels):
        # ========================================================================
        # CLIP Forward with Multi-layer Feature Extraction
        # ========================================================================
        output = self.model(**inputs, output_attentions=False, output_hidden_states=True)

        # Extract multi-layer features (last 4 layers)
        text_hidden_states = output['text_model_output']['hidden_states']  # Tuple of [B, Lt, 512]
        vision_hidden_states = output['vision_model_output']['hidden_states']  # Tuple of [B, Lv, 768]

        # Fuse last 4 layers with learnable weights (LAFF-style)
        text_layers = text_hidden_states[-4:]  # Last 4 layers
        vision_layers = vision_hidden_states[-4:]  # Last 4 layers

        # Normalize weights to sum to 1 (convex combination)
        weights_t = F.softmax(self.layer_weights_text, dim=0)
        weights_v = F.softmax(self.layer_weights_vision, dim=0)

        # Weighted sum of layers
        text_features = sum(w * layer for w, layer in zip(weights_t, text_layers))  # [B, Lt, 512]
        image_features = sum(w * layer for w, layer in zip(weights_v, vision_layers))  # [B, Lv, 768]

        # Also extract pooled outputs for classification head
        text_feature = output['text_model_output']['pooler_output']  # [B, 512]
        image_feature = output['vision_model_output']['pooler_output']  # [B, 768]
        text_feature = self.text_linear(text_feature)  # 64，768
        image_feature = self.image_linear(image_feature)  # 64,768

        # text_list, image_list, label_list, id_list = batch

        attn_mask = inputs.get("attention_mask", None)
        if attn_mask is None:
            input_ids = inputs.get("input_ids", None)
            if input_ids is not None:
                attn_mask = (input_ids != 0).long()
            else:
                attn_mask = torch.ones(
                    text_features.size(0),
                    text_features.size(1),
                    dtype=torch.long,
                    device=text_features.device
                )
        else:
            attn_mask = attn_mask.to(text_features.device)
        attn_mask = align_attention_mask(attn_mask, text_features.size(1))
        pad_mask = (attn_mask == 0)

        # ========================================================================
        # CID-DIMM Pipeline: Token-level cross attention before CID (Scheme A)
        # ========================================================================
        T_proj = self.cid.text_proj(text_features)
        pre_alpha = self.alpha_pre

        # Text cross-attend to vision (no key_padding_mask needed for vision)
        T_attn = self.cross_att(T_proj, image_features, image_features)
        # Mask out padding positions in query (text) side after attention
        T_attn = T_attn.masked_fill(pad_mask.unsqueeze(-1), 0.0)
        T_ref = T_proj + pre_alpha * T_attn

        # Vision cross-attend to text (with key_padding_mask for text)
        V_ref = image_features + pre_alpha * self.cross_att(
            image_features, T_proj, T_proj, key_padding_mask=pad_mask
        )

        T_ref = self.pre_ln_t(T_ref)
        V_ref = self.pre_ln_v(V_ref)

        # Run CID module (bilateral decomposition)
        T_con, T_inc, V_con, V_inc, m_t, m_v, loss_ratio, loss_itm = self.cid(
            T_ref,
            V_ref,
            attn_mask,
            labels=labels
        )

        valid = (~pad_mask).float()
        valid_den = valid.sum().clamp_min(1.0)
        m_t_valid = m_t * valid
        m_t_mean = m_t_valid.sum() / valid_den
        m_t_var = ((m_t_valid - m_t_mean) ** 2 * valid).sum() / valid_den
        self.last_cid_stats = {
            "m_t_mean": m_t_mean.detach().item(),
            "m_t_var": m_t_var.detach().item(),
            "m_v_mean": m_v.mean().detach().item(),
            "m_v_var": m_v.var(unbiased=False).detach().item(),
        }

        # Run DIMM module (bilateral interaction)
        z_cid = self.dimm(T_con, T_inc, V_con, V_inc, pad_mask, m_v=m_v)  # [B, 768]

        # Project and normalize CID output with residual connection
        cid_hat = self.cid_ln(z_cid + self.cid_proj(z_cid))  # LN(z_cid + W*z_cid)
        # ========================================================================

        # ========================================================================
        # CID-DIMM Connection: Align and integrate with residual connection
        # ========================================================================
        # Scheme A: use CID as the main branch
        # res_new = cid_hat  # [B, 768]


        z_final = self.post_ln(cid_hat)  # [B, 768]

        # Predict
        logits_fuse = self.classifier_fuse(z_final)  # [B, num_labels]
        score = nn.functional.softmax(logits_fuse, dim=-1)

        outputs = (score,) # (64,2)
        if labels is not None:
            loss_fuse = self.loss_fct(logits_fuse, labels)
            # ====================================================================
            # CID-DIMM Loss Integration
            # ====================================================================
            # Combine classification loss with CID consistency losses
            loss = loss_fuse + self.lambda_ratio * loss_ratio + self.lambda_itm * loss_itm
            # ====================================================================
            outputs = (loss,) + outputs
        return outputs
