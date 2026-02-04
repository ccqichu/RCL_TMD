import argparse
import os
import sys
from typing import List, Dict, Tuple
import html as _html
import re

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from transformers import CLIPProcessor

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =========================
# 0) Project imports
# =========================
SRC_PATH = "/home/user/chengtaiyu/RCLMuFN-main_copy/src"
if SRC_PATH not in sys.path and os.path.exists(SRC_PATH):
    sys.path.append(SRC_PATH)

from data_set import MyDataset
from model import RCLMuFN as ModelClass, align_attention_mask


CLIP_LOCAL_PATHS = [
    "/home/user/chengtaiyu/models/clip-vit-base-patch32",
]


def load_processor():
    try:
        return CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    except Exception:
        for local_path in CLIP_LOCAL_PATHS:
            if os.path.exists(local_path):
                return CLIPProcessor.from_pretrained(local_path)
    raise RuntimeError("Failed to load CLIPProcessor; check local paths.")


# =========================
# 1) Image helpers
# =========================
def _tensor_to_pil(image_tensor: torch.Tensor, image_mean, image_std) -> Image.Image:
    """
    Convert CLIP pixel_values back to an RGB PIL (note: this is the model input view, e.g., center-crop).
    """
    image = image_tensor.detach().cpu().float()
    for c in range(3):
        image[c] = image[c] * image_std[c] + image_mean[c]
    image = image.clamp(0.0, 1.0)
    image = (image * 255.0).byte().permute(1, 2, 0).numpy()
    return Image.fromarray(image)


# =========================
# 2) Normalization helpers (paper-stable)
# =========================
def _percentile_clip(scores: np.ndarray, lo: float = 5.0, hi: float = 95.0) -> np.ndarray:
    if scores.size == 0:
        return scores
    a = scores.astype(np.float32)
    p_lo = np.percentile(a, lo)
    p_hi = np.percentile(a, hi)
    if p_hi - p_lo < 1e-8:
        return np.zeros_like(a)
    return np.clip(a, p_lo, p_hi)


def _normalize_scores(
    scores: np.ndarray,
    mode: str = "minmax",
    gamma: float = 1.0,
    log_scale: float = 0.0,
    use_percentile_clip: bool = True
) -> np.ndarray:
    """
    Safer normalization than your original (default log=0, gamma=1).
    """
    if scores.size == 0:
        return scores
    s = scores.astype(np.float32)

    if use_percentile_clip:
        s = _percentile_clip(s, 5, 95)

    if log_scale > 0.0:
        s = np.log1p(np.maximum(s, 0.0) * log_scale)

    if mode == "minmax":
        min_v = float(s.min())
        max_v = float(s.max())
        if max_v - min_v < 1e-6:
            s = np.zeros_like(s)
        else:
            s = (s - min_v) / (max_v - min_v)
    elif mode == "none":
        pass
    else:
        raise ValueError(f"Unknown normalize mode: {mode}")

    if gamma != 1.0:
        # gamma < 1 amplifies; >1 compresses
        s = np.power(np.clip(s, 0.0, 1.0), gamma)

    return s


def _rank_normalize(scores: np.ndarray) -> np.ndarray:
    if scores.size == 0:
        return scores
    order = scores.argsort()
    ranks = np.empty_like(order, dtype=np.float32)
    ranks[order] = np.arange(scores.size, dtype=np.float32)
    return ranks / max(scores.size - 1, 1)


# =========================
# 3) Token -> readable words (fix "somepeopleare" issue)
# =========================
def _clip_valid_tokens(tokenizer, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> List[str]:
    ids = input_ids.detach().cpu().tolist()
    mask = attention_mask.detach().cpu().tolist()
    toks = []
    for token_id, keep in zip(ids, mask):
        if keep == 0:
            continue
        tok = tokenizer.convert_ids_to_tokens([token_id])[0]
        toks.append(tok)
    return toks


def _tokens_to_readable_words(tokenizer, toks: List[str]) -> List[str]:
    """
    Convert BPE tokens to a readable string with proper spaces, then split to words.
    """
    text = tokenizer.convert_tokens_to_string(toks)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) == 0:
        return []
    return text.split(" ")


def _map_token_scores_to_word_scores(tokenizer, toks: List[str], token_scores: np.ndarray) -> Tuple[List[str], np.ndarray]:
    """
    Map token-level scores -> word-level scores by grouping tokens that form each word.
    We do a conservative heuristic: rebuild string pieces sequentially and assign tokens to words.

    Returns:
      words: List[str]
      word_scores: np.ndarray shape [len(words)]
    """
    # Build readable text and word list
    readable = tokenizer.convert_tokens_to_string(toks)
    readable = re.sub(r"\s+", " ", readable).strip()
    words = readable.split(" ") if readable else []
    if not words:
        return [], np.array([], dtype=np.float32)

    # Also rebuild token pieces (string) to align. This is heuristic but works well for CLIP tokenizer display.
    token_pieces = []
    for t in toks:
        # convert single token to string piece (may include leading space)
        piece = tokenizer.convert_tokens_to_string([t])
        token_pieces.append(piece)

    # Align tokens to words by accumulating chars ignoring multiple spaces
    word_scores = np.zeros(len(words), dtype=np.float32)
    word_counts = np.zeros(len(words), dtype=np.float32)

    # Target string with single spaces
    target = " ".join(words)

    wi = 0
    pos = 0  # position in target
    for ti, piece in enumerate(token_pieces):
        if wi >= len(words):
            break

        # normalize piece spaces
        piece_norm = re.sub(r"\s+", " ", piece)
        if piece_norm == "":
            continue

        # advance pos by matching piece chars against target
        # strip leading spaces for matching logic, but if piece begins with space we may move to next word boundary
        p = piece_norm

        # If piece starts with space and we're mid-word, jump to next word boundary
        if p.startswith(" ") and pos < len(target) and target[pos] != " ":
            # move pos to next space then skip it
            while pos < len(target) and target[pos] != " ":
                pos += 1
            while pos < len(target) and target[pos] == " ":
                pos += 1
            wi = min(wi + 1, len(words) - 1)

        # now remove leading spaces for char matching
        p2 = p.lstrip(" ")

        # assign this token to current word index wi
        word_scores[wi] += float(token_scores[ti])
        word_counts[wi] += 1.0

        # advance pos by p2 length (best-effort)
        pos += len(p2)
        # if we passed a space boundary, update wi
        # move pos through spaces
        if pos < len(target) and target[pos:pos+1] == " ":
            while pos < len(target) and target[pos] == " ":
                pos += 1
            wi = min(wi + 1, len(words) - 1)

    # average token scores within each word
    word_counts = np.maximum(word_counts, 1.0)
    word_scores = word_scores / word_counts
    return words, word_scores


# =========================
# 4) Word highlighting render (PNG) — no HTML required
# =========================
def _render_word_highlight_png(
    words: List[str],
    scores: np.ndarray,
    out_path: str,
    title: str,
    cmap_name: str = "coolwarm",
    norm_mode: str = "minmax",
    gamma: float = 1.0,
    log_scale: float = 0.0,
    max_chars: int = 70
):
    scores = _normalize_scores(scores, mode=norm_mode, gamma=gamma, log_scale=log_scale)

    # wrap words into lines
    lines = []
    cur = []
    cur_len = 0
    for w in words:
        add = len(w) + (1 if cur else 0)
        if cur_len + add > max_chars:
            lines.append(cur)
            cur = [w]
            cur_len = len(w)
        else:
            cur.append(w)
            cur_len += add
    if cur:
        lines.append(cur)

    # figure height adapt
    n_lines = max(1, len(lines))
    fig_h = 1.2 + 0.55 * n_lines
    fig = plt.figure(figsize=(10.5, fig_h))
    ax = plt.gca()
    ax.axis("off")
    ax.text(0.01, 0.93, title, fontsize=14, weight="bold", transform=ax.transAxes)

    cmap = plt.get_cmap(cmap_name)

    y = 0.78
    line_h = 0.18 if n_lines <= 2 else 0.14

    idx = 0
    for line in lines:
        x = 0.01
        for w in line:
            s = float(scores[idx]) if idx < len(scores) else 0.0
            col = cmap(s)
            ax.text(
                x, y, w + " ",
                fontsize=13, va="top", ha="left",
                transform=ax.transAxes,
                bbox=dict(boxstyle="round,pad=0.22", facecolor=col, edgecolor="none", alpha=0.78)
            )
            x += 0.0125 * (len(w) + 1.4)
            idx += 1
        y -= line_h

    fig.savefig(out_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


# =========================
# 5) Patch heatmap overlay (transparent top-quantile)
# =========================
def _make_patch_heatmap_overlay(
    patch_grid: np.ndarray,
    image: Image.Image,
    out_path: str,
    cmap_name: str = "turbo",
    norm_mode: str = "minmax",
    gamma: float = 1.0,
    log_scale: float = 0.0,
    top_quantile: float = 0.70,
    alpha_scale: float = 0.60,
    overlay_all: bool = False
):
    """
    Paper-friendly overlay:
      - normalize patch grid
      - only show values above top_quantile; low values become transparent
      - overlay on base image without tinting the whole picture
    """
    s = _normalize_scores(patch_grid, mode=norm_mode, gamma=gamma, log_scale=log_scale)
    if overlay_all:
        alpha = np.ones_like(s, dtype=np.float32)
    else:
        thr = float(np.quantile(s, top_quantile))
        alpha = np.clip((s - thr) / (s.max() - thr + 1e-6), 0.0, 1.0) ** 0.85

    # to RGBA
    cmap = matplotlib.colormaps.get_cmap(cmap_name)
    rgba = cmap(s)  # HxWx4 float
    rgba[..., 3] = alpha_scale * alpha  # transparency

    # resize rgba to image size
    w, h = image.size
    rgba_img = Image.fromarray((rgba * 255).astype(np.uint8)).resize((w, h), resample=Image.BILINEAR)

    # blend with base
    base = image.convert("RGBA")
    out = Image.alpha_composite(base, rgba_img)
    out.convert("RGB").save(out_path)


# =========================
# 6) CID/DIMM computations (your model)
# =========================
def _compute_cid_outputs(model, inputs, labels):
    output = model.model(**inputs, output_attentions=False)

    text_features = output["text_model_output"]["last_hidden_state"]
    image_features = output["vision_model_output"]["last_hidden_state"]

    attn_mask = inputs.get("attention_mask")
    if attn_mask is None:
        input_ids = inputs.get("input_ids")
        if input_ids is not None:
            attn_mask = (input_ids != 0).long()
        else:
            attn_mask = torch.ones(
                text_features.size(0),
                text_features.size(1),
                dtype=torch.long,
                device=text_features.device,
            )
    else:
        attn_mask = attn_mask.to(text_features.device)

    attn_mask = align_attention_mask(attn_mask, text_features.size(1))
    pad_mask = (attn_mask == 0)

    T_proj = model.cid.text_proj(text_features)
    if getattr(model, "disable_pre_crossatt", False):
        T_ref = model.pre_ln_t(T_proj)
        V_ref = model.pre_ln_v(image_features)
    else:
        pre_alpha = model.alpha_pre
        T_attn = model.cross_att(T_proj, image_features, image_features)
        T_attn = T_attn.masked_fill(pad_mask.unsqueeze(-1), 0.0)
        T_ref = model.pre_ln_t(T_proj + pre_alpha * T_attn)
        V_ref = model.pre_ln_v(
            image_features + pre_alpha * model.cross_att(
                image_features, T_proj, T_proj, key_padding_mask=pad_mask
            )
        )

    if getattr(model, "disable_cid", False):
        valid = (~pad_mask).float()
        m_t = valid
        m_v = torch.ones(
            image_features.size(0),
            image_features.size(1),
            dtype=image_features.dtype,
            device=image_features.device,
        )
        T_con, T_inc = T_ref, torch.zeros_like(T_ref)
        V_con, V_inc = V_ref, torch.zeros_like(V_ref)
    else:
        T_con, T_inc, V_con, V_inc, m_t, m_v, _, _ = model.cid(
            T_ref, V_ref, attn_mask, labels=labels
        )

    return T_con, T_inc, V_con, V_inc, m_t, m_v, pad_mask


def _compute_dimm_evidence(model, T_con, T_inc, V_con, V_inc, pad_mask, m_v):
    dimm = model.dimm

    T_match, attn_match = dimm.mha_match(
        query=T_con, key=V_con, value=V_con, key_padding_mask=None, need_weights=True
    )
    T_match = dimm.ln_match(T_con + T_match)

    T_mis1, attn_mis1 = dimm.mha_mis_tc_vi(
        query=T_con, key=V_inc, value=V_inc, key_padding_mask=None, need_weights=True
    )
    T_mis1 = dimm.ln_mis(T_con + T_mis1)

    T_mis2, attn_mis2 = dimm.mha_mis_ti_vc(
        query=T_inc, key=V_con, value=V_con, key_padding_mask=None, need_weights=True
    )
    T_mis2 = dimm.ln_mis(T_inc + T_mis2)
    T_mis = (T_mis1 + T_mis2) / 2.0

    T_conf, attn_conf = dimm.mha_tconf(
        query=T_con, key=T_inc, value=T_inc, key_padding_mask=pad_mask, need_weights=True
    )
    T_conf = dimm.ln_tconf(T_con + T_conf)

    v_mask = (m_v < dimm.vision_conf_threshold)
    V_conf, attn_vconf = dimm.mha_vconf(
        query=V_con, key=V_inc, value=V_inc, key_padding_mask=v_mask, need_weights=True
    )
    V_conf = dimm.ln_vconf(V_con + V_conf)

    return {
        "T_match": T_match,
        "T_mis": T_mis,
        "T_conf": T_conf,
        "T_con": T_con,
        "attn_match": attn_match,
        "attn_mis1": attn_mis1,
        "attn_mis2": attn_mis2,
        "attn_conf": attn_conf,
        "attn_vconf": attn_vconf,
        "V_conf": V_conf,
    }


# =========================
# 7) Attention aggregation — FIXED (no more mean=constant)
# =========================
def _attn_to_2d(attn: torch.Tensor, sample_i: int) -> np.ndarray:
    """
    Convert attention weights to a single [tgt_len, src_len] array for sample i.
    Supports shapes:
      [B, H, tgt, src] or [B, tgt, src] or [H, tgt, src] or [tgt, src]
    """
    a = attn.detach().cpu().numpy()
    if a.ndim == 4:      # [B,H,tgt,src]
        a = a[sample_i].mean(axis=0)  # -> [tgt,src]
    elif a.ndim == 3:
        # could be [B,tgt,src] OR [H,tgt,src]
        if a.shape[0] > 1 and a.shape[1] > 1 and a.shape[2] > 1:
            # ambiguous; assume [B,tgt,src] if first dim equals batch indexable
            # We'll handle sample_i if possible
            if sample_i < a.shape[0]:
                a = a[sample_i]  # -> [tgt,src]
            else:
                a = a.mean(axis=0)
        else:
            a = np.array(a, ndmin=2)
    elif a.ndim == 2:
        pass
    else:
        a = np.array(a, ndmin=2)
    return a


def _token_evidence_from_attn(attn_2d: np.ndarray, src_weights: np.ndarray = None, mode: str = "weighted_sum") -> np.ndarray:
    """
    token evidence = aggregate attention over src positions.
    - mode="weighted_sum": sum_j attn[i,j]*src_weights[j]  (best for "evidence aligned" visualization)
    - mode="max": max_j attn[i,j]
    """
    if src_weights is not None:
        src_weights = src_weights.reshape(1, -1)
        ev = (attn_2d * src_weights).sum(axis=-1)
    else:
        if mode == "max":
            ev = attn_2d.max(axis=-1)
        else:
            # default: sum (still meaningful; unlike mean)
            ev = attn_2d.sum(axis=-1)
    return ev.astype(np.float32)


def _patch_evidence_from_attn(attn_2d: np.ndarray, token_weights: np.ndarray = None, mode: str = "weighted_sum") -> np.ndarray:
    """
    patch evidence = aggregate attention over tgt tokens.
    - mode="weighted_sum": sum_i attn[i,j]*token_weights[i]
    - mode="max": max_i attn[i,j]
    """
    if token_weights is not None:
        token_weights = token_weights.reshape(-1, 1)
        ev = (attn_2d * token_weights).sum(axis=0)
    else:
        if mode == "max":
            ev = attn_2d.max(axis=0)
        else:
            ev = attn_2d.sum(axis=0)
    return ev.astype(np.float32)


# =========================
# 8) Patch grid builder (handles CLS)
# =========================
def _patch_grid_from_vector(vec: np.ndarray) -> np.ndarray:
    """
    vec: [P] or [1+P], where P is square number.
    If first element is CLS, drop it.
    """
    v = vec.reshape(-1).astype(np.float32)
    if v.size > 1:
        # if v includes CLS and remaining forms a square
        p = v.size - 1
        if int(np.sqrt(p)) ** 2 == p:
            v = v[1:]
    g = int(np.sqrt(v.size))
    if g * g != v.size:
        raise ValueError(f"Patch grid is not square: {v.size}")
    return v.reshape(g, g)


# =========================
# 9) Plotting DIMM channel weights (optional)
# =========================
def _bar_plot(weights: List[float], labels: List[str], out_path: str, title: str):
    plt.figure(figsize=(4, 3))
    plt.bar(labels, weights)
    plt.ylim(0, 1)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


# =========================
# 10) Main
# =========================
def main():
    print("[visualize_cid_dimm] main start", flush=True)
    parser = argparse.ArgumentParser(description="CID + DIMM visualization (paper-ready)")
    parser.add_argument("--device", default="0", type=str, help="Device id, -1 for CPU")
    parser.add_argument("--model_path", type=str, required=True, help="Model checkpoint path")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--mode", default="test", type=str, choices=["train", "valid", "test"])
    parser.add_argument("--text_name", default="text_final", type=str)
    parser.add_argument("--max_len", type=int, default=77)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--sample_ids", type=str, default="", help="Comma-separated sample ids")

    # Paper-safe normalization defaults:
    parser.add_argument("--norm_mode", type=str, default="minmax", choices=["minmax", "none"])
    parser.add_argument("--gamma", type=float, default=1.0, help="Try 0.7~1.0 if you want slightly more contrast")
    parser.add_argument("--log_scale", type=float, default=0.0, help="Keep 0 for paper; if needed try 5~10")
    parser.add_argument("--top_quantile", type=float, default=0.70, help="Only show top quantile region in patch heatmap")
    parser.add_argument("--alpha_scale", type=float, default=0.60, help="Overlay alpha scale")
    parser.add_argument("--overlay_all", action="store_true", help="Overlay full heatmap (no top-quantile masking)")

    # What to show as CID token scores:
    parser.add_argument("--cid_token_mode", type=str, default="inconsistency",
                        choices=["consistency", "inconsistency", "threshold", "rank"],
                        help="CID token visualization source")
    parser.add_argument("--token_threshold", type=float, default=0.5)

    # Attention aggregation mode:
    parser.add_argument("--attn_agg", type=str, default="weighted_sum", choices=["weighted_sum", "max", "sum"])

    args = parser.parse_args()

    # Fill in model args expected by RCLMuFN (your original defaults)
    for k, v in {
        "simple_linear": False, "text_size": 512, "image_size": 768, "dropout_rate": 0.1,
        "label_number": 2, "layers": 3, "neg_sampling": "label_aware", "tau_min": 0.4,
        "tau_decay": 0.9995, "tau_schedule_mode": "step", "num_heads": 8,
        "disable_cid": False, "disable_dimm": False, "disable_pre_crossatt": False,
        "disable_cid_dimm": False, "dimm_drop_channel": "none", "cid_random_mask": False,
        "cid_random_mask_seed": 42, "disable_cid_loss": False
    }.items():
        if not hasattr(args, k):
            setattr(args, k, v)

    os.makedirs(args.output_dir, exist_ok=True)

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    use_cpu = args.device == "-1"
    device = torch.device("cpu" if use_cpu or not torch.cuda.is_available() else "cuda")

    processor = load_processor()
    tokenizer = processor.tokenizer
    image_mean = processor.image_processor.image_mean
    image_std = processor.image_processor.image_std

    dataset = MyDataset(mode=args.mode, text_name=args.text_name, limit=None)
    dataset.max_len = args.max_len
    if args.sample_ids:
        ids = [int(x) for x in args.sample_ids.split(",") if x.strip()]
        dataset.image_ids = [i for i in dataset.image_ids if i in ids]
        print(f"[visualize_cid_dimm] filter sample_ids={ids} -> kept={len(dataset.image_ids)}", flush=True)
    if args.limit:
        dataset.image_ids = dataset.image_ids[: args.limit]
    print(f"[visualize_cid_dimm] dataset size={len(dataset.image_ids)} mode={args.mode}", flush=True)
    if len(dataset.image_ids) == 0:
        print("[visualize_cid_dimm] no samples to visualize; check --mode/--sample_ids/data paths", flush=True)
        return

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        collate_fn=MyDataset.collate_func,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model = ModelClass(args)
    model.load_state_dict(torch.load(args.model_path, map_location=device), strict=False)
    model.to(device)
    model.eval()

    with torch.no_grad():
        for batch in loader:
            text_list, image_list, label_list, id_list = batch
            inputs = processor(
                text=text_list,
                images=image_list,
                padding="max_length",
                truncation=True,
                max_length=dataset.max_len,
                return_tensors="pt",
            ).to(device)
            labels = torch.tensor(label_list, device=device)

            T_con, T_inc, V_con, V_inc, m_t, m_v, pad_mask = _compute_cid_outputs(model, inputs, labels)
            evidence = _compute_dimm_evidence(model, T_con, T_inc, V_con, V_inc, pad_mask, m_v)

            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]
            pixel_values = inputs["pixel_values"]

            for bi, sample_id in enumerate(id_list):
                out_dir = os.path.join(args.output_dir, str(sample_id))
                os.makedirs(out_dir, exist_ok=True)
                print(f"[visualize_cid_dimm] sample_id={sample_id} -> {out_dir}", flush=True)

                # Base image: model input view
                pil_img = _tensor_to_pil(pixel_values[bi], image_mean, image_std)

                # ============ CID patch ============
                mv_vec = m_v[bi].detach().cpu().float().numpy()
                mv_grid = _patch_grid_from_vector(mv_vec)
                _make_patch_heatmap_overlay(
                    mv_grid, pil_img, os.path.join(out_dir, "cid_patch.png"),
                    cmap_name="turbo",
                    norm_mode=args.norm_mode, gamma=args.gamma, log_scale=args.log_scale,
                    top_quantile=args.top_quantile, alpha_scale=args.alpha_scale,
                    overlay_all=args.overlay_all
                )

                # ============ CID token (word-level) ============
                toks = _clip_valid_tokens(tokenizer, input_ids[bi], attention_mask[bi])
                # token scores: mt for valid tokens
                mt = m_t[bi].detach().cpu().float().numpy()
                keep = attention_mask[bi].detach().cpu().numpy().astype(bool)
                mt_valid = mt[keep]

                if args.cid_token_mode == "consistency":
                    token_scores = mt_valid
                elif args.cid_token_mode == "inconsistency":
                    token_scores = 1.0 - mt_valid
                elif args.cid_token_mode == "threshold":
                    token_scores = (mt_valid >= args.token_threshold).astype(np.float32)
                elif args.cid_token_mode == "rank":
                    token_scores = _rank_normalize(mt_valid)
                else:
                    token_scores = 1.0 - mt_valid

                words, word_scores = _map_token_scores_to_word_scores(tokenizer, toks, token_scores)
                _render_word_highlight_png(
                    words, word_scores, os.path.join(out_dir, "cid_token.png"),
                    title=f"CID Token ({args.cid_token_mode})",
                    cmap_name="coolwarm",
                    norm_mode=args.norm_mode, gamma=args.gamma, log_scale=args.log_scale,
                    max_chars=70
                )

                # ============ DIMM channel weights (optional) ============
                def _token_energy(x):
                    x = x[bi].detach().cpu()
                    keep_t = attention_mask[bi].detach().cpu().bool()
                    x = x[keep_t]
                    if x.numel() == 0:
                        return 0.0
                    return float(torch.norm(x, dim=-1).mean().item())

                w_match = _token_energy(evidence["T_match"])
                w_mis = _token_energy(evidence["T_mis"])
                w_conf = _token_energy(evidence["T_conf"])
                w_base = _token_energy(evidence["T_con"])
                weights = np.array([w_match, w_mis, w_conf, w_base], dtype=np.float32)
                weights = weights / (weights.sum() + 1e-6)
                _bar_plot(weights.tolist(), ["Match", "Mismatch", "T-Conflict", "Base"],
                          os.path.join(out_dir, "dimm_channel_weights.png"),
                          "DIMM Channel Weights (normed)")

                # ============ DIMM attentions -> evidence ============
                # Prepare patch weights from CID (mv) for weighted token evidence
                mv_for_patch = mv_grid.reshape(-1)  # [P]
                # Prepare token weights from CID (mt) for weighted patch evidence
                mt_for_token = mt_valid  # [L_valid]

                # Attentions as [tgt, src]
                A_match = _attn_to_2d(evidence["attn_match"], bi)
                A_mis1 = _attn_to_2d(evidence["attn_mis1"], bi)
                A_mis2 = _attn_to_2d(evidence["attn_mis2"], bi)
                A_mis = 0.5 * (A_mis1 + A_mis2)
                A_conf = _attn_to_2d(evidence["attn_conf"], bi)
                A_vconf = _attn_to_2d(evidence["attn_vconf"], bi)

                # Make sure lengths


if __name__ == "__main__":
    print("[visualize_cid_dimm] __main__ entry", flush=True)
    main()
