import argparse
import json
import os
import sys
from typing import Dict, List

import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader
from transformers import CLIPProcessor

# Add model path to sys.path
SRC_PATH = "/home/user/chengtaiyu/RCLMuFN-main_copy/src"
if SRC_PATH not in sys.path and os.path.exists(SRC_PATH):
    sys.path.append(SRC_PATH)

from data_set import MyDataset
from model import RCLMuFN as ModelClass, align_attention_mask, masked_mean


CLIP_LOCAL_PATHS = [
    "/home/user/chengtaiyu/models/clip-vit-base-patch32",
    "/home/user/2024_cty/RCLMuFN-main/src/models/clip-vit-base-patch32",
]


def load_processor():
    try:
        return CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    except Exception:
        for local_path in CLIP_LOCAL_PATHS:
            if os.path.exists(local_path):
                return CLIPProcessor.from_pretrained(local_path)
    raise RuntimeError("Failed to load CLIPProcessor; check local paths.")


def compute_metrics(labels: List[int], preds: List[int]) -> Dict[str, float]:
    accuracy = accuracy_score(labels, preds) if labels else 0.0
    precision = precision_score(labels, preds, zero_division=0) if labels else 0.0
    recall = recall_score(labels, preds, zero_division=0) if labels else 0.0
    f1 = f1_score(labels, preds, zero_division=0) if labels else 0.0
    precision_macro = precision_score(labels, preds, average="macro", zero_division=0) if labels else 0.0
    recall_macro = recall_score(labels, preds, average="macro", zero_division=0) if labels else 0.0
    f1_macro = f1_score(labels, preds, average="macro", zero_division=0) if labels else 0.0
    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "precision_macro": float(precision_macro),
        "recall_macro": float(recall_macro),
        "f1_macro": float(f1_macro),
        "total": int(len(labels)),
    }


def _random_mask_from_counts(valid_mask: torch.Tensor, counts: torch.Tensor, generator: torch.Generator):
    bsz, seq_len = valid_mask.shape
    mask = torch.zeros_like(valid_mask, dtype=torch.float)
    for i in range(bsz):
        valid_idx = torch.nonzero(valid_mask[i], as_tuple=False).squeeze(1)
        if valid_idx.numel() == 0:
            continue
        k = int(counts[i].item())
        if k <= 0:
            continue
        k = min(k, valid_idx.numel())
        perm = torch.randperm(valid_idx.numel(), device=valid_mask.device, generator=generator)
        chosen = valid_idx[perm[:k]]
        mask[i, chosen] = 1.0
    return mask


def _apply_cid_random_mask(T_ref, V_ref, m_t, m_v, valid_mask, generator):
    m_t_counts = torch.round(m_t.sum(dim=1)).long()
    m_v_counts = torch.round(m_v.sum(dim=1)).long()
    m_t_rand = _random_mask_from_counts(valid_mask.bool(), m_t_counts, generator)
    v_valid = torch.ones_like(m_v, dtype=torch.bool)
    m_v_rand = _random_mask_from_counts(v_valid, m_v_counts, generator)

    T_con = m_t_rand.unsqueeze(-1) * T_ref
    T_inc = (1.0 - m_t_rand).unsqueeze(-1) * T_ref
    V_con = m_v_rand.unsqueeze(-1) * V_ref
    V_inc = (1.0 - m_v_rand).unsqueeze(-1) * V_ref
    return T_con, T_inc, V_con, V_inc, m_t_rand, m_v_rand


def _dimm_forward(dimm, T_con, T_inc, V_con, V_inc, pad_mask, m_v=None, drop_channel=None):
    # Channel 1: Inter-Match
    T_match, _ = dimm.mha_match(
        query=T_con,
        key=V_con,
        value=V_con,
        key_padding_mask=None,
    )
    T_match = dimm.ln_match(T_con + T_match)

    # Channel 2: Inter-Mismatch
    T_mis1, _ = dimm.mha_mis_tc_vi(
        query=T_con,
        key=V_inc,
        value=V_inc,
        key_padding_mask=None,
    )
    T_mis1 = dimm.ln_mis(T_con + T_mis1)

    T_mis2, _ = dimm.mha_mis_ti_vc(
        query=T_inc,
        key=V_con,
        value=V_con,
        key_padding_mask=None,
    )
    T_mis2 = dimm.ln_mis(T_inc + T_mis2)

    T_mis = (T_mis1 + T_mis2) / 2.0

    # Channel 3: Intra-Text Conflict
    T_conf, _ = dimm.mha_tconf(
        query=T_con,
        key=T_inc,
        value=T_inc,
        key_padding_mask=pad_mask,
    )
    T_conf = dimm.ln_tconf(T_con + T_conf)

    if drop_channel == "match":
        T_match = torch.zeros_like(T_match)
    elif drop_channel == "mismatch":
        T_mis = torch.zeros_like(T_mis)
    elif drop_channel == "conflict":
        T_conf = torch.zeros_like(T_conf)

    T_all = torch.cat([T_match, T_mis, T_conf, T_con], dim=1)
    pad_mask_extended = torch.cat([pad_mask, pad_mask, pad_mask, pad_mask], dim=1)
    T_fused, _ = dimm.seq_fusion_mha(
        query=T_all,
        key=T_all,
        value=T_all,
        key_padding_mask=pad_mask_extended,
    )
    T_fused = dimm.seq_fusion_ln(T_all + T_fused)
    z_text = masked_mean(T_fused, pad_mask_extended)

    if m_v is None:
        m_v = torch.ones(V_con.size(0), V_con.size(1), dtype=V_con.dtype, device=V_con.device)
    if m_v.size(1) != V_con.size(1):
        if m_v.size(1) > V_con.size(1):
            m_v = m_v[:, :V_con.size(1)]
        else:
            extra_len = V_con.size(1) - m_v.size(1)
            extra = torch.ones(m_v.size(0), extra_len, dtype=m_v.dtype, device=m_v.device)
            m_v = torch.cat([m_v, extra], dim=1)
    v_mask = (m_v < dimm.vision_conf_threshold)
    V_conf, _ = dimm.mha_vconf(
        query=V_con,
        key=V_inc,
        value=V_inc,
        key_padding_mask=v_mask,
    )
    V_conf = dimm.ln_vconf(V_con + V_conf)
    z_vision = dimm._weighted_pool(V_conf, m_v)

    z_cid = dimm.final_mlp(torch.cat([z_text, z_vision], dim=-1))
    return z_cid


def forward_with_ablation(model, inputs, batch, labels, ablation, generator):
    output = model.model(**inputs, output_attentions=False)

    text_features = output["text_model_output"]["last_hidden_state"]
    image_features = output["vision_model_output"]["last_hidden_state"]
    text_feature = output["text_model_output"]["pooler_output"]
    image_feature = output["vision_model_output"]["pooler_output"]

    text_feature = model.text_linear(text_feature)
    image_feature = model.image_linear(image_feature)

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

    # CID pre-cross attention
    T_proj = model.cid.text_proj(text_features)
    if ablation.get("disable_pre_crossatt"):
        T_ref = T_proj
        V_ref = image_features
    else:
        pre_alpha = model.alpha_pre
        T_attn = model.cross_att(T_proj, image_features, image_features)
        T_attn = T_attn.masked_fill(pad_mask.unsqueeze(-1), 0.0)
        T_ref = T_proj + pre_alpha * T_attn
        V_ref = image_features + pre_alpha * model.cross_att(
            image_features, T_proj, T_proj, key_padding_mask=pad_mask
        )
        T_ref = model.pre_ln_t(T_ref)
        V_ref = model.pre_ln_v(V_ref)

    if ablation.get("disable_cid"):
        valid = (~pad_mask).float()
        m_t = valid
        m_v = torch.ones(
            image_features.size(0), image_features.size(1), dtype=image_features.dtype, device=image_features.device
        )
        T_con, T_inc = T_ref, torch.zeros_like(T_ref)
        V_con, V_inc = V_ref, torch.zeros_like(V_ref)
    else:
        T_con, T_inc, V_con, V_inc, m_t, m_v, _, _ = model.cid(
            T_ref,
            V_ref,
            attn_mask,
            labels=labels,
        )
        if ablation.get("cid_random_mask"):
            valid = (~pad_mask)
            T_con, T_inc, V_con, V_inc, m_t, m_v = _apply_cid_random_mask(
                T_ref, V_ref, m_t, m_v, valid, generator
            )

    if ablation.get("disable_cid_dimm"):
        z_final = model.post_ln((text_feature + image_feature) / 2.0)
    elif ablation.get("disable_dimm"):
        z_cid = 0.5 * (masked_mean(T_con, pad_mask) + V_con.mean(dim=1))
        z_final = model.post_ln(z_cid)
    else:
        z_cid = _dimm_forward(
            model.dimm,
            T_con,
            T_inc,
            V_con,
            V_inc,
            pad_mask,
            m_v=m_v,
            drop_channel=ablation.get("dimm_drop_channel"),
        )
        z_final = model.post_ln(z_cid)

    logits = model.classifier_fuse(z_final)
    scores = torch.softmax(logits, dim=-1)
    return scores


def evaluate_ablation(model, processor, data_loader, device, ablation, random_seed):
    model.eval()
    labels_all = []
    preds_all = []

    generator = torch.Generator(device=device)
    generator.manual_seed(int(random_seed))

    with torch.no_grad():
        for batch in data_loader:
            text_list, image_list, label_list, _ = batch
            inputs = processor(
                text=text_list,
                images=image_list,
                padding="max_length",
                truncation=True,
                max_length=data_loader.dataset.max_len,
                return_tensors="pt",
            ).to(device, non_blocking=True)
            labels = torch.tensor(label_list, device=device)

            scores = forward_with_ablation(model, inputs, batch, labels, ablation, generator)
            preds = torch.argmax(scores, dim=-1).detach().cpu().tolist()

            labels_all.extend(label_list)
            preds_all.extend(preds)

    return compute_metrics(labels_all, preds_all)


def build_ablation_plan():
    return [
        {
            "name": "no_cid_loss",
            "description": "Remove CID auxiliary loss (requires a checkpoint trained with loss weights = 0)",
            "disable_cid_loss": True,
        },
        {
            "name": "no_dimm",
            "description": "Remove DIMM (keep CID decomposition)",
            "disable_dimm": True,
        },
        {
            "name": "no_cid_keep_dimm",
            "description": "Remove CID but keep DIMM (use pseudo decomposition)",
            "disable_cid": True,
        },
        {
            "name": "no_pre_crossatt",
            "description": "Remove pre-CID cross-attention",
            "disable_pre_crossatt": True,
        },
        {
            "name": "no_cid_no_dimm",
            "description": "Remove CID + DIMM",
            "disable_cid_dimm": True,
        },
        {
            "name": "w_o_match",
            "description": "Channel ablation: w/o match",
            "dimm_drop_channel": "match",
        },
        {
            "name": "w_o_mismatch",
            "description": "Channel ablation: w/o mismatch",
            "dimm_drop_channel": "mismatch",
        },
        {
            "name": "w_o_conflict",
            "description": "Channel ablation: w/o conflict",
            "dimm_drop_channel": "conflict",
        },
        {
            "name": "cid_random_mask",
            "description": "CID-random mask with matched mask counts",
            "cid_random_mask": True,
        },
    ]


def main():
    parser = argparse.ArgumentParser(description="RCLMuFN Ablation Tests")
    parser.add_argument("--device", default="0", type=str, help="Device id, -1 for CPU")
    parser.add_argument(
        "--model_path",
        type=str,
        default="/home/user/chengtaiyu/RCLMuFN-main_copy/output_dir/883_pt/model.pt",
        help="Model checkpoint path",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="/home/user/chengtaiyu/RCLMuFN-main_copy/src/883_ablation_results.json",
        help="Output JSON path",
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed")
    parser.add_argument("--text_name", default="text_clean", type=str, help="Text folder name")
    parser.add_argument("--max_len", type=int, default=77, help="Max text length")
    parser.add_argument("--text_size", default=512, type=int, help="Text hidden size")
    parser.add_argument("--image_size", default=768, type=int, help="Image hidden size")
    parser.add_argument("--dropout_rate", default=0.1, type=float, help="Dropout")
    parser.add_argument("--label_number", type=int, default=2, help="Num labels")
    parser.add_argument("--layers", default=3, type=int, help="Transformer layers")
    parser.add_argument("--simple_linear", default=False, type=bool, help="Use simple linear layers")
    parser.add_argument(
        "--neg_sampling",
        default="label_aware",
        type=str,
        choices=["shuffle", "label_aware", "low_sim"],
        help="CID negative sampling",
    )
    parser.add_argument("--tau_schedule_mode", default="epoch", type=str, choices=["step", "epoch"])
    parser.add_argument("--tau_min", default=0.4, type=float)
    parser.add_argument("--tau_decay", default=0.9995, type=float)
    parser.add_argument("--num_heads", default=8, type=int)
    parser.add_argument(
        "--ablation_model_paths",
        type=str,
        default=None,
        help="JSON path mapping ablation name to checkpoint path",
    )
    parser.add_argument(
        "--include_baseline",
        action="store_true",
        help="Also evaluate baseline (no ablation)",
    )

    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    use_cpu = args.device == "-1"
    device = torch.device("cpu" if use_cpu or not torch.cuda.is_available() else "cuda")
    print(f"Using device: {device}")

    processor = load_processor()

    dataset = MyDataset(mode="test", text_name=args.text_name, limit=None)
    dataset.max_len = args.max_len
    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        collate_fn=MyDataset.collate_func,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model = ModelClass(args)
    model.load_state_dict(torch.load(args.model_path, map_location=device), strict=False)
    model.to(device)

    ablations = build_ablation_plan()
    if args.include_baseline:
        ablations.insert(0, {"name": "baseline", "description": "No ablation"})

    model_paths = {}
    if args.ablation_model_paths and os.path.exists(args.ablation_model_paths):
        with open(args.ablation_model_paths, "r", encoding="utf-8") as f:
            model_paths = json.load(f)

    results = {
        "model_path_default": args.model_path,
        "ablation_results": {},
    }

    for ablation in ablations:
        ablation_name = ablation["name"]
        ablation_path = model_paths.get(ablation_name, args.model_path)
        if ablation_path != args.model_path:
            model.load_state_dict(torch.load(ablation_path, map_location=device), strict=False)
            model.to(device)
        print(f"\nRunning ablation: {ablation_name}")
        metrics = evaluate_ablation(model, processor, data_loader, device, ablation, args.random_seed)
        results["ablation_results"][ablation_name] = {
            "description": ablation.get("description", ""),
            "model_path": ablation_path,
            "metrics": metrics,
        }

    output_dir = os.path.dirname(os.path.abspath(args.output_file))
    if output_dir and output_dir != ".":
        os.makedirs(output_dir, exist_ok=True)

    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Saved results to: {args.output_file}")


if __name__ == "__main__":
    main()
