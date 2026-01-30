import argparse
import json
import os
import sys
from typing import Dict, List

import torch
from torch.utils.data import DataLoader
from transformers import CLIPProcessor

SRC_PATH = "/home/user/chengtaiyu/RCLMuFN-main_copy/src"
if SRC_PATH not in sys.path and os.path.exists(SRC_PATH):
    sys.path.append(SRC_PATH)

from data_set import MyDataset
from model import RCLMuFN as ModelClass, align_attention_mask


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


def compute_ratios(m_t, m_v, pad_mask, threshold: float):
    valid = (~pad_mask).float()
    valid_counts = valid.sum(dim=1).clamp_min(1.0)

    text_soft = ((1.0 - m_t) * valid).sum(dim=1) / valid_counts
    text_hard = ((m_t < threshold).float() * valid).sum(dim=1) / valid_counts

    vision_soft = (1.0 - m_v).mean(dim=1)
    vision_hard = (m_v < threshold).float().mean(dim=1)

    return text_soft, text_hard, vision_soft, vision_hard, valid_counts


def forward_cid_masks(model, inputs, labels):
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
    if model.disable_pre_crossatt:
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

    if model.disable_cid:
        valid = (~pad_mask).float()
        m_t = valid
        m_v = torch.ones(
            image_features.size(0), image_features.size(1), dtype=image_features.dtype, device=image_features.device
        )
    else:
        _, _, _, _, m_t, m_v, _, _ = model.cid(
            T_ref,
            V_ref,
            attn_mask,
            labels=labels,
        )

    return m_t, m_v, pad_mask


def main():
    parser = argparse.ArgumentParser(description="Compute CID inconsistency mask ratios per sample")
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
        default="/home/user/chengtaiyu/RCLMuFN-main_copy/src/inconsistency_ratios.json",
        help="Output JSON path",
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--text_name", default="text_clean", type=str, help="Text folder name")
    parser.add_argument("--max_len", type=int, default=77, help="Max text length")
    parser.add_argument("--text_size", type=int, default=512, help="Text hidden size")
    parser.add_argument("--image_size", type=int, default=768, help="Image hidden size")
    parser.add_argument("--dropout_rate", type=float, default=0.1, help="Dropout")
    parser.add_argument("--label_number", type=int, default=2, help="Num labels")
    parser.add_argument("--layers", type=int, default=3, help="Transformer layers")
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
    parser.add_argument("--mode", default="test", type=str, choices=["train", "valid", "test"])
    parser.add_argument("--sarcastic_label", default=1, type=int, help="Label id for sarcastic")
    parser.add_argument("--mask_threshold", default=0.5, type=float, help="Threshold for hard mask")

    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    use_cpu = args.device == "-1"
    device = torch.device("cpu" if use_cpu or not torch.cuda.is_available() else "cuda")
    print(f"Using device: {device}")

    processor = load_processor()

    dataset = MyDataset(mode=args.mode, text_name=args.text_name, limit=None)
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
    model.eval()

    per_sample: List[Dict] = []

    with torch.no_grad():
        for batch in data_loader:
            text_list, image_list, label_list, id_list = batch
            inputs = processor(
                text=text_list,
                images=image_list,
                padding="max_length",
                truncation=True,
                max_length=data_loader.dataset.max_len,
                return_tensors="pt",
            ).to(device, non_blocking=True)
            labels = torch.tensor(label_list, device=device)

            m_t, m_v, pad_mask = forward_cid_masks(model, inputs, labels)
            text_soft, text_hard, vision_soft, vision_hard, valid_counts = compute_ratios(
                m_t, m_v, pad_mask, args.mask_threshold
            )

            for i, sample_id in enumerate(id_list):
                per_sample.append(
                    {
                        "id": int(sample_id),
                        "label": int(label_list[i]),
                        "text_inconsistent_soft": float(text_soft[i].item()),
                        "text_inconsistent_hard": float(text_hard[i].item()),
                        "vision_inconsistent_soft": float(vision_soft[i].item()),
                        "vision_inconsistent_hard": float(vision_hard[i].item()),
                        "text_valid_tokens": int(valid_counts[i].item()),
                        "vision_patches": int(m_v.size(1)),
                    }
                )

    groups = {
        "sarcastic": [],
        "non_sarcastic": [],
    }
    for item in per_sample:
        key = "sarcastic" if item["label"] == args.sarcastic_label else "non_sarcastic"
        groups[key].append(item)

    def _avg(items, key):
        if not items:
            return 0.0
        return float(sum(x[key] for x in items) / len(items))

    summary = {
        "sarcastic": {
            "count": len(groups["sarcastic"]),
            "text_inconsistent_soft": _avg(groups["sarcastic"], "text_inconsistent_soft"),
            "text_inconsistent_hard": _avg(groups["sarcastic"], "text_inconsistent_hard"),
            "vision_inconsistent_soft": _avg(groups["sarcastic"], "vision_inconsistent_soft"),
            "vision_inconsistent_hard": _avg(groups["sarcastic"], "vision_inconsistent_hard"),
        },
        "non_sarcastic": {
            "count": len(groups["non_sarcastic"]),
            "text_inconsistent_soft": _avg(groups["non_sarcastic"], "text_inconsistent_soft"),
            "text_inconsistent_hard": _avg(groups["non_sarcastic"], "text_inconsistent_hard"),
            "vision_inconsistent_soft": _avg(groups["non_sarcastic"], "vision_inconsistent_soft"),
            "vision_inconsistent_hard": _avg(groups["non_sarcastic"], "vision_inconsistent_hard"),
        },
    }

    output_dir = os.path.dirname(os.path.abspath(args.output_file))
    if output_dir and output_dir != ".":
        os.makedirs(output_dir, exist_ok=True)

    payload = {
        "model_path": args.model_path,
        "mode": args.mode,
        "sarcastic_label": args.sarcastic_label,
        "mask_threshold": args.mask_threshold,
        "summary": summary,
        "per_sample": per_sample,
    }

    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"Saved per-sample ratios to: {args.output_file}")


if __name__ == "__main__":
    main()
