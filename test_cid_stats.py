import argparse
import json
import os
import sys

import torch
from torch.utils.data import DataLoader
from transformers import CLIPProcessor

SRC_PATH = "/home/user/chengtaiyu/RCLMuFN-main_copy/src"
if SRC_PATH not in sys.path and os.path.exists(SRC_PATH):
    sys.path.append(SRC_PATH)

from data_set import MyDataset
from model import RCLMuFN as ModelClass


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


def _fill_model_args(args):
    if not hasattr(args, "simple_linear"):
        args.simple_linear = False
    if not hasattr(args, "text_size"):
        args.text_size = 512
    if not hasattr(args, "image_size"):
        args.image_size = 768
    if not hasattr(args, "dropout_rate"):
        args.dropout_rate = 0.1
    if not hasattr(args, "label_number"):
        args.label_number = 2
    if not hasattr(args, "layers"):
        args.layers = 3
    if not hasattr(args, "neg_sampling"):
        args.neg_sampling = "label_aware"
    if not hasattr(args, "tau_min"):
        args.tau_min = 0.4
    if not hasattr(args, "tau_decay"):
        args.tau_decay = 0.9995
    if not hasattr(args, "tau_schedule_mode"):
        args.tau_schedule_mode = "step"
    if not hasattr(args, "rho"):
        args.rho = 0.3
    if not hasattr(args, "rho_t"):
        args.rho_t = 0.5
    if not hasattr(args, "num_heads"):
        args.num_heads = 8
    if not hasattr(args, "disable_cid"):
        args.disable_cid = False
    if not hasattr(args, "disable_dimm"):
        args.disable_dimm = False
    if not hasattr(args, "disable_pre_crossatt"):
        args.disable_pre_crossatt = False
    if not hasattr(args, "disable_cid_dimm"):
        args.disable_cid_dimm = False
    if not hasattr(args, "dimm_drop_channel"):
        args.dimm_drop_channel = "none"
    if not hasattr(args, "cid_random_mask"):
        args.cid_random_mask = False
    if not hasattr(args, "cid_random_mask_seed"):
        args.cid_random_mask_seed = 42
    if not hasattr(args, "disable_cid_loss"):
        args.disable_cid_loss = False


def main():
    parser = argparse.ArgumentParser(description="Evaluate CID stats and basic metrics")
    parser.add_argument("--device", default="0", type=str, help="Device id, -1 for CPU")
    parser.add_argument("--model_path", type=str, required=True, help="Model checkpoint path")
    parser.add_argument("--mode", default="test", type=str, choices=["train", "valid", "test"])
    parser.add_argument("--text_name", default="text_clean", type=str)
    parser.add_argument("--max_len", type=int, default=77)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output_file", type=str, default="", help="Optional JSON output path")
    parser.add_argument("--tau_min", type=float, default=0.4)
    parser.add_argument("--tau_decay", type=float, default=0.9995)
    parser.add_argument("--tau_schedule_mode", type=str, default="epoch", choices=["step", "epoch"])
    parser.add_argument("--rho", type=float, default=0.3)
    parser.add_argument("--rho_t", type=float, default=0.5)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--neg_sampling", type=str, default="label_aware",
                        choices=["shuffle", "label_aware", "low_sim"])
    args = parser.parse_args()

    _fill_model_args(args)

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    use_cpu = args.device == "-1"
    device = torch.device("cpu" if use_cpu or not torch.cuda.is_available() else "cuda")

    processor = load_processor()
    dataset = MyDataset(mode=args.mode, text_name=args.text_name, limit=args.limit)
    dataset.max_len = args.max_len
    loader = DataLoader(
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

    # Debug: confirm CID hyperparameters are applied
    if hasattr(model, "cid"):
        print("CID config:",
              f"rho={getattr(model.cid, 'rho', None)}",
              f"rho_t={getattr(model.cid, 'rho_t', None)}",
              f"tau_min={getattr(model.cid, 'tau_min', None)}",
              f"decay={getattr(model.cid, 'decay', None)}",
              f"tau_schedule_mode={getattr(model.cid, 'tau_schedule_mode', None)}")

    total_loss = 0.0
    total_correct = 0
    total_count = 0
    mt_mean_sum = 0.0
    mv_mean_sum = 0.0
    mt_var_sum = 0.0
    mv_var_sum = 0.0
    stat_count = 0

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
            ).to(device, non_blocking=True)
            labels = torch.tensor(label_list, device=device)

            # Debug: check average valid token count
            if "attention_mask" in inputs:
                attn_count = inputs["attention_mask"].sum(dim=1).float().mean().item()
                print(f"avg_valid_tokens_per_sample: {attn_count:.2f}")

            outputs = model(inputs, batch, labels)
            loss = outputs[0]
            scores = outputs[1]

            total_loss += float(loss.item()) * labels.size(0)
            preds = scores.argmax(dim=1)
            total_correct += int((preds == labels).sum().item())
            total_count += labels.size(0)

            if hasattr(model, "last_cid_stats"):
                stats = model.last_cid_stats
                mt_mean_sum += float(stats["m_t_mean"]) * labels.size(0)
                mv_mean_sum += float(stats["m_v_mean"]) * labels.size(0)
                mt_var_sum += float(stats["m_t_var"]) * labels.size(0)
                mv_var_sum += float(stats["m_v_var"]) * labels.size(0)
                stat_count += labels.size(0)

    avg_loss = total_loss / max(total_count, 1)
    acc = total_correct / max(total_count, 1)
    out = {
        "mode": args.mode,
        "count": total_count,
        "avg_loss": avg_loss,
        "accuracy": acc,
        "m_t_mean": mt_mean_sum / max(stat_count, 1),
        "m_v_mean": mv_mean_sum / max(stat_count, 1),
        "m_t_var": mt_var_sum / max(stat_count, 1),
        "m_v_var": mv_var_sum / max(stat_count, 1),
    }

    print(json.dumps(out, indent=2, ensure_ascii=False))

    if args.output_file:
        with open(args.output_file, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
