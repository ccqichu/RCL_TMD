import argparse
import gc
import json
import os
import random
import sys

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor
from sklearn.metrics import precision_score, recall_score, f1_score


src_path = "/home/user/chengtaiyu/RCLMuFN-main_copy/src"
if src_path not in sys.path and os.path.exists(src_path):
    sys.path.append(src_path)

try:
    from model import RCLMuFN as ModelClass
except ImportError as e:
    print(f"Failed to import model: {e}")
    sys.exit(1)


def _load_test_data(text_json_path, image_dirs):
    with open(text_json_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    data = []
    for item in raw_data:
        image_id = item.get("image_id")
        text = item.get("text")
        label = item.get("label")

        image_path = None
        for dir_path in image_dirs:
            candidate = os.path.join(dir_path, f"{image_id}.jpg")
            if os.path.exists(candidate):
                image_path = candidate
                break

        if image_path is None:
            continue

        record = dict(item)
        record.update(
            {
                "image_id": image_id,
                "text": text,
                "label": label,
                "image_path": image_path,
            }
        )
        data.append(record)

    return data


def _normalize(x):
    return x / (x.norm(dim=-1, keepdim=True) + 1e-12)


def _compute_text_features(texts, processor, clip_model, device, batch_size):
    features = []
    clip_model.eval()
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="CLIP text features"):
            batch_texts = texts[i : i + batch_size]
            inputs = processor(
                text=batch_texts,
                images=None,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            text_feats = clip_model.get_text_features(**inputs)
            text_feats = _normalize(text_feats)
            features.append(text_feats.cpu())
    return torch.cat(features, dim=0)


def _compute_image_features(image_paths, processor, clip_model, device, batch_size):
    features = []
    clip_model.eval()
    with torch.no_grad():
        for i in tqdm(range(0, len(image_paths), batch_size), desc="CLIP image features"):
            batch_paths = image_paths[i : i + batch_size]
            images = []
            for path in batch_paths:
                try:
                    images.append(Image.open(path).convert("RGB"))
                except Exception:
                    images.append(Image.new("RGB", (224, 224), color="black"))
            inputs = processor(
                text=None,
                images=images,
                return_tensors="pt",
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            image_feats = clip_model.get_image_features(**inputs)
            image_feats = _normalize(image_feats)
            features.append(image_feats.cpu())
    return torch.cat(features, dim=0)


def _build_exclusion_mask(index, labels, group_keys, label_mode, exclude_group):
    mask = torch.zeros(len(labels), dtype=torch.bool)
    if label_mode == "aware":
        mask |= labels == labels[index]
    if exclude_group and group_keys is not None:
        mask |= group_keys == group_keys[index]
    mask[index] = True
    return mask


def _select_hard_index(
    scores,
    index,
    labels,
    group_keys,
    label_mode,
    exclude_group,
    min_sim,
    max_sim,
    top_k,
    sample_from_topk,
    rng,
):
    neg_inf = -1e9
    scores = scores.clone()

    exclusion_mask = _build_exclusion_mask(index, labels, group_keys, label_mode, exclude_group)
    scores[exclusion_mask] = neg_inf

    if min_sim is not None:
        scores[scores < min_sim] = neg_inf
    if max_sim is not None:
        scores[scores > max_sim] = neg_inf

    k = min(top_k, scores.numel())
    topk_vals, topk_idx = torch.topk(scores, k=k)
    valid = topk_vals > neg_inf / 2

    if valid.any():
        valid_idx = topk_idx[valid].tolist()
        if sample_from_topk and len(valid_idx) > 1:
            return rng.choice(valid_idx), "topk_sample"
        return valid_idx[0], "top1"

    # Fallback: ignore sim thresholds, keep exclusion constraints.
    fallback_candidates = (~exclusion_mask).nonzero(as_tuple=False).squeeze(-1).tolist()
    if not fallback_candidates:
        return index, "fallback_self"
    return rng.choice(fallback_candidates), "fallback_random"


def build_hard_pairs(
    data,
    text_feats,
    image_feats,
    label_mode,
    exclude_group,
    min_sim,
    max_sim,
    top_k,
    sample_from_topk,
    seed,
    sim_batch,
):
    labels = torch.tensor([item["label"] for item in data], dtype=torch.long)
    group_keys = None
    if exclude_group:
        raw_keys = [item.get(exclude_group, None) for item in data]
        key_to_id = {}
        mapped = []
        for key in raw_keys:
            if key is None:
                mapped.append(-1)
                continue
            if key not in key_to_id:
                key_to_id[key] = len(key_to_id) + 1
            mapped.append(key_to_id[key])
        group_keys = torch.tensor(mapped, dtype=torch.long)
    rng = random.Random(seed)

    hard_text = []
    hard_text_reason = []
    hard_text_sim = []

    text_feats_t = text_feats.t()
    for start in tqdm(range(0, len(data), sim_batch), desc="Hard text pairing"):
        end = min(start + sim_batch, len(data))
        batch = image_feats[start:end]
        scores = batch @ text_feats_t
        for offset in range(end - start):
            idx = start + offset
            chosen_idx, reason = _select_hard_index(
                scores[offset],
                idx,
                labels,
                group_keys,
                label_mode,
                exclude_group,
                min_sim,
                max_sim,
                top_k,
                sample_from_topk,
                rng,
            )
            hard_text.append(chosen_idx)
            hard_text_reason.append(reason)
            hard_text_sim.append(float(scores[offset][chosen_idx].item()))

    hard_image = []
    hard_image_reason = []
    hard_image_sim = []

    image_feats_t = image_feats.t()
    for start in tqdm(range(0, len(data), sim_batch), desc="Hard image pairing"):
        end = min(start + sim_batch, len(data))
        batch = text_feats[start:end]
        scores = batch @ image_feats_t
        for offset in range(end - start):
            idx = start + offset
            chosen_idx, reason = _select_hard_index(
                scores[offset],
                idx,
                labels,
                group_keys,
                label_mode,
                exclude_group,
                min_sim,
                max_sim,
                top_k,
                sample_from_topk,
                rng,
            )
            hard_image.append(chosen_idx)
            hard_image_reason.append(reason)
            hard_image_sim.append(float(scores[offset][chosen_idx].item()))

    return {
        "hard_text_indices": hard_text,
        "hard_text_reason": hard_text_reason,
        "hard_text_sim": hard_text_sim,
        "hard_image_indices": hard_image,
        "hard_image_reason": hard_image_reason,
        "hard_image_sim": hard_image_sim,
    }


class HardPairingDataset(Dataset):
    def __init__(self, data, hard_indices):
        self.data = data
        self.hard_indices = hard_indices

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        base = self.data[idx]
        pair_idx = self.hard_indices[idx]
        paired = self.data[pair_idx]

        try:
            image = Image.open(base["image_path"]).convert("RGB")
        except Exception:
            image = Image.new("RGB", (224, 224), color="black")

        text = paired["text"]
        return (
            text,
            image,
            base["label"],
            base["image_id"],
            paired["image_id"],
            base["image_id"],
            base["image_path"],
        )

    @staticmethod
    def collate_func(batch_data):
        text_list = []
        image_list = []
        label_list = []
        id_list = []
        used_text_id_list = []
        used_image_id_list = []
        used_image_path_list = []
        for instance in batch_data:
            text_list.append(instance[0])
            image_list.append(instance[1])
            label_list.append(instance[2])
            id_list.append(instance[3])
            used_text_id_list.append(instance[4])
            used_image_id_list.append(instance[5])
            used_image_path_list.append(instance[6])
        return (
            text_list,
            image_list,
            label_list,
            id_list,
            used_text_id_list,
            used_image_id_list,
            used_image_path_list,
        )


class HardImagePairingDataset(Dataset):
    def __init__(self, data, hard_indices):
        self.data = data
        self.hard_indices = hard_indices

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        base = self.data[idx]
        pair_idx = self.hard_indices[idx]
        paired = self.data[pair_idx]

        try:
            image = Image.open(paired["image_path"]).convert("RGB")
        except Exception:
            image = Image.new("RGB", (224, 224), color="black")

        text = base["text"]
        return (
            text,
            image,
            base["label"],
            base["image_id"],
            base["image_id"],
            paired["image_id"],
            paired["image_path"],
        )

    @staticmethod
    def collate_func(batch_data):
        text_list = []
        image_list = []
        label_list = []
        id_list = []
        used_text_id_list = []
        used_image_id_list = []
        used_image_path_list = []
        for instance in batch_data:
            text_list.append(instance[0])
            image_list.append(instance[1])
            label_list.append(instance[2])
            id_list.append(instance[3])
            used_text_id_list.append(instance[4])
            used_image_id_list.append(instance[5])
            used_image_path_list.append(instance[6])
        return (
            text_list,
            image_list,
            label_list,
            id_list,
            used_text_id_list,
            used_image_id_list,
            used_image_path_list,
        )


def run_test(args, model, dataset, processor, device):
    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        collate_fn=dataset.collate_func,
        shuffle=False,
    )

    n_correct, n_total = 0, 0
    t_targets_all, t_outputs_all = None, None

    model.eval()
    base_ids = []
    keys = []
    text = []
    used_text_ids = []
    used_image_ids = []
    used_image_paths = []
    logit = []

    with torch.no_grad():
        for t_batch in tqdm(data_loader, desc="Testing"):
            (
                text_list,
                image_list,
                label_list,
                id_list,
                used_text_id_list,
                used_image_id_list,
                used_image_path_list,
            ) = t_batch

            keys_batch = [
                f"{text_id}::{used_text_id}::{used_image_id}"
                for text_id, used_text_id, used_image_id in zip(
                    id_list, used_text_id_list, used_image_id_list
                )
            ]
            base_ids.extend(id_list)
            keys.extend(keys_batch)
            text.extend(text_list)
            used_text_ids.extend(used_text_id_list)
            used_image_ids.extend(used_image_id_list)
            used_image_paths.extend(used_image_path_list)

            inputs = processor(
                text=text_list,
                images=image_list,
                padding="max_length",
                truncation=True,
                max_length=args.max_len,
                return_tensors="pt",
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            labels = torch.tensor(label_list).to(device)

            model_batch = (text_list, image_list, label_list, id_list)
            _, t_outputs = model(inputs, model_batch, labels=labels)

            logit.extend(t_outputs.cpu().tolist())
            n_correct += (torch.argmax(t_outputs, -1) == labels).sum().item()
            n_total += len(t_outputs)

            if t_targets_all is None:
                t_targets_all = labels
                t_outputs_all = t_outputs
            else:
                t_targets_all = torch.cat((t_targets_all, labels), dim=0)
                t_outputs_all = torch.cat((t_outputs_all, t_outputs), dim=0)

            del inputs, labels, t_outputs
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if t_targets_all is not None and t_outputs_all is not None:
        predictions = torch.argmax(t_outputs_all.cpu(), -1).numpy().tolist()
        labels = t_targets_all.cpu().numpy().tolist()
    else:
        predictions = []
        labels = []

    return {
        "predictions": predictions,
        "labels": labels,
        "ids": base_ids,
        "keys": keys,
        "texts": text,
        "used_text_ids": used_text_ids,
        "used_image_ids": used_image_ids,
        "used_image_paths": used_image_paths,
        "logits": logit,
    }


def calculate_metrics(predictions, labels):
    if not predictions or not labels:
        return {
            "accuracy": 0,
            "precision": 0,
            "recall": 0,
            "f1": 0,
            "precision_macro": 0,
            "recall_macro": 0,
            "f1_macro": 0,
            "correct": 0,
            "total": 0,
        }

    y_pred = np.array(predictions)
    y_true = np.array(labels)

    correct = np.sum(y_pred == y_true)
    total = len(y_true)
    accuracy = correct / total if total > 0 else 0

    try:
        precision = precision_score(y_true, y_pred, average="binary", zero_division=0)
        recall = recall_score(y_true, y_pred, average="binary", zero_division=0)
        f1 = f1_score(y_true, y_pred, average="binary", zero_division=0)

        precision_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
        recall_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)
        f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    except Exception:
        precision = recall = f1 = precision_macro = recall_macro = f1_macro = 0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "correct": int(correct),
        "total": total,
    }


def main():
    parser = argparse.ArgumentParser(description="Hard pairing test")
    parser.add_argument("--device", default="1", type=str, help="GPU id, -1 for CPU")
    parser.add_argument(
        "--model_path",
        type=str,
        default="/home/user/chengtaiyu/RCLMuFN-main_copy/output_dir/854_pt/model.pt",
        help="model checkpoint path",
    )
    parser.add_argument(
        "--text_json",
        type=str,
        default="/home/user/chengtaiyu/RCLMuFN-main/data/text_final/test.json",
        help="test json path",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="/home/user/chengtaiyu/RCLMuFN-main_copy/src/hard_pairing_results.json",
        help="output json path",
    )
    parser.add_argument("--batch_size", type=int, default=8, help="test batch size")
    parser.add_argument("--clip_batch_size", type=int, default=32, help="CLIP batch size")
    parser.add_argument("--sim_batch", type=int, default=128, help="similarity batch size")
    parser.add_argument("--max_len", type=int, default=77, help="text max length")
    parser.add_argument("--test_subset", type=int, default=0, help="limit samples")

    parser.add_argument("--text_size", default=512, type=int, help="text hidden size")
    parser.add_argument("--image_size", default=768, type=int, help="image hidden size")
    parser.add_argument("--dropout_rate", default=0.1, type=float, help="dropout rate")
    parser.add_argument("--label_number", type=int, default=2, help="label count")
    parser.add_argument("--layers", default=3, type=int, help="transformer layers")
    parser.add_argument("--simple_linear", default=False, type=bool, help="simple linear flag")
    parser.add_argument(
        "--neg_sampling",
        default="label_aware",
        type=str,
        choices=["shuffle", "label_aware", "low_sim"],
        help="neg sampling strategy",
    )

    parser.add_argument(
        "--image_dirs",
        type=str,
        default="/home/user/chengtaiyu/Dataset/Dataset/MMSD2.0/dataset_image",
        help="comma-separated image dirs",
    )
    parser.add_argument(
        "--hard_pairs_cache",
        type=str,
        default="",
        help="json cache for hard pairs",
    )
    parser.add_argument(
        "--hard_label_mode",
        type=str,
        default="agnostic",
        choices=["agnostic", "aware"],
        help="label-aware hard pairing",
    )
    parser.add_argument("--hard_top_k", type=int, default=10, help="top-k pool size")
    parser.add_argument(
        "--hard_sample_from_topk",
        action="store_true",
        help="randomly sample from top-k",
    )
    parser.add_argument("--hard_min_sim", type=float, default=None, help="min cosine sim")
    parser.add_argument("--hard_max_sim", type=float, default=None, help="max cosine sim")
    parser.add_argument(
        "--exclude_group_field",
        type=str,
        default="",
        help="field name to exclude same group",
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed")

    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    use_cpu = args.device == "-1"
    device = torch.device("cpu" if use_cpu or not torch.cuda.is_available() else "cuda")

    image_dirs = [p.strip() for p in args.image_dirs.split(",") if p.strip()]
    for dir_path in image_dirs:
        if not os.path.exists(dir_path):
            print(f"Warning: image dir not found: {dir_path}")

    if not os.path.exists(args.text_json):
        print(f"Missing test json: {args.text_json}")
        return

    data = _load_test_data(args.text_json, image_dirs)
    if args.test_subset > 0:
        data = data[: args.test_subset]
    if not data:
        print("No valid samples found.")
        return

    try:
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    except Exception as e:
        print(f"Failed to load CLIP from hub: {e}")
        local_clip_path = "/home/user/chengtaiyu/models/clip-vit-base-patch32"
        processor = CLIPProcessor.from_pretrained(local_clip_path)
        clip_model = CLIPModel.from_pretrained(local_clip_path).to(device)

    cache = None
    if args.hard_pairs_cache and os.path.exists(args.hard_pairs_cache):
        with open(args.hard_pairs_cache, "r", encoding="utf-8") as f:
            cache = json.load(f)
    if cache is None:
        texts = [item["text"] for item in data]
        image_paths = [item["image_path"] for item in data]

        text_feats = _compute_text_features(
            texts, processor, clip_model, device, args.clip_batch_size
        )
        image_feats = _compute_image_features(
            image_paths, processor, clip_model, device, args.clip_batch_size
        )

        cache = build_hard_pairs(
            data=data,
            text_feats=text_feats,
            image_feats=image_feats,
            label_mode=args.hard_label_mode,
            exclude_group=args.exclude_group_field or None,
            min_sim=args.hard_min_sim,
            max_sim=args.hard_max_sim,
            top_k=args.hard_top_k,
            sample_from_topk=args.hard_sample_from_topk,
            seed=args.seed,
            sim_batch=args.sim_batch,
        )

        if args.hard_pairs_cache:
            with open(args.hard_pairs_cache, "w", encoding="utf-8") as f:
                json.dump(cache, f, indent=2)

    hard_text_indices = cache["hard_text_indices"]
    hard_image_indices = cache["hard_image_indices"]

    model = ModelClass(args)
    if use_cpu:
        model.load_state_dict(torch.load(args.model_path, map_location="cpu"), strict=False)
    else:
        model.load_state_dict(torch.load(args.model_path, map_location=device), strict=False)
    model.to(device)

    conditions = [
        ("hard_pairing_text", HardPairingDataset(data, hard_text_indices)),
        ("hard_pairing_image", HardImagePairingDataset(data, hard_image_indices)),
    ]

    all_results = {}
    all_metrics = {}

    for name, dataset in conditions:
        results = run_test(args, model, dataset, processor, device)
        all_results[name] = results
        all_metrics[name] = calculate_metrics(results["predictions"], results["labels"])

    output = {
        "metrics": all_metrics,
        "results": all_results,
        "hard_pairs": {
            "text_reason": cache.get("hard_text_reason", []),
            "text_sim": cache.get("hard_text_sim", []),
            "image_reason": cache.get("hard_image_reason", []),
            "image_sim": cache.get("hard_image_sim", []),
        },
    }

    output_dir = os.path.dirname(os.path.abspath(args.output_file))
    if output_dir and output_dir != ".":
        os.makedirs(output_dir, exist_ok=True)
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    for name, metrics in all_metrics.items():
        print(f"{name}: acc={metrics['accuracy']:.4f} ({metrics['correct']}/{metrics['total']})")


if __name__ == "__main__":
    main()
