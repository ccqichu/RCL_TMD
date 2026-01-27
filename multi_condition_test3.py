import argparse
import gc
import json
import os
import random
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

# Add model path to sys.path
src_path = "/home/user/chengtaiyu/RCLMuFN-main_copy/src"
if src_path not in sys.path and os.path.exists(src_path):
    sys.path.append(src_path)
    print(f"添加路径: {src_path}")

# Import model
try:
    from model import RCLMuFN as ModelClass
    print("成功导入 RCLMuFN 从 model 模块")
except ImportError as e:
    print(f"导入模型失败: {e}")
    sys.exit(1)


def build_derangement(n, rng):
    if n <= 1:
        return list(range(n))
    perm = list(range(n))
    for _ in range(1000):
        rng.shuffle(perm)
        if all(i != perm[i] for i in range(n)):
            return perm
    return perm[1:] + perm[:1]


def choose_random_token_id(tokenizer, generator):
    if tokenizer is None or getattr(tokenizer, "vocab_size", None) is None:
        return None
    special_ids = set(getattr(tokenizer, "all_special_ids", []) or [])
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is not None:
        special_ids.add(pad_token_id)
    vocab_size = tokenizer.vocab_size
    for _ in range(10):
        token_id = int(torch.randint(0, vocab_size, (1,), generator=generator).item())
        if token_id not in special_ids:
            return token_id
    if getattr(tokenizer, "unk_token_id", None) is not None and tokenizer.unk_token_id not in special_ids:
        return tokenizer.unk_token_id
    fallback = (pad_token_id + 1) % vocab_size if pad_token_id is not None else 1
    return fallback


def _build_special_mask(input_ids, tokenizer):
    special_ids = list(getattr(tokenizer, "all_special_ids", []) if tokenizer is not None else [])
    if tokenizer is not None and tokenizer.pad_token_id in special_ids:
        special_ids.remove(tokenizer.pad_token_id)
    if not special_ids:
        return torch.zeros_like(input_ids, dtype=torch.bool)
    special_ids_tensor = torch.tensor(special_ids, dtype=input_ids.dtype, device=input_ids.device)
    return (input_ids.unsqueeze(-1) == special_ids_tensor).any(-1)


def _replace_special_ids(rand_ids, tokenizer, fallback_id):
    if tokenizer is None:
        return rand_ids
    special_ids = list(getattr(tokenizer, "all_special_ids", []) or [])
    if tokenizer.pad_token_id in special_ids:
        special_ids.remove(tokenizer.pad_token_id)
    if not special_ids:
        return rand_ids
    special_ids_tensor = torch.tensor(special_ids, dtype=rand_ids.dtype, device=rand_ids.device)
    is_special = (rand_ids.unsqueeze(-1) == special_ids_tensor).any(-1)
    if is_special.any():
        rand_ids = rand_ids.clone()
        rand_ids[is_special] = fallback_id
    return rand_ids


def apply_text_controls(inputs, processor, zero_text=False, random_tokens_mode=None, fixed_attention_len=None,
                        generator=None, shared_token_id=None, fill_token_id=None):
    if zero_text is False and random_tokens_mode is None and fixed_attention_len is None:
        return inputs
    if generator is None:
        generator = torch.Generator()
        generator.manual_seed(0)

    attention_mask = inputs.get("attention_mask")
    if attention_mask is None:
        return inputs

    pad_token_id = None
    vocab_size = None
    tokenizer = None
    if processor is not None and hasattr(processor, "tokenizer"):
        tokenizer = processor.tokenizer
        pad_token_id = tokenizer.pad_token_id
        vocab_size = getattr(tokenizer, "vocab_size", None)

    input_ids = inputs.get("input_ids")
    special_mask = _build_special_mask(input_ids, tokenizer) if input_ids is not None else None
    if fill_token_id is None and tokenizer is not None and vocab_size is not None:
        fill_token_id = choose_random_token_id(tokenizer, generator)

    orig_attention_mask = attention_mask.clone()
    if fixed_attention_len is not None:
        seq_len = attention_mask.shape[1]
        keep_len = min(max(int(fixed_attention_len), 0), seq_len)
        fixed_mask = torch.zeros_like(attention_mask)
        if keep_len > 0:
            fixed_mask[:, :keep_len] = 1
        attention_mask = fixed_mask
        inputs["attention_mask"] = attention_mask
        if input_ids is not None:
            input_ids = input_ids.clone()
            new_unmasked = (attention_mask == 1) & (orig_attention_mask == 0)
            if special_mask is not None:
                new_unmasked = new_unmasked & (~special_mask)
            if fill_token_id is not None:
                input_ids[new_unmasked] = fill_token_id
            if pad_token_id is not None:
                input_ids[attention_mask == 0] = pad_token_id
            inputs["input_ids"] = input_ids

    if zero_text and input_ids is not None:
        if fill_token_id is not None:
            mask = attention_mask.bool()
            if special_mask is not None:
                mask = mask & (~special_mask)
            input_ids = input_ids.clone()
            input_ids[mask] = fill_token_id
            inputs["input_ids"] = input_ids

    if random_tokens_mode is not None and vocab_size is not None and input_ids is not None:
        input_ids = input_ids.clone()
        mask = attention_mask.bool()
        if special_mask is not None:
            mask = mask & (~special_mask)
        if random_tokens_mode == "per_sample":
            rand_ids = torch.randint(0, vocab_size, input_ids.shape, dtype=input_ids.dtype, generator=generator)
            if fill_token_id is not None:
                rand_ids = _replace_special_ids(rand_ids, tokenizer, fill_token_id)
            if special_mask is not None and special_mask.any():
                rand_ids[special_mask] = input_ids[special_mask]
            if pad_token_id is not None:
                rand_ids = torch.where(rand_ids == pad_token_id, (rand_ids + 1) % vocab_size, rand_ids)
            if fill_token_id is not None:
                rand_ids = _replace_special_ids(rand_ids, tokenizer, fill_token_id)
            input_ids = torch.where(mask, rand_ids, input_ids)
        elif random_tokens_mode == "shared":
            token_id = shared_token_id
            if token_id is None:
                token_id = choose_random_token_id(tokenizer, generator)
            if token_id is not None:
                input_ids[mask] = token_id
        inputs["input_ids"] = input_ids

    return inputs


CLIP_LOCAL_PATHS = [
    "/home/user/chengtaiyu/models/clip-vit-base-patch32",
    "/home/user/2024_cty/RCLMuFN-main/src/models/clip-vit-base-patch32",
]


def load_clip_model(device):
    try:
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    except Exception:
        model = None
        for local_path in CLIP_LOCAL_PATHS:
            if os.path.exists(local_path):
                model = CLIPModel.from_pretrained(local_path)
                break
        if model is None:
            raise
    model.to(device)
    model.eval()
    return model


def _normalize_embeddings(embeds):
    return F.normalize(embeds, p=2, dim=-1)


def compute_clip_embeddings(texts, image_paths, processor, model, device, batch_size=32):
    text_embeds = []
    image_embeds = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start:start + batch_size]
            inputs = processor(
                text=batch_texts,
                padding=True,
                truncation=True,
                return_tensors="pt"
            ).to(device)
            batch_embeds = model.get_text_features(**inputs)
            text_embeds.append(batch_embeds.detach().cpu())
        for start in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[start:start + batch_size]
            images = []
            for path in batch_paths:
                try:
                    images.append(Image.open(path).convert("RGB"))
                except Exception:
                    images.append(Image.new("RGB", (224, 224), color="black"))
            inputs = processor(images=images, return_tensors="pt").to(device)
            batch_embeds = model.get_image_features(**inputs)
            image_embeds.append(batch_embeds.detach().cpu())
    text_embeds = _normalize_embeddings(torch.cat(text_embeds, dim=0))
    image_embeds = _normalize_embeddings(torch.cat(image_embeds, dim=0))
    return text_embeds, image_embeds


def build_hard_mismatch_indices(src_embeds, tgt_embeds, chunk_size=256):
    if src_embeds.shape[0] <= 1:
        return list(range(src_embeds.shape[0]))
    indices = []
    tgt_t = tgt_embeds.t()
    for start in range(0, src_embeds.shape[0], chunk_size):
        end = min(start + chunk_size, src_embeds.shape[0])
        sims = src_embeds[start:end] @ tgt_t
        for row, i in enumerate(range(start, end)):
            if i < sims.shape[1]:
                sims[row, i] = -1e4
        best = torch.argmax(sims, dim=1)
        indices.extend(best.tolist())
    return indices


def compute_text_mask_ratio(full_image, masked_image, diff_threshold=10):
    full_arr = np.asarray(full_image).astype(np.int16)
    masked_arr = np.asarray(masked_image).astype(np.int16)
    if full_arr.shape != masked_arr.shape:
        masked_image = masked_image.resize((full_arr.shape[1], full_arr.shape[0]))
        masked_arr = np.asarray(masked_image).astype(np.int16)
    diff = np.abs(full_arr - masked_arr).mean(axis=2)
    ratio = float((diff > diff_threshold).mean())
    return max(0.0, min(1.0, ratio))


def build_random_block_mask(height, width, target_ratio, rng):
    target_area = int(round(target_ratio * height * width))
    if target_area <= 0:
        return np.zeros((height, width), dtype=bool)
    mask = np.zeros((height, width), dtype=bool)
    max_attempts = 60
    for _ in range(max_attempts):
        if mask.sum() >= target_area:
            break
        remaining = target_area - int(mask.sum())
        if remaining <= 0:
            break
        min_area = max(1, int(0.2 * remaining))
        block_area = rng.randint(min_area, max(1, remaining))
        aspect = rng.uniform(0.3, 3.0)
        block_w = int(max(1, min(width, round((block_area * aspect) ** 0.5))))
        block_h = int(max(1, min(height, round((block_area / aspect) ** 0.5))))
        x = rng.randint(0, max(0, width - block_w))
        y = rng.randint(0, max(0, height - block_h))
        mask[y:y + block_h, x:x + block_w] = True
    return mask


def apply_random_mask(image, target_ratio, rng, mode="mask", fill_color=(0, 0, 0)):
    if target_ratio <= 0:
        if mode == "keep":
            return Image.new("RGB", image.size, color=fill_color)
        return image
    width, height = image.size
    mask = build_random_block_mask(height, width, target_ratio, rng)
    image_arr = np.asarray(image).copy()
    fill_arr = np.array(fill_color, dtype=image_arr.dtype)
    if mode == "mask":
        image_arr[mask] = fill_arr
        return Image.fromarray(image_arr)
    if mode == "keep":
        out = np.zeros_like(image_arr)
        out[:] = fill_arr
        out[mask] = image_arr[mask]
        return Image.fromarray(out)
    return image


class CustomTestDataset(Dataset):
    """Dataset for conditional tests."""
    def __init__(self, text_json_path, image_dirs, processor, zero_text=False):
        self.data = []
        self.image_dirs = image_dirs
        self.processor = processor
        self.zero_text = zero_text

        with open(text_json_path, "r", encoding="utf-8") as f:
            test_data = json.load(f)

        for item in test_data:
            image_id = item["image_id"]
            text = item["text"]
            label = item["label"]

            image_path = None
            for dir_path in image_dirs:
                potential_path = os.path.join(dir_path, f"{image_id}.jpg")
                if os.path.exists(potential_path):
                    image_path = potential_path
                    break

            if image_path:
                self.data.append({
                    "image_id": image_id,
                    "text": text,
                    "label": label,
                    "image_path": image_path,
                })

        print(f"加载了 {len(self.data)} 个有效测试样本")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        try:
            image = Image.open(item["image_path"]).convert("RGB")
        except Exception as e:
            print(f"加载图像 {item['image_path']} 时出错: {e}")
            image = Image.new("RGB", (224, 224), color="black")

        text = item["text"]
        text_id = item["image_id"]
        used_text_id = item["image_id"]
        used_image_id = item["image_id"]
        used_image_path = item["image_path"]
        return text, image, item["label"], text_id, used_text_id, used_image_id, used_image_path

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

        return text_list, image_list, label_list, id_list, used_text_id_list, used_image_id_list, used_image_path_list


class RandomMaskDataset(CustomTestDataset):
    """Random masking based on TextMasked ratio."""
    def __init__(self, text_json_path, image_dirs, processor, masked_dir, zero_text=False, mask_mode="mask",
                 diff_threshold=10):
        super().__init__(text_json_path, image_dirs, processor, zero_text)
        self.masked_dir = masked_dir
        self.mask_mode = mask_mode
        self.diff_threshold = diff_threshold
        self.mask_ratios = self._build_mask_ratios()

    def _build_mask_ratios(self):
        ratios = {}
        values = []
        missing = []
        for item in self.data:
            masked_path = os.path.join(self.masked_dir, f"{item['image_id']}.jpg")
            if not os.path.exists(masked_path):
                missing.append(item["image_id"])
                continue
            try:
                full_image = Image.open(item["image_path"]).convert("RGB")
                masked_image = Image.open(masked_path).convert("RGB")
                ratio = compute_text_mask_ratio(full_image, masked_image, diff_threshold=self.diff_threshold)
                ratios[item["image_id"]] = ratio
                values.append(ratio)
            except Exception as e:
                print(f"计算遮挡比例失败 {item['image_id']}: {e}")
                missing.append(item["image_id"])
        avg_ratio = float(np.mean(values)) if values else 0.0
        for image_id in missing:
            ratios[image_id] = avg_ratio
        return ratios

    def __getitem__(self, idx):
        item = self.data[idx]
        try:
            image = Image.open(item["image_path"]).convert("RGB")
        except Exception as e:
            print(f"加载图像 {item['image_path']} 时出错: {e}")
            image = Image.new("RGB", (224, 224), color="black")

        ratio = self.mask_ratios.get(item["image_id"], 0.0)
        rng = random.Random(getattr(self, "random_seed", 0) + idx)
        image = apply_random_mask(image, ratio, rng, mode=self.mask_mode)

        text = item["text"]
        text_id = item["image_id"]
        used_text_id = item["image_id"]
        used_image_id = item["image_id"]
        used_image_path = item["image_path"]
        return text, image, item["label"], text_id, used_text_id, used_image_id, used_image_path


HARD_PAIR_CACHE = {}


def get_hard_pair_indices(data, processor, device, batch_size, cache_key):
    if cache_key in HARD_PAIR_CACHE:
        return HARD_PAIR_CACHE[cache_key]
    texts = [item["text"] for item in data]
    image_paths = [item["image_path"] for item in data]
    clip_model = load_clip_model(device)
    text_embeds, image_embeds = compute_clip_embeddings(
        texts,
        image_paths,
        processor,
        clip_model,
        device,
        batch_size=batch_size,
    )
    image_to_text = build_hard_mismatch_indices(image_embeds, text_embeds)
    text_to_image = build_hard_mismatch_indices(text_embeds, image_embeds)
    HARD_PAIR_CACHE[cache_key] = (image_to_text, text_to_image)
    return image_to_text, text_to_image


class HardMismatchPairingDataset(CustomTestDataset):
    """Hard mismatch: fix image, choose most similar text excluding self."""
    def __init__(self, text_json_path, image_dirs, processor, zero_text=False, clip_device=None, clip_batch_size=32):
        super().__init__(text_json_path, image_dirs, processor, zero_text)
        self.clip_device = clip_device or torch.device("cpu")
        self.clip_batch_size = clip_batch_size
        cache_key = ("hard_pairs", text_json_path, tuple(image_dirs))
        self.image_to_text, _ = get_hard_pair_indices(
            self.data,
            processor,
            self.clip_device,
            self.clip_batch_size,
            cache_key,
        )

    def __getitem__(self, idx):
        item = self.data[idx]
        try:
            image = Image.open(item["image_path"]).convert("RGB")
        except Exception as e:
            print(f"加载图像 {item['image_path']} 时出错: {e}")
            image = Image.new("RGB", (224, 224), color="black")

        match_idx = self.image_to_text[idx] if idx < len(self.image_to_text) else idx
        if match_idx == idx and len(self.data) > 1:
            match_idx = (idx + 1) % len(self.data)
        matched = self.data[match_idx]
        text = matched["text"]

        text_id = item["image_id"]
        used_text_id = matched["image_id"]
        used_image_id = item["image_id"]
        used_image_path = item["image_path"]
        return text, image, item["label"], text_id, used_text_id, used_image_id, used_image_path


class HardMismatchImagePairingDataset(CustomTestDataset):
    """Hard mismatch: fix text, choose most similar image excluding self."""
    def __init__(self, text_json_path, image_dirs, processor, zero_text=False, clip_device=None, clip_batch_size=32):
        super().__init__(text_json_path, image_dirs, processor, zero_text)
        self.clip_device = clip_device or torch.device("cpu")
        self.clip_batch_size = clip_batch_size
        cache_key = ("hard_pairs", text_json_path, tuple(image_dirs))
        _, self.text_to_image = get_hard_pair_indices(
            self.data,
            processor,
            self.clip_device,
            self.clip_batch_size,
            cache_key,
        )

    def __getitem__(self, idx):
        item = self.data[idx]
        match_idx = self.text_to_image[idx] if idx < len(self.text_to_image) else idx
        if match_idx == idx and len(self.data) > 1:
            match_idx = (idx + 1) % len(self.data)
        matched = self.data[match_idx]
        random_image_path = matched["image_path"]
        used_image_id = matched["image_id"]

        try:
            image = Image.open(random_image_path).convert("RGB")
        except Exception as e:
            print(f"加载图像 {random_image_path} 时出错: {e}")
            image = Image.new("RGB", (224, 224), color="black")

        text = item["text"]
        text_id = item["image_id"]
        used_text_id = item["image_id"]
        used_image_path = random_image_path
        return text, image, item["label"], text_id, used_text_id, used_image_id, used_image_path


def run_test(args, model, dataset, processor, device, use_cpu=False):
    batch_size = min(args.batch_size, 4) if not use_cpu else 1

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=CustomTestDataset.collate_func,
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

    if use_cpu:
        model = model.cpu()
        print("模型已移至CPU进行测试")

    generator = torch.Generator()
    generator.manual_seed(getattr(dataset, "random_seed", 0))
    shared_token_id = getattr(dataset, "shared_random_token_id", None)
    fill_token_id = getattr(dataset, "fill_token_id", None)
    if fill_token_id is None:
        fill_token_id = choose_random_token_id(getattr(processor, "tokenizer", None), generator)

    def prepare_inputs(text_list, image_list):
        inputs = processor(
            text=text_list,
            images=image_list,
            padding="max_length",
            truncation=True,
            max_length=args.max_len,
            return_tensors="pt",
        )
        inputs = apply_text_controls(
            inputs,
            processor,
            zero_text=getattr(dataset, "zero_text", False),
            random_tokens_mode=getattr(dataset, "random_tokens_mode", None),
            fixed_attention_len=getattr(dataset, "fixed_attention_len", None),
            generator=generator,
            shared_token_id=shared_token_id,
            fill_token_id=fill_token_id,
        )
        if getattr(dataset, "no_text_branch", False):
            inputs["attention_mask"] = torch.zeros_like(inputs["attention_mask"])
        return inputs

    with torch.no_grad():
        for i_batch, t_batch in enumerate(tqdm(data_loader, desc="测试中")):
            text_list, image_list, label_list, id_list, used_text_id_list, used_image_id_list, used_image_path_list = t_batch

            keys_batch = [
                f"{text_id}::{used_text_id}::{used_image_id}"
                for text_id, used_text_id, used_image_id in zip(id_list, used_text_id_list, used_image_id_list)
            ]
            base_ids.extend(id_list)
            keys.extend(keys_batch)
            text.extend(text_list)
            used_text_ids.extend(used_text_id_list)
            used_image_ids.extend(used_image_id_list)
            used_image_paths.extend(used_image_path_list)

            inputs = prepare_inputs(text_list, image_list)
            model_batch = (text_list, image_list, label_list, id_list)

            if use_cpu:
                inputs = {k: v.cpu() for k, v in inputs.items()}
                labels = torch.tensor(label_list).cpu()
            else:
                inputs = {k: v.to(device) for k, v in inputs.items()}
                labels = torch.tensor(label_list).to(device)

            t_targets = labels

            try:
                loss, t_outputs = model(
                    inputs,
                    model_batch,
                    labels=labels,
                    zero_text=getattr(dataset, "zero_text_vector", False),
                    no_text_branch=getattr(dataset, "no_text_branch", False),
                    zero_image=getattr(dataset, "zero_image_vector", False),
                )

                logit.extend(t_outputs.cpu().tolist())
                n_correct += (torch.argmax(t_outputs, -1) == t_targets).sum().item()
                n_total += len(t_outputs)

                if t_targets_all is None:
                    t_targets_all = t_targets
                    t_outputs_all = t_outputs
                else:
                    t_targets_all = torch.cat((t_targets_all, t_targets), dim=0)
                    t_outputs_all = torch.cat((t_outputs_all, t_outputs), dim=0)

                del inputs, labels, t_outputs
                if i_batch % 10 == 0:
                    gc.collect()
                    if not use_cpu and torch.cuda.is_available():
                        torch.cuda.empty_cache()

            except RuntimeError as e:
                if "out of memory" in str(e) and not use_cpu:
                    print("GPU内存不足，尝试在CPU上处理这个批次")
                    model = model.cpu()

                    inputs = prepare_inputs(text_list, image_list)
                    labels = torch.tensor(label_list)
                    t_targets = labels

                    loss, t_outputs = model(
                        inputs,
                        model_batch,
                        labels=labels,
                        zero_text=getattr(dataset, "zero_text_vector", False),
                        no_text_branch=getattr(dataset, "no_text_branch", False),
                        zero_image=getattr(dataset, "zero_image_vector", False),
                    )

                    logit.extend(t_outputs.cpu().tolist())
                    n_correct += (torch.argmax(t_outputs, -1) == t_targets).sum().item()
                    n_total += len(t_outputs)

                    if t_targets_all is None:
                        t_targets_all = t_targets
                        t_outputs_all = t_outputs
                    else:
                        t_targets_all = torch.cat((t_targets_all, t_targets), dim=0)
                        t_outputs_all = torch.cat((t_outputs_all, t_outputs), dim=0)

                    if not use_cpu and torch.cuda.is_available():
                        model = model.to(device)

                    del inputs, labels, t_outputs
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                else:
                    raise e

    if t_targets_all is not None and t_outputs_all is not None:
        predictions = torch.argmax(t_outputs_all.cpu(), -1).numpy().tolist()
        labels = t_targets_all.cpu().numpy().tolist()
    else:
        predictions = []
        labels = []

    del t_targets_all, t_outputs_all
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

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
    except Exception as e:
        print(f"计算指标时出错: {e}")
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
    parser = argparse.ArgumentParser(description="多条件测试3")
    parser.add_argument("--device", default="0", type=str, help="设备号，使用-1表示强制使用CPU")
    parser.add_argument(
        "--model_path",
        type=str,
        default="/home/user/chengtaiyu/RCLMuFN-main_copy/output_dir/883_pt/model.pt",
        help="保存的模型路径",
    )
    parser.add_argument(
        "--text_json",
        type=str,
        default="/home/user/chengtaiyu/RCLMuFN-main/data/text_clean/test.json",
        help="测试JSON文件的路径",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="/home/user/chengtaiyu/RCLMuFN-main_copy/src/multi_condition_results3.json",
        help="结果输出文件",
    )
    parser.add_argument("--batch_size", type=int, default=32, help="测试的批次大小")
    parser.add_argument("--fixed_attention_len", type=int, default=32, help="固定attention_mask的长度")
    parser.add_argument("--random_seed", type=int, default=42, help="随机种子（控制随机token实验）")
    parser.add_argument("--force_cpu", action="store_true", help="强制在CPU上运行，即使有GPU可用")
    parser.add_argument("--test_subset", type=int, default=0, help="只测试指定数量的样本，0表示测试所有样本")

    parser.add_argument("--max_len", type=int, default=77, help="文本的最大长度")
    parser.add_argument("--text_size", default=512, type=int, help="文本隐藏大小")
    parser.add_argument("--image_size", default=768, type=int, help="图像隐藏大小")
    parser.add_argument("--dropout_rate", default=0.1, type=float, help="dropout率")
    parser.add_argument("--label_number", type=int, default=2, help="分类标签的数量")
    parser.add_argument("--layers", default=3, type=int, help="transformer的层数")
    parser.add_argument("--simple_linear", default=False, type=bool, help="线性实现选择")
    parser.add_argument(
        "--neg_sampling",
        default="label_aware",
        type=str,
        choices=["shuffle", "label_aware", "low_sim"],
        help="CID负样本采样策略",
    )

    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    use_cpu = args.force_cpu or args.device == "-1"
    if use_cpu:
        device = torch.device("cpu")
        print("强制使用CPU进行测试")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() and int(args.device) >= 0 else "cpu")

    print(f"使用设备: {device}")

    output_dir = os.path.dirname(os.path.abspath(args.output_file))
    if output_dir and output_dir != ".":
        os.makedirs(output_dir, exist_ok=True)

    try:
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        print("成功加载CLIP处理器")
    except Exception as e:
        print(f"加载CLIP处理器失败: {e}")
        print("尝试从本地路径加载...")
        try:
            local_clip_path = "/home/user/chengtaiyu/models/clip-vit-base-patch32"
            processor = CLIPProcessor.from_pretrained(local_clip_path)
            print(f"从 {local_clip_path} 加载了CLIP处理器")
        except Exception as e2:
            print(f"从本地路径加载CLIP处理器也失败: {e2}")
            return

    try:
        model = ModelClass(args)
        print(f"创建了模型: {model.__class__.__name__}")

        if use_cpu:
            model.load_state_dict(torch.load(args.model_path, map_location="cpu"), strict=False)
        else:
            model.load_state_dict(torch.load(args.model_path, map_location=device), strict=False)

        model.to(device)
        print(f"从 {args.model_path} 加载了模型")
    except Exception as e:
        print(f"创建或加载模型失败: {e}")
        return

    base_data_dir = "/home/user/chengtaiyu/Dataset/Dataset/MMSD2.0"
    special_dir = "/home/user/chengtaiyu/Dataset/Dataset"
    image_dirs = [
        os.path.join(base_data_dir, "dataset_image"),
        os.path.join(special_dir, "MMSD_masked"),
        os.path.join(special_dir, "MMSD_textonly"),
    ]

    for dir_path in image_dirs:
        if not os.path.exists(dir_path):
            print(f"警告: 图像目录 {dir_path} 不存在")

    if not os.path.exists(args.text_json):
        print(f"警告: 测试JSON文件 {args.text_json} 未找到。请提供正确的路径。")
        return

    test_conditions = [
        {
            "name": "rand_masked_normal_text",
            "description": "RandMasked：Full随机遮挡（面积=TextMasked）和正常文本输入",
        },
        {
            "name": "rand_only_normal_text",
            "description": "RandOnly：Full随机保留（面积=TextMasked）和正常文本输入",
        },
        {
            "name": "hard_pairing",
            "description": "硬错配（相似度最高）的图像和文本",
        },
        {
            "name": "hard_image_pairing",
            "description": "硬错配（相似度最高）的文本和图像",
        },
    ]

    all_results = {}
    all_metrics = {}

    for condition in test_conditions:
        print(f"\n运行测试: {condition['description']}")

        zero_text_vector = False
        no_text_branch = False
        zero_image_vector = False

        if condition["name"] == "rand_masked_normal_text":
            dataset = RandomMaskDataset(
                args.text_json,
                [image_dirs[0]],
                processor,
                masked_dir=image_dirs[1],
                zero_text=False,
                mask_mode="mask",
            )
        elif condition["name"] == "rand_only_normal_text":
            dataset = RandomMaskDataset(
                args.text_json,
                [image_dirs[0]],
                processor,
                masked_dir=image_dirs[1],
                zero_text=False,
                mask_mode="keep",
            )
        elif condition["name"] == "hard_pairing":
            clip_device = device if not use_cpu else torch.device("cpu")
            dataset = HardMismatchPairingDataset(
                args.text_json,
                [image_dirs[0]],
                processor,
                zero_text=False,
                clip_device=clip_device,
            )
        elif condition["name"] == "hard_image_pairing":
            clip_device = device if not use_cpu else torch.device("cpu")
            dataset = HardMismatchImagePairingDataset(
                args.text_json,
                [image_dirs[0]],
                processor,
                zero_text=False,
                clip_device=clip_device,
            )
        else:
            raise ValueError(f"未识别的测试条件: {condition['name']}")

        dataset.random_tokens_mode = None
        dataset.zero_text_vector = zero_text_vector
        dataset.no_text_branch = no_text_branch
        dataset.zero_image_vector = zero_image_vector
        dataset.random_seed = args.random_seed
        dataset.shared_random_token_id = None
        dataset.fill_token_id = None
        if dataset.random_tokens_mode == "shared":
            generator = torch.Generator()
            generator.manual_seed(args.random_seed)
            dataset.shared_random_token_id = choose_random_token_id(getattr(processor, "tokenizer", None), generator)
            dataset.fill_token_id = choose_random_token_id(getattr(processor, "tokenizer", None), generator)
        if dataset.fill_token_id is None:
            generator = torch.Generator()
            generator.manual_seed(args.random_seed)
            dataset.fill_token_id = choose_random_token_id(getattr(processor, "tokenizer", None), generator)
        dataset.fixed_attention_len = None

        if args.test_subset > 0 and args.test_subset < len(dataset):
            dataset.data = dataset.data[:args.test_subset]
            print(f"限制测试样本数量为 {args.test_subset}")
            if isinstance(dataset, HardMismatchPairingDataset):
                cache_key = ("hard_pairs_subset", args.text_json, tuple(dataset.image_dirs), len(dataset.data))
                dataset.image_to_text, _ = get_hard_pair_indices(
                    dataset.data,
                    processor,
                    dataset.clip_device,
                    dataset.clip_batch_size,
                    cache_key,
                )
            if isinstance(dataset, HardMismatchImagePairingDataset):
                cache_key = ("hard_pairs_subset", args.text_json, tuple(dataset.image_dirs), len(dataset.data))
                _, dataset.text_to_image = get_hard_pair_indices(
                    dataset.data,
                    processor,
                    dataset.clip_device,
                    dataset.clip_batch_size,
                    cache_key,
                )

        try:
            results = run_test(args, model, dataset, processor, device, use_cpu=use_cpu)
            all_results[condition["name"]] = results

            metrics = calculate_metrics(results["predictions"], results["labels"])
            all_metrics[condition["name"]] = metrics

            print(f"测试结果 - {condition['description']}:")
            print(f"  - 准确率: {metrics['accuracy']:.4f} ({metrics['correct']}/{metrics['total']})")
            print(f"  - 二分类 - 精确率: {metrics['precision']:.4f}, 召回率: {metrics['recall']:.4f}, F1分数: {metrics['f1']:.4f}")
            print(f"  - 宏平均 - 精确率: {metrics['precision_macro']:.4f}, 召回率: {metrics['recall_macro']:.4f}, F1分数: {metrics['f1_macro']:.4f}")

        except Exception as e:
            print(f"测试 {condition['name']} 失败: {e}")
            all_results[condition["name"]] = {"predictions": [], "labels": [], "ids": []}
            all_metrics[condition["name"]] = {
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

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    consolidated_results = {}

    all_ids = set()
    for condition_name, results in all_results.items():
        all_ids.update(results["ids"])

    id_to_index = {}
    for condition_name, results in all_results.items():
        id_to_index[condition_name] = {id_val: idx for idx, id_val in enumerate(results["ids"])}

    for sample_id in all_ids:
        consolidated_results[sample_id] = {
            "labels": {},
            "predictions": {},
            "used_text_id": {},
            "used_image_id": {},
            "used_image_path": {},
            "key": {},
        }

        for condition_name, results in all_results.items():
            if sample_id in id_to_index[condition_name]:
                idx = id_to_index[condition_name][sample_id]
                prediction = results["predictions"][idx]
                true_label = results["labels"][idx]

                consolidated_results[sample_id]["predictions"][condition_name] = prediction
                consolidated_results[sample_id]["labels"][condition_name] = true_label
                consolidated_results[sample_id]["used_text_id"][condition_name] = results["used_text_ids"][idx]
                consolidated_results[sample_id]["used_image_id"][condition_name] = results["used_image_ids"][idx]
                consolidated_results[sample_id]["used_image_path"][condition_name] = results["used_image_paths"][idx]
                consolidated_results[sample_id]["key"][condition_name] = results["keys"][idx]

    output = {
        "test_conditions": [c["description"] for c in test_conditions],
        "results": consolidated_results,
        "metrics": all_metrics,
    }

    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"\n结果已保存到 {args.output_file}")

    print("\n多条件测试统计信息:")
    for condition in test_conditions:
        condition_name = condition["name"]
        metrics = all_metrics.get(condition_name, {})

        if metrics:
            correct = metrics["correct"]
            total = metrics["total"]
            accuracy = metrics["accuracy"]
            precision = metrics["precision"]
            recall = metrics["recall"]
            f1 = metrics["f1"]
            precision_macro = metrics["precision_macro"]
            recall_macro = metrics["recall_macro"]
            f1_macro = metrics["f1_macro"]

            print(f"{condition['description']}:")
            print(f"  - 准确率: {accuracy:.4f} ({correct}/{total})")
            print(f"  - 二分类 - 召回率: {recall:.4f}, 精确率: {precision:.4f}, F1分数: {f1:.4f}")
            print(f"  - 宏平均 - 召回率: {recall_macro:.4f}, 精确率: {precision_macro:.4f}, F1分数: {f1_macro:.4f}")
        else:
            print(f"{condition['description']}: 没有有效的指标数据")


if __name__ == "__main__":
    main()
