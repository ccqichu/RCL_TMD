import argparse
import gc
import inspect
import json
import math
import os
import random
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFile
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

from model import RCLMuFN as ModelClass

ImageFile.LOAD_TRUNCATED_IMAGES = True

CLIP_LOCAL_PATHS = [
    "/home/user/chengtaiyu/RCLMuFN-main_copy/src/models/clip-vit-base-patch32",
    "/home/user/chengtaiyu/models/clip-vit-base-patch32",
]


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


def apply_text_controls(
    inputs,
    processor,
    zero_text=False,
    random_tokens_mode=None,
    fixed_attention_len=None,
    generator=None,
    shared_token_id=None,
    fill_token_id=None,
):
    if zero_text is False and random_tokens_mode is None and fixed_attention_len is None:
        return inputs
    if generator is None:
        generator = torch.Generator()
        generator.manual_seed(0)

    attention_mask = inputs.get("attention_mask")
    if attention_mask is None:
        return inputs

    vocab_size = None
    tokenizer = None
    if processor is not None and hasattr(processor, "tokenizer"):
        tokenizer = processor.tokenizer
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
            if tokenizer is not None and tokenizer.pad_token_id is not None:
                input_ids[attention_mask == 0] = tokenizer.pad_token_id
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
            if tokenizer is not None and tokenizer.pad_token_id is not None:
                rand_ids = torch.where(
                    rand_ids == tokenizer.pad_token_id, (rand_ids + 1) % vocab_size, rand_ids
                )
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


class CustomTestDataset(Dataset):
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
                self.data.append(
                    {
                        "image_id": image_id,
                        "text": text,
                        "label": label,
                        "image_path": image_path,
                    }
                )

        print(f"Loaded {len(self.data)} samples")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        try:
            image = Image.open(item["image_path"]).convert("RGB")
        except Exception as e:
            print(f"Failed to load image {item['image_path']}: {e}")
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

        return (
            text_list,
            image_list,
            label_list,
            id_list,
            used_text_id_list,
            used_image_id_list,
            used_image_path_list,
        )


class MaskedImageDataset(CustomTestDataset):
    def __init__(self, text_json_path, image_dirs, processor, zero_text=False, mask_type="full"):
        super().__init__(text_json_path, image_dirs, processor, zero_text)
        self.mask_type = mask_type

    def __getitem__(self, idx):
        item = self.data[idx]

        if self.mask_type == "full":
            image = Image.new("RGB", (224, 224), color="black")
        else:
            try:
                image = Image.open(item["image_path"]).convert("RGB")
            except Exception as e:
                print(f"Failed to load image {item['image_path']}: {e}")
                image = Image.new("RGB", (224, 224), color="black")

        text = item["text"]
        text_id = item["image_id"]
        used_text_id = item["image_id"]
        used_image_id = item["image_id"]
        used_image_path = item["image_path"]

        return text, image, item["label"], text_id, used_text_id, used_image_id, used_image_path


class RandomPairingDataset(CustomTestDataset):
    def __init__(self, text_json_path, image_dirs, processor, zero_text=False, seed=1122):
        super().__init__(text_json_path, image_dirs, processor, zero_text)
        self.seed = seed
        self.rng = random.Random(seed)

        self.text_pool = [item["text"] for item in self.data]
        self.text_ids = [item["image_id"] for item in self.data]
        self.perm = build_derangement(len(self.text_pool), self.rng)

    def __getitem__(self, idx):
        item = self.data[idx]

        try:
            image = Image.open(item["image_path"]).convert("RGB")
        except Exception as e:
            print(f"Failed to load image {item['image_path']}: {e}")
            image = Image.new("RGB", (224, 224), color="black")

        if len(self.text_pool) > 0:
            text = self.text_pool[self.perm[idx]]
            used_text_id = self.text_ids[self.perm[idx]]
        else:
            text = item["text"]
            used_text_id = item["image_id"]
        text_id = item["image_id"]
        used_image_id = item["image_id"]
        used_image_path = item["image_path"]
        return text, image, item["label"], text_id, used_text_id, used_image_id, used_image_path


class RandomImagePairingDataset(CustomTestDataset):
    def __init__(self, text_json_path, image_dirs, processor, zero_text=False, seed=1122):
        super().__init__(text_json_path, image_dirs, processor, zero_text)
        self.seed = seed
        self.rng = random.Random(seed)

        self.image_paths = [item["image_path"] for item in self.data]
        self.image_ids = [item["image_id"] for item in self.data]
        self.perm = build_derangement(len(self.image_paths), self.rng)

    def __getitem__(self, idx):
        item = self.data[idx]

        if len(self.image_paths) > 0:
            perm_index = self.perm[idx]
            random_image_path = self.image_paths[perm_index]
            used_image_id = self.image_ids[perm_index]
        else:
            random_image_path = item["image_path"]
            used_image_id = item["image_id"]

        try:
            image = Image.open(random_image_path).convert("RGB")
        except Exception as e:
            print(f"Failed to load image {random_image_path}: {e}")
            image = Image.new("RGB", (224, 224), color="black")

        text = item["text"]
        text_id = item["image_id"]
        used_text_id = item["image_id"]
        used_image_path = random_image_path

        return text, image, item["label"], text_id, used_text_id, used_image_id, used_image_path


def _normalize_embeddings(embeds):
    return F.normalize(embeds, p=2, dim=-1)


def load_clip_model(device, clip_path=""):
    if clip_path:
        model = CLIPModel.from_pretrained(clip_path)
        model.to(device)
        model.eval()
        return model
    for local_path in CLIP_LOCAL_PATHS:
        if os.path.exists(local_path):
            model = CLIPModel.from_pretrained(local_path)
            model.to(device)
            model.eval()
            return model
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    model.to(device)
    model.eval()
    return model


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
                return_tensors="pt",
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
        side = max(1, int(round(remaining ** 0.5)))
        h = min(height, rng.randint(1, max(1, side)))
        w = min(width, rng.randint(1, max(1, side)))
        top = rng.randint(0, max(0, height - h))
        left = rng.randint(0, max(0, width - w))
        mask[top:top + h, left:left + w] = True
    return mask


def apply_random_mask(image, target_ratio, rng, mode="mask", fill_color=(0, 0, 0)):
    width, height = image.size
    mask = build_random_block_mask(height, width, target_ratio, rng)
    arr = np.asarray(image).copy()
    if mode == "mask":
        arr[mask] = fill_color
    elif mode == "keep":
        arr[~mask] = fill_color
    else:
        raise ValueError(f"Unknown mask mode: {mode}")
    return Image.fromarray(arr)


class RandomMaskDataset(CustomTestDataset):
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
                print(f"Failed to compute mask ratio {item['image_id']}: {e}")
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
            print(f"Failed to load image {item['image_path']}: {e}")
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


def get_hard_pair_indices(data, processor, device, batch_size, cache_key, clip_path=""):
    if cache_key in HARD_PAIR_CACHE:
        return HARD_PAIR_CACHE[cache_key]
    texts = [item["text"] for item in data]
    image_paths = [item["image_path"] for item in data]
    clip_model = load_clip_model(device, clip_path=clip_path)
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
    def __init__(self, text_json_path, image_dirs, processor, zero_text=False, clip_device=None,
                 clip_batch_size=32, clip_path=""):
        super().__init__(text_json_path, image_dirs, processor, zero_text)
        self.clip_device = clip_device or torch.device("cpu")
        self.clip_batch_size = clip_batch_size
        self.clip_path = clip_path
        cache_key = ("hard_pairs", text_json_path, tuple(image_dirs))
        self.image_to_text, _ = get_hard_pair_indices(
            self.data,
            processor,
            self.clip_device,
            self.clip_batch_size,
            cache_key,
            clip_path=self.clip_path,
        )

    def __getitem__(self, idx):
        item = self.data[idx]
        try:
            image = Image.open(item["image_path"]).convert("RGB")
        except Exception as e:
            print(f"Failed to load image {item['image_path']}: {e}")
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
    def __init__(self, text_json_path, image_dirs, processor, zero_text=False, clip_device=None,
                 clip_batch_size=32, clip_path=""):
        super().__init__(text_json_path, image_dirs, processor, zero_text)
        self.clip_device = clip_device or torch.device("cpu")
        self.clip_batch_size = clip_batch_size
        self.clip_path = clip_path
        cache_key = ("hard_pairs", text_json_path, tuple(image_dirs))
        _, self.text_to_image = get_hard_pair_indices(
            self.data,
            processor,
            self.clip_device,
            self.clip_batch_size,
            cache_key,
            clip_path=self.clip_path,
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
            print(f"Failed to load image {random_image_path}: {e}")
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

    t_targets_all, t_outputs_all = None, None

    model.eval()

    if use_cpu:
        model = model.cpu()
        print("Testing on CPU")

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
        if getattr(dataset, "zero_image_vector", False):
            inputs["pixel_values"] = torch.zeros_like(inputs["pixel_values"])
        if isinstance(dataset, MaskedImageDataset) and getattr(dataset, "mask_type", "") == "full":
            inputs["pixel_values"] = torch.zeros_like(inputs["pixel_values"])
        return inputs

    def _supports_kwarg(callable_obj, name):
        try:
            sig = inspect.signature(callable_obj)
        except (TypeError, ValueError):
            return False
        for param in sig.parameters.values():
            if param.kind == inspect.Parameter.VAR_KEYWORD:
                return True
        return name in sig.parameters

    forward_fn = getattr(model, "forward", None)
    supports_zero_text = _supports_kwarg(forward_fn, "zero_text") if forward_fn else False
    supports_no_text_branch = _supports_kwarg(forward_fn, "no_text_branch") if forward_fn else False
    supports_zero_image = _supports_kwarg(forward_fn, "zero_image") if forward_fn else False

    with torch.no_grad():
        for i_batch, t_batch in enumerate(tqdm(data_loader, desc="Testing")):
            text_list, image_list, label_list, id_list, used_text_id_list, used_image_id_list, used_image_path_list = (
                t_batch
            )
            inputs = prepare_inputs(text_list, image_list)
            model_batch = (text_list, image_list, label_list, id_list)

            if use_cpu:
                inputs = {k: v.cpu() for k, v in inputs.items()}
                labels = torch.tensor(label_list).cpu()
            else:
                inputs = {k: v.to(device) for k, v in inputs.items()}
                labels = torch.tensor(label_list).to(device)

            if getattr(dataset, "zero_text_vector", False) and not supports_zero_text:
                tokenizer = getattr(processor, "tokenizer", None)
                pad_token_id = getattr(tokenizer, "pad_token_id", 0) if tokenizer is not None else 0
                if inputs.get("input_ids") is not None:
                    inputs["input_ids"] = torch.full_like(inputs["input_ids"], pad_token_id)

            model_kwargs = {"labels": labels}
            if supports_zero_text:
                model_kwargs["zero_text"] = getattr(dataset, "zero_text_vector", False)
            if supports_no_text_branch:
                model_kwargs["no_text_branch"] = getattr(dataset, "no_text_branch", False)
            if supports_zero_image:
                model_kwargs["zero_image"] = getattr(dataset, "zero_image_vector", False)

            loss, t_outputs = model(inputs, model_batch, **model_kwargs)

            if t_targets_all is None:
                t_targets_all = labels
                t_outputs_all = t_outputs
            else:
                t_targets_all = torch.cat((t_targets_all, labels), dim=0)
                t_outputs_all = torch.cat((t_outputs_all, t_outputs), dim=0)

            del inputs, labels, t_outputs
            if i_batch % 10 == 0:
                gc.collect()
                if not use_cpu and torch.cuda.is_available():
                    torch.cuda.empty_cache()

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

    return {"predictions": predictions, "labels": labels}


def calculate_metrics(predictions, labels):
    if not predictions or not labels:
        return {
            "accuracy": 0,
            "correct": 0,
            "total": 0,
        }

    y_pred = np.array(predictions)
    y_true = np.array(labels)

    correct = np.sum(y_pred == y_true)
    total = len(y_true)
    accuracy = correct / total if total > 0 else 0

    return {
        "accuracy": accuracy,
        "correct": int(correct),
        "total": total,
    }


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def comp_max(A_full: float, A0txt: float, A0img: float) -> float:
    denom = max(A_full, 1e-12)
    return 100.0 * (A_full - max(A0txt, A0img)) / denom


def align_hard(A_full: float, AhI: float, AhT: float) -> float:
    A_hard = 0.5 * (AhI + AhT)
    denom = max(A_full, 1e-12)
    return 100.0 * (A_full - A_hard) / denom


def struct_score(A_full: float, A_RS: float, A_CT: float) -> float:
    A_struct = 0.5 * (A_RS + A_CT)
    denom = max(A_full, 1e-12)
    return 100.0 * (A_full - A_struct) / denom


def cler_soft(
    A_full: float,
    A_noimg: float,
    A_textmasked: float,
    A_randmasked: float,
    A_textonly: float,
    k: float = 0.02,
    lam: float = 0.05,
    eps: float = 1e-6,
) -> float:
    N = max(0.0, abs(A_randmasked - A_textmasked)) / max(A_full - A_textmasked, eps)
    S = _sigmoid((A_textonly - A_noimg) / k)
    V = (A_full - A_noimg)
    G = V / (V + lam) if (V + lam) > 0 else 0.0
    return 100.0 * N * S * G


def compute_all_diagnostics(acc: dict) -> dict:
    out = {}
    out["COMP_max"] = comp_max(acc["A_full"], acc["A0txt"], acc["A0img"])

    if "AhI" in acc and "AhT" in acc:
        out["ALIGN_hard"] = align_hard(acc["A_full"], acc["AhI"], acc["AhT"])

    out["STRUCT"] = struct_score(acc["A_full"], acc["A_RS"], acc["A_CT"])

    out["CLER_Soft"] = cler_soft(
        A_full=acc["A_full"],
        A_noimg=acc["A_noimg"],
        A_textmasked=acc["A_textmasked"],
        A_randmasked=acc["A_randmasked"],
        A_textonly=acc["A_textonly"],
        k=acc.get("k", 0.02),
        lam=acc.get("lam", 0.05),
        eps=acc.get("eps", 1e-6),
    )
    return out


def _supports_kwarg(callable_obj, name):
    try:
        sig = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return False
    for param in sig.parameters.values():
        if param.kind == inspect.Parameter.VAR_KEYWORD:
            return True
    return name in sig.parameters


def build_test_conditions(fixed_attention_len):
    return [
        {
            "name": "dataset_image_normal_text",
            "description": "1. dataset_image + normal text",
            "dataset_params": {"image_dir_index": 0, "zero_text": False},
        },
        {
            "name": "dataset_image_zero_text",
            "description": "2. dataset_image + zero-text vector",
            "dataset_params": {"image_dir_index": 0, "zero_text": True},
        },
        {
            "name": "dataset_image_zero_image",
            "description": "3. dataset_image + zero-image vector",
            "dataset_params": {"image_dir_index": 0, "zero_text": False},
        },
        {
            "name": "mmsd_masked_normal_text",
            "description": "4. MMSD_masked + normal text",
            "dataset_params": {"image_dir_index": 1, "zero_text": False},
        },
        {
            "name": "mmsd_textonly_normal_text",
            "description": "5. MMSD_textonly + normal text",
            "dataset_params": {"image_dir_index": 2, "zero_text": False},
        },
        {
            "name": "random_tokens_per_sample",
            "description": "6. random tokens (per sample)",
            "dataset_params": {"random_tokens_mode": "per_sample"},
        },
        {
            "name": "random_tokens_shared",
            "description": "7. random tokens (shared)",
            "dataset_params": {"random_tokens_mode": "shared"},
        },
        {
            "name": "rand_masked_normal_text",
            "description": "RandMasked: full random mask (area=TextMasked) + normal text",
            "dataset_params": {"rand_masked": True},
        },
        {
            "name": "hard_pairing",
            "description": "8. hard mismatch: image->text (most similar)",
            "dataset_params": {"hard_pairing": True},
        },
        {
            "name": "hard_image_pairing",
            "description": "9. hard mismatch: text->image (most similar)",
            "dataset_params": {"hard_image_pairing": True},
        },
    ]


DEFAULT_DIAG_MAP = {
    "A_full": "dataset_image_normal_text",
    "A0txt": "dataset_image_zero_text",
    "A0img": "dataset_image_zero_image",
    "AhI": "hard_pairing",
    "AhT": "hard_image_pairing",
    "A_RS": "random_tokens_per_sample",
    "A_CT": "random_tokens_shared",
    "A_noimg": "dataset_image_zero_image",
    "A_textmasked": "mmsd_masked_normal_text",
    "A_randmasked": "rand_masked_normal_text",
    "A_textonly": "mmsd_textonly_normal_text",
}


def load_diag_map(path: Optional[str]) -> Dict[str, str]:
    if not path:
        return DEFAULT_DIAG_MAP.copy()
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    out = DEFAULT_DIAG_MAP.copy()
    out.update(data)
    return out


def main():
    parser = argparse.ArgumentParser(description="Multi-condition diagnostics")
    parser.add_argument("--device", default="0", type=str, help="GPU id, use -1 for CPU")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="path to model checkpoint",
    )
    parser.add_argument(
        "--text_json",
        type=str,
        default="/home/user/chengtaiyu/RCLMuFN-main/data/text_clean/test.json",
        help="test JSON path",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="/home/user/chengtaiyu/RCLMuFN-main_copy/src/multi_condition_diagnostics.json",
        help="output file",
    )
    parser.add_argument("--batch_size", type=int, default=8, help="batch size")
    parser.add_argument("--fixed_attention_len", type=int, default=32, help="fixed attention len")
    parser.add_argument("--random_seed", type=int, default=1122, help="random seed")
    parser.add_argument("--force_cpu", action="store_true", help="force CPU")
    parser.add_argument("--test_subset", type=int, default=0, help="limit samples, 0 for all")

    parser.add_argument("--max_len", type=int, default=77, help="max text len")
    parser.add_argument("--text_size", default=512, type=int, help="text hidden size")
    parser.add_argument("--image_size", default=768, type=int, help="image hidden size")
    parser.add_argument("--dropout_rate", default=0.1, type=float, help="dropout rate")
    parser.add_argument("--label_number", type=int, default=2, help="num labels")
    parser.add_argument("--layers", default=3, type=int, help="transformer layers")
    parser.add_argument("--simple_linear", default=False, type=bool, help="simple linear")
    parser.add_argument(
        "--neg_sampling",
        default="label_aware",
        type=str,
        choices=["shuffle", "label_aware", "low_sim"],
        help="CID negative sampling",
    )
    parser.add_argument("--clip_path", type=str, default="", help="local CLIP model path")
    parser.add_argument("--base_data_dir", type=str, default="/home/user/chengtaiyu/Dataset/Dataset/MMSD2.0")
    parser.add_argument("--special_dir", type=str, default="/home/user/chengtaiyu/Dataset/Dataset")
    parser.add_argument("--diag_map", type=str, default="", help="JSON file mapping diagnostics to conditions")
    parser.add_argument("--diag_k", type=float, default=0.02, help="CLER-Soft k")
    parser.add_argument("--diag_lam", type=float, default=0.05, help="CLER-Soft lam")
    parser.add_argument("--diag_eps", type=float, default=1e-6, help="CLER-Soft eps")
    parser.add_argument("--clip_batch_size", type=int, default=32, help="CLIP batch size for hard mismatch")
    parser.add_argument("--rand_mask_diff_threshold", type=int, default=10, help="diff threshold for RandMasked")
    # Ablation switches (match training args)
    parser.add_argument("--disable_cid", action="store_true", help="remove CID module (keep DIMM with pseudo input)")
    parser.add_argument("--disable_dimm", action="store_true", help="remove DIMM module (keep CID)")
    parser.add_argument("--disable_pre_crossatt", action="store_true", help="remove pre-CID cross-attention")
    parser.add_argument("--disable_cid_dimm", action="store_true", help="remove CID + DIMM (use CLIP only)")
    parser.add_argument(
        "--dimm_drop_channel",
        default="none",
        type=str,
        choices=["none", "match", "mismatch", "conflict"],
        help="drop a DIMM channel for ablation",
    )
    parser.add_argument("--cid_random_mask", action="store_true", help="use random CID mask with matched counts")
    parser.add_argument("--cid_random_mask_seed", default=1122, type=int, help="random seed for CID random mask")
    parser.add_argument("--disable_cid_loss", action="store_true", help="force CID auxiliary losses to 0")
    parser.add_argument("--exp_name", default=None, type=str, help="experiment name (auto-generated if not set)")

    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    use_cpu = args.force_cpu or args.device == "-1"
    if use_cpu:
        device = torch.device("cpu")
        print("Force CPU")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() and int(args.device) >= 0 else "cpu")

    output_dir = os.path.dirname(os.path.abspath(args.output_file))
    if output_dir and output_dir != ".":
        os.makedirs(output_dir, exist_ok=True)

    clip_path = args.clip_path.strip()
    if not clip_path:
        for local_path in CLIP_LOCAL_PATHS:
            if os.path.exists(local_path):
                clip_path = local_path
                break
        if not clip_path:
            clip_path = "openai/clip-vit-base-patch32"

    try:
        processor = CLIPProcessor.from_pretrained(clip_path)
        print(f"Loaded CLIP processor from {clip_path}")
    except Exception as e:
        print(f"Failed to load CLIP processor: {e}")
        return

    try:
        model = ModelClass(args)
        if use_cpu:
            model.load_state_dict(torch.load(args.model_path, map_location="cpu"), strict=False)
        else:
            model.load_state_dict(torch.load(args.model_path, map_location=device), strict=False)
        model.to(device)
        print(f"Loaded model from {args.model_path}")
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    image_dirs = [
        os.path.join(args.base_data_dir, "dataset_image"),
        os.path.join(args.special_dir, "MMSD_masked"),
        os.path.join(args.special_dir, "MMSD_textonly"),
    ]

    for dir_path in image_dirs:
        if not os.path.exists(dir_path):
            print(f"Warning: image dir not found: {dir_path}")

    if not os.path.exists(args.text_json):
        print(f"Test JSON not found: {args.text_json}")
        return

    test_conditions = build_test_conditions(args.fixed_attention_len)
    all_metrics = {}

    for condition in test_conditions:
        print(f"\nRunning: {condition['description']}")

        zero_text_vector = condition["name"] == "dataset_image_zero_text"
        no_text_branch = False
        zero_image_vector = condition["name"] == "dataset_image_zero_image"

        if condition["dataset_params"].get("masked", False):
            dataset = MaskedImageDataset(args.text_json, image_dirs, processor, zero_text=False)
        elif condition["dataset_params"].get("rand_masked", False):
            dataset = RandomMaskDataset(
                args.text_json,
                [image_dirs[0]],
                processor,
                masked_dir=image_dirs[1],
                zero_text=False,
                mask_mode="mask",
                diff_threshold=args.rand_mask_diff_threshold,
            )
        elif condition["dataset_params"].get("hard_pairing", False):
            clip_device = device if not use_cpu else torch.device("cpu")
            dataset = HardMismatchPairingDataset(
                args.text_json,
                [image_dirs[0]],
                processor,
                zero_text=False,
                clip_device=clip_device,
                clip_batch_size=args.clip_batch_size,
                clip_path=clip_path,
            )
        elif condition["dataset_params"].get("hard_image_pairing", False):
            clip_device = device if not use_cpu else torch.device("cpu")
            dataset = HardMismatchImagePairingDataset(
                args.text_json,
                [image_dirs[0]],
                processor,
                zero_text=False,
                clip_device=clip_device,
                clip_batch_size=args.clip_batch_size,
                clip_path=clip_path,
            )
        elif condition["dataset_params"].get("random_pairing", False):
            dataset = RandomPairingDataset(args.text_json, [image_dirs[0]], processor, zero_text=False)
        elif condition["dataset_params"].get("random_image_pairing", False):
            dataset = RandomImagePairingDataset(args.text_json, [image_dirs[0]], processor, zero_text=False)
        else:
            image_dir_index = condition["dataset_params"].get("image_dir_index", 0)
            dataset = CustomTestDataset(
                args.text_json,
                [image_dirs[image_dir_index]],
                processor,
                zero_text=False if (zero_text_vector or no_text_branch) else condition["dataset_params"].get("zero_text", False),
            )

        dataset.random_tokens_mode = condition["dataset_params"].get("random_tokens_mode")
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

        if condition["name"] == "fixed_attention_mask":
            dataset.fixed_attention_len = args.fixed_attention_len
        else:
            dataset.fixed_attention_len = condition["dataset_params"].get("fixed_attention_len")

        if args.test_subset > 0 and args.test_subset < len(dataset):
            dataset.data = dataset.data[: args.test_subset]
            if isinstance(dataset, RandomPairingDataset):
                dataset.text_pool = [item["text"] for item in dataset.data]
                dataset.text_ids = [item["image_id"] for item in dataset.data]
                dataset.rng = random.Random(dataset.seed)
                dataset.perm = build_derangement(len(dataset.text_pool), dataset.rng)
            if isinstance(dataset, RandomImagePairingDataset):
                dataset.image_paths = [item["image_path"] for item in dataset.data]
                dataset.image_ids = [item["image_id"] for item in dataset.data]
                dataset.rng = random.Random(dataset.seed)
                dataset.perm = build_derangement(len(dataset.image_paths), dataset.rng)
            if isinstance(dataset, HardMismatchPairingDataset):
                cache_key = ("hard_pairs_subset", args.text_json, tuple(dataset.image_dirs), len(dataset.data))
                dataset.image_to_text, _ = get_hard_pair_indices(
                    dataset.data,
                    processor,
                    dataset.clip_device,
                    dataset.clip_batch_size,
                    cache_key,
                    clip_path=dataset.clip_path,
                )
            if isinstance(dataset, HardMismatchImagePairingDataset):
                cache_key = ("hard_pairs_subset", args.text_json, tuple(dataset.image_dirs), len(dataset.data))
                _, dataset.text_to_image = get_hard_pair_indices(
                    dataset.data,
                    processor,
                    dataset.clip_device,
                    dataset.clip_batch_size,
                    cache_key,
                    clip_path=dataset.clip_path,
                )

        try:
            results = run_test(args, model, dataset, processor, device, use_cpu=use_cpu)
            metrics = calculate_metrics(results["predictions"], results["labels"])
            all_metrics[condition["name"]] = metrics

            print(
                f"Accuracy: {metrics['accuracy']:.4f} ({metrics['correct']}/{metrics['total']})"
            )
        except Exception as e:
            print(f"Condition {condition['name']} failed: {e}")
            all_metrics[condition["name"]] = {
                "accuracy": 0,
                "correct": 0,
                "total": 0,
            }

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    acc_map = {name: metrics["accuracy"] for name, metrics in all_metrics.items()}

    diag_map = load_diag_map(args.diag_map)
    diag_inputs = {"k": args.diag_k, "lam": args.diag_lam, "eps": args.diag_eps}
    missing = []
    for key, cond in diag_map.items():
        if cond not in acc_map:
            missing.append((key, cond))
        else:
            diag_inputs[key] = acc_map[cond]

    diagnostics = None
    if missing:
        print("Missing diagnostics inputs:")
        for key, cond in missing:
            print(f"  - {key}: {cond} (not found)")
    else:
        diagnostics = compute_all_diagnostics(diag_inputs)
        print("\nDiagnostics:")
        for k, v in diagnostics.items():
            print(f"  {k}: {v:.4f}")

    output = {
        "metrics": all_metrics,
        "diagnostic_map": diag_map,
        "diagnostic_inputs": diag_inputs,
        "diagnostics": diagnostics,
    }

    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved results to {args.output_file}")


if __name__ == "__main__":
    main()
