import os
import sys
import torch
import json
import numpy as np
import inspect
from PIL import Image
from tqdm import tqdm
import argparse
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPProcessor
from sklearn.metrics import precision_score, recall_score, f1_score
import gc
import random

# 添加模型路径到系统路径
src_path = "/home/user/chengtaiyu/RCLMuFN-main_copy/src"
if src_path not in sys.path and os.path.exists(src_path):
    sys.path.append(src_path)
    print(f"添加路径: {src_path}")

# 导入模型
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


class CustomTestDataset(Dataset):
    """自定义数据集用于不同条件下的测试"""
    def __init__(self, text_json_path, image_dirs, processor, zero_text=False):
        """
        参数:
            text_json_path (str): 测试JSON文件的路径
            image_dirs (list): 要搜索图像的目录列表
            processor: CLIP处理器，用于分词
            zero_text (bool): 如果为True，将文本替换为空字符串
        """
        self.data = []
        self.image_dirs = image_dirs
        self.processor = processor
        self.zero_text = zero_text

        with open(text_json_path, 'r', encoding='utf-8') as f:
            test_data = json.load(f)

        for item in test_data:
            image_id = item['image_id']
            text = item['text']
            label = item['label']

            image_path = None
            for dir_path in image_dirs:
                potential_path = os.path.join(dir_path, f"{image_id}.jpg")
                if os.path.exists(potential_path):
                    image_path = potential_path
                    break

            if image_path:
                self.data.append({
                    'image_id': image_id,
                    'text': text,
                    'label': label,
                    'image_path': image_path
                })

        print(f"加载了 {len(self.data)} 个有效测试样本")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        try:
            image = Image.open(item['image_path']).convert('RGB')
        except Exception as e:
            print(f"加载图像 {item['image_path']} 时出错: {e}")
            image = Image.new('RGB', (224, 224), color='black')

        text = item['text']
        text_id = item['image_id']
        used_text_id = item['image_id']
        used_image_id = item['image_id']
        used_image_path = item['image_path']

        return text, image, item['label'], text_id, used_text_id, used_image_id, used_image_path

    @staticmethod
    def collate_func(batch_data):
        """
        参考predict.py中的collate_func实现
        """
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


class MaskedImageDataset(CustomTestDataset):
    """使用黑色像素掩盖图像的数据集"""
    def __init__(self, text_json_path, image_dirs, processor, zero_text=False, mask_type='full'):
        super().__init__(text_json_path, image_dirs, processor, zero_text)
        self.mask_type = mask_type

    def __getitem__(self, idx):
        item = self.data[idx]

        if self.mask_type == 'full':
            image = Image.new('RGB', (224, 224), color='black')
        else:
            try:
                image = Image.open(item['image_path']).convert('RGB')
            except Exception as e:
                print(f"加载图像 {item['image_path']} 时出错: {e}")
                image = Image.new('RGB', (224, 224), color='black')

        text = item['text']
        text_id = item['image_id']
        used_text_id = item['image_id']
        used_image_id = item['image_id']
        used_image_path = item['image_path']

        return text, image, item['label'], text_id, used_text_id, used_image_id, used_image_path


class RandomPairingDataset(CustomTestDataset):
    """随机配对图像和文本的数据集"""
    def __init__(self, text_json_path, image_dirs, processor, zero_text=False, seed=42):
        super().__init__(text_json_path, image_dirs, processor, zero_text)
        self.seed = seed
        self.rng = random.Random(seed)

        self.text_pool = [item['text'] for item in self.data]
        self.text_ids = [item['image_id'] for item in self.data]
        self.perm = build_derangement(len(self.text_pool), self.rng)

        print(f"创建了随机配对数据集，文本池大小: {len(self.text_pool)}")

    def __getitem__(self, idx):
        item = self.data[idx]

        try:
            image = Image.open(item['image_path']).convert('RGB')
        except Exception as e:
            print(f"加载图像 {item['image_path']} 时出错: {e}")
            image = Image.new('RGB', (224, 224), color='black')

        if len(self.text_pool) > 0:
            text = self.text_pool[self.perm[idx]]
            used_text_id = self.text_ids[self.perm[idx]]
        else:
            text = item['text']
            used_text_id = item['image_id']
        text_id = item['image_id']
        used_image_id = item['image_id']
        used_image_path = item['image_path']
        return text, image, item['label'], text_id, used_text_id, used_image_id, used_image_path


class RandomImagePairingDataset(CustomTestDataset):
    """随机配对文本和图像的数据集，但保持文本不变"""
    def __init__(self, text_json_path, image_dirs, processor, zero_text=False, seed=42):
        super().__init__(text_json_path, image_dirs, processor, zero_text)
        self.seed = seed
        self.rng = random.Random(seed)

        self.image_paths = [item['image_path'] for item in self.data]
        self.image_ids = [item['image_id'] for item in self.data]

        self.perm = build_derangement(len(self.image_paths), self.rng)

        print(f"创建了随机图像配对数据集，图像池大小: {len(self.image_paths)}")

    def __getitem__(self, idx):
        item = self.data[idx]

        if len(self.image_paths) > 0:
            perm_index = self.perm[idx]
            random_image_path = self.image_paths[perm_index]
            used_image_id = self.image_ids[perm_index]
        else:
            random_image_path = item['image_path']
            used_image_id = item['image_id']

        try:
            image = Image.open(random_image_path).convert('RGB')
        except Exception as e:
            print(f"加载图像 {random_image_path} 时出错: {e}")
            image = Image.new('RGB', (224, 224), color='black')

        text = item['text']
        text_id = item['image_id']
        used_text_id = item['image_id']
        used_image_path = random_image_path

        return text, image, item['label'], text_id, used_text_id, used_image_id, used_image_path


def run_test(args, model, dataset, processor, device, use_cpu=False):
    """在数据集上运行测试并返回预测结果"""
    batch_size = min(args.batch_size, 4) if not use_cpu else 1

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=CustomTestDataset.collate_func,
        shuffle=False
    )

    n_correct, n_total = 0, 0
    t_targets_all, t_outputs_all = None, None

    model.eval()
    data = []
    image = []
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
            padding='max_length',
            truncation=True,
            max_length=args.max_len,
            return_tensors="pt"
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
        for i_batch, t_batch in enumerate(tqdm(data_loader, desc="测试中")):
            text_list, image_list, label_list, id_list, used_text_id_list, used_image_id_list, used_image_path_list = t_batch
            effective_text_list = text_list

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

            inputs = prepare_inputs(effective_text_list, image_list)
            model_batch = (text_list, image_list, label_list, id_list)

            if use_cpu:
                inputs = {k: v.cpu() for k, v in inputs.items()}
                labels = torch.tensor(label_list).cpu()
            else:
                inputs = {k: v.to(device) for k, v in inputs.items()}
                labels = torch.tensor(label_list).to(device)

            t_targets = labels

            if getattr(dataset, "zero_text_vector", False) and not supports_zero_text:
                tokenizer = getattr(processor, "tokenizer", None)
                pad_token_id = getattr(tokenizer, "pad_token_id", 0) if tokenizer is not None else 0
                if inputs.get("input_ids") is not None:
                    inputs["input_ids"] = torch.full_like(inputs["input_ids"], pad_token_id)

            try:
                model_kwargs = {"labels": labels}
                if supports_zero_text:
                    model_kwargs["zero_text"] = getattr(dataset, "zero_text_vector", False)
                if supports_no_text_branch:
                    model_kwargs["no_text_branch"] = getattr(dataset, "no_text_branch", False)
                if supports_zero_image:
                    model_kwargs["zero_image"] = getattr(dataset, "zero_image_vector", False)

                loss, t_outputs = model(inputs, model_batch, **model_kwargs)

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
                    print(f"GPU内存不足，尝试在CPU上处理这个批次")
                    model = model.cpu()

                    inputs = prepare_inputs(text_list, image_list)
                    labels = torch.tensor(label_list)
                    t_targets = labels

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
        'predictions': predictions,
        'labels': labels,
        'ids': base_ids,
        'keys': keys,
        'texts': text,
        'used_text_ids': used_text_ids,
        'used_image_ids': used_image_ids,
        'used_image_paths': used_image_paths,
        'logits': logit
    }


def calculate_metrics(predictions, labels):
    if not predictions or not labels:
        return {
            'accuracy': 0,
            'precision': 0,
            'recall': 0,
            'f1': 0,
            'precision_macro': 0,
            'recall_macro': 0,
            'f1_macro': 0,
            'correct': 0,
            'total': 0
        }

    y_pred = np.array(predictions)
    y_true = np.array(labels)

    correct = np.sum(y_pred == y_true)
    total = len(y_true)
    accuracy = correct / total if total > 0 else 0

    try:
        precision = precision_score(y_true, y_pred, average='binary', zero_division=0)
        recall = recall_score(y_true, y_pred, average='binary', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='binary', zero_division=0)

        precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
        recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
        f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    except Exception as e:
        print(f"计算指标时出错: {e}")
        precision = recall = f1 = precision_macro = recall_macro = f1_macro = 0

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
        'correct': int(correct),
        'total': total
    }


def main():
    parser = argparse.ArgumentParser(description='多条件测试')
    parser.add_argument('--device', default='0', type=str, help='设备号，使用-1表示强制使用CPU')
    parser.add_argument('--model_path', type=str,
                        default="/home/user/chengtaiyu/RCLMuFN-main_copy/output_dir/856_pt/model.pt",
                        help='保存的模型路径')
    parser.add_argument('--text_json', type=str,
                        default="/home/user/chengtaiyu/RCLMuFN-main/data/text_final/test.json",
                        help='测试JSON文件的路径')
    parser.add_argument('--output_file', type=str,
                        default="/home/user/chengtaiyu/RCLMuFN-main_copy/src/multi_condition_results.json",
                        help='结果输出文件')
    parser.add_argument('--batch_size', type=int, default=8, help='测试的批次大小')
    parser.add_argument('--fixed_attention_len', type=int, default=32, help='固定attention_mask的长度')
    parser.add_argument('--random_seed', type=int, default=42, help='随机种子（控制随机token实验）')
    parser.add_argument('--force_cpu', action='store_true', help='强制在CPU上运行，即使有GPU可用')
    parser.add_argument('--test_subset', type=int, default=0, help='只测试指定数量的样本，0表示测试所有样本')

    parser.add_argument('--max_len', type=int, default=77, help='文本的最大长度')
    parser.add_argument('--text_size', default=512, type=int, help='文本隐藏大小')
    parser.add_argument('--image_size', default=768, type=int, help='图像隐藏大小')
    parser.add_argument('--dropout_rate', default=0.1, type=float, help='dropout率')
    parser.add_argument('--label_number', type=int, default=2, help='分类标签的数量')
    parser.add_argument('--layers', default=3, type=int, help='transformer的层数')
    parser.add_argument('--simple_linear', default=False, type=bool, help='线性实现选择')
    parser.add_argument('--neg_sampling', default='label_aware', type=str,
                        choices=['shuffle', 'label_aware', 'low_sim'],
                        help='CID负样本采样策略')

    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    use_cpu = args.force_cpu or args.device == '-1'
    if use_cpu:
        device = torch.device("cpu")
        print("强制使用CPU进行测试")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() and int(args.device) >= 0 else "cpu")

    print(f"使用设备: {device}")

    output_dir = os.path.dirname(os.path.abspath(args.output_file))
    if output_dir and output_dir != '.':
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
            'name': 'dataset_image_normal_text',
            'description': '1. 输入dataset_image中的图像和正常文本输入',
            'dataset_params': {
                'image_dir_index': 0,
                'zero_text': False
            }
        },
        {
            'name': 'dataset_image_zero_text',
            'description': '2. 输入dataset_image中的图像和文本置零向量（模型侧，保留attention_mask）',
            'dataset_params': {
                'image_dir_index': 0,
                'zero_text': True
            }
        },
        {
            'name': 'dataset_image_no_text_branch',
            'description': '3. 输入dataset_image中的图像，文本分支关闭（attention_mask置0）',
            'dataset_params': {
                'image_dir_index': 0,
                'zero_text': False
            }
        },
        {
            'name': 'dataset_image_zero_image',
            'description': '4. 输入dataset_image中的图像置零向量（模型侧）',
            'dataset_params': {
                'image_dir_index': 0,
                'zero_text': False
            }
        },
        {
            'name': 'mmsd_masked_normal_text',
            'description': '5. 输入MMSD_masked中的图像和正常文本输入',
            'dataset_params': {
                'image_dir_index': 1,
                'zero_text': False
            }
        },
        {
            'name': 'mmsd_masked_zero_text',
            'description': '6. 输入MMSD_masked中的图像和文本置零向量（模型侧，保留attention_mask）',
            'dataset_params': {
                'image_dir_index': 1,
                'zero_text': True
            }
        },
        {
            'name': 'mmsd_masked_no_text_branch',
            'description': '7. 输入MMSD_masked中的图像，文本分支关闭（attention_mask置0）',
            'dataset_params': {
                'image_dir_index': 1,
                'zero_text': False
            }
        },
        {
            'name': 'mmsd_masked_zero_image',
            'description': '8. 输入MMSD_masked中的图像置零向量（模型侧）',
            'dataset_params': {
                'image_dir_index': 1,
                'zero_text': False
            }
        },
        {
            'name': 'full_masked_normal_text',
            'description': '9. 输入mask掉整张图的图像和正常文本输入',
            'dataset_params': {
                'masked': True,
                'zero_text': False
            }
        },
        {
            'name': 'mmsd_textonly_normal_text',
            'description': '10. 输入MMSD_textonly中的图像和正常文本输入',
            'dataset_params': {
                'image_dir_index': 2,
                'zero_text': False
            }
        },
        {
            'name': 'mmsd_textonly_zero_text',
            'description': '11. 输入MMSD_textonly中的图像和文本置零向量（模型侧，保留attention_mask）',
            'dataset_params': {
                'image_dir_index': 2,
                'zero_text': True
            }
        },
        {
            'name': 'mmsd_textonly_no_text_branch',
            'description': '12. 输入MMSD_textonly中的图像，文本分支关闭（attention_mask置0）',
            'dataset_params': {
                'image_dir_index': 2,
                'zero_text': False
            }
        },
        {
            'name': 'mmsd_textonly_zero_image',
            'description': '13. 输入MMSD_textonly中的图像置零向量（模型侧）',
            'dataset_params': {
                'image_dir_index': 2,
                'zero_text': False
            }
        },
        {
            'name': 'random_pairing',
            'description': '14. 输入随机配对的图像和文本',
            'dataset_params': {
                'random_pairing': True
            }
        },
        {
            'name': 'random_image_pairing',
            'description': '15. 输入随机配对的文本和图像',
            'dataset_params': {
                'random_image_pairing': True
            }
        },
        {
            'name': 'random_tokens_per_sample',
            'description': '16. 随机token序列（保持attention_mask）',
            'dataset_params': {
                'random_tokens_mode': 'per_sample'
            }
        },
        {
            'name': 'random_tokens_shared',
            'description': '17. 同一随机token（保持attention_mask）',
            'dataset_params': {
                'random_tokens_mode': 'shared'
            }
        },
        {
            'name': 'fixed_attention_mask',
            'description': '18. 固定attention_mask模式',
            'dataset_params': {
                'fixed_attention_len': None
            }
        },
        {
            'name': 'fixed_attention_mask_4',
            'description': '19. 固定attention_mask模式 (len=4)',
            'dataset_params': {
                'fixed_attention_len': 4
            }
        },
        {
            'name': 'fixed_attention_mask_8',
            'description': '20. 固定attention_mask模式 (len=8)',
            'dataset_params': {
                'fixed_attention_len': 8
            }
        },
        {
            'name': 'fixed_attention_mask_16',
            'description': '21. 固定attention_mask模式 (len=16)',
            'dataset_params': {
                'fixed_attention_len': 16
            }
        },
        {
            'name': 'fixed_attention_mask_32',
            'description': '22. 固定attention_mask模式 (len=32)',
            'dataset_params': {
                'fixed_attention_len': 32
            }
        },
        {
            'name': 'fixed_attention_mask_64',
            'description': '23. 固定attention_mask模式 (len=64)',
            'dataset_params': {
                'fixed_attention_len': 64
            }
        }
    ]

    all_results = {}
    all_metrics = {}

    for condition in test_conditions:
        print(f"\n运行测试: {condition['description']}")

        zero_text_vector = condition['name'] in {
            'dataset_image_zero_text',
            'mmsd_masked_zero_text',
            'mmsd_textonly_zero_text',
        }
        no_text_branch = condition['name'] in {
            'dataset_image_no_text_branch',
            'mmsd_masked_no_text_branch',
            'mmsd_textonly_no_text_branch',
        }
        zero_image_vector = condition['name'] in {
            'dataset_image_zero_image',
            'mmsd_masked_zero_image',
            'mmsd_textonly_zero_image',
        }
        if condition['dataset_params'].get('masked', False):
            dataset = MaskedImageDataset(
                args.text_json,
                image_dirs,
                processor,
                zero_text=False
            )
        elif condition['dataset_params'].get('random_pairing', False):
            dataset = RandomPairingDataset(
                args.text_json,
                [image_dirs[0]],
                processor,
                zero_text=False
            )
        elif condition['dataset_params'].get('random_image_pairing', False):
            dataset = RandomImagePairingDataset(
                args.text_json,
                [image_dirs[0]],
                processor,
                zero_text=False
            )
        else:
            image_dir_index = condition['dataset_params'].get('image_dir_index', 0)
            dataset = CustomTestDataset(
                args.text_json,
                [image_dirs[image_dir_index]],
                processor,
                zero_text=False if (zero_text_vector or no_text_branch) else condition['dataset_params'].get('zero_text', False)
            )

        dataset.random_tokens_mode = condition['dataset_params'].get('random_tokens_mode')
        dataset.zero_text_vector = zero_text_vector
        dataset.no_text_branch = no_text_branch
        dataset.zero_image_vector = zero_image_vector
        dataset.random_seed = args.random_seed
        dataset.shared_random_token_id = None
        dataset.fill_token_id = None
        if dataset.random_tokens_mode == 'shared':
            generator = torch.Generator()
            generator.manual_seed(args.random_seed)
            dataset.shared_random_token_id = choose_random_token_id(getattr(processor, "tokenizer", None), generator)
            dataset.fill_token_id = choose_random_token_id(getattr(processor, "tokenizer", None), generator)
        if dataset.fill_token_id is None:
            generator = torch.Generator()
            generator.manual_seed(args.random_seed)
            dataset.fill_token_id = choose_random_token_id(getattr(processor, "tokenizer", None), generator)
        if condition['name'] == 'fixed_attention_mask':
            dataset.fixed_attention_len = args.fixed_attention_len
        else:
            dataset.fixed_attention_len = condition['dataset_params'].get('fixed_attention_len')

        if args.test_subset > 0 and args.test_subset < len(dataset):
            dataset.data = dataset.data[:args.test_subset]
            print(f"限制测试样本数量为 {args.test_subset}")
            if isinstance(dataset, RandomPairingDataset):
                dataset.text_pool = [item['text'] for item in dataset.data]
                dataset.text_ids = [item['image_id'] for item in dataset.data]
                dataset.rng = random.Random(dataset.seed)
                dataset.perm = build_derangement(len(dataset.text_pool), dataset.rng)
            if isinstance(dataset, RandomImagePairingDataset):
                dataset.image_paths = [item['image_path'] for item in dataset.data]
                dataset.image_ids = [item['image_id'] for item in dataset.data]
                dataset.rng = random.Random(dataset.seed)
                dataset.perm = build_derangement(len(dataset.image_paths), dataset.rng)

        try:
            results = run_test(args, model, dataset, processor, device, use_cpu=use_cpu)
            all_results[condition['name']] = results

            metrics = calculate_metrics(results['predictions'], results['labels'])
            all_metrics[condition['name']] = metrics

            print(f"测试结果 - {condition['description']}:")
            print(f"  - 准确率: {metrics['accuracy']:.4f} ({metrics['correct']}/{metrics['total']})")
            print(f"  - 二分类 - 精确率: {metrics['precision']:.4f}, 召回率: {metrics['recall']:.4f}, F1分数: {metrics['f1']:.4f}")
            print(f"  - 宏平均 - 精确率: {metrics['precision_macro']:.4f}, 召回率: {metrics['recall_macro']:.4f}, F1分数: {metrics['f1_macro']:.4f}")

        except Exception as e:
            print(f"测试 {condition['name']} 失败: {e}")
            all_results[condition['name']] = {'predictions': [], 'labels': [], 'ids': []}
            all_metrics[condition['name']] = {
                'accuracy': 0,
                'precision': 0,
                'recall': 0,
                'f1': 0,
                'precision_macro': 0,
                'recall_macro': 0,
                'f1_macro': 0,
                'correct': 0,
                'total': 0
            }

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    consolidated_results = {}

    all_ids = set()
    for condition_name, results in all_results.items():
        all_ids.update(results['ids'])

    id_to_index = {}
    for condition_name, results in all_results.items():
        id_to_index[condition_name] = {id_val: idx for idx, id_val in enumerate(results['ids'])}

    for sample_id in all_ids:
        consolidated_results[sample_id] = {
            'labels': {},
            'predictions': {},
            'used_text_id': {},
            'used_image_id': {},
            'used_image_path': {},
            'key': {}
        }

        for condition_name, results in all_results.items():
            if sample_id in id_to_index[condition_name]:
                idx = id_to_index[condition_name][sample_id]
                prediction = results['predictions'][idx]
                true_label = results['labels'][idx]

                consolidated_results[sample_id]['predictions'][condition_name] = prediction
                consolidated_results[sample_id]['labels'][condition_name] = true_label
                consolidated_results[sample_id]['used_text_id'][condition_name] = results['used_text_ids'][idx]
                consolidated_results[sample_id]['used_image_id'][condition_name] = results['used_image_ids'][idx]
                consolidated_results[sample_id]['used_image_path'][condition_name] = results['used_image_paths'][idx]
                consolidated_results[sample_id]['key'][condition_name] = results['keys'][idx]

    output = {
        'test_conditions': [c['description'] for c in test_conditions],
        'results': consolidated_results,
        'metrics': all_metrics
    }

    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2)

    print(f"\n结果已保存到 {args.output_file}")

    print("\n多条件测试统计信息:")
    for condition in test_conditions:
        condition_name = condition['name']
        metrics = all_metrics.get(condition_name, {})

        if metrics:
            correct = metrics['correct']
            total = metrics['total']
            accuracy = metrics['accuracy']
            precision = metrics['precision']
            recall = metrics['recall']
            f1 = metrics['f1']
            f1_macro = metrics['f1_macro']

            print(f"{condition['description']}:")
            print(f"  - 准确率: {accuracy:.4f} ({correct}/{total})")
            print(f"  - 二分类 - 召回率: {recall:.4f}, 精确率: {precision:.4f}, F1分数: {f1:.4f}")
            print(f"  - 宏平均 - F1分数: {f1_macro:.4f}")
        else:
            print(f"{condition['description']}: 没有有效的指标数据")


if __name__ == "__main__":
    main()
