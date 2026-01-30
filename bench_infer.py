import os
import time
import argparse

import torch
from torch.utils.data import DataLoader
from transformers import CLIPProcessor

from model import RCLMuFN
from data_set import MyDataset


class BackboneOnly(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        from transformers import CLIPModel
        self.model = CLIPModel.from_pretrained("/home/user/chengtaiyu/models/clip-vit-base-patch32")
        if args.simple_linear:
            self.text_linear = torch.nn.Linear(args.text_size, args.image_size)
            self.image_linear = torch.nn.Linear(args.image_size, args.image_size)
        else:
            self.text_linear = torch.nn.Sequential(
                torch.nn.Linear(args.text_size, args.image_size),
                torch.nn.Dropout(args.dropout_rate),
                torch.nn.GELU(),
            )
            self.image_linear = torch.nn.Sequential(
                torch.nn.Linear(args.image_size, args.image_size),
                torch.nn.Dropout(args.dropout_rate),
                torch.nn.GELU(),
            )
        self.post_ln = torch.nn.LayerNorm(args.image_size)
        self.classifier_fuse = torch.nn.Linear(args.image_size, args.label_number)
        self.loss_fct = torch.nn.CrossEntropyLoss()

    def forward(self, inputs, batch, labels=None):
        output = self.model(**inputs, output_attentions=False)
        text_feature = output["text_model_output"]["pooler_output"]
        image_feature = output["vision_model_output"]["pooler_output"]
        text_feature = self.text_linear(text_feature)
        image_feature = self.image_linear(image_feature)
        z_final = self.post_ln((text_feature + image_feature) / 2.0)
        logits_fuse = self.classifier_fuse(z_final)
        score = torch.softmax(logits_fuse, dim=-1)
        outputs = (score,)
        if labels is not None:
            loss_fuse = self.loss_fct(logits_fuse, labels)
            outputs = (loss_fuse,) + outputs
        return outputs


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def to_million(n):
    return n / 1e6


@torch.no_grad()
def run_infer_once(model, batch, processor, device, max_len):
    text_list, image_list, label_list, _ = batch
    inputs = processor(
        text=text_list,
        images=image_list,
        padding="max_length",
        truncation=True,
        max_length=max_len,
        return_tensors="pt",
    ).to(device, non_blocking=True)
    labels = torch.as_tensor(label_list).to(device, non_blocking=True)
    _loss, _logits = model(inputs, batch, labels=labels)
    return _loss, _logits


@torch.no_grad()
def run_backbone_once(model, batch, processor, device, max_len):
    text_list, image_list, label_list, _ = batch
    inputs = processor(
        text=text_list,
        images=image_list,
        padding="max_length",
        truncation=True,
        max_length=max_len,
        return_tensors="pt",
    ).to(device, non_blocking=True)
    labels = torch.as_tensor(label_list).to(device, non_blocking=True)
    output = model.model(**inputs, output_attentions=False)
    text_feature = output["text_model_output"]["pooler_output"]
    image_feature = output["vision_model_output"]["pooler_output"]
    text_feature = model.text_linear(text_feature)
    image_feature = model.image_linear(image_feature)
    z_final = model.post_ln((text_feature + image_feature) / 2.0)
    logits = model.classifier_fuse(z_final)
    score = torch.softmax(logits, dim=-1)
    loss = model.loss_fct(logits, labels)
    return loss, score


@torch.no_grad()
def run_cached_full(model, batch, inputs, labels):
    _loss, _logits = model(inputs, batch, labels=labels)
    return _loss, _logits


@torch.no_grad()
def run_cached_backbone(model, inputs, labels):
    output = model.model(**inputs, output_attentions=False)
    text_feature = output["text_model_output"]["pooler_output"]
    image_feature = output["vision_model_output"]["pooler_output"]
    text_feature = model.text_linear(text_feature)
    image_feature = model.image_linear(image_feature)
    z_final = model.post_ln((text_feature + image_feature) / 2.0)
    logits = model.classifier_fuse(z_final)
    score = torch.softmax(logits, dim=-1)
    loss = model.loss_fct(logits, labels)
    return loss, score


def _prepare_cached_batch(loader, processor, device, max_len):
    batch = next(iter(loader))
    text_list, image_list, label_list, _ = batch
    inputs = processor(
        text=text_list,
        images=image_list,
        padding="max_length",
        truncation=True,
        max_length=max_len,
        return_tensors="pt",
    ).to(device, non_blocking=True)
    labels = torch.as_tensor(label_list).to(device, non_blocking=True)
    return batch, inputs, labels


def benchmark_latency(model, loader, processor, device, max_len, warmup, iters, model_only, backbone_only):
    model.eval()
    times_ms = []

    if model_only:
        batch, inputs, labels = _prepare_cached_batch(loader, processor, device, max_len)
        for _ in range(warmup):
            if backbone_only:
                run_cached_backbone(model, inputs, labels)
            else:
                run_cached_full(model, batch, inputs, labels)
            if device.type == "cuda":
                torch.cuda.synchronize()

        for _ in range(iters):
            if device.type == "cuda":
                starter = torch.cuda.Event(enable_timing=True)
                ender = torch.cuda.Event(enable_timing=True)
                starter.record()
                if backbone_only:
                    run_cached_backbone(model, inputs, labels)
                else:
                    run_cached_full(model, batch, inputs, labels)
                ender.record()
                torch.cuda.synchronize()
                times_ms.append(starter.elapsed_time(ender))
            else:
                t0 = time.time()
                if backbone_only:
                    run_cached_backbone(model, inputs, labels)
                else:
                    run_cached_full(model, batch, inputs, labels)
                times_ms.append((time.time() - t0) * 1000.0)
    else:
        for i, batch in enumerate(loader):
            if i >= warmup:
                break
            if backbone_only:
                run_backbone_once(model, batch, processor, device, max_len)
            else:
                run_infer_once(model, batch, processor, device, max_len)
            if device.type == "cuda":
                torch.cuda.synchronize()

        it = 0
        for batch in loader:
            if it >= iters:
                break
            if device.type == "cuda":
                starter = torch.cuda.Event(enable_timing=True)
                ender = torch.cuda.Event(enable_timing=True)
                starter.record()
                if backbone_only:
                    run_backbone_once(model, batch, processor, device, max_len)
                else:
                    run_infer_once(model, batch, processor, device, max_len)
                ender.record()
                torch.cuda.synchronize()
                times_ms.append(starter.elapsed_time(ender))
            else:
                t0 = time.time()
                if backbone_only:
                    run_backbone_once(model, batch, processor, device, max_len)
                else:
                    run_infer_once(model, batch, processor, device, max_len)
                times_ms.append((time.time() - t0) * 1000.0)
            it += 1

    avg_ms = sum(times_ms) / len(times_ms)
    return avg_ms


def peak_memory_gb():
    if not torch.cuda.is_available():
        return None
    mem_bytes = torch.cuda.max_memory_allocated()
    return mem_bytes / (1024 ** 3)


def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="0", type=str)
    parser.add_argument("--model_path", required=True, type=str)
    parser.add_argument("--text_name", default="text_final", type=str)
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--max_len", default=77, type=int)
    parser.add_argument("--warmup", default=10, type=int)
    parser.add_argument("--iters", default=100, type=int)
    parser.add_argument("--model_only", action="store_true", help="benchmark model-only forward (no data/processor in loop)")
    parser.add_argument("--backbone_only", action="store_true", help="benchmark CLIP backbone baseline without CID-DIMM")
    parser.add_argument("--simple_linear", default=False, type=bool)
    parser.add_argument("--label_number", default=2, type=int)
    parser.add_argument("--text_size", default=512, type=int)
    parser.add_argument("--image_size", default=768, type=int)
    parser.add_argument("--layers", default=3, type=int)
    parser.add_argument("--num_heads", default=8, type=int)
    parser.add_argument("--dropout_rate", default=0.1, type=float)
    parser.add_argument("--neg_sampling", default="label_aware", type=str)
    parser.add_argument("--tau_schedule_mode", default="epoch", type=str)
    parser.add_argument("--tau_min", default=0.4, type=float)
    parser.add_argument("--tau_decay", default=0.9995, type=float)
    parser.add_argument("--num_workers", default=4, type=int)
    return parser.parse_args()


def _load_checkpoint(model, model_path):
    checkpoint = torch.load(model_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    for k in [
        "model.text_model.embeddings.position_ids",
        "model.vision_model.embeddings.position_ids",
    ]:
        if isinstance(checkpoint, dict) and k in checkpoint:
            del checkpoint[k]
    model.load_state_dict(checkpoint, strict=False)


def main():
    args = set_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    device = torch.device("cuda" if torch.cuda.is_available() and int(args.device) >= 0 else "cpu")

    processor = CLIPProcessor.from_pretrained("/home/user/chengtaiyu/models/clip-vit-base-patch32")
    if args.backbone_only:
        model = BackboneOnly(args)
    else:
        model = RCLMuFN(args)
    _load_checkpoint(model, args.model_path)
    model.to(device, non_blocking=True)
    model.eval()

    total, trainable = count_params(model)
    print("Total Params (M):", f"{to_million(total):.3f}")
    print("Trainable Params (M):", f"{to_million(trainable):.3f}")
    print("End-to-end fine-tune:", "Yes" if trainable == total else "No")

    dataset = MyDataset(mode="test", text_name=args.text_name, limit=None)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=MyDataset.collate_func,
    )

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    avg_ms = benchmark_latency(
        model, loader, processor, device,
        max_len=args.max_len,
        warmup=args.warmup,
        iters=args.iters,
        model_only=args.model_only,
        backbone_only=args.backbone_only,
    )
    latency_ms = avg_ms / args.batch_size
    throughput = (args.batch_size * 1000.0) / avg_ms

    print(f"Latency (ms/sample): {latency_ms:.3f}")
    print(f"Throughput (samples/s): {throughput:.3f}")

    if device.type == "cuda":
        peak_gb = peak_memory_gb()
        print(f"Peak GPU memory (GB): {peak_gb:.3f}")
    else:
        print("Peak GPU memory (GB): N/A (CPU)")


if __name__ == "__main__":
    main()
