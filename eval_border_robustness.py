"""
Border/Center Robustness Evaluation Script

Evaluates model performance under different border/center region transformations
to assess dependency on spatial layout cues.

Experiments:
    A) Border-only vs Center-only (k=0.10)
    B) Edge-mask / Crop curves (k ∈ {0, 0.02, 0.05, 0.08, 0.10, 0.15})

Output:
    - results/border_center_table.csv
    - results/mask_curve.csv
    - results/crop_curve.csv
    - results/mask_curve_macro_f1.png
    - results/crop_curve_macro_f1.png

Author: Auto-generated for CID-DIMM robustness evaluation
"""

import os
import sys
import argparse
import json
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import CLIPProcessor
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# Import project modules
from data_set import MyDataset
from model import RCLMuFN
from transforms_border import border_only, center_only, edge_mask, crop_resize


# ============================================================================
# Evaluation utilities
# ============================================================================

def compute_metrics(preds: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    """
    Compute classification metrics.

    Args:
        preds: predicted labels [N]
        labels: ground truth labels [N]

    Returns:
        Dictionary of metrics
    """
    acc = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average='macro', zero_division=0)
    macro_prec = precision_score(labels, preds, average='macro', zero_division=0)
    macro_rec = recall_score(labels, preds, average='macro', zero_division=0)

    return {
        'acc': acc,
        'macro_f1': macro_f1,
        'macro_precision': macro_prec,
        'macro_recall': macro_rec
    }


def predict_with_transform(
    model: nn.Module,
    dataloader: DataLoader,
    processor: CLIPProcessor,
    device: torch.device,
    transform_fn=None,
    transform_kwargs: Optional[Dict] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run model prediction on dataset with optional image transformation.

    Args:
        model: trained model
        dataloader: test data loader
        processor: CLIP processor for tokenization
        device: torch device
        transform_fn: optional transformation function (img -> img)
        transform_kwargs: optional kwargs for transform_fn

    Returns:
        (predictions, labels) as numpy arrays
    """
    model.eval()
    all_preds = []
    all_labels = []

    if transform_kwargs is None:
        transform_kwargs = {}

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            text_list, image_list, label_list, id_list = batch

            # Apply transformation if provided
            if transform_fn is not None:
                image_list = [transform_fn(img, **transform_kwargs) for img in image_list]

            # Process inputs
            inputs = processor(
                text=text_list,
                images=image_list,
                return_tensors="pt",
                padding=True,
                truncation=True
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            labels_tensor = torch.tensor(label_list, dtype=torch.long).to(device)

            # Forward pass
            outputs = model(inputs, batch, labels_tensor)
            # outputs = (score,) in eval mode
            scores = outputs[0]  # [B, num_classes]

            # Get predictions
            preds = scores.argmax(dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(label_list)

    return np.array(all_preds), np.array(all_labels)


# ============================================================================
# Experiment A: Border-only vs Center-only (fixed k=0.10)
# ============================================================================

def run_experiment_A(
    model: nn.Module,
    dataloader: DataLoader,
    processor: CLIPProcessor,
    device: torch.device,
    k: float = 0.10,
    fill_mode: str = "per_image_mean"
) -> pd.DataFrame:
    """
    Experiment A: Compare original, border_only, and center_only at fixed k.

    Returns:
        DataFrame with columns: [setting, k, acc, macro_f1, ...]
    """
    print(f"\n{'='*60}")
    print(f"Experiment A: Border vs Center (k={k})")
    print(f"{'='*60}")

    results = []

    # 1. Original (no transformation)
    print("\n[1/3] Evaluating: original")
    preds, labels = predict_with_transform(model, dataloader, processor, device)
    metrics = compute_metrics(preds, labels)
    results.append({
        'setting': 'original',
        'k': 0.0,
        **metrics
    })
    print(f"  ✅ Acc: {metrics['acc']:.4f}, Macro-F1: {metrics['macro_f1']:.4f}")

    # 2. Border-only
    print(f"\n[2/3] Evaluating: border_only (k={k})")
    preds, labels = predict_with_transform(
        model, dataloader, processor, device,
        transform_fn=border_only,
        transform_kwargs={'k': k, 'fill_mode': fill_mode}
    )
    metrics = compute_metrics(preds, labels)
    results.append({
        'setting': 'border_only',
        'k': k,
        **metrics
    })
    print(f"  ✅ Acc: {metrics['acc']:.4f}, Macro-F1: {metrics['macro_f1']:.4f}")

    # 3. Center-only
    print(f"\n[3/3] Evaluating: center_only (k={k})")
    preds, labels = predict_with_transform(
        model, dataloader, processor, device,
        transform_fn=center_only,
        transform_kwargs={'k': k, 'fill_mode': fill_mode}
    )
    metrics = compute_metrics(preds, labels)
    results.append({
        'setting': 'center_only',
        'k': k,
        **metrics
    })
    print(f"  ✅ Acc: {metrics['acc']:.4f}, Macro-F1: {metrics['macro_f1']:.4f}")

    # Compute deltas
    df = pd.DataFrame(results)
    acc_orig = df[df['setting'] == 'original']['acc'].values[0]
    f1_orig = df[df['setting'] == 'original']['macro_f1'].values[0]

    df['delta_acc'] = acc_orig - df['acc']
    df['delta_f1'] = f1_orig - df['macro_f1']

    return df


# ============================================================================
# Experiment B: Edge-mask / Crop curves (varying k)
# ============================================================================

def run_experiment_B(
    model: nn.Module,
    dataloader: DataLoader,
    processor: CLIPProcessor,
    device: torch.device,
    k_list: List[float],
    fill_mode: str = "per_image_mean"
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Experiment B: Evaluate performance curves for edge_mask and crop_resize.

    Returns:
        (mask_curve_df, crop_curve_df)
    """
    print(f"\n{'='*60}")
    print(f"Experiment B: Mask & Crop Curves")
    print(f"{'='*60}")

    mask_results = []
    crop_results = []

    for k in k_list:
        print(f"\n--- k = {k} ---")

        # 1. Edge-mask (same as center_only semantically)
        if k == 0:
            # k=0 means no masking (original)
            print(f"  [Mask] Evaluating: original (k=0)")
            preds, labels = predict_with_transform(model, dataloader, processor, device)
        else:
            print(f"  [Mask] Evaluating: edge_mask (k={k})")
            preds, labels = predict_with_transform(
                model, dataloader, processor, device,
                transform_fn=edge_mask,
                transform_kwargs={'k': k, 'fill_mode': fill_mode}
            )
        metrics = compute_metrics(preds, labels)
        mask_results.append({
            'transform': 'edge_mask',
            'k': k,
            **metrics
        })
        print(f"    ✅ Acc: {metrics['acc']:.4f}, Macro-F1: {metrics['macro_f1']:.4f}")

        # 2. Crop-resize
        if k == 0:
            # k=0 means no cropping (original)
            print(f"  [Crop] Evaluating: original (k=0)")
            preds, labels = predict_with_transform(model, dataloader, processor, device)
        else:
            print(f"  [Crop] Evaluating: crop_resize (k={k})")
            preds, labels = predict_with_transform(
                model, dataloader, processor, device,
                transform_fn=crop_resize,
                transform_kwargs={'k': k}
            )
        metrics = compute_metrics(preds, labels)
        crop_results.append({
            'transform': 'crop',
            'k': k,
            **metrics
        })
        print(f"    ✅ Acc: {metrics['acc']:.4f}, Macro-F1: {metrics['macro_f1']:.4f}")

    mask_df = pd.DataFrame(mask_results)
    crop_df = pd.DataFrame(crop_results)

    # Compute delta from k=0
    for df in [mask_df, crop_df]:
        baseline = df[df['k'] == 0]
        acc_0 = baseline['acc'].values[0]
        f1_0 = baseline['macro_f1'].values[0]
        df['delta_acc'] = acc_0 - df['acc']
        df['delta_f1'] = f1_0 - df['macro_f1']

    return mask_df, crop_df


# ============================================================================
# Plotting utilities
# ============================================================================

def plot_curve(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    hue_col: Optional[str],
    title: str,
    ylabel: str,
    save_path: str
):
    """
    Plot performance curve.

    Args:
        df: DataFrame with results
        x_col: column name for x-axis (e.g., 'k')
        y_col: column name for y-axis (e.g., 'macro_f1')
        hue_col: column name for grouping (e.g., 'model')
        title: plot title
        ylabel: y-axis label
        save_path: path to save figure
    """
    plt.figure(figsize=(8, 6))
    sns.set_style("whitegrid")

    if hue_col and hue_col in df.columns and df[hue_col].nunique() > 1:
        sns.lineplot(data=df, x=x_col, y=y_col, hue=hue_col, marker='o', linewidth=2)
        plt.legend(title=hue_col, loc='best')
    else:
        sns.lineplot(data=df, x=x_col, y=y_col, marker='o', linewidth=2, color='steelblue')

    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel(x_col, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  📊 Plot saved to: {save_path}")
    plt.close()


# ============================================================================
# Main evaluation pipeline
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Border/Center Robustness Evaluation")

    # Model and data
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--text_name', type=str, default='text_final',
                        help='Text data folder name')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Evaluation batch size')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device to use (e.g., cuda:0, cpu)')

    # Experiment settings
    parser.add_argument('--k_fixed', type=float, default=0.10,
                        help='Fixed k for Experiment A (border vs center)')
    parser.add_argument('--k_list', type=str, default='0,0.02,0.05,0.08,0.10,0.15',
                        help='Comma-separated k values for Experiment B')
    parser.add_argument('--fill_mode', type=str, default='per_image_mean',
                        choices=['per_image_mean', 'gray', 'imagenet_mean'],
                        help='Fill mode for masked regions')

    # Output
    parser.add_argument('--out_dir', type=str, default='../results',
                        help='Output directory for results')

    args = parser.parse_args()

    # Parse k_list
    k_list = [float(x.strip()) for x in args.k_list.split(',')]

    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() and 'cuda' in args.device else 'cpu')
    print(f"Using device: {device}")

    # Create output directory
    os.makedirs(args.out_dir, exist_ok=True)

    # ========================================================================
    # Load model
    # ========================================================================
    print("\n" + "="*60)
    print("Loading model...")
    print("="*60)

    # Create model with default args (we only need inference)
    class DummyArgs:
        simple_linear = False
        text_size = 512
        image_size = 768
        label_number = 2
        dropout_rate = 0.1
        neg_sampling = 'label_aware'
        tau_schedule_mode = 'epoch'
        use_dimm_adapter = False
        use_cf = False

    model_args = DummyArgs()
    model = RCLMuFN(model_args)

    # Load checkpoint
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    model.load_state_dict(checkpoint, strict=False)
    model.to(device)
    model.eval()

    print(f"✅ Model loaded from: {args.checkpoint}")

    # ========================================================================
    # Load data
    # ========================================================================
    print("\n" + "="*60)
    print("Loading test data...")
    print("="*60)

    test_dataset = MyDataset(mode='test', text_name=args.text_name)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=MyDataset.collate_func,
        num_workers=4
    )

    processor = CLIPProcessor.from_pretrained("/home/user/chengtaiyu/models/clip-vit-base-patch32")

    print(f"✅ Test set loaded: {len(test_dataset)} samples")

    # ========================================================================
    # Experiment A: Border vs Center (k=0.10)
    # ========================================================================
    df_A = run_experiment_A(
        model, test_loader, processor, device,
        k=args.k_fixed,
        fill_mode=args.fill_mode
    )

    # Save Experiment A results
    csv_A = os.path.join(args.out_dir, 'border_center_table.csv')
    df_A['model'] = 'CID-DIMM'  # Add model column
    df_A = df_A[['model', 'setting', 'k', 'acc', 'macro_f1', 'macro_precision', 'macro_recall', 'delta_acc', 'delta_f1']]
    df_A.to_csv(csv_A, index=False)
    print(f"\n✅ Experiment A results saved to: {csv_A}")
    print(df_A.to_string(index=False))

    # ========================================================================
    # Experiment B: Mask & Crop curves
    # ========================================================================
    mask_df, crop_df = run_experiment_B(
        model, test_loader, processor, device,
        k_list=k_list,
        fill_mode=args.fill_mode
    )

    # Add model column
    mask_df['model'] = 'CID-DIMM'
    crop_df['model'] = 'CID-DIMM'

    # Save Experiment B results
    csv_mask = os.path.join(args.out_dir, 'mask_curve.csv')
    csv_crop = os.path.join(args.out_dir, 'crop_curve.csv')

    mask_df = mask_df[['model', 'transform', 'k', 'acc', 'macro_f1', 'macro_precision', 'macro_recall', 'delta_acc', 'delta_f1']]
    crop_df = crop_df[['model', 'transform', 'k', 'acc', 'macro_f1', 'macro_precision', 'macro_recall', 'delta_acc', 'delta_f1']]

    mask_df.to_csv(csv_mask, index=False)
    crop_df.to_csv(csv_crop, index=False)

    print(f"\n✅ Mask curve results saved to: {csv_mask}")
    print(mask_df.to_string(index=False))

    print(f"\n✅ Crop curve results saved to: {csv_crop}")
    print(crop_df.to_string(index=False))

    # ========================================================================
    # Generate plots
    # ========================================================================
    print("\n" + "="*60)
    print("Generating plots...")
    print("="*60)

    # Plot mask curve
    plot_curve(
        mask_df,
        x_col='k',
        y_col='macro_f1',
        hue_col='model',
        title='Edge-Mask Performance Curve',
        ylabel='Macro F1',
        save_path=os.path.join(args.out_dir, 'mask_curve_macro_f1.png')
    )

    # Plot crop curve
    plot_curve(
        crop_df,
        x_col='k',
        y_col='macro_f1',
        hue_col='model',
        title='Crop-Resize Performance Curve',
        ylabel='Macro F1',
        save_path=os.path.join(args.out_dir, 'crop_curve_macro_f1.png')
    )

    print("\n" + "="*60)
    print("✅ Evaluation complete!")
    print("="*60)
    print(f"\nResults saved to: {args.out_dir}")
    print(f"  - border_center_table.csv")
    print(f"  - mask_curve.csv")
    print(f"  - crop_curve.csv")
    print(f"  - mask_curve_macro_f1.png")
    print(f"  - crop_curve_macro_f1.png")


if __name__ == '__main__':
    main()
