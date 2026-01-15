"""
Model Evaluation Script for RCL_TMD
Evaluates trained model checkpoints on test/validation sets
Generates detailed performance reports and confusion matrices
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn import metrics
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from transformers import CLIPProcessor
from model import RCLMuFN
from data_set import MyDataset
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def set_args():
    parser = argparse.ArgumentParser(description='Evaluate trained RCLMuFN model')

    # Model and data
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint (.pt file)')
    parser.add_argument('--text_name', default='text_clean', type=str,
                        help='Text data folder name (e.g., text_clean, text_final)')
    parser.add_argument('--device', default='0', type=str, help='GPU device number')

    # Model architecture (must match training config)
    parser.add_argument('--model', default='RCLMuFN', type=str)
    parser.add_argument('--simple_linear', default=False, type=bool)
    parser.add_argument('--text_size', default=512, type=int)
    parser.add_argument('--image_size', default=768, type=int)
    parser.add_argument('--max_len', default=77, type=int)
    parser.add_argument('--layers', default=3, type=int)
    parser.add_argument('--label_number', default=2, type=int)
    parser.add_argument('--dropout_rate', default=0.15, type=float)

    # Optimizer params (not used in eval but needed for model init)
    parser.add_argument('--optimizer_name', default='adam', type=str)
    parser.add_argument('--learning_rate', default=3e-4, type=float)
    parser.add_argument('--clip_learning_rate', default=5e-6, type=float)
    parser.add_argument('--weight_decay', default=0.05, type=float)
    parser.add_argument('--warmup_proportion', default=0.15, type=float)
    parser.add_argument('--max_grad_norm', default=3.0, type=float)
    parser.add_argument('--adam_epsilon', default=1e-8, type=float)

    # Lambda parameters (not used in eval but needed for model init)
    parser.add_argument('--lambda_ratio_start', default=0.0, type=float)
    parser.add_argument('--lambda_ratio_end', default=5e-3, type=float)
    parser.add_argument('--lambda_itm_start', default=0.0, type=float)
    parser.add_argument('--lambda_itm_end', default=3e-3, type=float)
    parser.add_argument('--lambda_warmup_epochs', default=3, type=int)
    parser.add_argument('--lambda_ramp_epochs', default=5, type=int)
    parser.add_argument('--lambda_schedule', default='linear', type=str,
                        choices=['none', 'linear', 'cosine'])

    # CID parameters (must match training config)
    parser.add_argument('--neg_sampling', default='label_aware', type=str,
                        choices=['shuffle', 'label_aware', 'low_sim'])
    parser.add_argument('--tau_schedule_mode', default='epoch', type=str,
                        choices=['step', 'epoch'])

    # Evaluation settings
    parser.add_argument('--batch_size', default=32, type=int,
                        help='Batch size for evaluation')
    parser.add_argument('--splits', default='test,valid', type=str,
                        help='Comma-separated list of splits to evaluate (test,valid,train)')
    parser.add_argument('--output_dir', default='./eval_results/', type=str,
                        help='Directory to save evaluation results')
    parser.add_argument('--save_predictions', action='store_true',
                        help='Save predictions to file')

    return parser.parse_args()


def evaluate_model(model, data_loader, processor, device, split_name='test'):
    """
    Evaluate model on a dataset split

    Returns:
        dict: Evaluation metrics including accuracy, F1, precision, recall
    """
    model.eval()

    all_predictions = []
    all_labels = []
    all_probabilities = []

    with torch.no_grad():
        for batch in data_loader:
            text_list, image_list, label_list, id_list = batch

            # Process inputs
            inputs = processor(
                text=text_list,
                images=image_list,
                padding='max_length',
                truncation=True,
                max_length=77,
                return_tensors="pt"
            ).to(device)

            labels = torch.tensor(label_list).to(device)

            # Forward pass
            _, logits = model(inputs, batch, labels=labels)
            probabilities = torch.softmax(logits, dim=-1)
            predictions = torch.argmax(logits, dim=-1)

            # Collect results
            all_predictions.extend(predictions.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())
            all_probabilities.extend(probabilities.cpu().numpy().tolist())

    # Convert to numpy arrays
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_probabilities = np.array(all_probabilities)

    # Calculate metrics
    accuracy = metrics.accuracy_score(all_labels, all_predictions)

    # Micro metrics (overall)
    micro_precision = metrics.precision_score(all_labels, all_predictions, average='micro')
    micro_recall = metrics.recall_score(all_labels, all_predictions, average='micro')
    micro_f1 = metrics.f1_score(all_labels, all_predictions, average='micro')

    # Macro metrics (per-class average)
    macro_precision = metrics.precision_score(all_labels, all_predictions, average='macro')
    macro_recall = metrics.recall_score(all_labels, all_predictions, average='macro')
    macro_f1 = metrics.f1_score(all_labels, all_predictions, average='macro')

    # Per-class metrics
    per_class_precision = metrics.precision_score(all_labels, all_predictions, average=None)
    per_class_recall = metrics.recall_score(all_labels, all_predictions, average=None)
    per_class_f1 = metrics.f1_score(all_labels, all_predictions, average=None)

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)

    # Classification report
    class_names = ['Non-Hate', 'Hate']
    report = classification_report(
        all_labels, all_predictions,
        target_names=class_names,
        output_dict=True
    )

    results = {
        'split': split_name,
        'num_samples': len(all_labels),
        'accuracy': float(accuracy),
        'micro': {
            'precision': float(micro_precision),
            'recall': float(micro_recall),
            'f1': float(micro_f1)
        },
        'macro': {
            'precision': float(macro_precision),
            'recall': float(macro_recall),
            'f1': float(macro_f1)
        },
        'per_class': {
            'Non-Hate': {
                'precision': float(per_class_precision[0]),
                'recall': float(per_class_recall[0]),
                'f1': float(per_class_f1[0]),
                'support': int(report['Non-Hate']['support'])
            },
            'Hate': {
                'precision': float(per_class_precision[1]),
                'recall': float(per_class_recall[1]),
                'f1': float(per_class_f1[1]),
                'support': int(report['Hate']['support'])
            }
        },
        'confusion_matrix': cm.tolist(),
        'predictions': all_predictions.tolist() if args.save_predictions else None,
        'labels': all_labels.tolist() if args.save_predictions else None,
        'probabilities': all_probabilities.tolist() if args.save_predictions else None
    }

    return results


def plot_confusion_matrix(cm, class_names, output_path, title='Confusion Matrix'):
    """
    Plot and save confusion matrix
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Count'}
    )
    plt.title(title, fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Confusion matrix saved to: {output_path}")


def print_results_table(results):
    """
    Print formatted results table
    """
    print("\n" + "="*70)
    print(f"  {results['split'].upper()} SET RESULTS ({results['num_samples']} samples)")
    print("="*70)

    print(f"\n{'Metric':<20} {'Value':>15}")
    print("-" * 37)
    print(f"{'Accuracy':<20} {results['accuracy']:>14.4f}")

    print(f"\n{'MICRO AVERAGE':<20}")
    print("-" * 37)
    print(f"{'  Precision':<20} {results['micro']['precision']:>14.4f}")
    print(f"{'  Recall':<20} {results['micro']['recall']:>14.4f}")
    print(f"{'  F1-Score':<20} {results['micro']['f1']:>14.4f}")

    print(f"\n{'MACRO AVERAGE':<20}")
    print("-" * 37)
    print(f"{'  Precision':<20} {results['macro']['precision']:>14.4f}")
    print(f"{'  Recall':<20} {results['macro']['recall']:>14.4f}")
    print(f"{'  F1-Score':<20} {results['macro']['f1']:>14.4f}")

    print(f"\n{'PER-CLASS METRICS':<20}")
    print("-" * 70)
    print(f"{'Class':<15} {'Precision':>12} {'Recall':>12} {'F1-Score':>12} {'Support':>12}")
    print("-" * 70)

    for class_name, metrics in results['per_class'].items():
        print(f"{class_name:<15} {metrics['precision']:>12.4f} {metrics['recall']:>12.4f} "
              f"{metrics['f1']:>12.4f} {metrics['support']:>12}")

    print("\n" + "="*70 + "\n")


def main():
    global args
    args = set_args()

    # Check if checkpoint exists
    if not os.path.exists(args.checkpoint):
        print(f"❌ Error: Checkpoint not found at {args.checkpoint}")
        sys.exit(1)

    # Setup device
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() and int(args.device) >= 0 else "cpu")
    print(f"\n{'='*70}")
    print(f"  RCL_TMD Model Evaluation")
    print(f"{'='*70}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Dataset: {args.text_name}")
    print(f"Device: {device}")
    print(f"Splits: {args.splits}")
    print(f"{'='*70}\n")

    # Load CLIP processor
    clip_path = "/home/user/2024_cty/RCL/src/models/clip-vit-base-patch32"
    processor = CLIPProcessor.from_pretrained(clip_path)
    print(f"✓ Loaded CLIP processor from {clip_path}")

    # Initialize model
    model = RCLMuFN(args)

    # Load checkpoint
    print(f"✓ Loading checkpoint...")
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    model.load_state_dict(checkpoint)
    model.to(device)
    print(f"✓ Model loaded successfully")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Generate unique result ID
    checkpoint_name = os.path.basename(os.path.dirname(args.checkpoint))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_id = f"{checkpoint_name}_{args.text_name}_{timestamp}"

    # Evaluate on specified splits
    splits_to_eval = [s.strip() for s in args.splits.split(',')]
    all_results = {
        'checkpoint': args.checkpoint,
        'dataset': args.text_name,
        'timestamp': timestamp,
        'splits': {}
    }

    class_names = ['Non-Hate', 'Hate']

    for split in splits_to_eval:
        print(f"\n{'='*70}")
        print(f"  Evaluating on {split.upper()} set...")
        print(f"{'='*70}")

        # Load dataset
        try:
            dataset = MyDataset(mode=split, text_name=args.text_name, limit=None)
            data_loader = DataLoader(
                dataset,
                batch_size=args.batch_size,
                collate_fn=MyDataset.collate_func,
                shuffle=False,
                num_workers=4,
                pin_memory=True
            )
            print(f"✓ Loaded {split} dataset: {len(dataset)} samples")
        except Exception as e:
            print(f"❌ Error loading {split} dataset: {e}")
            continue

        # Evaluate
        results = evaluate_model(model, data_loader, processor, device, split)
        all_results['splits'][split] = results

        # Print results
        print_results_table(results)

        # Plot confusion matrix
        cm_path = os.path.join(args.output_dir, f"{result_id}_{split}_confusion_matrix.png")
        plot_confusion_matrix(
            np.array(results['confusion_matrix']),
            class_names,
            cm_path,
            title=f'Confusion Matrix - {split.upper()} Set'
        )

    # Save results to JSON
    json_path = os.path.join(args.output_dir, f"{result_id}_results.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n✓ Results saved to: {json_path}")

    # Generate summary report
    report_path = os.path.join(args.output_dir, f"{result_id}_report.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("  RCL_TMD MODEL EVALUATION REPORT\n")
        f.write("="*70 + "\n\n")
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"Dataset: {args.text_name}\n")
        f.write(f"Evaluation Time: {timestamp}\n")
        f.write(f"Device: {device}\n\n")

        for split, results in all_results['splits'].items():
            f.write(f"\n{'='*70}\n")
            f.write(f"  {split.upper()} SET ({results['num_samples']} samples)\n")
            f.write(f"{'='*70}\n\n")

            f.write(f"Overall Accuracy: {results['accuracy']:.4f}\n\n")

            f.write("Micro Average:\n")
            f.write(f"  Precision: {results['micro']['precision']:.4f}\n")
            f.write(f"  Recall:    {results['micro']['recall']:.4f}\n")
            f.write(f"  F1-Score:  {results['micro']['f1']:.4f}\n\n")

            f.write("Macro Average:\n")
            f.write(f"  Precision: {results['macro']['precision']:.4f}\n")
            f.write(f"  Recall:    {results['macro']['recall']:.4f}\n")
            f.write(f"  F1-Score:  {results['macro']['f1']:.4f}\n\n")

            f.write("Per-Class Metrics:\n")
            f.write("-" * 70 + "\n")
            f.write(f"{'Class':<15} {'Precision':>12} {'Recall':>12} {'F1-Score':>12} {'Support':>12}\n")
            f.write("-" * 70 + "\n")
            for class_name, metrics in results['per_class'].items():
                f.write(f"{class_name:<15} {metrics['precision']:>12.4f} {metrics['recall']:>12.4f} "
                       f"{metrics['f1']:>12.4f} {metrics['support']:>12}\n")

            f.write(f"\nConfusion Matrix:\n")
            cm = np.array(results['confusion_matrix'])
            f.write(f"                Predicted\n")
            f.write(f"              Non-Hate  Hate\n")
            f.write(f"Actual Non-Hate  {cm[0][0]:>6}  {cm[0][1]:>6}\n")
            f.write(f"       Hate      {cm[1][0]:>6}  {cm[1][1]:>6}\n")

    print(f"✓ Report saved to: {report_path}")

    print(f"\n{'='*70}")
    print("  Evaluation Complete!")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
