"""
Compare and analyze evaluation results across multiple experiments
Generates comparative tables and visualizations
"""

import os
import json
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob
from datetime import datetime


def load_all_results(eval_dir):
    """
    Load all evaluation results from JSON files
    """
    json_files = glob(os.path.join(eval_dir, '*_results.json'))

    if not json_files:
        print(f"❌ No evaluation results found in {eval_dir}")
        return []

    results = []
    for json_file in json_files:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # Extract experiment name from checkpoint path
            checkpoint = data['checkpoint']
            exp_name = os.path.basename(os.path.dirname(os.path.dirname(checkpoint)))
            data['experiment'] = exp_name
            data['result_file'] = json_file
            results.append(data)

    return results


def create_comparison_table(results, split='test'):
    """
    Create comparison table for a specific split
    """
    rows = []

    for result in results:
        if split not in result['splits']:
            continue

        split_data = result['splits'][split]
        exp_name = result['experiment']

        row = {
            'Experiment': exp_name,
            'Accuracy': split_data['accuracy'],
            'Macro F1': split_data['macro']['f1'],
            'Macro Precision': split_data['macro']['precision'],
            'Macro Recall': split_data['macro']['recall'],
            'Micro F1': split_data['micro']['f1'],
            'Hate F1': split_data['per_class']['Hate']['f1'],
            'Hate Recall': split_data['per_class']['Hate']['recall'],
            'Hate Precision': split_data['per_class']['Hate']['precision'],
            'Non-Hate F1': split_data['per_class']['Non-Hate']['f1'],
            'Samples': split_data['num_samples']
        }
        rows.append(row)

    df = pd.DataFrame(rows)

    # Sort by Macro F1 descending
    df = df.sort_values('Macro F1', ascending=False)

    return df


def print_comparison_table(df, split='test'):
    """
    Print formatted comparison table
    """
    print(f"\n{'='*120}")
    print(f"  EXPERIMENT COMPARISON - {split.upper()} SET")
    print(f"{'='*120}\n")

    # Main metrics
    print("OVERALL METRICS:")
    print("-" * 120)
    cols = ['Experiment', 'Accuracy', 'Macro F1', 'Macro Precision', 'Macro Recall', 'Micro F1']
    print(df[cols].to_string(index=False, float_format=lambda x: f'{x:.4f}'))

    print(f"\n\nPER-CLASS METRICS (HATE):")
    print("-" * 120)
    cols = ['Experiment', 'Hate F1', 'Hate Precision', 'Hate Recall']
    print(df[cols].to_string(index=False, float_format=lambda x: f'{x:.4f}'))

    print(f"\n\nPER-CLASS METRICS (NON-HATE):")
    print("-" * 120)
    cols = ['Experiment', 'Non-Hate F1']
    print(df[cols].to_string(index=False, float_format=lambda x: f'{x:.4f}'))

    print(f"\n{'='*120}\n")

    # Best models
    print("BEST MODELS:")
    print("-" * 60)
    print(f"  Best Macro F1:       {df.iloc[0]['Experiment']} ({df.iloc[0]['Macro F1']:.4f})")
    print(f"  Best Accuracy:       {df.loc[df['Accuracy'].idxmax()]['Experiment']} "
          f"({df['Accuracy'].max():.4f})")
    print(f"  Best Hate Recall:    {df.loc[df['Hate Recall'].idxmax()]['Experiment']} "
          f"({df['Hate Recall'].max():.4f})")
    print(f"  Best Hate F1:        {df.loc[df['Hate F1'].idxmax()]['Experiment']} "
          f"({df['Hate F1'].max():.4f})")
    print("-" * 60 + "\n")


def plot_comparison_charts(df, split, output_dir):
    """
    Create comparison visualizations
    """
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (14, 10)

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Model Performance Comparison - {split.upper()} Set',
                 fontsize=16, fontweight='bold')

    # Sort by Macro F1 for consistent ordering
    df_sorted = df.sort_values('Macro F1', ascending=True)

    # 1. Macro metrics comparison
    ax1 = axes[0, 0]
    metrics = ['Macro F1', 'Macro Precision', 'Macro Recall']
    x = range(len(df_sorted))
    width = 0.25

    for i, metric in enumerate(metrics):
        ax1.barh([xi + i * width for xi in x], df_sorted[metric],
                width, label=metric, alpha=0.8)

    ax1.set_yticks([xi + width for xi in x])
    ax1.set_yticklabels(df_sorted['Experiment'], fontsize=9)
    ax1.set_xlabel('Score', fontsize=11)
    ax1.set_title('Macro Average Metrics', fontsize=12, fontweight='bold')
    ax1.legend(loc='lower right')
    ax1.set_xlim([0, 1])
    ax1.grid(axis='x', alpha=0.3)

    # 2. Accuracy and Micro F1
    ax2 = axes[0, 1]
    metrics = ['Accuracy', 'Micro F1']
    x = range(len(df_sorted))
    width = 0.35

    for i, metric in enumerate(metrics):
        ax2.barh([xi + i * width for xi in x], df_sorted[metric],
                width, label=metric, alpha=0.8)

    ax2.set_yticks([xi + width / 2 for xi in x])
    ax2.set_yticklabels(df_sorted['Experiment'], fontsize=9)
    ax2.set_xlabel('Score', fontsize=11)
    ax2.set_title('Overall Performance', fontsize=12, fontweight='bold')
    ax2.legend(loc='lower right')
    ax2.set_xlim([0, 1])
    ax2.grid(axis='x', alpha=0.3)

    # 3. Per-class F1 comparison
    ax3 = axes[1, 0]
    metrics = ['Hate F1', 'Non-Hate F1']
    x = range(len(df_sorted))
    width = 0.35

    colors = ['#e74c3c', '#3498db']
    for i, metric in enumerate(metrics):
        ax3.barh([xi + i * width for xi in x], df_sorted[metric],
                width, label=metric, alpha=0.8, color=colors[i])

    ax3.set_yticks([xi + width / 2 for xi in x])
    ax3.set_yticklabels(df_sorted['Experiment'], fontsize=9)
    ax3.set_xlabel('F1-Score', fontsize=11)
    ax3.set_title('Per-Class F1-Score', fontsize=12, fontweight='bold')
    ax3.legend(loc='lower right')
    ax3.set_xlim([0, 1])
    ax3.grid(axis='x', alpha=0.3)

    # 4. Hate class detailed metrics
    ax4 = axes[1, 1]
    metrics = ['Hate Precision', 'Hate Recall', 'Hate F1']
    x = range(len(df_sorted))
    width = 0.25

    colors = ['#e67e22', '#9b59b6', '#e74c3c']
    for i, metric in enumerate(metrics):
        ax4.barh([xi + i * width for xi in x], df_sorted[metric],
                width, label=metric, alpha=0.8, color=colors[i])

    ax4.set_yticks([xi + width for xi in x])
    ax4.set_yticklabels(df_sorted['Experiment'], fontsize=9)
    ax4.set_xlabel('Score', fontsize=11)
    ax4.set_title('Hate Class Metrics (Important for Imbalanced Data)',
                 fontsize=12, fontweight='bold')
    ax4.legend(loc='lower right')
    ax4.set_xlim([0, 1])
    ax4.grid(axis='x', alpha=0.3)

    plt.tight_layout()

    # Save figure
    output_path = os.path.join(output_dir, f'comparison_{split}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Comparison chart saved: {output_path}")


def save_comparison_report(results, output_dir):
    """
    Save comprehensive comparison report
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(output_dir, f'comparison_report_{timestamp}.txt')

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*100 + "\n")
        f.write("  COMPREHENSIVE MODEL COMPARISON REPORT\n")
        f.write("="*100 + "\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Experiments: {len(results)}\n\n")

        for split in ['test', 'valid']:
            df = create_comparison_table(results, split)
            if df.empty:
                continue

            f.write(f"\n{'='*100}\n")
            f.write(f"  {split.upper()} SET COMPARISON\n")
            f.write(f"{'='*100}\n\n")

            # Overall metrics
            f.write("OVERALL METRICS:\n")
            f.write("-" * 100 + "\n")
            cols = ['Experiment', 'Accuracy', 'Macro F1', 'Macro Precision', 'Macro Recall']
            f.write(df[cols].to_string(index=False, float_format=lambda x: f'{x:.4f}'))
            f.write("\n\n")

            # Per-class metrics
            f.write("PER-CLASS METRICS:\n")
            f.write("-" * 100 + "\n")
            cols = ['Experiment', 'Hate F1', 'Hate Precision', 'Hate Recall',
                   'Non-Hate F1']
            f.write(df[cols].to_string(index=False, float_format=lambda x: f'{x:.4f}'))
            f.write("\n\n")

            # Best models
            f.write("BEST MODELS:\n")
            f.write("-" * 100 + "\n")
            f.write(f"  Best Macro F1:       {df.iloc[0]['Experiment']} ({df.iloc[0]['Macro F1']:.4f})\n")
            f.write(f"  Best Accuracy:       {df.loc[df['Accuracy'].idxmax()]['Experiment']} "
                   f"({df['Accuracy'].max():.4f})\n")
            f.write(f"  Best Hate Recall:    {df.loc[df['Hate Recall'].idxmax()]['Experiment']} "
                   f"({df['Hate Recall'].max():.4f})\n")
            f.write(f"  Best Hate F1:        {df.loc[df['Hate F1'].idxmax()]['Experiment']} "
                   f"({df['Hate F1'].max():.4f})\n")
            f.write("-" * 100 + "\n\n")

    print(f"  ✓ Comparison report saved: {report_path}")

    # Also save CSV
    for split in ['test', 'valid']:
        df = create_comparison_table(results, split)
        if not df.empty:
            csv_path = os.path.join(output_dir, f'comparison_{split}_{timestamp}.csv')
            df.to_csv(csv_path, index=False, float_format='%.4f')
            print(f"  ✓ CSV saved: {csv_path}")


def main():
    parser = argparse.ArgumentParser(description='Compare evaluation results')
    parser.add_argument('--eval_dir', default='./eval_results', type=str,
                       help='Directory containing evaluation results')
    parser.add_argument('--output_dir', default=None, type=str,
                       help='Output directory (default: same as eval_dir)')
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = args.eval_dir

    print("\n" + "="*100)
    print("  MODEL COMPARISON ANALYSIS")
    print("="*100 + "\n")

    # Load all results
    print(f"Loading results from: {args.eval_dir}")
    results = load_all_results(args.eval_dir)

    if not results:
        print("❌ No results found!")
        return

    print(f"✓ Loaded {len(results)} evaluation results\n")

    # List experiments
    print("Experiments found:")
    for i, result in enumerate(results, 1):
        print(f"  {i}. {result['experiment']}")
    print("")

    # Create comparison tables and charts for each split
    for split in ['test', 'valid']:
        df = create_comparison_table(results, split)
        if df.empty:
            print(f"⊗ No {split} results found")
            continue

        print_comparison_table(df, split)
        plot_comparison_charts(df, split, args.output_dir)

    # Save comprehensive report
    save_comparison_report(results, args.output_dir)

    print("\n" + "="*100)
    print("  ANALYSIS COMPLETE!")
    print("="*100 + "\n")
    print(f"Results saved to: {args.output_dir}")
    print("  - comparison_report_*.txt")
    print("  - comparison_test_*.csv")
    print("  - comparison_valid_*.csv")
    print("  - comparison_test.png")
    print("  - comparison_valid.png")
    print("")


if __name__ == '__main__':
    main()
