#!/usr/bin/env python3
"""
阶段1结果分析脚本
自动分析架构变体实验结果，推荐最佳配置
"""

import json
import os
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import sys

# ANSI颜色代码
class Colors:
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    RED = '\033[0;31m'
    BLUE = '\033[0;34m'
    CYAN = '\033[0;36m'
    BOLD = '\033[1m'
    NC = '\033[0m'

def load_result(variant: str, seed: int, results_dir: str) -> Dict:
    """加载单个实验结果"""
    result_file = Path(results_dir) / f"{variant}_seed{seed}.json"

    if not result_file.exists():
        return None

    try:
        with open(result_file, 'r') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"{Colors.RED}Error loading {result_file}: {e}{Colors.NC}")
        return None

def print_separator(char='=', length=80):
    print(char * length)

def analyze_batch1(results_dir: str):
    """分析批次1的结果 (C4/C2/C1, seed=96)"""

    print(f"\n{Colors.BOLD}{Colors.CYAN}")
    print_separator()
    print("阶段1 - 批次1结果分析")
    print_separator()
    print(f"{Colors.NC}\n")

    # 基准线：extreme_seed96
    baseline_acc = 0.8514
    baseline_f1 = 0.8501

    variants = ['C4', 'C2', 'C1']
    seed = 96

    results = {}
    for variant in variants:
        data = load_result(variant, seed, results_dir)
        if data:
            results[variant] = data

    if not results:
        print(f"{Colors.RED}❌ No results found in {results_dir}{Colors.NC}")
        print(f"\n请确保批次1已完成训练！")
        return None

    # 打印对比表格
    print(f"{Colors.BOLD}📊 架构变体性能对比{Colors.NC}\n")
    print(f"{'Variant':<10} {'Config':<30} {'Test Acc':<12} {'Macro F1':<12} {'Δ Acc':<10} {'Δ F1':<10} {'Epoch':<8}")
    print_separator('-')

    # 基准线
    print(f"{'Baseline':<10} {'extreme (heads=8, layers=4)':<30} "
          f"{baseline_acc:.4f} ({baseline_acc*100:.2f}%)  "
          f"{baseline_f1:.4f}      "
          f"{'--':<10} {'--':<10} {'--':<8}")
    print_separator('-')

    # 各变体结果
    variant_scores = {}
    for variant in variants:
        if variant not in results:
            print(f"{variant:<10} {'NOT FOUND':<30} {'--':<12} {'--':<12} {'--':<10} {'--':<10} {'--':<8}")
            continue

        data = results[variant]
        best_test = data.get('best_test', data)  # Support both formats
        acc = best_test.get('best_test_acc', 0)
        f1 = best_test.get('macro_test_f1', 0)
        epoch = best_test.get('best_epoch', 0)

        delta_acc = (acc - baseline_acc) * 100
        delta_f1 = (f1 - baseline_f1) * 100

        variant_scores[variant] = {'acc': acc, 'f1': f1, 'delta_acc': delta_acc, 'delta_f1': delta_f1}

        # 配置描述
        config_map = {
            'C4': 'heads=12, layers=5 (组合)',
            'C2': 'heads=8, layers=5 (加深度)',
            'C1': 'heads=12, layers=4 (增强注意力)'
        }
        config = config_map.get(variant, 'Unknown')

        # 颜色标记
        color = Colors.GREEN if delta_acc > 0 else Colors.RED if delta_acc < 0 else Colors.YELLOW

        print(f"{color}{variant:<10}{Colors.NC} {config:<30} "
              f"{acc:.4f} ({acc*100:.2f}%)  "
              f"{f1:.4f}      "
              f"{delta_acc:+.2f}%   "
              f"{delta_f1:+.2f}%   "
              f"{epoch:<8}")

    print_separator('-')

    # 找到最佳变体
    if not variant_scores:
        print(f"\n{Colors.RED}❌ 没有成功的实验结果{Colors.NC}\n")
        return None

    best_variant = max(variant_scores.items(), key=lambda x: x[1]['acc'])
    best_name = best_variant[0]
    best_data = best_variant[1]

    print(f"\n{Colors.BOLD}{Colors.GREEN}🏆 批次1最佳变体: {best_name}{Colors.NC}")
    print(f"   Test Accuracy: {best_data['acc']:.4f} ({best_data['acc']*100:.2f}%)")
    print(f"   Macro F1:      {best_data['f1']:.4f}")
    print(f"   提升:          {best_data['delta_acc']:+.2f}% (Acc), {best_data['delta_f1']:+.2f}% (F1)")

    # 决策建议
    print(f"\n{Colors.BOLD}{Colors.CYAN}📋 批次2执行建议{Colors.NC}\n")

    if best_data['delta_acc'] >= 0.2:
        # 有明显提升
        print(f"{Colors.GREEN}✅ {best_name} 有明显提升！{Colors.NC}")
        print(f"\n推荐批次2策略：")
        print(f"  1. {best_name}-seed42  (验证可复现性)")
        print(f"  2. {best_name}-seed100 (扩展验证)")
        print(f"  3. C3-seed96          (补充温度衰减实验)")
        print(f"\n执行命令：")
        print(f"  bash run_stage1_batch2.sh --best_variant {best_name}")

    elif best_data['delta_acc'] >= 0.0:
        # 有轻微提升
        print(f"{Colors.YELLOW}⚠️  {best_name} 有轻微提升 ({best_data['delta_acc']:+.2f}%)，但不明显{Colors.NC}")
        print(f"\n推荐批次2策略：")
        print(f"  选项A（保守）：直接跳过阶段1，进入阶段2（超参数搜索）")
        print(f"  选项B（激进）：验证 {best_name} 在seed=42是否稳定")
        print(f"\n如果选B，执行命令：")
        print(f"  bash run_stage1_batch2.sh --best_variant {best_name} --verify_only")

    else:
        # 无提升或下降
        print(f"{Colors.RED}❌ 所有架构改进都没有提升（甚至下降）{Colors.NC}")
        print(f"\n推荐行动：")
        print(f"  🔄 立即停止架构探索，回退到baseline架构")
        print(f"  ➡️  直接进入阶段2：超参数网格搜索")
        print(f"\n执行命令：")
        print(f"  # 跳过批次2，直接进入阶段2")
        print(f"  python generate_stage2_script.py --architecture baseline")

    print()

    # 保存分析结果
    analysis_result = {
        'batch': 1,
        'baseline': {'acc': baseline_acc, 'f1': baseline_f1},
        'variants': variant_scores,
        'best_variant': best_name,
        'best_acc': best_data['acc'],
        'best_f1': best_data['f1'],
        'improvement_acc': best_data['delta_acc'],
        'improvement_f1': best_data['delta_f1'],
        'recommendation': 'proceed' if best_data['delta_acc'] >= 0.2 else 'cautious' if best_data['delta_acc'] >= 0 else 'skip'
    }

    analysis_file = Path(results_dir).parent / "stage1_results" / "batch1_analysis.json"
    analysis_file.parent.mkdir(parents=True, exist_ok=True)
    with open(analysis_file, 'w') as f:
        json.dump(analysis_result, f, indent=2)

    print(f"{Colors.GREEN}✅ 分析结果已保存到: {analysis_file}{Colors.NC}\n")

    return best_name

def analyze_all(results_dir: str):
    """分析所有批次的完整结果"""

    print(f"\n{Colors.BOLD}{Colors.CYAN}")
    print_separator()
    print("阶段1完整结果分析")
    print_separator()
    print(f"{Colors.NC}\n")

    # 加载所有结果
    all_variants = ['C1', 'C2', 'C3', 'C4']
    seeds = [42, 96, 100]

    results = {}
    for variant in all_variants:
        results[variant] = {}
        for seed in seeds:
            data = load_result(variant, seed, results_dir)
            if data:
                results[variant][seed] = data

    # 基准线
    baseline_acc = 0.8514
    baseline_f1 = 0.8501

    # 打印完整对比表
    print(f"{Colors.BOLD}📊 所有架构变体完整对比{Colors.NC}\n")
    print(f"{'Variant':<10} {'Seed':<8} {'Test Acc':<12} {'Macro F1':<12} {'Δ Acc':<10} {'Δ F1':<10} {'Epoch':<8}")
    print_separator('-')

    # 基准线
    print(f"{'Baseline':<10} {'--':<8} {baseline_acc:.4f} ({baseline_acc*100:.2f}%)  {baseline_f1:.4f}      {'--':<10} {'--':<10} {'--':<8}")
    print_separator('-')

    # 统计每个变体的平均性能
    variant_stats = {}

    for variant in all_variants:
        if not results[variant]:
            continue

        accs = []
        f1s = []

        for seed in seeds:
            if seed not in results[variant]:
                print(f"{variant:<10} {seed:<8} {'NOT FOUND':<12} {'--':<12} {'--':<10} {'--':<10} {'--':<8}")
                continue

            data = results[variant][seed]
            best_test = data.get('best_test', data)  # Support both formats
            acc = best_test.get('best_test_acc', 0)
            f1 = best_test.get('macro_test_f1', 0)
            epoch = best_test.get('best_epoch', 0)

            delta_acc = (acc - baseline_acc) * 100
            delta_f1 = (f1 - baseline_f1) * 100

            accs.append(acc)
            f1s.append(f1)

            color = Colors.GREEN if delta_acc > 0 else Colors.RED if delta_acc < 0 else Colors.YELLOW

            print(f"{color}{variant:<10}{Colors.NC} {seed:<8} "
                  f"{acc:.4f} ({acc*100:.2f}%)  "
                  f"{f1:.4f}      "
                  f"{delta_acc:+.2f}%   "
                  f"{delta_f1:+.2f}%   "
                  f"{epoch:<8}")

        if accs:
            import numpy as np
            variant_stats[variant] = {
                'avg_acc': np.mean(accs),
                'std_acc': np.std(accs),
                'avg_f1': np.mean(f1s),
                'std_f1': np.std(f1s),
                'max_acc': max(accs),
                'max_f1': max(f1s)
            }

    print_separator('-')

    # 打印统计摘要
    if variant_stats:
        print(f"\n{Colors.BOLD}📈 架构变体统计摘要{Colors.NC}\n")
        print(f"{'Variant':<10} {'Avg Acc':<15} {'Std Acc':<12} {'Avg F1':<15} {'Max Acc':<12}")
        print_separator('-')

        for variant, stats in sorted(variant_stats.items(), key=lambda x: x[1]['avg_acc'], reverse=True):
            delta_avg = (stats['avg_acc'] - baseline_acc) * 100
            color = Colors.GREEN if delta_avg > 0 else Colors.RED if delta_avg < 0 else Colors.YELLOW

            print(f"{color}{variant:<10}{Colors.NC} "
                  f"{stats['avg_acc']:.4f} ({delta_avg:+.2f}%)  "
                  f"{stats['std_acc']:.4f}     "
                  f"{stats['avg_f1']:.4f}         "
                  f"{stats['max_acc']:.4f}")

        print_separator('-')

        # 最佳配置
        best_overall = max(variant_stats.items(), key=lambda x: x[1]['max_acc'])
        best_variant = best_overall[0]
        best_stats = best_overall[1]

        print(f"\n{Colors.BOLD}{Colors.GREEN}🏆 阶段1最佳架构: {best_variant}{Colors.NC}")
        print(f"   平均准确率: {best_stats['avg_acc']:.4f} ({best_stats['avg_acc']*100:.2f}%)")
        print(f"   峰值准确率: {best_stats['max_acc']:.4f} ({best_stats['max_acc']*100:.2f}%)")
        print(f"   标准差:     {best_stats['std_acc']:.4f}")
        print(f"   vs Baseline: {(best_stats['avg_acc'] - baseline_acc)*100:+.2f}%")

        # 进入阶段2的建议
        print(f"\n{Colors.BOLD}{Colors.CYAN}➡️  进入阶段2：超参数搜索{Colors.NC}\n")

        if best_stats['avg_acc'] > baseline_acc:
            print(f"{Colors.GREEN}✅ 架构改进有效！{Colors.NC}")
            print(f"   推荐使用 {best_variant} 架构进入阶段2")
            print(f"\n执行命令：")
            print(f"   python generate_stage2_script.py --architecture {best_variant}")
        else:
            print(f"{Colors.YELLOW}⚠️  架构改进效果不明显{Colors.NC}")
            print(f"   推荐使用baseline架构进入阶段2")
            print(f"\n执行命令：")
            print(f"   python generate_stage2_script.py --architecture baseline")

        print()

    print_separator()
    print()

def main():
    parser = argparse.ArgumentParser(description='阶段1结果分析')
    parser.add_argument('--batch', type=str, default='1',
                       choices=['1', 'all'],
                       help='分析批次：1=只分析批次1, all=分析所有完整结果')
    parser.add_argument('--results_dir', type=str,
                       default='../output_dir/stage1_results',
                       help='结果文件目录')

    args = parser.parse_args()

    results_dir = args.results_dir

    if not Path(results_dir).exists():
        print(f"{Colors.RED}❌ 结果目录不存在: {results_dir}{Colors.NC}")
        print(f"\n请确保批次1已完成训练！")
        print(f"运行: bash run_stage1_batch1.sh")
        sys.exit(1)

    if args.batch == '1':
        analyze_batch1(results_dir)
    else:
        analyze_all(results_dir)

if __name__ == '__main__':
    main()
