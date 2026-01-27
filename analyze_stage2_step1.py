#!/usr/bin/env python3
"""
阶段2 - Step 2.1 结果分析脚本
分析学习率搜索结果，推荐最佳配置
"""

import json
import os
from pathlib import Path
from typing import Dict, List
import sys

# 颜色代码
class Colors:
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    RED = '\033[0;31m'
    BLUE = '\033[0;34m'
    CYAN = '\033[0;36m'
    BOLD = '\033[1m'
    NC = '\033[0m'

def load_result(config_name: str, results_dir: str) -> Dict:
    """加载单个实验结果"""
    result_file = Path(results_dir) / f"{config_name}.json"

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

def analyze_lr_search(results_dir: str = '../output_dir/stage2_results'):
    """分析学习率搜索结果"""

    print(f"\n{Colors.BOLD}{Colors.CYAN}")
    print_separator()
    print("阶段2 - Step 2.1：学习率搜索结果分析")
    print_separator()
    print(f"{Colors.NC}\n")

    # 基准线
    baseline_acc = 0.8514
    baseline_f1 = 0.8501
    c4_seed96_acc = 0.8518

    # 学习率配置
    lr_values = ['1.3e-3', '1.4e-3', '1.5e-3', '1.6e-3', '1.7e-3']
    configs = [f"lr_{lr}_seed42" for lr in lr_values]

    # 加载所有结果
    results = {}
    for config in configs:
        data = load_result(config, results_dir)
        if data:
            results[config] = data

    if not results:
        print(f"{Colors.RED}❌ 没有找到结果文件在 {results_dir}{Colors.NC}")
        print(f"\n请确保Step 2.1已完成训练！")
        sys.exit(1)

    # 打印对比表格
    print(f"{Colors.BOLD}📊 学习率性能对比{Colors.NC}\n")
    print(f"{'LR':<12} {'Test Acc':<15} {'Macro F1':<12} {'Δ vs C4-96':<12} {'Δ vs Baseline':<15} {'Epoch':<8}")
    print_separator('-')

    # 参考线
    print(f"{'Baseline':<12} {baseline_acc:.4f} ({baseline_acc*100:.2f}%)  {baseline_f1:.4f}      "
          f"{'--':<12} {'--':<15} {'--':<8}")
    print(f"{'C4-seed96':<12} {c4_seed96_acc:.4f} ({c4_seed96_acc*100:.2f}%)  {0.8500:.4f}      "
          f"{'--':<12} {(c4_seed96_acc-baseline_acc)*100:+.2f}%         {'4':<8}")
    print_separator('-')

    # 各学习率结果
    lr_scores = {}
    for i, lr in enumerate(lr_values):
        config = f"lr_{lr}_seed42"

        if config not in results:
            print(f"{lr:<12} {'NOT FOUND':<15} {'--':<12} {'--':<12} {'--':<15} {'--':<8}")
            continue

        data = results[config]
        best_test = data.get('best_test', {})
        acc = best_test.get('best_test_acc', 0)
        f1 = best_test.get('macro_test_f1', 0)
        epoch = best_test.get('best_epoch', 0)

        delta_c4 = (acc - c4_seed96_acc) * 100
        delta_baseline = (acc - baseline_acc) * 100

        lr_scores[lr] = {
            'acc': acc,
            'f1': f1,
            'epoch': epoch,
            'delta_c4': delta_c4,
            'delta_baseline': delta_baseline
        }

        # 颜色标记
        if delta_baseline > 0.15:
            color = Colors.GREEN
        elif delta_baseline > 0:
            color = Colors.YELLOW
        else:
            color = Colors.RED

        print(f"{color}{lr:<12}{Colors.NC} "
              f"{acc:.4f} ({acc*100:.2f}%)  "
              f"{f1:.4f}      "
              f"{delta_c4:+.2f}%      "
              f"{delta_baseline:+.2f}%         "
              f"{epoch:<8}")

    print_separator('-')

    # 找到最佳学习率
    if not lr_scores:
        print(f"\n{Colors.RED}❌ 没有成功的实验结果{Colors.NC}\n")
        sys.exit(1)

    best_lr = max(lr_scores.items(), key=lambda x: x[1]['acc'])
    best_lr_name = best_lr[0]
    best_lr_data = best_lr[1]

    print(f"\n{Colors.BOLD}{Colors.GREEN}🏆 最佳学习率: {best_lr_name}{Colors.NC}")
    print(f"   Test Accuracy: {best_lr_data['acc']:.4f} ({best_lr_data['acc']*100:.2f}%)")
    print(f"   Macro F1:      {best_lr_data['f1']:.4f}")
    print(f"   vs Baseline:   {best_lr_data['delta_baseline']:+.2f}%")
    print(f"   vs C4-seed96:  {best_lr_data['delta_c4']:+.2f}%")
    print(f"   Best Epoch:    {best_lr_data['epoch']}")

    # 决策建议
    print(f"\n{Colors.BOLD}{Colors.CYAN}📋 Step 2.2 执行建议{Colors.NC}\n")

    if best_lr_data['delta_baseline'] >= 0.3:
        # 显著提升
        print(f"{Colors.GREEN}✅ 找到显著提升的学习率！{Colors.NC}")
        print(f"\n推荐策略：")
        print(f"  1. 使用LR={best_lr_name}进入Step 2.2（lambda搜索）")
        print(f"  2. 预期lambda搜索可带来额外 +0.1-0.3% 提升")
        print(f"  3. 最终目标：85.5-85.8%")
        print(f"\n执行命令：")
        print(f"  python generate_stage2_step2.py --best_lr {best_lr_name}")
        print(f"  bash run_stage2_step2_lambda_search.sh")

    elif best_lr_data['delta_baseline'] >= 0.15:
        # 中等提升
        print(f"{Colors.YELLOW}⚠️  找到中等提升的学习率 ({best_lr_data['delta_baseline']:+.2f}%)${Colors.NC}")
        print(f"\n推荐策略：")
        print(f"  选项A（推荐）：继续Step 2.2（lambda搜索），可能还有提升空间")
        print(f"  选项B（保守）：直接用LR={best_lr_name}训练多个seed验证")
        print(f"\n如果选A，执行命令：")
        print(f"  python generate_stage2_step2.py --best_lr {best_lr_name}")

    elif best_lr_data['delta_baseline'] > 0:
        # 轻微提升
        print(f"{Colors.YELLOW}⚠️  只有轻微提升 ({best_lr_data['delta_baseline']:+.2f}%)${Colors.NC}")
        print(f"\n推荐策略：")
        print(f"  选项A：继续Step 2.2（激进），寻求lambda组合的突破")
        print(f"  选项B：接受当前结果，用LR={best_lr_name}训练多seed")
        print(f"\n风险提示：")
        print(f"  - 轻微提升可能在统计误差范围内")
        print(f"  - 建议至少用seed=96验证一次")

    else:
        # 无提升或下降
        print(f"{Colors.RED}❌ 所有学习率都没有提升（甚至下降）{Colors.NC}")
        print(f"\n问题诊断：")
        print(f"  1. 检查是否有训练失败的实验")
        print(f"  2. C4架构可能不适合batch_size=128（显存问题）")
        print(f"  3. 考虑回退到baseline架构重新搜索")
        print(f"\n建议行动：")
        print(f"  - 检查训练日志：tail -f ../output_dir/stage2_lr_search/*/training.log")
        print(f"  - 或者回退baseline架构：python generate_stage2_step1.py --architecture baseline")

    print()

    # 保存分析结果
    analysis_result = {
        'step': '2.1',
        'search_type': 'learning_rate',
        'baseline': {'acc': baseline_acc, 'f1': baseline_f1},
        'c4_seed96': {'acc': c4_seed96_acc, 'f1': 0.8500},
        'lr_results': lr_scores,
        'best_lr': best_lr_name,
        'best_acc': best_lr_data['acc'],
        'best_f1': best_lr_data['f1'],
        'improvement_vs_baseline': best_lr_data['delta_baseline'],
        'improvement_vs_c4': best_lr_data['delta_c4'],
        'recommendation': 'proceed' if best_lr_data['delta_baseline'] >= 0.15 else 'cautious' if best_lr_data['delta_baseline'] > 0 else 'failed'
    }

    analysis_file = Path(results_dir) / "step2.1_analysis.json"
    with open(analysis_file, 'w') as f:
        json.dump(analysis_result, f, indent=2)

    print(f"{Colors.GREEN}✅ 分析结果已保存到: {analysis_file}{Colors.NC}\n")

    print_separator()
    print()

if __name__ == '__main__':
    analyze_lr_search()
