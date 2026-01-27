#!/usr/bin/env python3
"""
Batch Size 128 实验结果分析脚本
自动读取所有实验结果，生成对比表格和可视化
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import statistics

# 配置
RESULTS_DIR = Path("../output_dir/seed_runs")
METHODS = ["conservative", "aggressive", "extreme"]
SEEDS = [42, 96, 100]

# ANSI颜色
class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'


def load_results(method: str, seed: int) -> Dict:
    """加载单个实验结果"""
    results_file = RESULTS_DIR / f"{method}_seed{seed}.json"

    if not results_file.exists():
        return None

    try:
        with open(results_file, 'r') as f:
            data = json.load(f)
            return data.get('best_test', {})
    except Exception as e:
        print(f"Error loading {results_file}: {e}")
        return None


def format_metric(value, is_best=False):
    """格式化指标值"""
    if value is None:
        return "N/A"

    formatted = f"{value:.4f}" if isinstance(value, float) else str(value)

    if is_best:
        return f"{Colors.GREEN}{Colors.BOLD}{formatted}{Colors.END}"
    return formatted


def print_header(title: str):
    """打印章节标题"""
    print(f"\n{Colors.BLUE}{'='*80}{Colors.END}")
    print(f"{Colors.BLUE}{Colors.BOLD}{title:^80}{Colors.END}")
    print(f"{Colors.BLUE}{'='*80}{Colors.END}\n")


def print_comparison_table():
    """打印所有结果对比表"""
    print_header("Batch Size 128 实验结果对比")

    # 表头
    header = f"{'Method':<15} {'Seed':<6} {'Acc':>8} {'Macro F1':>8} {'Macro P':>8} {'Macro R':>8} {'Epoch':>6}"
    print(header)
    print("-" * len(header))

    all_results = {}

    # 收集所有结果
    for method in METHODS:
        all_results[method] = {}
        for seed in SEEDS:
            result = load_results(method, seed)
            all_results[method][seed] = result

    # 找出最佳值
    all_accs = []
    all_f1s = []
    for method_results in all_results.values():
        for result in method_results.values():
            if result:
                all_accs.append(result.get('best_test_acc', 0))
                all_f1s.append(result.get('macro_test_f1', 0))

    best_acc = max(all_accs) if all_accs else 0
    best_f1 = max(all_f1s) if all_f1s else 0

    # 打印结果
    for method in METHODS:
        for seed in SEEDS:
            result = all_results[method][seed]

            if result:
                acc = result.get('best_test_acc', 0)
                f1 = result.get('macro_test_f1', 0)
                precision = result.get('macro_test_precision', 0)
                recall = result.get('macro_test_recall', 0)
                epoch = result.get('best_epoch', 0)

                acc_str = format_metric(acc, acc == best_acc)
                f1_str = format_metric(f1, f1 == best_f1)

                print(f"{method:<15} {seed:<6} {acc_str:>8} {f1_str:>8} "
                      f"{format_metric(precision):>8} {format_metric(recall):>8} {epoch:>6}")
            else:
                print(f"{method:<15} {seed:<6} {'FAILED':>8} {'FAILED':>8} "
                      f"{'FAILED':>8} {'FAILED':>8} {'N/A':>6}")

    return all_results


def print_method_statistics(all_results: Dict):
    """打印每个方法的统计信息"""
    print_header("各方法统计摘要")

    header = f"{'Method':<15} {'Avg Acc':>10} {'Std Acc':>10} {'Avg F1':>10} {'Std F1':>10} {'Avg Epoch':>10}"
    print(header)
    print("-" * len(header))

    best_avg_acc = 0
    best_method = ""

    for method in METHODS:
        accs = []
        f1s = []
        epochs = []

        for seed in SEEDS:
            result = all_results[method][seed]
            if result:
                accs.append(result.get('best_test_acc', 0))
                f1s.append(result.get('macro_test_f1', 0))
                epochs.append(result.get('best_epoch', 0))

        if accs:
            avg_acc = statistics.mean(accs)
            std_acc = statistics.stdev(accs) if len(accs) > 1 else 0
            avg_f1 = statistics.mean(f1s)
            std_f1 = statistics.stdev(f1s) if len(f1s) > 1 else 0
            avg_epoch = statistics.mean(epochs)

            if avg_acc > best_avg_acc:
                best_avg_acc = avg_acc
                best_method = method

            is_best = (method == best_method)

            print(f"{method:<15} "
                  f"{format_metric(avg_acc, is_best):>10} "
                  f"{format_metric(std_acc):>10} "
                  f"{format_metric(avg_f1, is_best):>10} "
                  f"{format_metric(std_f1):>10} "
                  f"{avg_epoch:>10.1f}")
        else:
            print(f"{method:<15} {'NO DATA':>10} {'NO DATA':>10} {'NO DATA':>10} {'NO DATA':>10} {'N/A':>10}")

    return best_method


def print_seed_comparison(all_results: Dict):
    """打印不同种子的表现对比"""
    print_header("种子稳定性分析")

    header = f"{'Seed':<6} {'Conservative':>15} {'Aggressive':>15} {'Extreme':>15} {'Best Method':>15}"
    print(header)
    print("-" * len(header))

    for seed in SEEDS:
        row = f"{seed:<6}"

        seed_accs = {}
        for method in METHODS:
            result = all_results[method][seed]
            if result:
                acc = result.get('best_test_acc', 0)
                seed_accs[method] = acc
                row += f" {acc:>14.4f}"
            else:
                row += f" {'FAILED':>14}"

        # 找出该种子下的最佳方法
        if seed_accs:
            best_method_for_seed = max(seed_accs, key=seed_accs.get)
            row += f" {best_method_for_seed:>15}"
        else:
            row += f" {'N/A':>15}"

        print(row)


def find_overall_best(all_results: Dict) -> Tuple[str, int, Dict]:
    """找出总体最佳配置"""
    best_acc = 0
    best_config = None
    best_result = None

    for method in METHODS:
        for seed in SEEDS:
            result = all_results[method][seed]
            if result:
                acc = result.get('best_test_acc', 0)
                if acc > best_acc:
                    best_acc = acc
                    best_config = (method, seed)
                    best_result = result

    return best_config, best_result


def print_recommendations(all_results: Dict, best_method: str, best_config: Tuple, best_result: Dict):
    """打印推荐和建议"""
    print_header("结论与建议")

    if best_config and best_result:
        method, seed = best_config
        acc = best_result.get('best_test_acc', 0)
        f1 = best_result.get('macro_test_f1', 0)
        epoch = best_result.get('best_epoch', 0)

        print(f"{Colors.GREEN}🏆 最佳单次配置:{Colors.END}")
        print(f"   Method: {Colors.BOLD}{method.upper()}{Colors.END}")
        print(f"   Seed: {Colors.BOLD}{seed}{Colors.END}")
        print(f"   Test Accuracy: {Colors.BOLD}{acc:.4f}{Colors.END}")
        print(f"   Macro F1: {Colors.BOLD}{f1:.4f}{Colors.END}")
        print(f"   Best Epoch: {Colors.BOLD}{epoch}{Colors.END}")
        print(f"   Model Path: ../output_dir/batch128_experiments/{method}_seed{seed}/RCLMuFN/model.pt")
        print()

    print(f"{Colors.YELLOW}📊 最稳定方法:{Colors.END} {Colors.BOLD}{best_method.upper()}{Colors.END}")
    print()

    # 计算各方法的成功率
    print(f"{Colors.BLUE}💡 实验建议:{Colors.END}")

    success_rates = {}
    for method in METHODS:
        success_count = sum(1 for seed in SEEDS if all_results[method][seed] is not None)
        success_rates[method] = success_count / len(SEEDS) * 100

    for method in METHODS:
        rate = success_rates[method]
        symbol = "✅" if rate == 100 else "⚠️" if rate >= 66 else "❌"
        print(f"   {symbol} {method.capitalize()}: {rate:.0f}% 成功率")

    print()

    # 给出下一步建议
    print(f"{Colors.BLUE}🎯 下一步优化方向:{Colors.END}")

    # 分析哪个方法最好
    method_accs = {}
    for method in METHODS:
        accs = [all_results[method][seed].get('best_test_acc', 0)
                for seed in SEEDS if all_results[method][seed]]
        if accs:
            method_accs[method] = statistics.mean(accs)

    if method_accs:
        best = max(method_accs, key=method_accs.get)
        worst = min(method_accs, key=method_accs.get)

        print(f"   1. 如果追求最高指标: 使用 {Colors.BOLD}{best.upper()}{Colors.END} 方法")
        print(f"   2. 如果追求稳定性: 使用 {Colors.BOLD}{best_method.upper()}{Colors.END} 方法")
        print(f"   3. 考虑ensemble: 融合top-3配置的预测结果")
        print(f"   4. 进一步调优: 在 {best} 基础上微调学习率 ±20%")
        print(f"   5. 如果batch=128显存充足: 可尝试batch=160或192")


def export_to_csv(all_results: Dict):
    """导出结果到CSV文件"""
    output_file = RESULTS_DIR / "batch128_comparison.csv"

    with open(output_file, 'w') as f:
        # 写入表头
        f.write("Method,Seed,Accuracy,Macro_F1,Macro_Precision,Macro_Recall,"
                "Micro_F1,Micro_Precision,Micro_Recall,Best_Epoch\n")

        # 写入数据
        for method in METHODS:
            for seed in SEEDS:
                result = all_results[method][seed]
                if result:
                    f.write(f"{method},{seed},"
                           f"{result.get('best_test_acc', 0)},"
                           f"{result.get('macro_test_f1', 0)},"
                           f"{result.get('macro_test_precision', 0)},"
                           f"{result.get('macro_test_recall', 0)},"
                           f"{result.get('micro_test_f1', 0)},"
                           f"{result.get('micro_test_precision', 0)},"
                           f"{result.get('micro_test_recall', 0)},"
                           f"{result.get('best_epoch', 0)}\n")
                else:
                    f.write(f"{method},{seed},FAILED,FAILED,FAILED,FAILED,FAILED,FAILED,FAILED,N/A\n")

    print(f"\n{Colors.GREEN}✅ Results exported to: {output_file}{Colors.END}")


def main():
    print(f"{Colors.BOLD}")
    print(r"""
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║        Batch Size 128 实验结果分析                           ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
    print(f"{Colors.END}")

    # 检查结果目录
    if not RESULTS_DIR.exists():
        print(f"{Colors.RED}错误: 结果目录不存在: {RESULTS_DIR}{Colors.END}")
        sys.exit(1)

    # 1. 打印详细对比表
    all_results = print_comparison_table()

    # 2. 打印方法统计
    best_method = print_method_statistics(all_results)

    # 3. 打印种子对比
    print_seed_comparison(all_results)

    # 4. 找出最佳配置
    best_config, best_result = find_overall_best(all_results)

    # 5. 打印建议
    print_recommendations(all_results, best_method, best_config, best_result)

    # 6. 导出CSV
    export_to_csv(all_results)

    print(f"\n{Colors.GREEN}{'='*80}{Colors.END}")
    print(f"{Colors.GREEN}Analysis complete!{Colors.END}")
    print(f"{Colors.GREEN}{'='*80}{Colors.END}\n")


if __name__ == "__main__":
    main()
