#!/usr/bin/env python3
"""
根据批次1结果，动态生成批次2训练脚本
"""

import json
import argparse
from pathlib import Path

def load_batch1_analysis():
    """加载批次1分析结果"""
    analysis_file = Path("../output_dir/stage1_results/batch1_analysis.json")

    if not analysis_file.exists():
        print("❌ 批次1分析结果不存在！")
        print("请先运行: python analyze_stage1.py --batch 1")
        return None

    with open(analysis_file, 'r') as f:
        return json.load(f)

def generate_batch2_script(best_variant, strategy='full'):
    """
    生成批次2脚本

    Args:
        best_variant: 批次1的最佳变体 (C1/C2/C3/C4)
        strategy: 'full' = 完整验证, 'verify' = 只验证seed=42
    """

    # 配置映射
    config_map = {
        'C1': {'num_heads': 12, 'layers': 4, 'batch_size': 128},
        'C2': {'num_heads': 8, 'layers': 5, 'batch_size': 128},
        'C3': {'num_heads': 8, 'layers': 4, 'batch_size': 128, 'tau_min': 0.3, 'tau_decay': 0.999},
        'C4': {'num_heads': 12, 'layers': 5, 'batch_size': 96}
    }

    config = config_map.get(best_variant, config_map['C2'])  # 默认C2

    if strategy == 'full':
        # 完整策略：最佳变体seed42+100 + C3补充
        experiments = [
            (best_variant, 42),
            (best_variant, 100),
            ('C3', 96) if best_variant != 'C3' else (best_variant, 42)
        ]
        gpu_allocation = [0, 1, 2]
    else:
        # 验证策略：只跑seed=42
        experiments = [(best_variant, 42)]
        gpu_allocation = [0]

    script_content = f"""#!/bin/bash
# =============================================================================
# 阶段1 - 批次2：基于批次1结果的验证实验
# =============================================================================
# 最佳变体: {best_variant}
# 策略: {strategy}
# =============================================================================

set -e

GREEN='\\033[0;32m'
YELLOW='\\033[1;33m'
RED='\\033[0;31m'
NC='\\033[0m'

BASE_DIR="/home/user/chengtaiyu/RCLMuFN-main_copy"
OUTPUT_BASE="${{BASE_DIR}}/output_dir/stage1_architecture"
RESULTS_DIR="${{BASE_DIR}}/output_dir/stage1_results"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${{OUTPUT_BASE}}/batch2_master_${{TIMESTAMP}}.log"

log_info() {{
    echo -e "${{GREEN}}[$(date +'%Y-%m-%d %H:%M:%S')] [INFO]${{NC}} $1" | tee -a "${{LOG_FILE}}"
}}

log_error() {{
    echo -e "${{RED}}[$(date +'%Y-%m-%d %H:%M:%S')] [ERROR]${{NC}} $1" | tee -a "${{LOG_FILE}}"
}}

# =============================================================================
# 训练函数
# =============================================================================
train_variant() {{
    local variant_name=$1
    local gpu_id=$2
    local seed=$3
    local num_heads=$4
    local layers=$5
    local batch_size=$6
    local tau_min=${{7:-0.4}}
    local tau_decay=${{8:-0.9995}}

    local output_dir="${{OUTPUT_BASE}}/${{variant_name}}_seed${{seed}}"
    local result_file="${{RESULTS_DIR}}/${{variant_name}}_seed${{seed}}.json"
    local train_log="${{output_dir}}/training.log"

    mkdir -p "${{output_dir}}"

    log_info "Starting ${{variant_name}}-seed${{seed}} on GPU ${{gpu_id}}"

    # Extreme配置超参数
    python main.py \\
        --device ${{gpu_id}} \\
        --model RCLMuFN \\
        --text_name text_final \\
        --seed ${{seed}} \\
        --train_batch_size ${{batch_size}} \\
        --dev_batch_size ${{batch_size}} \\
        --optimizer_name adam \\
        --learning_rate 1.5e-3 \\
        --clip_learning_rate 2e-6 \\
        --weight_decay 0.025 \\
        --warmup_proportion 0.3 \\
        --max_grad_norm 5.0 \\
        --adam_epsilon 1e-8 \\
        --num_train_epochs 15 \\
        --max_len 77 \\
        --text_size 512 \\
        --image_size 768 \\
        --layers ${{layers}} \\
        --num_heads ${{num_heads}} \\
        --dropout_rate 0.15 \\
        --label_number 2 \\
        --lambda_ratio_start 0.0 \\
        --lambda_ratio_end 3e-3 \\
        --lambda_itm_start 0.0 \\
        --lambda_itm_end 2.5e-3 \\
        --lambda_warmup_epochs 3 \\
        --lambda_ramp_epochs 5 \\
        --lambda_schedule cosine \\
        --neg_sampling label_aware \\
        --tau_schedule_mode epoch \\
        --tau_min ${{tau_min}} \\
        --tau_decay ${{tau_decay}} \\
        --output_dir "${{output_dir}}" \\
        --results_path "${{result_file}}" \\
        > "${{train_log}}" 2>&1

    local exit_code=$?

    if [ $exit_code -eq 0 ]; then
        log_info "✅ ${{variant_name}}-seed${{seed}} completed"

        if [ -f "${{result_file}}" ]; then
            local best_acc=$(python -c "import json; d=json.load(open('${{result_file}}')); print(f\\"{d.get('best_test_acc', 0):.4f}\\")")
            log_info "  Best Acc: ${{best_acc}}"
        fi
    else
        log_error "❌ ${{variant_name}}-seed${{seed}} failed"
    fi

    return $exit_code
}}

# =============================================================================
# 主执行流程
# =============================================================================
log_info "========================================="
log_info "阶段1 - 批次2：验证实验"
log_info "========================================="
log_info "最佳变体: {best_variant}"
log_info "开始时间: $(date)"
echo ""

"""

    # 添加并行训练命令
    for i, (variant, seed) in enumerate(experiments):
        gpu = gpu_allocation[i] if i < len(gpu_allocation) else 0
        cfg = config_map.get(variant, config)

        tau_min = cfg.get('tau_min', 0.4)
        tau_decay = cfg.get('tau_decay', 0.9995)

        script_content += f"""# {variant}-seed{seed}
train_variant "{variant}" {gpu} {seed} {cfg['num_heads']} {cfg['layers']} {cfg['batch_size']} {tau_min} {tau_decay} &
PID_{variant}_{seed}=$!
sleep 5

"""

    script_content += """
# 等待所有任务完成
log_info "等待所有任务完成..."
echo ""

"""

    for variant, seed in experiments:
        script_content += f"""wait $PID_{variant}_{seed}
EXIT_{variant}_{seed}=$?

"""

    script_content += f"""
# 汇总结果
log_info "========================================="
log_info "批次2完成"
log_info "========================================="

"""

    for variant, seed in experiments:
        script_content += f"""if [ $EXIT_{variant}_{seed} -eq 0 ]; then
    log_info "✅ {variant}-seed{seed}: SUCCESS"
else
    log_error "❌ {variant}-seed{seed}: FAILED"
fi

"""

    script_content += """
log_info "下一步: python analyze_stage1.py --batch all"
echo ""

exit 0
"""

    # 保存脚本
    script_file = Path(f"run_stage1_batch2_{best_variant.lower()}.sh")
    with open(script_file, 'w') as f:
        f.write(script_content)

    script_file.chmod(0o755)

    return str(script_file)

def main():
    parser = argparse.ArgumentParser(description='生成批次2训练脚本')
    parser.add_argument('--best_variant', type=str,
                       choices=['C1', 'C2', 'C3', 'C4'],
                       help='最佳变体（如果不指定，从batch1_analysis.json读取）')
    parser.add_argument('--strategy', type=str, default='full',
                       choices=['full', 'verify'],
                       help='full=完整验证(3个实验), verify=只验证seed=42(1个实验)')

    args = parser.parse_args()

    # 确定最佳变体
    if args.best_variant:
        best_variant = args.best_variant
        print(f"使用指定的最佳变体: {best_variant}")
    else:
        # 从分析结果读取
        analysis = load_batch1_analysis()
        if not analysis:
            return

        best_variant = analysis.get('best_variant', 'C2')
        improvement = analysis.get('improvement_acc', 0)

        print(f"\n从批次1分析结果读取:")
        print(f"  最佳变体: {best_variant}")
        print(f"  提升: {improvement:+.2f}%")
        print(f"  推荐: {analysis.get('recommendation', 'cautious')}")
        print()

    # 生成脚本
    script_file = generate_batch2_script(best_variant, args.strategy)

    print(f"✅ 批次2脚本已生成: {script_file}")
    print()
    print(f"执行命令:")
    print(f"  bash {script_file}")
    print()

    if args.strategy == 'full':
        print(f"批次2将运行3个实验，预计耗时: ~2.5小时（3并行）")
    else:
        print(f"批次2将运行1个实验，预计耗时: ~2.5小时")
    print()

if __name__ == '__main__':
    main()
