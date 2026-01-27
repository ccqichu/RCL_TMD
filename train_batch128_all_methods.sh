#!/bin/bash
# =============================================================================
# Batch Size 128 全方案训练脚本
# =============================================================================
# 功能：按顺序测试三种优化方案（保守、激进、极致）
# 种子：42, 96, 100（基于历史最优表现）
# 输出：以 {方案}_{种子} 命名区分
# =============================================================================

set -e  # 遇到错误立即退出

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 工作目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 随机种子列表（按历史性能排序）
SEEDS=(42 96 100)

# 基础输出目录
BASE_OUTPUT_DIR="../output_dir/batch128_experiments"
RESULTS_DIR="../output_dir/seed_runs"
mkdir -p "$BASE_OUTPUT_DIR"
mkdir -p "$RESULTS_DIR"

# 创建主日志文件
MASTER_LOG="${BASE_OUTPUT_DIR}/master_training_log_$(date +%Y%m%d_%H%M%S).log"
touch "$MASTER_LOG"

# 日志函数
log_info() {
    echo -e "${GREEN}[INFO]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "$MASTER_LOG"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "$MASTER_LOG"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "$MASTER_LOG"
}

log_section() {
    echo -e "\n${BLUE}========================================${NC}" | tee -a "$MASTER_LOG"
    echo -e "${BLUE}$1${NC}" | tee -a "$MASTER_LOG"
    echo -e "${BLUE}========================================${NC}\n" | tee -a "$MASTER_LOG"
}

# 检查GPU可用性
if ! nvidia-smi &> /dev/null; then
    log_error "GPU not available! Exiting..."
    exit 1
fi

log_info "GPU Status:"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader | tee -a "$MASTER_LOG"

# =============================================================================
# 训练函数
# =============================================================================

train_model() {
    local method=$1
    local seed=$2
    local lr=$3
    local clip_lr=$4
    local warmup=$5
    local epochs=$6
    local lambda_ratio=$7
    local lambda_itm=$8
    local lambda_warmup=$9
    local lambda_ramp=${10}
    local lambda_schedule=${11}
    local layers=${12}
    local dropout=${13}
    local weight_decay=${14}
    local max_grad_norm=${15}

    local output_dir="${BASE_OUTPUT_DIR}/${method}_seed${seed}"
    local results_file="${RESULTS_DIR}/${method}_seed${seed}.json"
    local run_log="${output_dir}/training.log"

    mkdir -p "$output_dir"

    log_section "Method: ${method} | Seed: ${seed}"
    log_info "Output Dir: $output_dir"
    log_info "Results File: $results_file"
    log_info "Learning Rate: $lr"
    log_info "Epochs: $epochs"

    # 记录开始时间
    local start_time=$(date +%s)

    # 执行训练
    python main.py \
        --device 2 \
        --model RCLMuFN \
        --text_name text_final \
        --seed $seed \
        --train_batch_size 128 \
        --dev_batch_size 128 \
        --optimizer_name adam \
        --learning_rate $lr \
        --clip_learning_rate $clip_lr \
        --weight_decay $weight_decay \
        --warmup_proportion $warmup \
        --max_grad_norm $max_grad_norm \
        --adam_epsilon 1e-8 \
        --num_train_epochs $epochs \
        --max_len 77 \
        --text_size 512 \
        --image_size 768 \
        --layers $layers \
        --dropout_rate $dropout \
        --label_number 2 \
        --lambda_ratio_start 0.0 \
        --lambda_ratio_end $lambda_ratio \
        --lambda_itm_start 0.0 \
        --lambda_itm_end $lambda_itm \
        --lambda_warmup_epochs $lambda_warmup \
        --lambda_ramp_epochs $lambda_ramp \
        --lambda_schedule $lambda_schedule \
        --neg_sampling label_aware \
        --tau_schedule_mode epoch \
        --output_dir "$output_dir" \
        --results_path "$results_file" \
        2>&1 | tee "$run_log"

    local exit_code=${PIPESTATUS[0]}

    # 记录结束时间
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    local hours=$((duration / 3600))
    local minutes=$(((duration % 3600) / 60))

    if [ $exit_code -eq 0 ]; then
        log_info "✅ Training completed successfully in ${hours}h ${minutes}m"

        # 读取并显示结果
        if [ -f "$results_file" ]; then
            log_info "Results from $results_file:"
            cat "$results_file" | tee -a "$MASTER_LOG"
        fi
    else
        log_error "❌ Training failed with exit code $exit_code"
        return $exit_code
    fi

    # 清理显存
    log_info "Cleaning GPU memory..."
    python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true
    sleep 5

    return 0
}

# =============================================================================
# 方案配置
# =============================================================================

# 方案B：保守优化
run_conservative() {
    local seed=$1
    log_section "Running CONSERVATIVE method with seed=$seed"

    train_model \
        "conservative" \
        $seed \
        1.05e-3 \
        1e-6 \
        0.22 \
        10 \
        2e-3 \
        1.5e-3 \
        2 \
        3 \
        "linear" \
        3 \
        0.1 \
        0.03 \
        3.0
}

# 方案A：激进优化
run_aggressive() {
    local seed=$1
    log_section "Running AGGRESSIVE method with seed=$seed"

    train_model \
        "aggressive" \
        $seed \
        1.2e-3 \
        1e-6 \
        0.25 \
        12 \
        2.5e-3 \
        2e-3 \
        2 \
        4 \
        "linear" \
        3 \
        0.1 \
        0.03 \
        3.0
}

# 方案C：极致优化
run_extreme() {
    local seed=$1
    log_section "Running EXTREME method with seed=$seed"

    train_model \
        "extreme" \
        $seed \
        1.5e-3 \
        2e-6 \
        0.3 \
        15 \
        3e-3 \
        2.5e-3 \
        3 \
        5 \
        "cosine" \
        4 \
        0.15 \
        0.025 \
        5.0
}

# =============================================================================
# 主训练循环
# =============================================================================

log_section "Starting Batch Size 128 Experiments"
log_info "Total configurations: 3 methods × ${#SEEDS[@]} seeds = $((3 * ${#SEEDS[@]})) runs"
log_info "Seeds: ${SEEDS[*]}"
log_info "Master log: $MASTER_LOG"

# 记录实验开始时间
EXPERIMENT_START=$(date +%s)

# 计数器
total_runs=0
successful_runs=0
failed_runs=0

# 按方案顺序执行（每个方案测试所有种子）
for method in conservative aggressive extreme; do
    log_section "Starting method: ${method^^}"

    for seed in "${SEEDS[@]}"; do
        total_runs=$((total_runs + 1))
        log_info "Run $total_runs of $((3 * ${#SEEDS[@]}))"

        case $method in
            conservative)
                if run_conservative $seed; then
                    successful_runs=$((successful_runs + 1))
                else
                    failed_runs=$((failed_runs + 1))
                    log_warn "Failed run will be recorded but continuing..."
                fi
                ;;
            aggressive)
                if run_aggressive $seed; then
                    successful_runs=$((successful_runs + 1))
                else
                    failed_runs=$((failed_runs + 1))
                    log_warn "Failed run will be recorded but continuing..."
                fi
                ;;
            extreme)
                if run_extreme $seed; then
                    successful_runs=$((successful_runs + 1))
                else
                    failed_runs=$((failed_runs + 1))
                    log_warn "Failed run will be recorded but continuing..."
                fi
                ;;
        esac

        # 显示当前进度
        log_info "Progress: $successful_runs succeeded, $failed_runs failed out of $total_runs completed"
    done
done

# =============================================================================
# 生成实验总结
# =============================================================================

EXPERIMENT_END=$(date +%s)
TOTAL_DURATION=$((EXPERIMENT_END - EXPERIMENT_START))
TOTAL_HOURS=$((TOTAL_DURATION / 3600))
TOTAL_MINUTES=$(((TOTAL_DURATION % 3600) / 60))

log_section "Experiment Summary"
log_info "Total runs: $total_runs"
log_info "Successful: $successful_runs"
log_info "Failed: $failed_runs"
log_info "Total time: ${TOTAL_HOURS}h ${TOTAL_MINUTES}m"
log_info "Master log: $MASTER_LOG"

# 生成结果汇总文件
SUMMARY_FILE="${BASE_OUTPUT_DIR}/summary_$(date +%Y%m%d_%H%M%S).txt"
{
    echo "==================================================="
    echo "Batch Size 128 Experiments Summary"
    echo "==================================================="
    echo ""
    echo "Experiment Date: $(date)"
    echo "Total Duration: ${TOTAL_HOURS}h ${TOTAL_MINUTES}m"
    echo "Total Runs: $total_runs"
    echo "Successful: $successful_runs"
    echo "Failed: $failed_runs"
    echo ""
    echo "==================================================="
    echo "Results by Method and Seed"
    echo "==================================================="
    echo ""

    for method in conservative aggressive extreme; do
        echo "--- ${method^^} METHOD ---"
        for seed in "${SEEDS[@]}"; do
            results_file="${RESULTS_DIR}/${method}_seed${seed}.json"
            if [ -f "$results_file" ]; then
                echo "Seed $seed:"
                cat "$results_file" | python -m json.tool 2>/dev/null | grep -E "(best_test_acc|macro_test_f1|best_epoch)" || echo "  Results file exists but could not parse"
            else
                echo "Seed $seed: No results file found"
            fi
            echo ""
        done
        echo ""
    done

    echo "==================================================="
    echo "File Locations"
    echo "==================================================="
    echo "Base output directory: $BASE_OUTPUT_DIR"
    echo "Results directory: $RESULTS_DIR"
    echo "Master log: $MASTER_LOG"
    echo ""
    echo "All models saved in: ${BASE_OUTPUT_DIR}/{method}_seed{seed}/RCLMuFN/model.pt"
    echo "All results saved in: ${RESULTS_DIR}/{method}_seed{seed}.json"

} > "$SUMMARY_FILE"

log_info "Summary saved to: $SUMMARY_FILE"
cat "$SUMMARY_FILE" | tee -a "$MASTER_LOG"

# =============================================================================
# 找出最佳配置
# =============================================================================

log_section "Finding Best Configuration"

BEST_ACC=0
BEST_CONFIG=""
BEST_RESULTS_FILE=""

for method in conservative aggressive extreme; do
    for seed in "${SEEDS[@]}"; do
        results_file="${RESULTS_DIR}/${method}_seed${seed}.json"
        if [ -f "$results_file" ]; then
            acc=$(python -c "
import json
import sys
try:
    with open('$results_file', 'r') as f:
        data = json.load(f)
        print(data.get('best_test', {}).get('best_test_acc', 0))
except:
    print(0)
" 2>/dev/null)

            if [ -n "$acc" ] && (( $(echo "$acc > $BEST_ACC" | bc -l) )); then
                BEST_ACC=$acc
                BEST_CONFIG="${method}_seed${seed}"
                BEST_RESULTS_FILE=$results_file
            fi
        fi
    done
done

if [ -n "$BEST_CONFIG" ]; then
    log_info "🏆 Best configuration: $BEST_CONFIG"
    log_info "🏆 Best accuracy: $BEST_ACC"
    log_info "🏆 Results file: $BEST_RESULTS_FILE"
    echo ""
    log_info "Full results:"
    cat "$BEST_RESULTS_FILE" | python -m json.tool 2>/dev/null | tee -a "$MASTER_LOG"
else
    log_warn "Could not determine best configuration"
fi

log_section "All Experiments Complete! 🎉"
log_info "Check $SUMMARY_FILE for detailed results"
log_info "Check $MASTER_LOG for complete training logs"

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Next Steps:${NC}"
echo -e "${GREEN}1. Review summary: cat $SUMMARY_FILE${NC}"
echo -e "${GREEN}2. Check best model: ${BASE_OUTPUT_DIR}/${BEST_CONFIG}/RCLMuFN/model.pt${NC}"
echo -e "${GREEN}3. View training curves in wandb${NC}"
echo -e "${GREEN}========================================${NC}"
