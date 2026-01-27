#!/bin/bash
# =============================================================================
# 阶段1 - 批次1：架构变体快速验证（3个并行）
# =============================================================================
# 基于extreme配置，测试3个架构改进：
#   - C4-seed96: heads=12, layers=5 (容量最大化)
#   - C2-seed96: heads=8, layers=5 (加深度)
#   - C1-seed96: heads=12, layers=4 (增强注意力)
# =============================================================================

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

# =============================================================================
# 配置
# =============================================================================
BASE_DIR="/home/user/chengtaiyu/RCLMuFN-main_copy"
SRC_DIR="${BASE_DIR}/src"
OUTPUT_BASE="${BASE_DIR}/output_dir/stage1_architecture"
RESULTS_DIR="${BASE_DIR}/output_dir/stage1_results"

# 创建输出目录
mkdir -p "${OUTPUT_BASE}"
mkdir -p "${RESULTS_DIR}"

# GPU分配
GPU_C4=3
GPU_C2=1
GPU_C1=2

SEED=96  # 历史最佳种子

# 时间戳
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${OUTPUT_BASE}/batch1_master_${TIMESTAMP}.log"

# =============================================================================
# 日志函数
# =============================================================================
log_info() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] [INFO]${NC} $1" | tee -a "${LOG_FILE}"
}

log_warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] [WARN]${NC} $1" | tee -a "${LOG_FILE}"
}

log_error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] [ERROR]${NC} $1" | tee -a "${LOG_FILE}"
}

# =============================================================================
# 训练函数
# =============================================================================
train_variant() {
    local variant_name=$1
    local gpu_id=$2
    local seed=$3
    local num_heads=$4
    local layers=$5
    local batch_size=$6

    local output_dir="${OUTPUT_BASE}/${variant_name}_seed${seed}"
    local result_file="${RESULTS_DIR}/${variant_name}_seed${seed}.json"
    local train_log="${output_dir}/training.log"

    mkdir -p "${output_dir}"

    log_info "Starting ${variant_name} on GPU ${gpu_id}"
    log_info "  Config: heads=${num_heads}, layers=${layers}, batch_size=${batch_size}"

    # Extreme配置的超参数
    local lr=1.5e-3
    local clip_lr=2e-6
    local warmup_prop=0.3
    local epochs=15
    local lambda_ratio_end=3e-3
    local lambda_itm_end=2.5e-3
    local lambda_warmup_epochs=3
    local lambda_ramp_epochs=5
    local lambda_schedule="cosine"
    local dropout=0.15
    local weight_decay=0.025
    local max_grad_norm=5.0

    # 运行训练
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=${gpu_id} \
    python "${SRC_DIR}/main.py" \
        --device 0 \
        --device ${gpu_id} \
        --model RCLMuFN \
        --text_name text_final \
        --seed ${seed} \
        --train_batch_size ${batch_size} \
        --dev_batch_size ${batch_size} \
        --optimizer_name adam \
        --learning_rate ${lr} \
        --clip_learning_rate ${clip_lr} \
        --weight_decay ${weight_decay} \
        --warmup_proportion ${warmup_prop} \
        --max_grad_norm ${max_grad_norm} \
        --adam_epsilon 1e-8 \
        --num_train_epochs ${epochs} \
        --max_len 77 \
        --text_size 512 \
        --image_size 768 \
        --layers ${layers} \
        --num_heads ${num_heads} \
        --dropout_rate ${dropout} \
        --label_number 2 \
        --lambda_ratio_start 0.0 \
        --lambda_ratio_end ${lambda_ratio_end} \
        --lambda_itm_start 0.0 \
        --lambda_itm_end ${lambda_itm_end} \
        --lambda_warmup_epochs ${lambda_warmup_epochs} \
        --lambda_ramp_epochs ${lambda_ramp_epochs} \
        --lambda_schedule ${lambda_schedule} \
        --neg_sampling label_aware \
        --tau_schedule_mode epoch \
        --output_dir "${output_dir}" \
        --results_path "${result_file}" \
        > "${train_log}" 2>&1

    local exit_code=$?

    if [ $exit_code -eq 0 ]; then
        log_info "✅ ${variant_name} completed successfully"

        # 提取最佳结果
        if [ -f "${result_file}" ]; then
            local best_acc=$(python -c "import json; d=json.load(open('${result_file}')); print(f\"{d.get('best_test_acc', 0):.4f}\")")
            local best_f1=$(python -c "import json; d=json.load(open('${result_file}')); print(f\"{d.get('macro_test_f1', 0):.4f}\")")
            log_info "  Best Acc: ${best_acc}, Macro F1: ${best_f1}"
        fi
    else
        log_error "❌ ${variant_name} failed with exit code ${exit_code}"
        log_error "  Check log: ${train_log}"
    fi

    return $exit_code
}

# =============================================================================
# 主执行流程
# =============================================================================
echo ""
log_info "========================================="
log_info "阶段1 - 批次1：架构变体验证"
log_info "========================================="
log_info "Variants: C4, C2, C1"
log_info "Seed: ${SEED}"
log_info "Parallel GPUs: ${GPU_C4}, ${GPU_C2}, ${GPU_C1}"
log_info "Start time: $(date)"
echo ""

# 检查GPU
log_info "Checking GPUs..."
if ! nvidia-smi &> /dev/null; then
    log_error "GPU not available!"
    exit 1
fi
log_info "✓ GPUs available"

# 检查CLIP模型
CLIP_PATH="/home/user/chengtaiyu/models/clip-vit-base-patch32"
if [ ! -d "$CLIP_PATH" ]; then
    log_error "CLIP model not found at $CLIP_PATH"
    exit 1
fi
log_info "✓ CLIP model found"

# 检查数据集
DATA_PATH="/home/user/chengtaiyu/RCLMuFN-main/data/text_final"
if [ ! -d "$DATA_PATH" ]; then
    log_error "Dataset not found at $DATA_PATH"
    exit 1
fi
log_info "✓ Dataset found"

echo ""
log_info "-----------------------------------------"
log_info "Starting 3 parallel training jobs..."
log_info "-----------------------------------------"

# 启动3个并行训练
# C4: 最大容量 (heads=12, layers=5, batch可能需要降到96)
train_variant "C4" ${GPU_C4} ${SEED} 12 5 96 &
PID_C4=$!
sleep 5  # 错开启动，避免同时加载模型

# C2: 加深度 (heads=8, layers=5, batch=128)
train_variant "C2" ${GPU_C2} ${SEED} 8 5 128 &
PID_C2=$!
sleep 5

# C1: 增强注意力 (heads=12, layers=4, batch=128)
train_variant "C1" ${GPU_C1} ${SEED} 12 4 128 &
PID_C1=$!

# 等待所有任务完成
log_info ""
log_info "Waiting for all 3 jobs to complete..."
log_info "  C4 (GPU ${GPU_C4}): PID ${PID_C4}"
log_info "  C2 (GPU ${GPU_C2}): PID ${PID_C2}"
log_info "  C1 (GPU ${GPU_C1}): PID ${PID_C1}"
log_info ""
log_info "Monitor progress:"
log_info "  watch -n 10 'nvidia-smi && tail -n 3 ${OUTPUT_BASE}/C*/training.log'"
echo ""

# 等待所有进程
wait $PID_C4
EXIT_C4=$?

wait $PID_C2
EXIT_C2=$?

wait $PID_C1
EXIT_C1=$?

# =============================================================================
# 汇总结果
# =============================================================================
echo ""
log_info "========================================="
log_info "批次1完成"
log_info "========================================="
log_info "End time: $(date)"
echo ""

SUCCESS_COUNT=0
FAIL_COUNT=0

if [ $EXIT_C4 -eq 0 ]; then
    log_info "✅ C4 (heads=12, layers=5): SUCCESS"
    SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
else
    log_error "❌ C4 (heads=12, layers=5): FAILED"
    FAIL_COUNT=$((FAIL_COUNT + 1))
fi

if [ $EXIT_C2 -eq 0 ]; then
    log_info "✅ C2 (heads=8, layers=5): SUCCESS"
    SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
else
    log_error "❌ C2 (heads=8, layers=5): FAILED"
    FAIL_COUNT=$((FAIL_COUNT + 1))
fi

if [ $EXIT_C1 -eq 0 ]; then
    log_info "✅ C1 (heads=12, layers=4): SUCCESS"
    SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
else
    log_error "❌ C1 (heads=12, layers=4): FAILED"
    FAIL_COUNT=$((FAIL_COUNT + 1))
fi

echo ""
log_info "Summary: ${SUCCESS_COUNT} succeeded, ${FAIL_COUNT} failed"
log_info "Results saved to: ${RESULTS_DIR}"
log_info "Logs saved to: ${OUTPUT_BASE}"
echo ""

if [ $SUCCESS_COUNT -gt 0 ]; then
    log_info "========================================="
    log_info "下一步：分析结果"
    log_info "========================================="
    log_info "Run: python analyze_stage1.py --batch 1"
    log_info ""
    log_info "This will generate a comparison table and recommend"
    log_info "which variant to use for Batch 2."
    echo ""
fi

exit 0
