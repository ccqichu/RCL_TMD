#!/bin/bash
# =============================================================================
# 阶段2 - Step 2.1：学习率网格搜索
# =============================================================================
# 架构：C4 (heads=12, layers=5)
# 固定：lambda_ratio=3.0e-3, lambda_itm=2.5e-3
# 搜索：lr = [1.3e-3, 1.4e-3, 1.5e-3, 1.6e-3, 1.7e-3]
# Seed：42
# =============================================================================

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

BASE_DIR="/home/user/chengtaiyu/RCLMuFN-main_copy"
SRC_DIR="${BASE_DIR}/src"
OUTPUT_BASE="${BASE_DIR}/output_dir/stage2_lr_search"
RESULTS_DIR="${BASE_DIR}/output_dir/stage2_results"

mkdir -p "${OUTPUT_BASE}"
mkdir -p "${RESULTS_DIR}"

# GPU分配（3个并行 + 2个顺序）
GPUS=(1 2 3)
SEED=42

# C4架构配置
NUM_HEADS=12
LAYERS=5
BATCH_SIZE=128

# 固定的lambda参数
LAMBDA_RATIO=3.0e-3
LAMBDA_ITM=2.5e-3

# 学习率搜索空间
LR_VALUES=(1.3e-3 1.4e-3 1.5e-3 1.6e-3 1.7e-3)

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${OUTPUT_BASE}/step1_master_${TIMESTAMP}.log"

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
train_config() {
    local config_name=$1
    local gpu_id=$2
    local lr=$3

    local output_dir="${OUTPUT_BASE}/${config_name}"
    local result_file="${RESULTS_DIR}/${config_name}.json"
    local train_log="${output_dir}/training.log"

    mkdir -p "${output_dir}"

    log_info "Starting ${config_name} on GPU ${gpu_id}"
    log_info "  LR=${lr}, Lambda_ratio=${LAMBDA_RATIO}, Lambda_itm=${LAMBDA_ITM}"

    # 运行训练
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=${gpu_id} \
    python "${SRC_DIR}/main.py" \
        --device 0 \
        --model RCLMuFN \
        --text_name text_final \
        --seed ${SEED} \
        --train_batch_size ${BATCH_SIZE} \
        --dev_batch_size ${BATCH_SIZE} \
        --optimizer_name adam \
        --learning_rate ${lr} \
        --clip_learning_rate 2e-6 \
        --weight_decay 0.025 \
        --warmup_proportion 0.3 \
        --max_grad_norm 5.0 \
        --adam_epsilon 1e-8 \
        --num_train_epochs 15 \
        --max_len 77 \
        --text_size 512 \
        --image_size 768 \
        --layers ${LAYERS} \
        --num_heads ${NUM_HEADS} \
        --dropout_rate 0.15 \
        --label_number 2 \
        --lambda_ratio_start 0.0 \
        --lambda_ratio_end ${LAMBDA_RATIO} \
        --lambda_itm_start 0.0 \
        --lambda_itm_end ${LAMBDA_ITM} \
        --lambda_warmup_epochs 3 \
        --lambda_ramp_epochs 5 \
        --lambda_schedule cosine \
        --neg_sampling label_aware \
        --tau_schedule_mode epoch \
        --tau_min 0.4 \
        --tau_decay 0.9995 \
        --output_dir "${output_dir}" \
        --results_path "${result_file}" \
        > "${train_log}" 2>&1

    local exit_code=$?

    if [ $exit_code -eq 0 ]; then
        log_info "✅ ${config_name} completed successfully"

        if [ -f "${result_file}" ]; then
            local best_acc=$(python3 -c "import json; d=json.load(open('${result_file}')); print(f\"{d['best_test']['best_test_acc']:.4f}\")" 2>/dev/null || echo "N/A")
            local best_f1=$(python3 -c "import json; d=json.load(open('${result_file}')); print(f\"{d['best_test']['macro_test_f1']:.4f}\")" 2>/dev/null || echo "N/A")
            log_info "  Best Acc: ${best_acc}, Macro F1: ${best_f1}"
        fi
    else
        log_error "❌ ${config_name} failed with exit code ${exit_code}"
        log_error "  Check log: ${train_log}"
    fi

    return $exit_code
}

# =============================================================================
# 主执行流程
# =============================================================================
echo ""
log_info "========================================="
log_info "阶段2 - Step 2.1：学习率搜索"
log_info "========================================="
log_info "架构: C4 (heads=${NUM_HEADS}, layers=${LAYERS})"
log_info "Batch size: ${BATCH_SIZE}"
log_info "Seed: ${SEED}"
log_info "固定参数: λ_ratio=${LAMBDA_RATIO}, λ_itm=${LAMBDA_ITM}"
log_info "搜索范围: LR=${LR_VALUES[@]}"
log_info "开始时间: $(date)"
echo ""

# 检查GPU
if ! nvidia-smi &> /dev/null; then
    log_error "GPU不可用！"
    exit 1
fi
log_info "✓ GPU可用"

# 检查CLIP模型
CLIP_PATH="/home/user/chengtaiyu/models/clip-vit-base-patch32"
if [ ! -d "$CLIP_PATH" ]; then
    log_error "CLIP模型不存在: $CLIP_PATH"
    exit 1
fi
log_info "✓ CLIP模型已找到"

echo ""
log_info "-----------------------------------------"
log_info "开始5个学习率实验（3并行）..."
log_info "-----------------------------------------"

# 批次1：前3个（并行）
log_info ""
log_info "批次1: lr=1.3e-3, 1.4e-3, 1.5e-3 (并行)"

train_config "lr_1.3e-3_seed42" ${GPUS[0]} 1.3e-3 &
PID_1=$!
sleep 5

train_config "lr_1.4e-3_seed42" ${GPUS[1]} 1.4e-3 &
PID_2=$!
sleep 5

train_config "lr_1.5e-3_seed42" ${GPUS[2]} 1.5e-3 &
PID_3=$!

log_info "等待批次1完成..."
wait $PID_1
EXIT_1=$?
wait $PID_2
EXIT_2=$?
wait $PID_3
EXIT_3=$?

log_info "批次1完成"

# 批次2：后2个（并行）
log_info ""
log_info "批次2: lr=1.6e-3, 1.7e-3 (并行)"

train_config "lr_1.6e-3_seed42" ${GPUS[0]} 1.6e-3 &
PID_4=$!
sleep 5

train_config "lr_1.7e-3_seed42" ${GPUS[1]} 1.7e-3 &
PID_5=$!

log_info "等待批次2完成..."
wait $PID_4
EXIT_4=$?
wait $PID_5
EXIT_5=$?

log_info "批次2完成"

# =============================================================================
# 汇总结果
# =============================================================================
echo ""
log_info "========================================="
log_info "Step 2.1 完成"
log_info "========================================="
log_info "结束时间: $(date)"
echo ""

SUCCESS_COUNT=0
FAIL_COUNT=0

declare -a EXIT_CODES=($EXIT_1 $EXIT_2 $EXIT_3 $EXIT_4 $EXIT_5)
declare -a CONFIG_NAMES=("lr_1.3e-3" "lr_1.4e-3" "lr_1.5e-3" "lr_1.6e-3" "lr_1.7e-3")

for i in "${!EXIT_CODES[@]}"; do
    if [ ${EXIT_CODES[$i]} -eq 0 ]; then
        log_info "✅ ${CONFIG_NAMES[$i]}: SUCCESS"
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
    else
        log_error "❌ ${CONFIG_NAMES[$i]}: FAILED"
        FAIL_COUNT=$((FAIL_COUNT + 1))
    fi
done

echo ""
log_info "总结: ${SUCCESS_COUNT} 成功, ${FAIL_COUNT} 失败"
echo ""

if [ $SUCCESS_COUNT -gt 0 ]; then
    log_info "========================================="
    log_info "下一步：分析结果"
    log_info "========================================="
    log_info "运行: python analyze_stage2_step1.py"
    log_info ""
    log_info "这将自动："
    log_info "  1. 对比5个学习率的性能"
    log_info "  2. 推荐最佳学习率"
    log_info "  3. 生成Step 2.2的lambda搜索脚本"
    echo ""
fi

exit 0
