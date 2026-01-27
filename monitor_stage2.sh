#!/bin/bash
# =============================================================================
# 阶段2实时监控脚本
# =============================================================================

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

OUTPUT_BASE="../output_dir/stage2_lr_search"
RESULTS_DIR="../output_dir/stage2_results"

clear

echo -e "${BOLD}${CYAN}"
echo "============================================================"
echo "阶段2 - Step 2.1：学习率搜索监控"
echo "============================================================"
echo -e "${NC}"

# GPU状态
echo -e "${BOLD}${BLUE}[1] GPU状态${NC}"
echo "------------------------------------------------------------"
nvidia-smi --query-gpu=index,name,temperature.gpu,utilization.gpu,utilization.memory,memory.used,memory.total --format=csv,noheader,nounits | awk -F', ' '{
    gpu_id=$1
    name=$2
    temp=$3
    gpu_util=$4
    mem_util=$5
    mem_used=$6
    mem_total=$7

    if (temp < 70) temp_color="\033[0;32m"
    else if (temp < 80) temp_color="\033[1;33m"
    else temp_color="\033[0;31m"

    if (gpu_util > 80) gpu_color="\033[0;32m"
    else if (gpu_util > 50) gpu_color="\033[1;33m"
    else gpu_color="\033[0;31m"

    printf "GPU %d | %s | Temp: %s%3d°C\033[0m | GPU: %s%3d%%\033[0m | MEM: %3d%% (%5d/%5d MB)\n",
           gpu_id, name, temp_color, temp, gpu_color, gpu_util, mem_util, mem_used, mem_total
}'
echo ""

# 训练进度
echo -e "${BOLD}${BLUE}[2] 训练进度${NC}"
echo "------------------------------------------------------------"

CONFIGS=("lr_1.3e-3_seed42" "lr_1.4e-3_seed42" "lr_1.5e-3_seed42" "lr_1.6e-3_seed42" "lr_1.7e-3_seed42")
COMPLETED=0
RUNNING=0
FAILED=0

for config in "${CONFIGS[@]}"; do
    RESULT_FILE="${RESULTS_DIR}/${config}.json"
    LOG_FILE="${OUTPUT_BASE}/${config}/training.log"

    if [ -f "$RESULT_FILE" ]; then
        # 已完成
        ACC=$(python3 -c "import json; d=json.load(open('$RESULT_FILE')); print(f\"{d['best_test']['best_test_acc']:.4f}\")" 2>/dev/null || echo "N/A")
        F1=$(python3 -c "import json; d=json.load(open('$RESULT_FILE')); print(f\"{d['best_test']['macro_test_f1']:.4f}\")" 2>/dev/null || echo "N/A")
        EPOCH=$(python3 -c "import json; d=json.load(open('$RESULT_FILE')); print(d['best_test']['best_epoch'])" 2>/dev/null || echo "N/A")

        echo -e "${GREEN}✓ ${config}${NC} | COMPLETED | Acc: ${ACC} | F1: ${F1} | Epoch: ${EPOCH}"
        COMPLETED=$((COMPLETED + 1))

    elif [ -f "$LOG_FILE" ]; then
        # 运行中
        LAST_LINE=$(tail -n 20 "$LOG_FILE" | grep -E "Epoch|epoch" | tail -n 1)

        if [ -z "$LAST_LINE" ]; then
            echo -e "${YELLOW}⟳ ${config}${NC} | RUNNING | Initializing..."
        else
            CURRENT_EPOCH=$(echo "$LAST_LINE" | grep -oP '(?<=Epoch )\d+|(?<=epoch )\d+' | head -n 1)
            if [ -z "$CURRENT_EPOCH" ]; then
                CURRENT_EPOCH="?"
            fi

            LOSS=$(echo "$LAST_LINE" | grep -oP 'loss[=: ]+\K[0-9.]+' | head -n 1)
            if [ -z "$LOSS" ]; then
                LOSS="--"
            fi

            ACC=$(echo "$LAST_LINE" | grep -oP 'acc[uracy]*[=: ]+\K[0-9.]+' | head -n 1)
            if [ -z "$ACC" ]; then
                ACC="--"
            fi

            echo -e "${YELLOW}⟳ ${config}${NC} | RUNNING | Epoch: ${CURRENT_EPOCH}/15 | Loss: ${LOSS} | Acc: ${ACC}"
        fi
        RUNNING=$((RUNNING + 1))

    else
        echo -e "${RED}✗ ${config}${NC} | NOT STARTED / FAILED"
        FAILED=$((FAILED + 1))
    fi
done

echo ""
echo -e "进度: ${GREEN}${COMPLETED}/5 已完成${NC} | ${YELLOW}${RUNNING} 运行中${NC} | ${RED}${FAILED} 未开始/失败${NC}"
echo ""

# 最新日志
if [ $RUNNING -gt 0 ]; then
    echo -e "${BOLD}${BLUE}[3] 最新训练日志${NC}"
    echo "------------------------------------------------------------"

    for config in "${CONFIGS[@]}"; do
        LOG_FILE="${OUTPUT_BASE}/${config}/training.log"

        if [ -f "$LOG_FILE" ]; then
            # 检查是否正在运行
            RESULT_FILE="${RESULTS_DIR}/${config}.json"
            if [ ! -f "$RESULT_FILE" ]; then
                echo -e "${CYAN}>>> ${config} 最新3行:${NC}"
                tail -n 3 "$LOG_FILE" | sed 's/^/    /'
                echo ""
            fi
        fi
    done
fi

# 预计完成时间
echo -e "${BOLD}${BLUE}[4] 预计完成时间${NC}"
echo "------------------------------------------------------------"

if [ $RUNNING -gt 0 ]; then
    # 估算剩余时间
    TOTAL_EXPERIMENTS=5
    AVG_TIME_PER_EXP=150  # 2.5小时 = 150分钟

    # 批次1：前3个并行（2.5小时）
    # 批次2：后2个并行（2.5小时）
    # 总计：5小时

    if [ $COMPLETED -lt 3 ]; then
        echo -e "批次1运行中（lr=1.3e-3, 1.4e-3, 1.5e-3）"
        echo -e "预计批次1还需: ${YELLOW}~2.5小时${NC}"
        echo -e "预计总时间: ~5小时（2批次）"
    elif [ $COMPLETED -ge 3 ] && [ $COMPLETED -lt 5 ]; then
        echo -e "批次2运行中（lr=1.6e-3, 1.7e-3）"
        echo -e "预计批次2还需: ${YELLOW}~2.5小时${NC}"
    fi
else
    if [ $COMPLETED -eq 5 ]; then
        echo -e "${GREEN}✅ Step 2.1已全部完成！${NC}"
        echo -e "\n下一步: python analyze_stage2_step1.py"
    else
        echo -e "${YELLOW}没有正在运行的任务${NC}"
    fi
fi

echo ""
echo "============================================================"
echo -e "${CYAN}刷新: 每10秒自动更新 (Ctrl+C退出)${NC}"
echo -e "${CYAN}手动刷新: watch -n 10 bash monitor_stage2.sh${NC}"
echo "============================================================"
