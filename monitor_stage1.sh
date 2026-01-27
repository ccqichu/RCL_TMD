#!/bin/bash
# =============================================================================
# 阶段1实时监控脚本
# =============================================================================

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

OUTPUT_BASE="../output_dir/stage1_architecture"
RESULTS_DIR="../output_dir/stage1_results"

clear

echo -e "${BOLD}${CYAN}"
echo "============================================================"
echo "阶段1 - 架构变体实验监控"
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

    # 温度颜色
    if (temp < 70) temp_color="\033[0;32m"
    else if (temp < 80) temp_color="\033[1;33m"
    else temp_color="\033[0;31m"

    # GPU利用率颜色
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

VARIANTS=("C4" "C2" "C1")
COMPLETED=0
RUNNING=0
FAILED=0

for variant in "${VARIANTS[@]}"; do
    RESULT_FILE="${RESULTS_DIR}/${variant}_seed96.json"
    LOG_FILE="${OUTPUT_BASE}/${variant}_seed96/training.log"

    if [ -f "$RESULT_FILE" ]; then
        # 已完成
        ACC=$(python3 -c "import json; d=json.load(open('$RESULT_FILE')); print(f\"{d.get('best_test_acc', 0):.4f}\")" 2>/dev/null)
        F1=$(python3 -c "import json; d=json.load(open('$RESULT_FILE')); print(f\"{d.get('macro_test_f1', 0):.4f}\")" 2>/dev/null)
        EPOCH=$(python3 -c "import json; d=json.load(open('$RESULT_FILE')); print(d.get('best_epoch', 0))" 2>/dev/null)

        echo -e "${GREEN}✓ ${variant}_seed96${NC} | COMPLETED | Acc: ${ACC} | F1: ${F1} | Epoch: ${EPOCH}"
        COMPLETED=$((COMPLETED + 1))

    elif [ -f "$LOG_FILE" ]; then
        # 运行中
        LAST_LINE=$(tail -n 20 "$LOG_FILE" | grep -E "Epoch|epoch" | tail -n 1)

        if [ -z "$LAST_LINE" ]; then
            echo -e "${YELLOW}⟳ ${variant}_seed96${NC} | RUNNING | Initializing..."
        else
            # 提取epoch信息
            CURRENT_EPOCH=$(echo "$LAST_LINE" | grep -oP '(?<=Epoch )\d+|(?<=epoch )\d+' | head -n 1)
            if [ -z "$CURRENT_EPOCH" ]; then
                CURRENT_EPOCH="?"
            fi

            # 提取loss信息
            LOSS=$(echo "$LAST_LINE" | grep -oP 'loss[=: ]+\K[0-9.]+' | head -n 1)
            if [ -z "$LOSS" ]; then
                LOSS="--"
            fi

            # 提取准确率信息
            ACC=$(echo "$LAST_LINE" | grep -oP 'acc[uracy]*[=: ]+\K[0-9.]+' | head -n 1)
            if [ -z "$ACC" ]; then
                ACC="--"
            fi

            echo -e "${YELLOW}⟳ ${variant}_seed96${NC} | RUNNING | Epoch: ${CURRENT_EPOCH}/15 | Loss: ${LOSS} | Acc: ${ACC}"
        fi
        RUNNING=$((RUNNING + 1))

    else
        # 未开始或失败
        echo -e "${RED}✗ ${variant}_seed96${NC} | NOT STARTED / FAILED"
        FAILED=$((FAILED + 1))
    fi
done

echo ""
echo -e "进度: ${GREEN}${COMPLETED} 已完成${NC} | ${YELLOW}${RUNNING} 运行中${NC} | ${RED}${FAILED} 未开始/失败${NC}"
echo ""

# 最新日志
echo -e "${BOLD}${BLUE}[3] 最新训练日志${NC}"
echo "------------------------------------------------------------"

for variant in "${VARIANTS[@]}"; do
    LOG_FILE="${OUTPUT_BASE}/${variant}_seed96/training.log"

    if [ -f "$LOG_FILE" ]; then
        echo -e "${CYAN}>>> ${variant}_seed96 最新3行:${NC}"
        tail -n 3 "$LOG_FILE" | sed 's/^/    /'
        echo ""
    fi
done

# 预计完成时间
echo -e "${BOLD}${BLUE}[4] 预计完成时间${NC}"
echo "------------------------------------------------------------"

if [ $RUNNING -gt 0 ]; then
    # 检查第一个运行中的任务开始时间
    for variant in "${VARIANTS[@]}"; do
        LOG_FILE="${OUTPUT_BASE}/${variant}_seed96/training.log"

        if [ -f "$LOG_FILE" ]; then
            START_TIME=$(stat -c %Y "$LOG_FILE" 2>/dev/null || stat -f %m "$LOG_FILE" 2>/dev/null)
            CURRENT_TIME=$(date +%s)
            ELAPSED=$((CURRENT_TIME - START_TIME))

            ELAPSED_MIN=$((ELAPSED / 60))

            # 估计每个任务2.5小时
            TOTAL_MIN=150
            REMAINING_MIN=$((TOTAL_MIN - ELAPSED_MIN))

            if [ $REMAINING_MIN -gt 0 ]; then
                HOURS=$((REMAINING_MIN / 60))
                MINS=$((REMAINING_MIN % 60))
                echo -e "运行中的任务预计还需: ${YELLOW}${HOURS}小时 ${MINS}分钟${NC}"
            else
                echo -e "${GREEN}任务即将完成...${NC}"
            fi

            break
        fi
    done

    if [ $COMPLETED -lt 3 ]; then
        echo -e "批次1预计总时间: ~2.5小时 (3个并行)"
    fi
else
    if [ $COMPLETED -eq 3 ]; then
        echo -e "${GREEN}✅ 批次1已全部完成！${NC}"
        echo -e "\n下一步: python analyze_stage1.py --batch 1"
    else
        echo -e "${YELLOW}没有正在运行的任务${NC}"
    fi
fi

echo ""
echo "============================================================"
echo -e "${CYAN}刷新: 每10秒自动更新 (Ctrl+C退出)${NC}"
echo -e "${CYAN}手动刷新: watch -n 10 bash monitor_stage1.sh${NC}"
echo "============================================================"
