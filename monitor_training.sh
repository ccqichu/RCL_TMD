#!/bin/bash
# =============================================================================
# 训练监控脚本
# 实时显示训练进度和GPU使用情况
# =============================================================================

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

BASE_DIR="../output_dir/batch128_experiments"
MASTER_LOG=$(ls -t ${BASE_DIR}/master_training_log_*.log 2>/dev/null | head -1)

clear

echo -e "${BLUE}╔═══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║          Batch Size 128 Training Monitor                     ║${NC}"
echo -e "${BLUE}╚═══════════════════════════════════════════════════════════════╝${NC}"
echo ""

# 检查训练是否在运行
if ! pgrep -f "python.*main.py.*batch_size.*128" > /dev/null; then
    echo -e "${RED}⚠️  No training process detected${NC}"
    echo ""
    echo "To start training, run:"
    echo "  nohup bash train_batch128_all_methods.sh > train.log 2>&1 &"
    echo ""
    exit 0
fi

echo -e "${GREEN}✅ Training is running${NC}"
echo ""

# 显示GPU状态
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${YELLOW}GPU Status${NC}"
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits | \
while IFS=, read -r idx name util mem_used mem_total temp; do
    util_bar=$(printf "%-20s" "$(printf '#%.0s' $(seq 1 $((util/5))))")
    mem_pct=$((mem_used * 100 / mem_total))
    mem_bar=$(printf "%-20s" "$(printf '#%.0s' $(seq 1 $((mem_pct/5))))")

    echo -e "GPU $idx: ${name}"
    echo -e "  Utilization: [${util_bar}] ${util}%"
    echo -e "  Memory:      [${mem_bar}] ${mem_used}/${mem_total}MB (${mem_pct}%)"
    echo -e "  Temperature: ${temp}°C"
    echo ""
done

# 显示当前训练进度
if [ -f "$MASTER_LOG" ]; then
    echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${YELLOW}Training Progress (Last 15 lines)${NC}"
    echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    tail -15 "$MASTER_LOG" | grep -E "(Method:|Seed:|Epoch|loss|acc|f1)" --color=never
    echo ""
fi

# 显示完成的实验
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${YELLOW}Completed Experiments${NC}"
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

RESULTS_DIR="../output_dir/seed_runs"
completed=0
failed=0

for method in conservative aggressive extreme; do
    for seed in 42 96 100; do
        result_file="${RESULTS_DIR}/${method}_seed${seed}.json"
        if [ -f "$result_file" ]; then
            acc=$(python3 -c "import json; print(json.load(open('$result_file'))['best_test']['best_test_acc'])" 2>/dev/null)
            if [ -n "$acc" ]; then
                echo -e "${GREEN}✓${NC} ${method}_seed${seed}: Acc=${acc}"
                completed=$((completed + 1))
            else
                echo -e "${RED}✗${NC} ${method}_seed${seed}: Parse error"
                failed=$((failed + 1))
            fi
        fi
    done
done

echo ""
echo -e "Completed: ${GREEN}${completed}${NC} | Failed: ${RED}${failed}${NC} | Remaining: $((9 - completed - failed))"
echo ""

# 显示估计剩余时间
if [ -f "$MASTER_LOG" ] && [ $completed -gt 0 ]; then
    start_time=$(head -20 "$MASTER_LOG" | grep -oP "\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}" | head -1)
    if [ -n "$start_time" ]; then
        start_epoch=$(date -d "$start_time" +%s 2>/dev/null)
        current_epoch=$(date +%s)
        elapsed=$((current_epoch - start_epoch))
        avg_time_per_run=$((elapsed / completed))
        remaining_runs=$((9 - completed - failed))
        estimated_remaining=$((avg_time_per_run * remaining_runs))

        hours=$((estimated_remaining / 3600))
        minutes=$(((estimated_remaining % 3600) / 60))

        echo -e "${BLUE}Estimated time remaining: ${hours}h ${minutes}m${NC}"
    fi
fi

echo ""
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo "Press Ctrl+C to exit monitor"
echo "To follow master log: tail -f $MASTER_LOG"
echo "To analyze results: python analyze_batch128_results.py"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
