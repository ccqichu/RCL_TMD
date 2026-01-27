#!/bin/bash
# =============================================================================
# 快速测试脚本 - 验证配置是否正确（只训练1个epoch）
# =============================================================================

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${YELLOW}========================================${NC}"
echo -e "${YELLOW}Quick Test - Running 1 epoch to verify setup${NC}"
echo -e "${YELLOW}========================================${NC}"
echo ""

# 检查GPU
echo -e "${GREEN}Checking GPU...${NC}"
if ! nvidia-smi &> /dev/null; then
    echo -e "${RED}❌ GPU not available${NC}"
    exit 1
fi
echo -e "${GREEN}✅ GPU available${NC}"
echo ""

# 检查CLIP模型
echo -e "${GREEN}Checking CLIP model...${NC}"
CLIP_PATH="/home/user/chengtaiyu/models/clip-vit-base-patch32"
if [ ! -d "$CLIP_PATH" ]; then
    echo -e "${RED}❌ CLIP model not found at $CLIP_PATH${NC}"
    exit 1
fi
echo -e "${GREEN}✅ CLIP model found${NC}"
echo ""

# 检查数据集
echo -e "${GREEN}Checking dataset...${NC}"
DATA_PATH="/home/user/chengtaiyu/RCLMuFN-main/data/text_final"
if [ ! -d "$DATA_PATH" ]; then
    echo -e "${RED}❌ Dataset not found at $DATA_PATH${NC}"
    exit 1
fi
echo -e "${GREEN}✅ Dataset found${NC}"
echo ""

# 创建测试输出目录
TEST_OUTPUT="../output_dir/quick_test"
mkdir -p "$TEST_OUTPUT"

echo -e "${YELLOW}Running quick test (1 epoch, aggressive config, seed=42)...${NC}"
echo ""

# 运行1个epoch的测试
python main.py \
    --device 2 \
    --model RCLMuFN \
    --text_name text_final \
    --seed 42 \
    --train_batch_size 128 \
    --dev_batch_size 128 \
    --optimizer_name adam \
    --learning_rate 1.2e-3 \
    --clip_learning_rate 1e-6 \
    --weight_decay 0.03 \
    --warmup_proportion 0.25 \
    --max_grad_norm 3.0 \
    --adam_epsilon 1e-8 \
    --num_train_epochs 1 \
    --max_len 77 \
    --text_size 512 \
    --image_size 768 \
    --layers 3 \
    --dropout_rate 0.1 \
    --label_number 2 \
    --lambda_ratio_start 0.0 \
    --lambda_ratio_end 2.5e-3 \
    --lambda_itm_start 0.0 \
    --lambda_itm_end 2e-3 \
    --lambda_warmup_epochs 2 \
    --lambda_ramp_epochs 4 \
    --lambda_schedule linear \
    --neg_sampling label_aware \
    --tau_schedule_mode epoch \
    --output_dir "$TEST_OUTPUT" \
    --results_path "${TEST_OUTPUT}/test_results.json"

exit_code=$?

echo ""
if [ $exit_code -eq 0 ]; then
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}✅ Quick test PASSED${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo ""
    echo "Your setup is ready!"
    echo "To start full training:"
    echo "  nohup bash train_batch128_all_methods.sh > train.log 2>&1 &"
    echo ""

    if [ -f "${TEST_OUTPUT}/test_results.json" ]; then
        echo "Test results:"
        cat "${TEST_OUTPUT}/test_results.json"
    fi
else
    echo -e "${RED}========================================${NC}"
    echo -e "${RED}❌ Quick test FAILED${NC}"
    echo -e "${RED}========================================${NC}"
    echo ""
    echo "Please check the error messages above and fix the issues before running full training."
fi

exit $exit_code
