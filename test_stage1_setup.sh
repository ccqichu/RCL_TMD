#!/bin/bash
# =============================================================================
# 阶段1环境测试脚本 - 在正式训练前验证配置
# =============================================================================

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${YELLOW}========================================${NC}"
echo -e "${YELLOW}阶段1环境测试${NC}"
echo -e "${YELLOW}========================================${NC}"
echo ""

# 检查GPU
echo -e "${GREEN}[1/6] 检查GPU...${NC}"
if ! nvidia-smi &> /dev/null; then
    echo -e "${RED}❌ GPU不可用${NC}"
    exit 1
fi

GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
echo -e "${GREEN}✅ 发现 ${GPU_COUNT} 个GPU${NC}"

if [ $GPU_COUNT -lt 3 ]; then
    echo -e "${YELLOW}⚠️  GPU数量少于3个，批次1将无法并行${NC}"
    echo -e "${YELLOW}   建议：顺序执行或调整脚本${NC}"
fi
echo ""

# 检查CLIP模型
echo -e "${GREEN}[2/6] 检查CLIP模型...${NC}"
CLIP_PATH="/home/user/chengtaiyu/models/clip-vit-base-patch32"
if [ ! -d "$CLIP_PATH" ]; then
    echo -e "${RED}❌ CLIP模型不存在: $CLIP_PATH${NC}"
    exit 1
fi

# 检查关键文件
if [ ! -f "$CLIP_PATH/config.json" ]; then
    echo -e "${RED}❌ CLIP模型文件不完整${NC}"
    exit 1
fi

echo -e "${GREEN}✅ CLIP模型正常${NC}"
echo ""

# 检查数据集
echo -e "${GREEN}[3/6] 检查数据集...${NC}"
DATA_PATH="/home/user/chengtaiyu/RCLMuFN-main/data/text_final"
if [ ! -d "$DATA_PATH" ]; then
    echo -e "${RED}❌ 数据集不存在: $DATA_PATH${NC}"
    exit 1
fi

# 检查数据文件
if [ ! -f "$DATA_PATH/test.json" ]; then
    echo -e "${RED}❌ 测试集文件不存在${NC}"
    exit 1
fi

echo -e "${GREEN}✅ 数据集正常${NC}"
echo ""

# 检查磁盘空间
echo -e "${GREEN}[4/6] 检查磁盘空间...${NC}"
OUTPUT_DIR="/home/user/chengtaiyu/RCLMuFN-main_copy/output_dir"
AVAILABLE_GB=$(df -BG "$OUTPUT_DIR" | tail -1 | awk '{print $4}' | sed 's/G//')

if [ $AVAILABLE_GB -lt 20 ]; then
    echo -e "${YELLOW}⚠️  可用空间不足20GB (当前: ${AVAILABLE_GB}GB)${NC}"
    echo -e "${YELLOW}   建议清理磁盘空间${NC}"
else
    echo -e "${GREEN}✅ 磁盘空间充足 (${AVAILABLE_GB}GB可用)${NC}"
fi
echo ""

# 检查Python环境
echo -e "${GREEN}[5/6] 检查Python环境...${NC}"

# 检查torch
if ! python3 -c "import torch" 2>/dev/null; then
    echo -e "${RED}❌ PyTorch未安装${NC}"
    exit 1
fi

# 检查transformers
if ! python3 -c "import transformers" 2>/dev/null; then
    echo -e "${RED}❌ Transformers未安装${NC}"
    exit 1
fi

# 检查CUDA
CUDA_AVAILABLE=$(python3 -c "import torch; print(torch.cuda.is_available())")
if [ "$CUDA_AVAILABLE" != "True" ]; then
    echo -e "${RED}❌ PyTorch无法访问CUDA${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Python环境正常${NC}"
echo ""

# 快速模型测试（1个epoch）
echo -e "${GREEN}[6/6] 快速模型测试 (1个epoch)...${NC}"
echo -e "${YELLOW}这将花费约3-5分钟...${NC}"

TEST_OUTPUT="../output_dir/stage1_test"
mkdir -p "$TEST_OUTPUT"

python main.py \
    --device 0 \
    --model RCLMuFN \
    --text_name text_final \
    --seed 42 \
    --train_batch_size 32 \
    --dev_batch_size 32 \
    --optimizer_name adam \
    --learning_rate 1e-3 \
    --clip_learning_rate 1e-6 \
    --weight_decay 0.01 \
    --warmup_proportion 0.1 \
    --max_grad_norm 3.0 \
    --adam_epsilon 1e-8 \
    --num_train_epochs 1 \
    --max_len 77 \
    --text_size 512 \
    --image_size 768 \
    --layers 3 \
    --num_heads 8 \
    --dropout_rate 0.1 \
    --label_number 2 \
    --lambda_ratio_start 0.0 \
    --lambda_ratio_end 1e-3 \
    --lambda_itm_start 0.0 \
    --lambda_itm_end 1e-3 \
    --lambda_warmup_epochs 1 \
    --lambda_ramp_epochs 1 \
    --lambda_schedule linear \
    --neg_sampling label_aware \
    --tau_schedule_mode step \
    --output_dir "$TEST_OUTPUT" \
    --results_path "${TEST_OUTPUT}/test_results.json" \
    > "${TEST_OUTPUT}/test.log" 2>&1

exit_code=$?

echo ""
if [ $exit_code -eq 0 ]; then
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}✅ 所有测试通过！${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo ""
    echo -e "${GREEN}你的环境已就绪，可以开始阶段1实验！${NC}"
    echo ""
    echo -e "下一步："
    echo -e "  1. 查看执行指南: cat STAGE1_GUIDE.md"
    echo -e "  2. 启动批次1:    bash run_stage1_batch1.sh"
    echo ""

    # 显示测试结果
    if [ -f "${TEST_OUTPUT}/test_results.json" ]; then
        echo -e "快速测试结果："
        cat "${TEST_OUTPUT}/test_results.json" | python3 -m json.tool
        echo ""
    fi
else
    echo -e "${RED}========================================${NC}"
    echo -e "${RED}❌ 模型测试失败${NC}"
    echo -e "${RED}========================================${NC}"
    echo ""
    echo -e "请检查错误日志："
    echo -e "  cat ${TEST_OUTPUT}/test.log"
    echo ""
    echo -e "常见问题："
    echo -e "  1. 显存不足 → 检查GPU使用情况"
    echo -e "  2. 模型维度错误 → 检查model.py是否正确"
    echo -e "  3. 数据加载失败 → 检查data_loader.py"
    echo ""
fi

exit $exit_code
