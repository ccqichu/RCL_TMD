#!/bin/bash
# =============================================================================
# Stage 2 Training Script (Improved): Full CID-DIMM Training
# =============================================================================
# Changes from original:
#   - Batch size: 24 -> 32 (for better hard negative mining)
#   - neg_sampling: label_aware -> hard_negative (⭐⭐⭐ Core improvement!)
#   - learning_rate: keep 3e-4 (good for multi-layer fusion)
# =============================================================================

echo "========================================="
echo "Starting Stage 2: Full CID-DIMM Training (Improved)"
echo "========================================="

# Check if Stage 1 checkpoint exists
STAGE1_CHECKPOINT="../output_dir/stage1_improved/RCLMuFN/model.pt"
if [ ! -f "$STAGE1_CHECKPOINT" ]; then
    echo "❌ Error: Stage 1 checkpoint not found at $STAGE1_CHECKPOINT"
    echo "Please run train_stage1_improved.sh first!"
    exit 1
fi

echo "✅ Found Stage 1 checkpoint: $STAGE1_CHECKPOINT"

python main.py \
    --device 1 \
    --model RCLMuFN \
    --text_name text_final \
    --num_train_epochs 8 \
    --train_batch_size 32 \
    --dev_batch_size 32 \
    --learning_rate 3e-4 \
    --clip_learning_rate 1e-6 \
    --max_len 77 \
    --layers 3 \
    --max_grad_norm 3.0 \
    --weight_decay 0.03 \
    --warmup_proportion 0.2 \
    --dropout_rate 0.1 \
    --lambda_ratio_start 0.0 \
    --lambda_ratio_end 2e-3 \
    --lambda_itm_start 0.0 \
    --lambda_itm_end 1.5e-3 \
    --lambda_warmup_epochs 2 \
    --lambda_ramp_epochs 3 \
    --lambda_schedule linear \
    --neg_sampling hard_negative \
    --tau_schedule_mode epoch \
    --resume_from "$STAGE1_CHECKPOINT" \
    --output_dir ../output_dir/stage2_improved/ \
    --seed 42

echo "========================================="
echo "Stage 2 Complete!"
echo "Final checkpoint saved to: ../output_dir/stage2_improved/RCLMuFN/model.pt"
echo ""
echo "📊 Check wandb for:"
echo "  - CID mask statistics (m_t_mean, m_v_mean)"
echo "  - Layer fusion weights evolution"
echo "  - Hard negative mining effects on loss_itm"
echo "========================================="
