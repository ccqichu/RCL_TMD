#!/bin/bash
# =============================================================================
# Stage 2 Training Script: Full CID-DIMM Training
# =============================================================================
# Goal: Complete training with CID losses and CLIP fine-tuning
#       - Resume from Stage 1 checkpoint
#       - Unfreeze CLIP for fine-tuning
#       - Enable CID losses (L_ratio, L_itm) with gradual warmup
#       - Lambda weights ramp from 0 to target over 5 epochs (2 warmup + 3 ramp)
#       - Temperature annealing active throughout
# Duration: 8 epochs
# =============================================================================

echo "========================================="
echo "Starting Stage 2: Full CID-DIMM Training"
echo "========================================="

# Check if Stage 1 checkpoint exists
STAGE1_CHECKPOINT="../output_dir/stage1/RCLMuFN/model.pt"
if [ ! -f "$STAGE1_CHECKPOINT" ]; then
    echo "❌ Error: Stage 1 checkpoint not found at $STAGE1_CHECKPOINT"
    echo "Please run train_stage1.sh first!"
    exit 1
fi

echo "✅ Found Stage 1 checkpoint: $STAGE1_CHECKPOINT"

python main.py \
    --device 1 \
    --model RCLMuFN \
    --text_name text_final \
    --num_train_epochs 8 \
    --train_batch_size 24 \
    --dev_batch_size 24 \
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
    --neg_sampling label_aware \
    --tau_schedule_mode epoch \
    --resume_from "$STAGE1_CHECKPOINT" \
    --output_dir ../output_dir/stage2/ \
    --seed 42

echo "========================================="
echo "Stage 2 Complete!"
echo "Final checkpoint saved to: ../output_dir/stage2/RCLMuFN/model.pt"
echo "========================================="
