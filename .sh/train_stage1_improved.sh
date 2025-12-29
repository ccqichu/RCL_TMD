#!/bin/bash
# =============================================================================
# Stage 1 Training Script (Improved): CID-DIMM Pretraining
# =============================================================================
# Changes from original:
#   - Batch size: 24 -> 32 (prepare for hard negative mining in Stage 2)
#   - Keep neg_sampling as 'shuffle' (simple strategy for stage 1)
# =============================================================================

echo "========================================="
echo "Starting Stage 1: CID-DIMM Pretraining (Improved)"
echo "========================================="

python main.py \
    --device 1 \
    --model RCLMuFN \
    --text_name text_final \
    --num_train_epochs 2 \
    --train_batch_size 32 \
    --dev_batch_size 32 \
    --learning_rate 1e-4 \
    --clip_learning_rate 0 \
    --max_len 77 \
    --layers 3 \
    --max_grad_norm 3.0 \
    --weight_decay 0.03 \
    --warmup_proportion 0.2 \
    --dropout_rate 0.1 \
    --lambda_ratio_start 0.0 \
    --lambda_ratio_end 0.0 \
    --lambda_itm_start 0.0 \
    --lambda_itm_end 0.0 \
    --lambda_warmup_epochs 0 \
    --lambda_ramp_epochs 0 \
    --lambda_schedule none \
    --neg_sampling shuffle \
    --tau_schedule_mode epoch \
    --freeze_clip \
    --output_dir ../output_dir/stage1_improved/ \
    --seed 42

echo "========================================="
echo "Stage 1 Complete!"
echo "Checkpoint saved to: ../output_dir/stage1_improved/RCLMuFN/model.pt"
echo "========================================="
