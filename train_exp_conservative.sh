#!/bin/bash
# =============================================================================
# Experiment C: Conservative Training for text_clean
# =============================================================================
# Moderate improvements over original settings
# Use this if you want a safer approach with minimal risk
# Optimized for text_clean dataset
# =============================================================================

echo "========================================="
echo "Experiment C: Conservative Training"
echo "========================================="

python main.py \
    --device 3 \
    --model RCLMuFN \
    --text_name text_clean \
    --num_train_epochs 12 \
    --train_batch_size 24 \
    --dev_batch_size 24 \
    --learning_rate 3e-4 \
    --clip_learning_rate 3e-6 \
    --max_len 77 \
    --layers 3 \
    --max_grad_norm 3.0 \
    --weight_decay 0.05 \
    --warmup_proportion 0.15 \
    --dropout_rate 0.15 \
    --lambda_ratio_start 0.0 \
    --lambda_ratio_end 3e-3 \
    --lambda_itm_start 0.0 \
    --lambda_itm_end 2e-3 \
    --lambda_warmup_epochs 2 \
    --lambda_ramp_epochs 3 \
    --lambda_schedule linear \
    --neg_sampling label_aware \
    --tau_schedule_mode epoch \
    --output_dir ../output_dir/text_clean_conservative/ \
    --seed 42

echo "========================================="
echo "Experiment C Complete!"
echo "Changes: 1.5x CID losses, 3x CLIP LR, 12 epochs"
echo "========================================="
