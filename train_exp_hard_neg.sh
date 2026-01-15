#!/bin/bash
# =============================================================================
# Experiment D: Hard Negative Sampling for text_clean
# =============================================================================
# Uses low_sim negative sampling strategy
# Selects hardest negatives (most dissimilar samples)
# Better for learning discriminative features
# Optimized for text_clean dataset
# =============================================================================

echo "========================================="
echo "Experiment D: Hard Negative Sampling"
echo "========================================="

python main.py \
    --device 3 \
    --model RCLMuFN \
    --text_name text_clean \
    --num_train_epochs 15 \
    --train_batch_size 24 \
    --dev_batch_size 24 \
    --learning_rate 3e-4 \
    --clip_learning_rate 5e-6 \
    --max_len 77 \
    --layers 3 \
    --max_grad_norm 3.0 \
    --weight_decay 0.05 \
    --warmup_proportion 0.15 \
    --dropout_rate 0.15 \
    --lambda_ratio_start 0.0 \
    --lambda_ratio_end 5e-3 \
    --lambda_itm_start 0.0 \
    --lambda_itm_end 3e-3 \
    --lambda_warmup_epochs 3 \
    --lambda_ramp_epochs 5 \
    --lambda_schedule linear \
    --neg_sampling low_sim \
    --tau_schedule_mode epoch \
    --output_dir ../output_dir/text_clean_hard_neg/ \
    --seed 42

echo "========================================="
echo "Experiment D Complete!"
echo "Changes: low_sim negative sampling (hardest negatives)"
echo "========================================="
