#!/bin/bash
# =============================================================================
# Dataset-Specific Training Scripts (Improved with 3 Enhancements)
# =============================================================================
# Improvements applied:
#   1. Hard Negative Mining (neg_sampling=hard_negative)
#   2. DIMM Channel-level Fusion (automatic)
#   3. Multi-layer Feature Fusion (automatic)
#
# Usage: Uncomment the dataset you want to train
# =============================================================================

# =============================================================================
# MMSD2.0 Dataset (Improved)
# =============================================================================
# Changes from original:
#   - Added CID loss parameters (lambda_ratio, lambda_itm)
#   - neg_sampling: hard_negative (⭐ Core improvement)
#   - Adjusted learning_rate: 5e-4 -> 3e-4 (more stable)
#   - weight_decay: 0.05 -> 0.03 (better for new parameters)
# =============================================================================

echo "========================================="
echo "Training on MMSD2.0 Dataset (Improved)"
echo "========================================="

python3 main.py \
    --device 0 \
    --model RCLMuFN \
    --text_name text_final \
    --num_train_epochs 20 \
    --train_batch_size 32 \
    --dev_batch_size 32 \
    --learning_rate 3e-4 \
    --clip_learning_rate 1e-6 \
    --max_len 77 \
    --layers 3 \
    --max_grad_norm 5 \
    --weight_decay 0.03 \
    --warmup_proportion 0.2 \
    --dropout_rate 0.1 \
    --optimizer_name adam \
    --text_size 512 \
    --image_size 768 \
    --lambda_ratio_start 0.0 \
    --lambda_ratio_end 2e-3 \
    --lambda_itm_start 0.0 \
    --lambda_itm_end 1.5e-3 \
    --lambda_warmup_epochs 4 \
    --lambda_ramp_epochs 6 \
    --lambda_schedule linear \
    --neg_sampling hard_negative \
    --tau_schedule_mode epoch \
    --output_dir ../output_dir/MMSD2.0_improved/ \
    --seed 42 \
    > RCLMuFN_MMSD2_improved.log 2>&1 &

echo "Training started in background (PID: $!)"
echo "Log file: RCLMuFN_MMSD2_improved.log"
echo "========================================="


# =============================================================================
# MMSD Dataset (Improved)
# =============================================================================
# Changes from original:
#   - Added CID loss parameters
#   - neg_sampling: hard_negative (⭐ Core improvement)
#   - Adjusted learning_rate: 3e-4 (keep original)
#   - clip_learning_rate: 3e-7 -> 1e-6 (more aggressive CLIP fine-tuning)
#   - weight_decay: 0.005 -> 0.03 (stronger regularization)
#   - layers: 5 -> 3 (align with model implementation)
#   - dropout_rate: 0.3 -> 0.1 (match other configs)
# =============================================================================

# Uncomment below to train on MMSD instead of MMSD2.0
# echo "========================================="
# echo "Training on MMSD Dataset (Improved)"
# echo "========================================="
#
# python3 main.py \
#     --device 0 \
#     --model RCLMuFN \
#     --text_name text_json_clean \
#     --num_train_epochs 20 \
#     --train_batch_size 32 \
#     --dev_batch_size 32 \
#     --learning_rate 3e-4 \
#     --clip_learning_rate 1e-6 \
#     --max_len 77 \
#     --layers 3 \
#     --max_grad_norm 6 \
#     --weight_decay 0.03 \
#     --warmup_proportion 0.2 \
#     --dropout_rate 0.1 \
#     --optimizer_name adam \
#     --text_size 512 \
#     --image_size 768 \
#     --lambda_ratio_start 0.0 \
#     --lambda_ratio_end 2e-3 \
#     --lambda_itm_start 0.0 \
#     --lambda_itm_end 1.5e-3 \
#     --lambda_warmup_epochs 4 \
#     --lambda_ramp_epochs 6 \
#     --lambda_schedule linear \
#     --neg_sampling hard_negative \
#     --tau_schedule_mode epoch \
#     --output_dir ../output_dir/MMSD_improved/ \
#     --seed 42 \
#     > RCLMuFN_MMSD_improved.log 2>&1 &
#
# echo "Training started in background (PID: $!)"
# echo "Log file: RCLMuFN_MMSD_improved.log"
# echo "========================================="
