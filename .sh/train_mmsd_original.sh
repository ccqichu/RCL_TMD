#!/bin/bash
# =============================================================================
# MMSD Original Dataset Training Script (Improved)
# =============================================================================
# This script is specifically optimized for the MMSD (text_clean) dataset,
# which is larger than MMSD2.0 and requires slightly different hyperparameters.
# =============================================================================

echo "========================================="
echo "Training on MMSD Dataset (Improved)"
echo "========================================="

python3 main.py \
    --device 0 \
    --model RCLMuFN \
    --text_name text_clean \
    --num_train_epochs 20 \
    --train_batch_size 32 \
    --dev_batch_size 32 \
    --learning_rate 3e-4 \
    --clip_learning_rate 1e-6 \
    --max_len 77 \
    --layers 3 \
    --max_grad_norm 6 \
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
    --output_dir ../output_dir/MMSD_improved/ \
    --seed 42 \
    > RCLMuFN_MMSD_improved.log 2>&1 &

echo "Training started in background (PID: $!)"
echo "Log file: RCLMuFN_MMSD_improved.log"
echo ""
echo "📊 Dataset: MMSD (text_clean)"
echo "📦 Training samples: ~larger dataset"
echo "⏱️  Expected time: ~20 epochs"
echo "========================================="
