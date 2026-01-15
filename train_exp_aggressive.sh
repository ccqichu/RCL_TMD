#!/bin/bash
# =============================================================================
# Experiment B: Aggressive Hyperparameters for MMSD
# =============================================================================
# Stronger CID supervision with higher learning rates
# Use this if the baseline (train_single_stage.sh) shows underfitting
# =============================================================================

echo "========================================="
echo "Experiment B: Aggressive Training"
echo "========================================="

python main.py \
    --device 1 \
    --model RCLMuFN \
    --text_name text_clean \
    --num_train_epochs 12 \
    --train_batch_size 24 \
    --dev_batch_size 24 \
    --learning_rate 3e-4 \
    --clip_learning_rate 1e-5 \
    --max_len 77 \
    --layers 3 \
    --max_grad_norm 3.0 \
    --weight_decay 0.03 \
    --warmup_proportion 0.2 \
    --dropout_rate 0.1 \
    --lambda_ratio_start 0.0 \
    --lambda_ratio_end 1e-2 \
    --lambda_itm_start 0.0 \
    --lambda_itm_end 5e-3 \
    --lambda_warmup_epochs 2 \
    --lambda_ramp_epochs 4 \
    --lambda_schedule cosine \
    --neg_sampling label_aware \
    --tau_schedule_mode step \
    --output_dir ../output_dir/exp_aggressive/ \
    --seed 42

echo "========================================="
echo "Experiment B Complete!"
echo "Changes: 10x CID losses, 10x CLIP LR, cosine schedule, step-based tau"
echo "========================================="
