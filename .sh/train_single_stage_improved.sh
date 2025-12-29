#!/bin/bash
# =============================================================================
# Single-Stage Training Script (Improved): Full Training in One Stage
# =============================================================================
# Recommended for:
#   - Quick experiments
#   - Limited time/resources
#   - When you want to skip stage 1 pretraining
#
# Changes from original:
#   - Added all CID parameters (lambda_ratio, lambda_itm, etc.)
#   - neg_sampling: hard_negative (⭐ Core improvement!)
#   - train_batch_size: 32 (for hard negative mining)
#   - Longer warmup (3 epochs instead of 2) for stability
#   - learning_rate: 3e-4 (optimized for multi-layer fusion)
# =============================================================================

echo "========================================="
echo "Starting Single-Stage Training (Improved)"
echo "========================================="
echo ""
echo "⚙️  Configuration:"
echo "  - Dataset: text_final"
echo "  - Epochs: 10"
echo "  - Batch size: 32"
echo "  - Negative sampling: hard_negative ⭐"
echo "  - CID losses: Enabled with 3-epoch warmup"
echo ""

python main.py \
    --device 1 \
    --model RCLMuFN \
    --text_name text_final \
    --num_train_epochs 10 \
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
    --optimizer_name adam \
    --text_size 512 \
    --image_size 768 \
    --lambda_ratio_start 0.0 \
    --lambda_ratio_end 2e-3 \
    --lambda_itm_start 0.0 \
    --lambda_itm_end 1.5e-3 \
    --lambda_warmup_epochs 3 \
    --lambda_ramp_epochs 4 \
    --lambda_schedule linear \
    --neg_sampling hard_negative \
    --tau_schedule_mode epoch \
    --output_dir ../output_dir/single_stage_improved/ \
    --seed 42

echo ""
echo "========================================="
echo "✅ Training Complete!"
echo "========================================="
echo "Checkpoint saved to: ../output_dir/single_stage_improved/RCLMuFN/model.pt"
echo ""
echo "📊 Check wandb for:"
echo "  - CID mask statistics (m_t_mean, m_v_mean)"
echo "  - Layer fusion weights evolution"
echo "  - Hard negative mining effects on loss_itm"
echo "========================================="
