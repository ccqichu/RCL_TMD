#!/bin/bash
# =============================================================================
# Stage 1 Training Script: CID-DIMM Pretraining
# =============================================================================
# Goal: Train basic fusion and classification with frozen CLIP
#       - CLIP frozen to save memory and computation
#       - No CID losses (L_ratio, L_itm) to focus on basic fusion
#       - alpha grows naturally from 0 but without CID supervision
#       - Model learns basic multimodal fusion patterns
# Duration: 2 epochs
# =============================================================================

echo "========================================="
echo "Starting Stage 1: CID-DIMM Pretraining"
echo "========================================="

python main.py \
    --device 1 \
    --model RCLMuFN \
    --text_name text_final \
    --num_train_epochs 2 \
    --train_batch_size 24 \
    --dev_batch_size 24 \
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
    --neg_sampling label_aware \
    --tau_schedule_mode epoch \
    --freeze_clip \
    --output_dir ../output_dir/stage1/ \
    --seed 42

echo "========================================="
echo "Stage 1 Complete!"
echo "Checkpoint saved to: ../output_dir/stage1/RCLMuFN/model.pt"
echo "========================================="
