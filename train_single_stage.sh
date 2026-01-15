#!/bin/bash
# =============================================================================
# Single-Stage Training Script: Optimized for text_clean Dataset
# =============================================================================
# Goal: Train the complete model from scratch in one go
#       - CLIP fine-tuning from the beginning (no freezing)
#       - CID losses with gradual warmup and ramp
#       - Optimized hyperparameters for text_clean (longer text, class imbalance)
#       - Enhanced regularization for longer text (+23% avg length)
# Duration: 15 epochs (~20-25 hours on single GPU)
# =============================================================================

echo "========================================="
echo "Single-Stage Training for text_clean"
echo "========================================="
echo "Dataset: text_clean (19,557 samples, avg_len=79.9)"
echo "Class imbalance: hate=42.9%, non-hate=57.1%"
echo "Training Duration: 15 epochs"
echo "========================================="

python main.py \
    --device 0 \
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
    --neg_sampling label_aware \
    --tau_schedule_mode epoch \
    --output_dir ../output_dir/text_clean_baseline/ \
    --seed 42

echo "========================================="
echo "Training Complete!"
echo "Model saved to: ../output_dir/text_clean_baseline/RCLMuFN/model.pt"
echo "========================================="
echo ""
echo "Optimizations for text_clean:"
echo "  ✓ Dataset: text_clean (longer text, 79.9 avg chars)"
echo "  ✓ Dropout: 0.1 → 0.15 (prevent overfitting on long text)"
echo "  ✓ Weight decay: 0.03 → 0.05 (stronger regularization)"
echo "  ✓ Warmup: 0.2 → 0.15 (faster training start)"
echo "  ✓ Epochs: 12 → 15 (more complex text needs longer training)"
echo "  ✓ CID targets: rho=0.4, rho_t=0.55 (adapted for longer text)"
echo "========================================="
