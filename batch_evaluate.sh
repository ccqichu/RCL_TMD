#!/bin/bash
# =============================================================================
# Batch Model Evaluation Script
# =============================================================================
# Automatically evaluates all trained model checkpoints
# Generates comparative analysis reports
# =============================================================================

echo "========================================="
echo "Batch Model Evaluation"
echo "========================================="

# Configuration
OUTPUT_DIR="../output_dir"
EVAL_RESULTS_DIR="./eval_results"
TEXT_NAME="text_clean"
DEVICE="0"

# Create evaluation results directory
mkdir -p "$EVAL_RESULTS_DIR"

# List of experiment directories to evaluate
EXPERIMENTS=(
    "text_clean_baseline"
    "text_clean_aggressive"
    "text_clean_conservative"
    "text_clean_hard_neg"
    "exp_aggressive"  # If you ran the old version
)

echo ""
echo "Searching for trained models..."
echo "Output directory: $OUTPUT_DIR"
echo "Text dataset: $TEXT_NAME"
echo ""

# Counter for found models
FOUND_MODELS=0
EVALUATED_MODELS=0

# Evaluate each experiment
for exp in "${EXPERIMENTS[@]}"; do
    CHECKPOINT_PATH="${OUTPUT_DIR}/${exp}/RCLMuFN/model.pt"

    if [ -f "$CHECKPOINT_PATH" ]; then
        ((FOUND_MODELS++))
        echo "========================================="
        echo "[$FOUND_MODELS] Found: $exp"
        echo "========================================="
        echo "Checkpoint: $CHECKPOINT_PATH"
        echo ""

        # Run evaluation
        python evaluate.py \
            --checkpoint "$CHECKPOINT_PATH" \
            --text_name "$TEXT_NAME" \
            --device "$DEVICE" \
            --splits "test,valid" \
            --batch_size 32 \
            --output_dir "$EVAL_RESULTS_DIR" \
            --save_predictions

        if [ $? -eq 0 ]; then
            ((EVALUATED_MODELS++))
            echo "✓ Evaluation complete for $exp"
        else
            echo "❌ Evaluation failed for $exp"
        fi
        echo ""
    else
        echo "⊗ Checkpoint not found: $CHECKPOINT_PATH"
    fi
done

echo ""
echo "========================================="
echo "Batch Evaluation Summary"
echo "========================================="
echo "Models found: $FOUND_MODELS"
echo "Models evaluated: $EVALUATED_MODELS"
echo ""

if [ $EVALUATED_MODELS -eq 0 ]; then
    echo "❌ No models were evaluated!"
    echo "Please check:"
    echo "  1. Models are trained and saved in $OUTPUT_DIR"
    echo "  2. Checkpoint paths are correct"
    exit 1
fi

echo "✓ All evaluations complete!"
echo ""
echo "Results saved to: $EVAL_RESULTS_DIR"
echo ""
echo "Generated files for each experiment:"
echo "  - *_results.json       (detailed metrics)"
echo "  - *_report.txt         (human-readable report)"
echo "  - *_confusion_matrix.png (visualization)"
echo ""
echo "========================================="
