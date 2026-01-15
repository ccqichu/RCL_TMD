#!/bin/bash
# =============================================================================
# Quick Evaluation Example
# =============================================================================
# This script demonstrates how to evaluate a single model
# Useful for testing if the evaluation pipeline works
# =============================================================================

echo "========================================="
echo "Quick Evaluation Example"
echo "========================================="

# Check if any model checkpoint exists
CHECKPOINT_DIRS=(
    "../output_dir/text_clean_baseline/RCLMuFN"
    "../output_dir/text_clean_aggressive/RCLMuFN"
    "../output_dir/text_clean_conservative/RCLMuFN"
    "../output_dir/stage2/RCLMuFN"
)

FOUND_CHECKPOINT=""

echo "Searching for trained models..."
for dir in "${CHECKPOINT_DIRS[@]}"; do
    if [ -f "$dir/model.pt" ]; then
        FOUND_CHECKPOINT="$dir/model.pt"
        echo "✓ Found checkpoint: $FOUND_CHECKPOINT"
        break
    fi
done

if [ -z "$FOUND_CHECKPOINT" ]; then
    echo ""
    echo "❌ No trained model found!"
    echo ""
    echo "Please train a model first using one of:"
    echo "  bash train_single_stage.sh"
    echo "  bash train_exp_aggressive.sh"
    echo "  bash train_exp_conservative.sh"
    echo ""
    exit 1
fi

echo ""
echo "Running evaluation on test set..."
echo "========================================="
echo ""

python evaluate.py \
    --checkpoint "$FOUND_CHECKPOINT" \
    --text_name text_clean \
    --device 0 \
    --splits test \
    --batch_size 32 \
    --output_dir ./eval_results

if [ $? -eq 0 ]; then
    echo ""
    echo "========================================="
    echo "✓ Evaluation complete!"
    echo "========================================="
    echo ""
    echo "Results saved to: ./eval_results/"
    echo ""
    echo "Next steps:"
    echo "  1. View the report: cat eval_results/*_report.txt"
    echo "  2. Run batch evaluation: bash batch_evaluate.sh"
    echo "  3. Compare all models: python compare_results.py"
    echo ""
else
    echo ""
    echo "❌ Evaluation failed!"
    echo "Please check the error messages above."
    echo ""
fi
