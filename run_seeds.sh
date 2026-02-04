#!/usr/bin/env bash
set -euo pipefail

SEEDS="${SEEDS:-"128 256 512 1024 2048 2025 2024 2026 3041 3041 3043 2027 2023 2020 2021"}"
EXTRA_ARGS="${EXTRA_ARGS:-""}"
OUTDIR="${OUTDIR:-"./seed_runs"}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

# 预设配置：PROFILE=final|clean|third（默认 third）
PROFILE="${PROFILE:-third}"
BASE_ARGS=()
if [ "$PROFILE" = "clean" ]; then
  BASE_ARGS=(
    --device 1
    --num_train_epochs 12
    --learning_rate 3e-4
    --clip_learning_rate 3e-7
    --weight_decay 0.005
    --max_grad_norm 1.0
    --dropout_rate 0.3
    --lambda_ratio_end 1.5e-3
    --lambda_itm_end 1.0e-3
    --lambda_warmup_epochs 3
    --lambda_ramp_epochs 6
    --lambda_schedule cosine
    --tau_schedule_mode epoch
    --use_ema
    --ema_decay 0.9999
    --ema_eval_mode both
    --lambda_chan_entropy 1e-4
  )
elif [ "$PROFILE" = "final" ]; then
  # final / default
  BASE_ARGS=(
    --device 3
    --num_train_epochs 12
    --learning_rate 3e-4
    --clip_learning_rate 1e-6
    --weight_decay 0.03
    --max_grad_norm 1.0
    --dropout_rate 0.1
    --lambda_ratio_end 2.5e-3
    --lambda_itm_end 2.0e-3
    --lambda_warmup_epochs 2
    --lambda_ramp_epochs 5
    --lambda_schedule cosine
    --tau_schedule_mode epoch
    --use_ema
    --ema_decay 0.9999
    --ema_eval_mode both
    --lambda_chan_entropy 1e-4
  )
else
  BASE_ARGS=(
  --num_train_epochs 1 \
  --lambda_ratio_start 0 \
  --lambda_itm_start 0 \
  --lambda_schedule none \
  --tau_schedule_mode step \
  --tau_decay 0.99995 \
  --neg_sampling shuffle \
  --output_dir /home/user/chengtaiyu/RCLMuFN-main_copy/output_dir/grid \
  --lambda_ratio_end 0.0001 \
  --lambda_itm_end 0.0001 \
  --tau_min 1.2 \
  --rho 0.7 \
  --rho_t 0.9 \
  --cid_smooth_beta 3.0 \
  --device 1
  )
fi

mkdir -p "$OUTDIR"
SUMMARY="$OUTDIR/seed_summary_${PROFILE}.csv"
echo "seed,dev_acc,test_acc_raw,test_acc_ema,test_acc_best,log_path" > "$SUMMARY"

for seed in $SEEDS; do
  LOG="$OUTDIR/${PROFILE}_seed_${seed}.log"
  echo "Running seed ${seed}..."
  python -u main.py --seed "${seed}" "${BASE_ARGS[@]}" ${EXTRA_ARGS} 2>&1 | tee "$LOG"

  dev_acc=$(grep -o "dev_acc is [0-9\\.]*" "$LOG" | tail -1 | awk '{print $4}')
  both_line=$(grep -o "test_acc raw [0-9\\.]*, ema [0-9\\.]*" "$LOG" | tail -1)
  if [ -n "$both_line" ]; then
    test_acc_raw=$(echo "$both_line" | awk '{print $4}' | tr -d ',')
    test_acc_ema=$(echo "$both_line" | awk '{print $6}')
  else
    test_acc_raw=$(grep -o "test_acc is [0-9\\.]*" "$LOG" | tail -1 | awk '{print $4}')
    test_acc_ema=""
  fi
  if [ -n "$test_acc_ema" ]; then
    test_acc_best=$(python - <<PY
raw = float("${test_acc_raw:-nan}")
ema = float("${test_acc_ema:-nan}")
print(max(raw, ema))
PY
)
  else
    test_acc_best="${test_acc_raw:-NA}"
  fi

  echo "${seed},${dev_acc:-NA},${test_acc_raw:-NA},${test_acc_ema:-NA},${test_acc_best:-NA},${LOG}" >> "$SUMMARY"
done

echo "Summary saved to ${SUMMARY}"
