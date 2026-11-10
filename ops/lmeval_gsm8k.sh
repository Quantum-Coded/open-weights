#!/usr/bin/env bash
# Pre-registered cross-check (eval_plan.md, "Cross-checks"): lm-evaluation-harness on GSM8K, base vs run-1 edit.
# vLLM backend, model chat template, reasoning stripped up to </think>, max tokens = token cap (16,384), greedy.
# Waits for the Amendment 3 runner; the GPU holds one job. Runs with no internet (data cached on 8 Nov).
# --limit 400 (first 400 GSM8K test problems): fixed 9 Nov before any lm-eval accuracy was seen (RUNLOG).
set -u
while pgrep -f "a3.sh|fresh.sh" > /dev/null; do sleep 60; done
cd ~/wsp-r1 && source .venv/bin/activate
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1 VLLM_BATCH_INVARIANT=1
mkdir -p results/lmeval
for arm in edit_r2 base edit; do   # edit_r2 added 10 Nov (before any lm-eval result), run first
  if ls results/lmeval/$arm/*/results_*.json > /dev/null 2>&1; then echo "SKIP lmeval $arm"; continue; fi
  echo "START lmeval $arm $(date -Is)"
  ( time lm_eval --model vllm \
      --model_args "pretrained=work/$arm,dtype=bfloat16,max_model_len=17408,gpu_memory_utilization=0.80,think_end_token=</think>,seed=0,attention_backend=FLASH_ATTN" \
      --tasks gsm8k_cot_zeroshot --apply_chat_template --batch_size auto \
      --gen_kwargs "max_gen_toks=16384,do_sample=False,temperature=0" \
      --log_samples --output_path results/lmeval/$arm --seed 0 --limit 400 ) > logs_lmeval_$arm.txt 2>&1
  echo "END lmeval $arm rc=$? $(date -Is)"
  grep -E "exact_match|gsm8k" logs_lmeval_$arm.txt | tail -4
done
echo "LMEVAL DONE $(date -Is)"
