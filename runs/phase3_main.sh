#!/usr/bin/env bash
# 第 3 阶段主流程（可续跑：已有 summary.json / final 的部分跳过）
cd ~/agent-rl-infra
E=".venv/bin/python -u rl/eval_rollout.py"
for r in 1 2; do
  [ -f runs/eval_base_r$r/summary.json ] || timeout -s KILL 1800 $E --out runs/eval_base_r$r > runs/eval_base_r$r.log 2>&1
done
if [ ! -d runs/main/final ]; then
  ( while true; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits >> runs/main_gpu_mem.txt; sleep 10; done ) &
  MON=$!
  timeout -s KILL 14400 .venv/bin/python -u rl/train_grpo.py --max-steps 300 --save-steps 100 --out runs/main > runs/main_train.log 2>&1
  kill $MON
fi
for ck in checkpoint-100 checkpoint-200 final; do
  [ -d runs/main/$ck ] || continue
  [ -f runs/eval_main_$ck/summary.json ] || timeout -s KILL 1800 $E --adapter runs/main/$ck --out runs/eval_main_$ck > runs/eval_main_$ck.log 2>&1
done
echo PHASE3_MAIN_DONE
