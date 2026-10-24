#!/usr/bin/env bash
# 重复实验：未优化 / 优化版各用种子 1 训练 200 步并评测（可续跑）
cd ~/agent-rl-infra
for cfg in "control200_s1:off" "fused200_s1:on"; do
  name=${cfg%%:*}; mode=${cfg#*:}
  if [ ! -d runs/$name/final ]; then
    rm -rf runs/$name
    timeout -s KILL 10800 .venv/bin/python -u rl/train_grpo.py --max-steps 200 --save-steps 200 --seed 1 --fused-is $mode --out runs/$name > runs/${name}_train.log 2>&1
  fi
  [ -f runs/eval_$name/summary.json ] || timeout -s KILL 1800 .venv/bin/python -u rl/eval_rollout.py --adapter runs/$name/final --out runs/eval_$name > runs/eval_$name.log 2>&1
done
echo REPLICATE_DONE
