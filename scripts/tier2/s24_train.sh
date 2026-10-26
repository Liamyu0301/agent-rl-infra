# 2.4：1.7B 服务端模式正式训练 + 完整测试集评测。与 0.6B 的 control200 / control200_s1 配对：
# 相同种子（42、1）、相同 200 步与学习率调度、不开冗余前向优化。种子数由 G1 关卡写进 $RES/g1.json。
source "$(dirname "$0")/common.sh"
M=Qwen/Qwen3-1.7B
SEEDS=$(python -c "import json;print(*json.load(open('$RES/g1.json'))['seeds'])" 2>/dev/null || echo 42)
for s in $SEEDS; do
  if [ ! -f "$RES/main17_s$s/DONE" ]; then
    serve_start $M logs/tier2_serve_main_s$s.log || { mark s24_fail; exit 1; }
    train main17_s$s 1 14400 --model $M --max-steps 200 --save-steps 200 --seed $s --vllm-mode server
    serve_stop
  fi
  evaluate eval_main17_s$s 1 --model $M --adapter "$RUNS/main17_s$s/final"
  mark s24_seed${s}_done
done
mark s24_done
