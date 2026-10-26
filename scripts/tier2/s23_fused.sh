# 2.3：冗余前向优化在服务端模式下再验证——先 check（只并行计算并记录差异），数值等价才跑 on 测提速。
source "$(dirname "$0")/common.sh"
M=Qwen/Qwen3-1.7B; P="--model $M --max-steps 10 --seed 42 --profile --vllm-mode server"
serve_start $M logs/tier2_serve_fused.log || { mark s23_fail; exit 1; }
train fused17_check 1 1500 $P --fused-is check
serve_stop
if run python scripts/tier2/analyze.py fused_check "$RES" fused17_check; then
  serve_start $M logs/tier2_serve_fused.log || { mark s23_fail; exit 1; }
  train fused17_on 1 1500 $P --fused-is on
  serve_stop
  run python scripts/tier2/analyze.py profile "$RES" prof17_server fused17_on
fi
mark s23_done
