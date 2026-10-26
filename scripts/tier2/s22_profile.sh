# 2.1 / 2.2：1.7B 基线评测 + 分段计时（同卡 vs 服务端，各 20 步，正式学习率），然后 G1 关卡决定跑几个种子。
# 混杂因素（写进结果表头）：同卡模式 vLLM 只占 0.3 张卡，服务端模式占整张 GPU 0 的 0.9 → 生成段的差异混有 KV 容量的差异；
# 两种模式间真正只反映架构的是"权重同步"这一段。
source "$(dirname "$0")/common.sh"
M=Qwen/Qwen3-1.7B; P="--model $M --max-steps 20 --seed 42 --profile"
evaluate eval_base17 1 --model $M
train prof17_colocate 1 2400 $P --vllm-mode colocate --vllm-mem 0.3
serve_start $M logs/tier2_serve_prof.log || { mark s22_fail; exit 1; }
run timeout -s KILL 600 python rl/test_rollout.py --server http://localhost:8000 --model $M > "$RES/test_rollout_17.txt" 2>&1
echo "test_rollout exit $?" | tee "$RES/test_rollout_17.rc" >&2
train prof17_server 1 2400 $P --vllm-mode server
serve_stop
run python scripts/tier2/analyze.py profile "$RES" prof17_colocate prof17_server
run python scripts/tier2/analyze.py g1 "$RES" prof17_server   # 写 $RES/g1.json：seeds = [42] 或 [42, 1]
mark s22_done
