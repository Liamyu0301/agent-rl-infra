# G0 关卡：用 0.6B 验证双卡服务端模式（NCCL 建组 + PEFT 权重同步），并补上第 1 档 1.3 欠下的 logprob 对比。
# 两种模式都用 lr 1e-3（比正式训练大 30 倍）：5 步内权重变化明显，如果服务端没收到新权重，
# vLLM 与训练端的 logprob 差会逐步变大，而同卡模式不会——这样才能在 5 步里暴露同步失败。
source "$(dirname "$0")/common.sh"
M=Qwen/Qwen3-0.6B; A="--model $M --max-steps 5 --seed 42 --lr 1e-3"
mkdir -p "$RES/g0"
train g0_colocate06 1 900 $A --vllm-mode colocate --vllm-mem 0.3
serve_start $M logs/tier2_serve_g0.log || { mark g0_fail; exit 1; }
run timeout -s KILL 600 python rl/test_rollout.py --server http://localhost:8000 --model $M > "$RES/g0/test_rollout_06.txt" 2>&1
echo "test_rollout exit $?" | tee "$RES/g0/test_rollout_06.rc" >&2
train g0_server06 1 900 $A --vllm-mode server
serve_stop; [ "$DRY" = 1 ] || cp logs/tier2_serve_g0.log "$RES/g0/serve.log"
run python scripts/tier2/analyze.py g0 "$RES" && mark g0_pass || mark g0_fail
