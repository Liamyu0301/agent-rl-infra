# 第 3 档：异步 rollout（落后一步的流水线，rl/async_rollout.py）在 2 卡训推分离上的实测。规则开机前定好，运行中不改：
#   3.0 实例上再跑一次流水线自检（假训练器，不用 GPU）
#   3.1 1.7B 同步基线 30 步（--profile）：同一台机器上重测，不拿第 2 档的数字比
#   3.2 1.7B 异步 30 步（--profile）→ G0：落后步数恰好为 1、权重同步与生成不重叠、训练跑完；不通过就停
#   3.3 异步 + 冗余前向优化 30 步（只计时）
#   G1：异步 30 步每步耗时必须小于同步（否则流水线没起作用，不花钱跑正式训练），且预测花费不超上限
#   3.4 异步 200 步种子 42 + 完整测试集评测 → G1：准确率 ≥ 78.2%（第 2 档同步同种子 81.2% − 3 个点）且预测花费不超上限，才跑种子 1
# 与第 2 档 main17 相同：200 步、lr 3e-5、不开冗余前向优化、IS 纠偏用 TRL 默认（sequence_mask，阈值 3.0），只差异步这一个变量。
# 用法（实例上，setup.sh 之后的新登录）：sg docker -c "bash scripts/tier3/chain.sh"
export TIER=tier3
source "$(dirname "$0")/../tier2/common.sh"
A3="python scripts/tier3/analyze.py"
cp "results/tier2_$TAG/machine.txt" "$RES/" 2>/dev/null
# setup.sh 把本机信息写进了第 2 档的目录；还原成已提交的版本，免得 Mac 端同步回去覆盖第 2 档的记录
[ "$DRY" = 1 ] || git checkout -- "results/tier2_$TAG/machine.txt" 2>/dev/null
M=Qwen/Qwen3-1.7B; P="--model $M --max-steps 30 --seed 42 --profile --vllm-mode server"

run python rl/test_async_rollout.py > "$RES/test_async_rollout.txt" 2>&1 || { mark g0_fail; exit 1; }

# 每次训练前重启服务端（TRL 第一步前不同步权重，见 common.sh）
serve_run() {  # serve_run NAME TIMEOUT ARGS...
  local name=$1 to=$2; shift 2
  [ -f "$RES/$name/DONE" ] && { echo "skip $name" >&2; return 0; }
  serve_start $M "logs/t3_serve_$name.log" || return 1
  train "$name" 1 "$to" "$@"; local rc=$?
  serve_stop
  [ "$DRY" = 1 ] || cp "logs/t3_serve_$name.log" "$RES/$name/serve.log"
  return $rc
}

serve_run sync17 2400 $P
serve_run async17 2400 $P --async-rollout
run $A3 g0 "$RES" && mark g0_pass || { mark g0_fail; exit 1; }
serve_run asyncfused17 2400 $P --async-rollout --fused-is on
run python scripts/tier2/analyze.py profile "$RES" sync17 async17
run python scripts/tier2/analyze.py profile "$RES" async17 asyncfused17
mark s31_done

run $A3 g1 "$RES" async17   # 写 $RES/g1.json：按实测耗时预测花费，决定种子 42 能否开跑
for s in 42 1; do
  if [ "$s" = 1 ]; then run $A3 g1b "$RES" || break; fi   # 种子 1 前：准确率 + 花费复核，写 $RES/g1b.json
  [ "$DRY" = 1 ] || python -c "import json,sys;sys.exit(0 if $s in json.load(open('$RES/g1.json'))['seeds'] else 1)" || break
  serve_run amain17_s$s 14400 --model $M --max-steps 200 --save-steps 200 --seed $s --vllm-mode server --async-rollout
  evaluate eval_amain17_s$s 1 --model $M --adapter "$RUNS/amain17_s$s/final"
  mark s34_seed${s}_done
done
run $A3 summary "$RES"
mark chain_done
