# 第 2 档（训推分离，2 卡）公用设置：各阶段脚本 source 本文件。
# GPU 0 跑 vLLM 服务端，GPU 1 跑训练（进程内看到的是 cuda:0；NCCL 按物理卡区分，所以能建组）。
# DRY=1：只打印命令、不执行（在 Mac 上自检脚本逻辑用）。
REPO=${REPO:-$HOME/agent-rl-infra}
cd "$REPO" || exit 1
export PATH=$REPO/.venv/bin:$HOME/.local/bin:$PATH
export VLLM_USE_FLASHINFER_SAMPLER=1 TRL_EXPERIMENTAL_SILENCE=1
TAG=$(cat "$HOME/tier2_gpu_tag" 2>/dev/null || echo unknown)  # setup.sh 写入，例如 2xh100sxm5
RES=results/tier2_$TAG       # 小文件：下载回 Mac 并提交
RUNS=runs/tier2              # 训练输出（含 LoRA 适配器，不进 git）
MK=$HOME/tier2_markers       # 阶段完成标记，Mac 端据此下载结果 / 决定是否终止
mkdir -p "$RES" "$RUNS" "$MK" logs

run() { echo "+ $*" >&2; [ "$DRY" = 1 ] && return 0; "$@"; }
mark() { echo "$(date '+%F %T') $1" >> "$MK/log"; touch "$MK/$1"; }

serve_stop() { [ "$DRY" = 1 ] && return 0; pkill -f "vllm serve"; sleep 3; pkill -9 -f "vllm serve"; sleep 2; true; }
# serve_start MODEL LOG：GPU 0 上起服务端。每次训练前都要重启：TRL 第一步生成前不会同步权重，
# 沿用上一次训练改过的服务端权重会污染新一轮的第一步
serve_start() {
  serve_stop
  echo "+ serve $1 on GPU0 -> $2" >&2
  [ "$DRY" = 1 ] && return 0
  CUDA_VISIBLE_DEVICES=0 VLLM_SERVER_DEV_MODE=1 nohup vllm serve "$1" --port 8000 --max-model-len 2048 \
    --gpu-memory-utilization 0.9 --weight-transfer-config '{"backend":"nccl"}' > "$2" 2>&1 &
  for i in $(seq 120); do curl -sf localhost:8000/health > /dev/null && return 0; sleep 5; done
  tail -20 "$2" >&2; return 1
}

# 每秒采样两张卡的利用率和显存（timestamp 列用来和 steps.jsonl 对齐）
gpumon_start() { [ "$DRY" = 1 ] && return 0; nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used \
  --format=csv,noheader,nounits -lms 1000 > "$1" & GPUMON=$!; }
gpumon_stop() { [ -n "$GPUMON" ] && kill "$GPUMON" 2>/dev/null; GPUMON=; }

# train NAME GPU TIMEOUT_S ARGS...：训练并把小文件拷进 $RES/NAME；成功后写 DONE，再次调用时跳过（可续跑）
train() {
  local name=$1 gpu=$2 to=$3; shift 3
  [ -f "$RES/$name/DONE" ] && { echo "skip $name" >&2; return 0; }
  rm -rf "$RUNS/$name"; mkdir -p "$RES/$name"
  gpumon_start "$RES/$name/gpu.csv"
  run env CUDA_VISIBLE_DEVICES="$gpu" timeout -s KILL "$to" python -u rl/train_grpo.py --out "$RUNS/$name" "$@" \
    > "$RES/$name/train.log" 2>&1
  local rc=$?
  gpumon_stop
  [ "$DRY" = 1 ] || cp "$RUNS/$name"/*.json* "$RES/$name/" 2>/dev/null
  echo "train $name exit $rc" >&2
  [ $rc = 0 ] && touch "$RES/$name/DONE"
  return $rc
}

# evaluate NAME GPU ARGS...：GSM8K 完整测试集（与第 3 阶段同一评测代码）
evaluate() {
  local name=$1 gpu=$2; shift 2
  [ -f "$RES/$name/summary.json" ] && { echo "skip $name" >&2; return 0; }
  mkdir -p "$RES/$name"
  run env CUDA_VISIBLE_DEVICES="$gpu" timeout -s KILL 1800 python -u rl/eval_rollout.py --out "$RUNS/$name" "$@" \
    > "$RES/$name/eval.log" 2>&1
  local rc=$?
  [ "$DRY" = 1 ] || cp "$RUNS/$name"/summary.json "$RUNS/$name"/episodes.jsonl "$RES/$name/" 2>/dev/null
  echo "eval $name exit $rc" >&2
  return $rc
}
