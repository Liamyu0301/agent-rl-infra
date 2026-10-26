#!/usr/bin/env bash
# 第 2 档总控（在 Mac 上运行；与会话无关，用 nohup caffeinate 启动）：
#   开机 + 保险 → 装环境 → 启动实例上的 chain.sh → 每分钟把结果同步回 Mac
#   → G0 失败或流程结束：推送结果到 GitHub → 终止实例 → 停保险 → 记录时长与花费
# 用法：mac_orchestrate.sh INSTANCE_TYPE REGION USD_PER_HOUR DEADMAN_S PRIOR_CNY
#   例：nohup caffeinate -i scripts/tier2/mac_orchestrate.sh gpu_2x_h100_sxm5 us-east-1 8.38 19800 85 > ~/tier2_mac.log 2>&1 &
# DRY=1：不调 API、不连实例，只打印将要执行的步骤（自检用）
set -u
TYPE=$1 REGION=$2 PRICE=$3 DEADMAN_S=$4 PRIOR_CNY=$5
REPO=$(cd "$(dirname "$0")/../.." && pwd)
A=${ARTIFACTS:-/Users/yumengli/code/agent-rl-infra-artifacts}/tier2; mkdir -p "$A"
API=~/.config/lambda/lambda_api.sh
KEY=(-i "$HOME/.ssh/id_lambda" -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20)
log() { echo "$(date '+%F %T') $*"; }
api() { if [ "${DRY:-0}" = 1 ]; then echo "DRY api $*" >&2; echo '{"data":{"instance_ids":["dry"],"status":"active","ip":"0.0.0.0"}}'; else "$API" "$@"; fi; }
rsh() { if [ "${DRY:-0}" = 1 ]; then echo "DRY ssh: $*" >&2; else ssh "${KEY[@]}" "ubuntu@$IP" "$@"; fi; }
pull() { if [ "${DRY:-0}" = 1 ]; then echo "DRY rsync $1 -> $2" >&2; else rsync -az -e "ssh ${KEY[*]}" "ubuntu@$IP:$1" "$2"; fi; }
push() { if [ "${DRY:-0}" = 1 ]; then echo "DRY scp $*" >&2; else scp -q "${KEY[@]}" "$@"; fi; }

# 1. 开机，同时挂保险（保险独立于本脚本：本脚本死了也会按时终止）
R=$(api POST /instance-operations/launch "{\"region_name\":\"$REGION\",\"instance_type_name\":\"$TYPE\",\"ssh_key_names\":[\"agent-rl-infra-lambda\"],\"quantity\":1,\"name\":\"agent-rl-tier2\"}")
ID=$(echo "$R" | python3 -c 'import json,sys;print(json.load(sys.stdin)["data"]["instance_ids"][0])') || { log "launch failed: $R"; exit 1; }
LAUNCH=$(date +%s); echo "$ID" > "$A/instance_id"; echo "$LAUNCH" > "$A/launch_epoch"
[ "${DRY:-0}" = 1 ] || nohup caffeinate -i /Users/yumengli/code/agent-rl-infra-artifacts/lambda/deadman.sh "$ID" "$DEADMAN_S" > "$A/deadman.log" 2>&1 &
log "launched $ID ($TYPE $REGION, \$$PRICE/h); deadman ${DEADMAN_S}s"

terminate() {
  log "terminate: $1"
  api POST /instance-operations/terminate "{\"instance_ids\":[\"$ID\"]}" > "$A/terminate.json"
  local end; end=$(date +%s)
  pkill -f "deadman.sh $ID" 2>/dev/null
  python3 -c "h=($end-$LAUNCH)/3600;usd=h*$PRICE;print(f'hours={h:.2f} usd={usd:.2f} cny={usd*7.2:.0f} cumulative_cny={$PRIOR_CNY+usd*7.2:.0f}')" | tee "$A/cost.txt"
  exit 0
}

# 2. 等实例就绪
IP=
for i in $(seq 80); do
  S=$(api GET "/instances/$ID" | python3 -c 'import json,sys;d=json.load(sys.stdin)["data"];print(d["status"],d.get("ip"))')
  case "$S" in active*None*) ;; active*) IP=${S#active }; break ;; terminated*|unhealthy*) log "instance $S"; exit 1 ;; esac
  sleep 15
done
[ -n "$IP" ] || terminate "not active after 20 min"
echo "$IP" > "$A/ip"; log "active $IP"

# 3. 装环境（前台，最多 30 分钟）；写入预算参数供 G1 使用
sleep 20  # sshd 刚起来时偶尔拒绝连接
push "$REPO/scripts/tier2/setup.sh" "ubuntu@$IP:"
echo "{\"launch_epoch\": $LAUNCH, \"usd_per_hour\": $PRICE, \"prior_cny\": $PRIOR_CNY, \"cap_cny\": 400, \"cny_per_usd\": 7.2}" > "$A/budget.json"
push "$A/budget.json" "ubuntu@$IP:tier2_budget.json"
rsh 'timeout 1800 bash ~/setup.sh > ~/setup.log 2>&1; tail -1 ~/setup.log'
pull setup.log "$A/"
[ "${DRY:-0}" = 1 ] || grep -q SETUP_DONE "$A/setup.log" || terminate "setup failed"

# 4. 新登录启动流程（docker 组此时已生效；再包一层 sg docker 以防万一）
printf '%s\n' 'pgrep -f "bash .*tier2/chain.sh" > /dev/null && echo alive || echo dead' > "$A/alive.sh"
push "$A/alive.sh" "ubuntu@$IP:"
rsh 'cd ~/agent-rl-infra && nohup sg docker -c "bash scripts/tier2/chain.sh > ~/chain.log 2>&1" > /dev/null 2>&1 < /dev/null & echo started'
log "chain started"

# 5. 每分钟同步结果；按标记决定何时终止
sync_all() { mkdir -p "$REPO/results"; pull 'agent-rl-infra/results/tier2_*' "$REPO/results/"; pull tier2_markers/ "$A/markers/"; pull chain.log "$A/"; }
while true; do
  sleep 60
  sync_all
  [ "${DRY:-0}" = 1 ] && break
  [ -f "$A/markers/g0_fail" ] && { sync_all; FINAL="G0 failed"; break; }
  [ -f "$A/markers/chain_done" ] && { FINAL="chain done"; break; }
  [ "$(rsh 'bash ~/alive.sh')" = dead ] && { sleep 5; sync_all; FINAL="chain exited without chain_done"; break; }
done

# 6. 收尾：LoRA 适配器下载回 Mac（不进 git），结果推送 GitHub，然后终止
for d in $(rsh 'ls -d ~/agent-rl-infra/runs/tier2/main17_s*/final 2>/dev/null'); do
  n=$(basename "$(dirname "$d")"); mkdir -p "$A/adapters/$n"; pull "$d/" "$A/adapters/$n/"
done
if [ "${DRY:-0}" != 1 ]; then
  cd "$REPO" && git add results/tier2_* && git commit -q -m "Tier 2 results ($TYPE): ${FINAL:-dry}

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" && git push -q origin main && log "pushed $(git log -1 --format=%h)"
fi
terminate "${FINAL:-dry run}"
