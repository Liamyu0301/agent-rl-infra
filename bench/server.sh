#!/usr/bin/env bash
# 压测用的服务启停：source bench/server.sh 后调用 start_server [vllm 额外参数...] / stop_server
start_server() {
  stop_server
  if ss -ltn | grep -q ":8000 "; then echo "port 8000 still busy" >&2; return 1; fi
  nohup scripts/serve.sh "$@" > logs/serve_bench.log 2>&1 &
  for i in $(seq 1 60); do
    curl -s localhost:8000/health >/dev/null 2>&1 && { echo "server up: $*"; return 0; }
    sleep 5
  done
  echo "server failed to start" >&2; tail -5 logs/serve_bench.log >&2; return 1
}
stop_server() { pkill -f "vllm serve" 2>/dev/null; sleep 3; pkill -9 -f "vllm serve" 2>/dev/null; sleep 1; }
