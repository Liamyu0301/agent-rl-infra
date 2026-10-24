#!/usr/bin/env bash
# 启动 vLLM 推理服务（OpenAI 兼容接口，端口 8000）。额外参数原样传给 vllm serve，例如：
#   scripts/serve.sh --no-enable-prefix-caching
# 环境变量 MODEL 可切换模型（默认 Qwen/Qwen3-0.6B）。
# VLLM_USE_FLASHINFER_SAMPLER=0：PyPI 上没有与 flashinfer 0.7.0.post1 匹配的 flashinfer-cubin，
# 而本机没有 nvcc，FlashInfer 采样器无法 JIT 编译，所以改用 PyTorch 采样器。
# --max-model-len 16384：Agent 会话多轮累积后上下文较长。
MODEL=${MODEL:-Qwen/Qwen3-0.6B}
cd "$(dirname "$0")/.."
VLLM_USE_FLASHINFER_SAMPLER=0 exec .venv/bin/vllm serve "$MODEL" \
  --gpu-memory-utilization 0.8 --max-model-len 16384 --port 8000 "$@"
