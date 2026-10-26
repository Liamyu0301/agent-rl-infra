# 第 2 档环境：与第 1 档相同的 cu129 软件栈（版本号与 3080 Ti 一致），驱动再新也不换，保持可比。
# 用法（实例上）：bash setup.sh > ~/setup.log 2>&1；成功时最后一行是 SETUP_DONE
set -x
sudo usermod -aG docker ubuntu   # 新登录才生效：后续阶段用 sg docker -c 启动
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH=$HOME/.local/bin:$PATH
[ -d ~/agent-rl-infra ] || git clone -q https://github.com/Liamyu0301/agent-rl-infra.git ~/agent-rl-infra
cd ~/agent-rl-infra && git log --oneline -1
uv python install 3.12 && uv venv -p 3.12 .venv && . .venv/bin/activate || exit 1
uv pip install "https://github.com/vllm-project/vllm/releases/download/v0.32.0/vllm-0.32.0%2Bcu129-cp38-abi3-manylinux_2_28_x86_64.whl" \
  torch==2.13.0 torchvision==0.28.0 torchaudio==2.11.0 "torchcodec==0.16.0+cu129" trl==1.16.0 peft==0.21.2 accelerate==1.15.0 \
  --extra-index-url https://download.pytorch.org/whl/cu129 --index-strategy unsafe-best-match || exit 1
# 结果目录名按卡型命名，不同卡型的数字不混表
n=$(nvidia-smi -L | wc -l); g=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | tr 'A-Z' 'a-z' | tr -cd 'a-z0-9')
echo "${n}x${g#nvidia}" > ~/tier2_gpu_tag; cat ~/tier2_gpu_tag
mkdir -p results/tier2_$(cat ~/tier2_gpu_tag)
{ nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv; nvidia-smi topo -m; nproc; free -g | head -2; } \
  > results/tier2_$(cat ~/tier2_gpu_tag)/machine.txt 2>&1
python -c "import torch,vllm,flashinfer,trl,peft;from torchcodec.decoders import AudioDecoder;print('torch',torch.__version__,torch.cuda.device_count(),'gpus;','vllm',vllm.__version__,'flashinfer',flashinfer.__version__,'trl',trl.__version__,'peft',peft.__version__)" || exit 1
# 预下载模型和数据集，避免把下载时间算进第一步
python -c "from huggingface_hub import snapshot_download as d; [d(m) for m in ('Qwen/Qwen3-0.6B','Qwen/Qwen3-1.7B')]; from datasets import load_dataset as l; l('openai/gsm8k','main')" || exit 1
sg docker -c "docker run --rm python:3.12-slim python -c 'print(6*7)'" || exit 1
echo SETUP_DONE
