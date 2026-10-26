"""CPU 复现：TRL 1.16 的 PEFT 权重迭代器 + vLLM 0.32 的 pack_tensors → 最后一块缓冲区拿到 unmerge 之后的旧权重。
pack_tensors 从 vLLM 源码原样取出（只依赖 torch）。用法：
  curl -sfL https://raw.githubusercontent.com/vllm-project/vllm/v0.32.0/vllm/distributed/weight_transfer/packed_tensor.py -o packed_tensor.py
  python scripts/tier2/repro_pack_race.py packed_tensor.py
预期：最后一块因迭代器耗尽而结束时（buf=50000 / 10000000），LoRA 作用的权重全部不等于合并后的值；clone 后全部相等。"""
import sys
from contextlib import nullcontext
from types import SimpleNamespace

import torch
from peft import LoraConfig, get_peft_model

# 从 vLLM v0.32.0 源码里原样取出 pack_tensors（只依赖 torch）
src = open(sys.argv[1]).read()
cls_start = src.index("@dataclass\nclass PackedChunk")
cls_src = src[cls_start:src.index("\n\n\n", cls_start)]
start = src.index("def pack_tensors(")
fn_src = src[start:src.index("\ndef ", start + 1)]
ns = {"torch": torch}
exec("from collections.abc import Callable, Iterator\nfrom dataclasses import dataclass\nfrom typing import Any\n"
     + cls_src + "\n\n" + fn_src, ns)
pack_tensors = ns["pack_tensors"]

from trl.generation.vllm_generation import VLLMGeneration  # noqa: E402

torch.manual_seed(0)
base = torch.nn.Sequential(*[torch.nn.Linear(64, 64) for _ in range(6)])
model = get_peft_model(base, LoraConfig(r=4, target_modules=[str(i) for i in range(6)]))
for n, p in model.named_parameters():
    if "lora_B" in n:
        p.data.normal_()  # 模拟训练过几步：B 不再是 0，合并后的权重 ≠ 原权重

model.merge_adapter()
expected = {n.removeprefix("base_model.model.").replace(".base_layer", ""): p.detach().clone()
            for n, p in model.named_parameters() if "lora_" not in n}
model.unmerge_adapter()

fake = SimpleNamespace(model=model, _dist=SimpleNamespace(gather_params=lambda ps: nullcontext(), is_fsdp=False),
                       _fix_param_name_to_vllm=lambda name, extra_prefixes=None: name)


def check(iterator, label, buf):
    got, chunk_id = {}, 0
    while True:
        chunk = pack_tensors(iterator, lambda item: item[1], buffer_size_bytes=buf)  # 小缓冲区 → 分成多块
        if chunk is None:
            break
        off = 0
        for name, shape, dt, size in zip(chunk.names, chunk.shapes, chunk.dtypes, chunk.tensor_sizes):
            got[name] = (chunk.packed_tensor[off:off + size].view(dt).view(shape), chunk_id)
            off += size
        chunk_id += 1
    bad = [(n, c) for n, (t, c) in got.items() if not torch.equal(t, expected[n])]
    print(f"{label} buf={buf}: {len(got)} tensors in {chunk_id} chunks; wrong = {len(bad)} -> {bad}")


for buf in (40_000, 50_000, 10_000_000):
    check(VLLMGeneration._iter_named_params(fake), "TRL iterator as-is", buf)



def cloned(it):
    for name, t in it:
        c = t.clone()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        yield name, c


for buf in (40_000, 50_000, 10_000_000):
    check(cloned(VLLMGeneration._iter_named_params(fake)), "with clone()      ", buf)
