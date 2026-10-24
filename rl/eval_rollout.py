"""第 3 阶段评测：用与训练完全相同的多轮 rollout 代码（温度 0，贪心）在 GSM8K 测试集上评测。

用法：
  .venv/bin/python rl/eval_rollout.py --out runs/eval_base                          # 未训练的基线
  .venv/bin/python rl/eval_rollout.py --adapter runs/main/final --out runs/eval_main  # 训练后的 LoRA
"""
import argparse
import json
import os
import re
import sys
import time
from types import SimpleNamespace

os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 代码围栏按出现顺序配对：第 1、3、5… 个是开头。只有"开头围栏没有语言标记"才算裸代码块。
# （旧写法 r"```\s*\n" 会把 python 代码块的结尾 ``` 也匹配上，导致第 0 阶段把"有代码块"误报成"裸代码块"）
FENCE_RE = re.compile(r"^\s*```(\S*)", re.M)


def has_bare_fence(text):
    return any(tag == "" for tag in FENCE_RE.findall(text)[0::2])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--adapter", default="", help="LoRA 适配器目录；为空则评测原始模型")
    ap.add_argument("--n", type=int, default=0, help="评测题数，0 表示完整测试集（1319 题）")
    ap.add_argument("--prompt", choices=["forced", "optional", "none"], default="forced",
                    help="forced=训练用的强制用工具；optional=可选用工具；none=不提工具")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    from transformers import AutoTokenizer
    from vllm import LLM

    import rl.agent_rollout as ar

    model_path = args.model
    if args.adapter:  # 把 LoRA 合并进权重后交给 vLLM，避免依赖 vLLM 的 LoRA 推理路径
        import torch
        from peft import AutoPeftModelForCausalLM
        merged = os.path.join(args.out, "merged")
        m = AutoPeftModelForCausalLM.from_pretrained(args.adapter, torch_dtype=torch.bfloat16).merge_and_unload()
        m.save_pretrained(merged)
        AutoTokenizer.from_pretrained(args.model).save_pretrained(merged)
        del m
        model_path = merged

    tok = AutoTokenizer.from_pretrained(model_path)
    llm = LLM(model_path, gpu_memory_utilization=0.6, max_model_len=4096)
    trainer = SimpleNamespace(processing_class=tok, vllm_generation=SimpleNamespace(llm=llm),
                              args=SimpleNamespace(temperature=0.0, top_p=1.0, max_completion_length=1024))
    ds = ar.load_gsm8k("test", prompt=args.prompt)
    if args.n:
        ds = ds.select(range(args.n))
    prompts, answers = [x["prompt"] for x in ds], [x["answer"] for x in ds]

    t0 = time.perf_counter()
    out = ar.make_rollout_func(log_path=os.path.join(args.out, "rollout.jsonl"))(prompts, trainer)
    wall = time.perf_counter() - t0
    correct = ar.correctness_reward(prompts, None, answers, out["final_text"])

    n = len(prompts)
    rows = []
    for i in range(n):
        model_text = tok.decode([t for t, m in zip(out["completion_ids"][i], out["env_mask"][i]) if m],
                                skip_special_tokens=True)
        rows.append({"answer": answers[i], "final_text": out["final_text"][i], "correct": correct[i],
                     "tool_calls": out["tool_calls"][i], "useful_calls": out["useful_calls"][i],
                     "turns": out["turns"][i], "bare_fence": has_bare_fence(model_text), "model_text": model_text,
                     "has_answer": ar.extract_answer(out["final_text"][i]) is not None,
                     "model_tokens": sum(out["env_mask"][i])})
    calls = sum(r["tool_calls"] for r in rows)
    summary = {
        "model": model_path, "adapter": args.adapter, "prompt": args.prompt, "n": n, "wall_s": round(wall, 1),
        "accuracy": round(sum(r["correct"] for r in rows) / n, 4),
        "tool_use_rate": round(sum(r["tool_calls"] > 0 for r in rows) / n, 4),
        "useful_episode_rate": round(sum(r["useful_calls"] > 0 for r in rows) / n, 4),
        "tool_calls": calls, "useful_call_rate": round(sum(r["useful_calls"] for r in rows) / max(1, calls), 4),
        "bare_fence_rate": round(sum(r["bare_fence"] for r in rows) / n, 4),
        "answer_format_rate": round(sum(r["has_answer"] for r in rows) / n, 4),
        "turns_mean": round(sum(r["turns"] for r in rows) / n, 3),
        "model_tokens_mean": round(sum(r["model_tokens"] for r in rows) / n, 1),
    }
    with open(os.path.join(args.out, "episodes.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary))

    if ar._pool is not None:  # 不提工具时可能一次代码都没执行，沙箱池没有创建
        import asyncio
        asyncio.run_coroutine_threadsafe(ar._pool.close(), ar._loop).result()


if __name__ == "__main__":
    main()
