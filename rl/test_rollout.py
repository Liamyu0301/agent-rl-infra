"""3.1 自检：不训练，用独立的 vLLM 实例跑一批 rollout，检查长度一致性、模板一致性、奖励分解和耗时。
加 --server URL 时改为通过 vLLM 服务端生成（训推分离模式的生成路径）。"""
import argparse
import os
import sys
import time
from types import SimpleNamespace

os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from transformers import AutoTokenizer  # noqa: E402

from rl.agent_rollout import correctness_reward, load_gsm8k, make_rollout_func, tool_reward  # noqa: E402
import rl.agent_rollout as ar  # noqa: E402

N_PROMPTS, G = 16, 4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default=None, help="vLLM 服务地址，如 http://localhost:8000；不给则在本进程起 vLLM")
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B", help="服务端模式下须与服务端加载的模型一致")
    a = ap.parse_args()
    server, MODEL = a.server, a.model

    tok = AutoTokenizer.from_pretrained(MODEL)
    if server:
        from trl.generation.vllm_client import VLLMClient
        gen = SimpleNamespace(mode="server", vllm_client=VLLMClient(base_url=server))
    else:
        from vllm import LLM
        gen = SimpleNamespace(llm=LLM(MODEL, gpu_memory_utilization=0.5, max_model_len=4096))
    trainer = SimpleNamespace(processing_class=tok, vllm_generation=gen,
                              args=SimpleNamespace(temperature=1.0, top_p=1.0, max_completion_length=1024))
    ds = load_gsm8k("train").select(range(N_PROMPTS))
    prompts = [x["prompt"] for x in ds for _ in range(G)]   # 与 GRPO 一样：每题重复 G 次
    answers = [x["answer"] for x in ds for _ in range(G)]

    t0 = time.perf_counter()
    out = make_rollout_func(log_path="logs/rollout_test.jsonl")(prompts, trainer)
    wall = time.perf_counter() - t0

    n = len(prompts)
    lens_ok = all(len(c) == len(l) == len(m) for c, l, m in zip(out["completion_ids"], out["logprobs"], out["env_mask"]))
    env_tokens = sum(len(m) - sum(m) for m in out["env_mask"])
    print(f"episodes={n} wall={wall:.1f}s  lengths consistent={lens_ok}  env(masked) tokens={env_tokens}")
    r_c = correctness_reward(prompts, None, answers, out["final_text"])
    r_t = tool_reward(prompts, None, out["useful_calls"])
    print(f"correctness mean={sum(r_c)/n:.3f}  tool_reward mean={sum(r_t)/n:.3f}  "
          f"episodes with >=1 tool call={sum(c > 0 for c in out['tool_calls'])/n:.2%}  "
          f"useful/total calls={sum(out['useful_calls'])}/{sum(out['tool_calls'])}  "
          f"turns mean={sum(out['turns'])/n:.2f}")
    # 每组 G 个样本的奖励是否有差异：全组相同的奖励在 GRPO 里优势为 0，不产生学习信号
    groups = [[r_c[i + j] + r_t[i + j] for j in range(G)] for i in range(0, n, G)]
    print(f"groups with reward variance (learning signal): {sum(len(set(g)) > 1 for g in groups)}/{len(groups)}")
    print(open("logs/rollout_test.jsonl").read().strip().splitlines()[-1])

    # 模板一致性：手工拼接的序列 vs 官方 chat 模板渲染同一段对话
    idx = next((i for i, c in enumerate(out["tool_calls"]) if c > 0), None)
    same = None
    if idx is None:
        print("no episode used a tool; template check skipped")
    else:
        full = tok.decode(out["prompt_ids"][idx] + out["completion_ids"][idx])
        msgs = list(prompts[idx])
        segs, cur, mask = [], [], out["env_mask"][idx]
        for t, m in zip(out["completion_ids"][idx], mask):
            if cur and m != cur[-1][1]:
                segs.append(cur)
                cur = []
            cur.append((t, m))
        segs.append(cur)
        for s in segs:
            text = tok.decode([t for t, _ in s], skip_special_tokens=True)
            if s[0][1] == 1:
                msgs.append({"role": "assistant", "content": text.strip()})
            else:
                body = text.split("<output>\n", 1)[-1].rsplit("\n</output>", 1)[0]
                msgs.append({"role": "user", "content": f"<output>\n{body}\n</output>"})
        ref = tok.apply_chat_template(msgs, tokenize=False, enable_thinking=False)
        # 已知差异：官方模板渲染历史 assistant 轮时去掉空 think 块，rollout 保留（模型确实在其后生成），比较时两边都去掉
        drop_think = lambda s: s.replace("<think>\n\n</think>\n\n", "").strip()  # noqa: E731
        same = drop_think(full) == drop_think(ref)
        print(f"template check on episode {idx}: identical={same}")
        if not same:
            import difflib
            for line in list(difflib.unified_diff(ref.splitlines(), full.splitlines(), "chat_template", "rollout",
                                                  lineterm="", n=1))[:30]:
                print("   ", line[:160])
        print("---- sample transcript (rollout side) ----")
        print(full[-1500:])

    if ar._pool is not None:
        import asyncio
        asyncio.run_coroutine_threadsafe(ar._pool.close(), ar._loop).result()
    # 检查不通过时以非 0 退出，脚本化运行时 exit code 才有意义（模板检查因无工具调用而跳过时不算失败）
    sys.exit(0 if lens_ok and same is not False else 1)


if __name__ == "__main__":  # vLLM 用 spawn 启动引擎子进程，主逻辑必须放在这里
    main()
