"""第 3 阶段：工具调用 Agent 的多轮 rollout 与奖励函数（供 TRL GRPOTrainer 的 rollout_func 使用）。

一次 rollout（一批回合并行推进，最多 MAX_TURNS 轮）：
  vLLM 一次性生成所有活跃回合的下一段 assistant 回复
  → 找出写了 ```python 代码块的回合，把代码并发送进第 2 阶段的长驻容器池执行
  → 把执行结果按 chat 模板包成一轮 user 消息接在后面（env_mask=0，不计入 loss，logprob 填 0.0）
  → 没写代码 / 超长 / 达到轮数上限的回合结束。
与评测脚本 eval/gsm8k_baseline.py 的格式一致：工具结果放在 user 消息的 <output></output> 里，答案写成 "Answer: 42"。

返回给 TRL 的字段：prompt_ids、completion_ids、logprobs（每个 token 一个数）、env_mask，
以及转给奖励函数的 final_text / tool_calls / useful_calls / turns。
"""
import asyncio
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sandbox.pool import WorkerPool  # noqa: E402

SYSTEM_TOOL_FORCED = (
    "You are a careful math solver. You can run Python code: write it in a ```python code block. "
    "The code will be executed and its printed output returned to you inside <output></output> tags. "
    "Use print() to see results. When you are done, write the final numeric answer on its own line "
    "in the form 'Answer: 42'. You MUST write a ```python code block to compute the result before giving the answer."
)
# 评测用的另外两种提示词：可选用工具（不强制），以及完全不提工具（检验数学能力本身）
SYSTEM_TOOL_OPTIONAL = SYSTEM_TOOL_FORCED.replace(
    " You MUST write a ```python code block to compute the result before giving the answer.", "")
SYSTEM_NO_TOOL = ("You are a careful math solver. Solve the problem step by step, then write the final numeric answer "
                  "on its own line in the form 'Answer: 42'.")
SYSTEMS = {"forced": SYSTEM_TOOL_FORCED, "optional": SYSTEM_TOOL_OPTIONAL, "none": SYSTEM_NO_TOOL}
CODE_RE = re.compile(r"```(?:python|py)\s*\n(.*?)```", re.S)
ANSWER_RE = re.compile(r"Answer:[^\d\-\n]{0,25}(-?[\d,]*\.?\d+)")
MAX_TURNS = 4
TOOL_TIMEOUT = 3          # 第 2 阶段结论：GSM8K 类代码超时设 2–3 s
TOOL_OUTPUT_CHARS = 1000  # 喂回模型的执行结果最多保留的字符数
POOL_SIZE = 12            # 第 2 阶段容量规则：12 × 512 MB ≈ 本机内存的 40%
# Qwen3 chat 模板：结束当前 assistant 轮 → 一轮 user 消息 → 开始新的 assistant 轮（非思考模式带空 think 块）
ENV_TEMPLATE = ("<|im_start|>user\n<output>\n{out}\n</output><|im_end|>\n"
                "<|im_start|>assistant\n<think>\n\n</think>\n\n")

_pool, _loop = None, None


def _get_pool():
    """长驻容器池跑在一个后台事件循环线程里；rollout（同步代码）通过 run_coroutine_threadsafe 提交任务。"""
    global _pool, _loop
    if _pool is None:
        _loop = asyncio.new_event_loop()
        threading.Thread(target=_loop.run_forever, daemon=True).start()
        _pool = WorkerPool(size=POOL_SIZE, timeout=TOOL_TIMEOUT)
        asyncio.run_coroutine_threadsafe(_pool.start(), _loop).result()
    return _pool, _loop


def run_codes(codes):
    """并发执行一批代码，返回 [(输出, 是否有效输出)]。有效 = 进程成功且输出非空（空输出即"忘写 print"）。"""
    pool, loop = _get_pool()
    futs = [asyncio.run_coroutine_threadsafe(pool.run(c, TOOL_TIMEOUT), loop) for c in codes]
    res = [f.result() for f in futs]
    return [(r["output"], r["ok"] and bool(r["output"].strip())) for r in res]


def extract_answer(text):
    m = ANSWER_RE.findall(text)
    if not m:
        return None
    try:
        return float(m[-1].replace(",", "").rstrip("."))
    except ValueError:
        return None


def _generate(trainer, prompts, max_tokens):
    """每个 prompt（token id 列表）生成一段回复，返回 [(token_ids, 每个 token 的 logprob)]。
    同卡模式（colocate）直接调本进程里的 vLLM；服务端模式（训推分离）经 HTTP 调 vLLM 服务。"""
    args, gen = trainer.args, trainer.vllm_generation
    if getattr(gen, "mode", "colocate") == "server":
        # VLLMClient.generate 一次请求只能用一个 max_tokens，而各回合剩余长度不同：
        # 每个回合单独发请求、并发发出，由服务端自己拼批
        def one(p, m):
            o = gen.vllm_client.generate([p], temperature=args.temperature, top_p=args.top_p, max_tokens=m, logprobs=0)
            return o["completion_ids"][0], [lp[0] for lp in o["logprobs"][0]]

        with ThreadPoolExecutor(max_workers=min(64, len(prompts))) as ex:
            return list(ex.map(one, prompts, max_tokens))
    from vllm import SamplingParams
    params = [SamplingParams(temperature=args.temperature, top_p=args.top_p, logprobs=0, max_tokens=m)
              for m in max_tokens]
    outs = gen.llm.generate([{"prompt_token_ids": p} for p in prompts], sampling_params=params, use_tqdm=False)
    res = []
    for o in outs:
        c = o.outputs[0]
        toks = list(c.token_ids)
        res.append((toks, [c.logprobs[i][t].logprob for i, t in enumerate(toks)] if c.logprobs else [0.0] * len(toks)))
    return res


def make_rollout_func(log_path=None):
    def rollout_func(prompts, trainer):
        tok = trainer.processing_class
        args = trainer.args
        im_end = tok.convert_tokens_to_ids("<|im_end|>")
        max_len = args.max_completion_length
        eps = []
        for p in prompts:
            # 先渲染成文本再编码：transformers 5 的 tokenize=True 返回 BatchEncoding（类字典但不是 dict）
            text = tok.apply_chat_template(p, add_generation_prompt=True, tokenize=False, enable_thinking=False)
            ids = tok.encode(text, add_special_tokens=False)
            eps.append({"prompt_ids": ids, "completion_ids": [], "logprobs": [], "env_mask": [], "done": False,
                        "final_text": "", "tool_calls": 0, "useful_calls": 0, "turns": 0})
        gen_s = tool_s = 0.0
        for _ in range(MAX_TURNS):
            active = [e for e in eps if not e["done"]]
            if not active:
                break
            t0 = time.perf_counter()
            outs = _generate(trainer, [e["prompt_ids"] + e["completion_ids"] for e in active],
                             [max(1, max_len - len(e["completion_ids"])) for e in active])
            gen_s += time.perf_counter() - t0
            pending = []
            for e, (toks, lps) in zip(active, outs):
                e["completion_ids"] += toks
                e["logprobs"] += lps
                e["env_mask"] += [1] * len(toks)
                e["turns"] += 1
                text = tok.decode(toks, skip_special_tokens=True)
                e["final_text"] = text
                code = CODE_RE.findall(text)
                ended_turn = bool(toks) and toks[-1] == im_end
                if not code or not ended_turn or len(e["completion_ids"]) >= max_len:
                    e["done"] = True  # 没写代码、被截断或已到长度上限：回合结束
                else:
                    pending.append((e, code[-1]))
            if not pending:
                continue
            t0 = time.perf_counter()
            results = run_codes([c for _, c in pending])
            tool_s += time.perf_counter() - t0
            for (e, _), (out, useful) in zip(pending, results):
                e["tool_calls"] += 1
                e["useful_calls"] += int(useful)
                seg = tok.encode("\n" + ENV_TEMPLATE.format(out=out.strip()[-TOOL_OUTPUT_CHARS:]),
                                 add_special_tokens=False)
                if len(e["completion_ids"]) + len(seg) >= max_len:
                    e["done"] = True
                    continue
                e["completion_ids"] += seg
                e["logprobs"] += [0.0] * len(seg)
                e["env_mask"] += [0] * len(seg)
        if log_path:
            with open(log_path, "a") as f:
                f.write(json.dumps({"t": time.time(), "episodes": len(eps), "gen_s": round(gen_s, 3),
                                    "tool_s": round(tool_s, 3), "tool_calls": sum(e["tool_calls"] for e in eps),
                                    "useful_calls": sum(e["useful_calls"] for e in eps),
                                    "turns_mean": sum(e["turns"] for e in eps) / len(eps),
                                    "completion_tokens": sum(len(e["completion_ids"]) for e in eps),
                                    "model_tokens": sum(sum(e["env_mask"]) for e in eps)}) + "\n")
        return {k: [e[k] for e in eps] for k in ("prompt_ids", "completion_ids", "logprobs", "env_mask",
                                                  "final_text", "tool_calls", "useful_calls", "turns")}
    return rollout_func


def correctness_reward(prompts, completions, answer, final_text, **kwargs):
    """主奖励：最后一轮回复里 "Answer:" 后的数字等于标准答案得 1 分。"""
    out = []
    for gold, text in zip(answer, final_text):
        pred = extract_answer(text)
        out.append(1.0 if pred is not None and abs(pred - float(gold)) < 1e-4 else 0.0)
    return out


def tool_reward(prompts, completions, useful_calls, **kwargs):
    """小额奖励：至少一次工具调用拿到有效输出（成功且非空）得 0.2，每题最多一次，防止刷调用次数。"""
    return [0.2 if n > 0 else 0.0 for n in useful_calls]


def load_gsm8k(split="train", prompt="forced"):
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split=split)

    def to_example(x):
        gold = float(x["answer"].split("####")[-1].strip().replace(",", ""))
        return {"prompt": [{"role": "system", "content": SYSTEMS[prompt]},
                           {"role": "user", "content": x["question"]}], "answer": gold}

    return ds.map(to_example, remove_columns=ds.column_names)
