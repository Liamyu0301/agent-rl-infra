"""GSM8K 基线评测：对比"不用工具"和"可调用 Python 工具"两种模式。

模型通过 OpenAI 兼容接口调用（先用 scripts/serve.sh 启动 vLLM）。
工具模式：模型写 ```python 代码块 -> 本脚本执行 -> 输出放进 <output> 返回 -> 模型继续，
直到写出 "Answer: <数字>" 或达到最大轮数。
注意：代码只做了临时目录 + 超时的简单隔离，正式的 Docker 沙箱在第 2 阶段实现。
"""
import argparse, json, os, re, subprocess, sys, tempfile, time
from concurrent.futures import ThreadPoolExecutor

from datasets import load_dataset
from openai import OpenAI

SYSTEM_TOOL = (
    "You are a careful math solver. You can run Python code: write it in a ```python code block. "
    "The code will be executed and its printed output returned to you inside <output></output> tags. "
    "Use print() to see results. When you are done, write the final numeric answer on its own line in the form 'Answer: 42'."
)
SYSTEM_TOOL_FORCED = SYSTEM_TOOL + (
    " You MUST write a ```python code block to compute the result before giving the answer."
)
SYSTEM_NOTOOL = (
    "You are a careful math solver. Solve the problem step by step, "
    "then write the final numeric answer on its own line in the form 'Answer: 42'."
)
CODE_RE = re.compile(r"```(?:python|py)\s*\n(.*?)```", re.S)
# 只看开头围栏（按顺序配对，第 1、3、5… 个）是否缺语言标记；旧写法会把 python 代码块的结尾 ``` 也算成裸代码块
FENCE_RE = re.compile(r"^\s*```(\S*)", re.M)
# Answer: 后 25 个字符内的第一个数字，兼容 "Answer: <18>"、"Answer: <output>18</output>"、"**Answer: $18**"
ANSWER_RE = re.compile(r"Answer:[^\d\-\n]{0,25}(-?[\d,]*\.?\d+)")
NUM_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def run_python(code, timeout):
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "main.py")
        with open(path, "w") as f:
            f.write(code)
        try:
            r = subprocess.run([sys.executable, path], cwd=tmp, capture_output=True, text=True, timeout=timeout)
            return (r.stdout + r.stderr)[-2000:], r.returncode == 0
        except subprocess.TimeoutExpired:
            return f"TimeoutError: code ran longer than {timeout}s", False


def init_sandbox(kind, size, timeout):
    """--sandbox docker：把 run_python 换成长驻容器池（sandbox/pool.py）。池跑在后台事件循环线程里，
    评测的线程池通过 run_coroutine_threadsafe 提交任务。返回 (pool, loop) 供结束时关闭。"""
    global run_python
    if kind != "docker":
        return None
    import asyncio, threading
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from sandbox.pool import WorkerPool
    loop = asyncio.new_event_loop()
    threading.Thread(target=loop.run_forever, daemon=True).start()
    pool = WorkerPool(size=size, timeout=timeout)
    asyncio.run_coroutine_threadsafe(pool.start(), loop).result()

    def run_in_pool(code, timeout):
        r = asyncio.run_coroutine_threadsafe(pool.run(code, timeout), loop).result()
        return r["output"][-2000:], r["ok"]

    run_python = run_in_pool
    return pool, loop


def to_float(s):
    try:
        return float(s.replace(",", "").rstrip("."))
    except (ValueError, AttributeError):
        return None


def extract(text):
    m = ANSWER_RE.findall(text)
    if m:
        return to_float(m[-1]), True
    nums = NUM_RE.findall(text)
    return (to_float(nums[-1]) if nums else None), False


def solve(client, args, question, use_tool, system):
    msgs = [{"role": "system", "content": system},
            {"role": "user", "content": question}]
    calls = ok = completion_tokens = 0
    text = ""
    bare_fence = False
    for _ in range(args.max_turns if use_tool else 1):
        r = client.chat.completions.create(
            model=args.model, messages=msgs, temperature=0, max_tokens=args.max_tokens,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}})
        text = r.choices[0].message.content or ""
        completion_tokens += r.usage.completion_tokens
        msgs.append({"role": "assistant", "content": text})
        bare_fence = bare_fence or any(tag == "" for tag in FENCE_RE.findall(text)[0::2])
        code = CODE_RE.findall(text)
        if not use_tool or not code:
            break
        out, success = run_python(code[-1], args.timeout)
        calls += 1
        ok += success
        msgs.append({"role": "user", "content": f"<output>\n{out}\n</output>"})
    pred, strict = extract(text)
    return {"pred": pred, "strict_format": strict, "tool_calls": calls, "tool_ok": ok,
            "completion_tokens": completion_tokens, "bare_fence": bare_fence, "messages": msgs}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--concurrency", type=int, default=16)
    p.add_argument("--max-turns", type=int, default=4)
    p.add_argument("--max-tokens", type=int, default=1024)
    p.add_argument("--timeout", type=int, default=10)
    p.add_argument("--out", default="results")
    p.add_argument("--sandbox", choices=["subprocess", "docker"], default="subprocess",
                   help="工具代码的执行方式：宿主子进程（第 0 阶段）或 Docker 长驻容器池（第 2 阶段）")
    args = p.parse_args()
    sandbox = init_sandbox(args.sandbox, args.concurrency, args.timeout)

    data = load_dataset("openai/gsm8k", "main", split="test").select(range(args.n))
    golds = [to_float(a.split("####")[-1].strip()) for a in data["answer"]]
    client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")
    summary = {}
    modes = [("no_tool", False, SYSTEM_NOTOOL), ("python_tool", True, SYSTEM_TOOL),
             ("python_tool_forced", True, SYSTEM_TOOL_FORCED)]
    for mode, use_tool, system in modes:
        t0 = time.perf_counter()  # 单调时钟：WSL 的系统时钟会跳变，time.time() 可能测出负耗时
        with ThreadPoolExecutor(args.concurrency) as ex:
            rs = list(ex.map(lambda q: solve(client, args, q, use_tool, system), data["question"]))
        wall = time.perf_counter() - t0
        for r, g, q in zip(rs, golds, data["question"]):
            r["gold"], r["question"] = g, q
            r["correct"] = r["pred"] is not None and abs(r["pred"] - g) < 1e-4
        n = len(rs)
        used = [r for r in rs if r["tool_calls"] > 0]
        summary[mode] = {
            "accuracy": sum(r["correct"] for r in rs) / n,
            "strict_format_rate": sum(r["strict_format"] for r in rs) / n,
            "tool_use_rate": len(used) / n,
            "tool_calls_total": sum(r["tool_calls"] for r in rs),
            "tool_success_rate": (sum(r["tool_ok"] for r in rs) / max(1, sum(r["tool_calls"] for r in rs))),
            "accuracy_when_tool_used": (sum(r["correct"] for r in used) / len(used)) if used else None,
            "bare_fence_rate": sum(r["bare_fence"] for r in rs) / n,
            "avg_completion_tokens": sum(r["completion_tokens"] for r in rs) / n,
            "wall_seconds": round(wall, 1),
            "output_tokens_per_second": round(sum(r["completion_tokens"] for r in rs) / wall, 1),
        }
        with open(os.path.join(args.out, f"gsm8k_{mode}.jsonl"), "w") as f:
            for r in rs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(mode, json.dumps(summary[mode], ensure_ascii=False), flush=True)
    if sandbox:
        import asyncio
        pool, loop = sandbox
        asyncio.run_coroutine_threadsafe(pool.close(), loop).result()
    with open(os.path.join(args.out, "gsm8k_baseline_summary.json"), "w") as f:
        json.dump({"model": args.model, "n": args.n, "config": vars(args), "results": summary}, f, indent=2)


if __name__ == "__main__":
    main()
