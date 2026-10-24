"""第 1 阶段：Agent 型负载的 vLLM 推理压测（闭环 closed-loop）。

所有 prompt 都按 token 精确构造，直接把 token id 发给 /v1/completions（绕开 chat 模板带来的长度误差）。
- agent：所有会话共享同一段 system prompt + 工具定义（--prefix 个 token）。每个会话先有一段任务描述（--task），
  之后每轮：模型输出 --output 个 token（ignore_eos 固定长度）→ 把真实生成的 token 和一段工具返回结果（--tool-result）
  追加进上下文 → 等待"工具执行"（--gap-min ~ --gap-max 秒，不计入任何延迟指标）→ 下一轮，共 --turns 轮。
- chat：对照组。每个请求的 prompt 长度与 agent 同一位置完全相同，但内容全部随机、互不重复，没有可复用的前缀。

并发（两种模式）：
- 闭环（默认）：--concurrency 个槽位同时运行，每个槽位依次跑会话，直到总请求数达到 --min-requests。
- 开环（--arrival-rate > 0）：--n-sessions 个会话按泊松过程到达（平均每秒 --arrival-rate 个），
  到达时间只由 --seed 决定，因此不同配置面对完全相同的到达序列。
会话级闸门（--gate K）：同时活跃的会话最多 K 个，新会话在闸门排队；排队时间计入该会话第 0 轮的
"用户感知 TTFT"（user_ttft = 闸门等待 + 服务端 TTFT），SLO 达标率和有效吞吐都按 user_ttft 计算。
服务端指标（/metrics）在压测前后各取一次做差；KV Cache 占用等瞬时值每 0.5 秒采样一次。
"""
import argparse, asyncio, json, math, os, random, time

import httpx

URL = "http://localhost:8000"
VOCAB = (1000, 100000)  # 随机 token id 范围，避开 Qwen 的特殊 token（>= 151643）
COUNTERS = ("vllm:prefix_cache_hits_total", "vllm:prefix_cache_queries_total", "vllm:num_preemptions_total")
GAUGES = ("vllm:kv_cache_usage_perc", "vllm:num_requests_running", "vllm:num_requests_waiting")


def rand_ids(rng, n):
    return [rng.randrange(*VOCAB) for _ in range(n)]


def pct(xs, p):
    xs = sorted(xs)
    return xs[max(0, math.ceil(p / 100 * len(xs)) - 1)] if xs else None


def dist_ms(xs):
    return {k: round(v * 1000, 1) for k, v in
            {"p50": pct(xs, 50), "p90": pct(xs, 90), "p99": pct(xs, 99), "mean": sum(xs) / len(xs)}.items()}


async def scrape(client):
    out = dict.fromkeys(COUNTERS + GAUGES, 0.0)
    for line in (await client.get(f"{URL}/metrics")).text.splitlines():
        if line.startswith("#"):
            continue
        name, _, val = line.rpartition(" ")
        name = name.split("{")[0]
        if name in out:
            out[name] += float(val)
    return out


async def complete(client, args, prompt_ids):
    body = {"model": args.model, "prompt": prompt_ids, "max_tokens": args.output, "ignore_eos": True,
            "temperature": 0, "stream": True, "stream_options": {"include_usage": True}, "return_token_ids": True}
    t0 = time.perf_counter()
    ttft, out_ids, usage = None, [], None
    async with client.stream("POST", f"{URL}/v1/completions", json=body) as r:
        r.raise_for_status()
        async for line in r.aiter_lines():
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            chunk = json.loads(line[6:])
            if chunk.get("usage"):
                usage = chunk["usage"]
            for c in chunk.get("choices", []):
                ids = c.get("token_ids") or []
                if ids and ttft is None:
                    ttft = time.perf_counter() - t0
                out_ids += ids
    return t0, out_ids, usage, ttft, time.perf_counter() - t0


async def run_session(client, args, sid, shared, records, origin, gate_wait=0.0):
    # 每次运行（tag）、每个会话的随机内容各自独立，避免跨运行/跨负载意外命中前缀缓存
    rng = random.Random(f"{args.tag}-{args.seed}-{sid}")
    ctx = shared + rand_ids(rng, args.task) if args.workload == "agent" else None
    expected = args.prefix + args.task
    for turn in range(args.turns):
        prompt = ctx if args.workload == "agent" else rand_ids(rng, expected)
        t0, out_ids, usage, ttft, e2e = await complete(client, args, prompt)
        n_out = len(out_ids)
        records.append({"session": sid, "turn": turn, "start": round(t0 - origin, 4),
                        "prompt_tokens": usage["prompt_tokens"], "expected_prompt_tokens": expected,
                        "completion_tokens": usage["completion_tokens"], "streamed_tokens": n_out,
                        "ttft": ttft, "e2e": e2e, "tpot": (e2e - ttft) / max(1, n_out - 1),
                        "user_ttft": ttft + (gate_wait if turn == 0 else 0.0)})
        if args.workload == "agent":
            ctx = ctx + out_ids + rand_ids(rng, args.tool_result)
        expected += args.output + args.tool_result
        if turn < args.turns - 1:
            await asyncio.sleep(rng.uniform(args.gap_min, args.gap_max))


async def bench(args):
    # 共享 system prompt 只由 seed 决定：像真实系统一样，跨运行保持不变
    shared = rand_ids(random.Random(f"shared-{args.seed}"), args.prefix)
    # 连接池上限要大于可能的最大在途请求数：开环时所有会话都可能同时在途，否则连接池会变成隐形闸门
    pool = (args.n_sessions if args.arrival_rate > 0 else args.concurrency) + 8
    limits = httpx.Limits(max_connections=pool, max_keepalive_connections=pool)
    async with httpx.AsyncClient(timeout=900, limits=limits) as client:
        if args.warmup:  # 预热：2 个短会话（会话 id 为负，不与正式会话重复）
            w = argparse.Namespace(**{**vars(args), "turns": 2, "gap_min": 0.0, "gap_max": 0.0})
            await asyncio.gather(*(run_session(client, w, -1 - i, shared, [], 0.0) for i in range(2)))

        open_loop = args.arrival_rate > 0
        n_sessions = args.n_sessions if open_loop else max(args.concurrency, math.ceil(args.min_requests / args.turns))
        queue = asyncio.Queue()
        for sid in range(n_sessions):
            queue.put_nowait(sid)
        records, sessions, samples, stop = [], [], [], asyncio.Event()
        gate = asyncio.Semaphore(args.gate) if args.gate > 0 else None

        async def sampler():
            while not stop.is_set():
                samples.append(await scrape(client))
                try:
                    await asyncio.wait_for(stop.wait(), 0.5)
                except asyncio.TimeoutError:
                    pass

        async def arrive(sid, at):
            await asyncio.sleep(max(0.0, at - (time.perf_counter() - origin)))
            t_arr = time.perf_counter()
            if gate:
                await gate.acquire()
            t_adm = time.perf_counter()
            try:
                await run_session(client, args, sid, shared, records, origin, gate_wait=t_adm - t_arr)
            finally:
                if gate:
                    gate.release()
            sessions.append({"session": sid, "arrive": t_arr - origin, "gate_wait": t_adm - t_arr,
                             "session_e2e": time.perf_counter() - t_arr})

        async def worker(slot):
            await asyncio.sleep(random.Random(slot).uniform(0, args.gap_max))  # 错开启动，避免所有会话同一时刻起步
            while not queue.empty():
                await run_session(client, args, queue.get_nowait(), shared, records, origin)

        m0 = await scrape(client)
        origin, cpu0 = time.perf_counter(), time.process_time()
        sampler_task = asyncio.create_task(sampler())
        if open_loop:  # 到达序列只依赖 seed：所有对比配置面对相同的到达时间
            arng, at, arrivals = random.Random(f"arrivals-{args.seed}"), 0.0, []
            for sid in range(n_sessions):
                at += arng.expovariate(args.arrival_rate)
                arrivals.append(arrive(sid, at))
            await asyncio.gather(*arrivals)
        else:
            await asyncio.gather(*(worker(i) for i in range(args.concurrency)))
        wall, cpu = time.perf_counter() - origin, time.process_time() - cpu0
        stop.set()
        await sampler_task
        m1 = await scrape(client)

    d = {k: m1[k] - m0[k] for k in COUNTERS}
    ttft = [r["ttft"] for r in records]
    summary = {
        "tag": args.tag, "config": vars(args), "n_requests": len(records), "n_sessions": n_sessions,
        "token_check_failures": sum(r["prompt_tokens"] != r["expected_prompt_tokens"]
                                    or r["completion_tokens"] != args.output
                                    or r["streamed_tokens"] != args.output for r in records),
        "wall_s": round(wall, 2), "client_cpu_util": round(cpu / wall, 3),
        "avg_in_flight": round(sum(r["e2e"] for r in records) / wall, 2),
        "req_per_s": round(len(records) / wall, 2),
        "output_tok_per_s": round(sum(r["completion_tokens"] for r in records) / wall, 1),
        "prompt_tok_per_s": round(sum(r["prompt_tokens"] for r in records) / wall, 1),
        "ttft_ms": dist_ms(ttft), "tpot_ms": dist_ms([r["tpot"] for r in records]),
        "e2e_ms": dist_ms([r["e2e"] for r in records]),
        "prefix_cache_hit_rate": round(d[COUNTERS[0]] / d[COUNTERS[1]], 4) if d[COUNTERS[1]] else None,
        "prefix_cache_query_tokens": d[COUNTERS[1]], "preemptions": d[COUNTERS[2]],
        "kv_usage_max": round(max(s[GAUGES[0]] for s in samples), 3),
        "kv_usage_mean": round(sum(s[GAUGES[0]] for s in samples) / len(samples), 3),
        "running_max": max(s[GAUGES[1]] for s in samples), "waiting_max": max(s[GAUGES[2]] for s in samples),
        "user_ttft_ms": dist_ms([r["user_ttft"] for r in records]),
        "slo_ttft_ms": args.slo_ttft_ms,
        "slo_attainment": round(sum(r["user_ttft"] * 1000 <= args.slo_ttft_ms for r in records) / len(records), 4),
        "goodput_req_per_s": round(sum(r["user_ttft"] * 1000 <= args.slo_ttft_ms for r in records) / wall, 2),
    }
    if sessions:
        summary["gate_wait_ms"] = dist_ms([x["gate_wait"] for x in sessions])
        summary["session_e2e_ms"] = dist_ms([x["session_e2e"] for x in sessions])
        summary["sessions_per_s"] = round(len(sessions) / wall, 3)
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, f"{args.tag}.jsonl"), "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    with open(os.path.join(args.out, f"{args.tag}.summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    brief = {k: summary[k] for k in ("tag", "n_requests", "token_check_failures", "req_per_s", "output_tok_per_s",
                                     "prefix_cache_hit_rate", "preemptions", "kv_usage_max", "waiting_max",
                                     "avg_in_flight", "client_cpu_util")}
    brief.update(ttft_p50=summary["ttft_ms"]["p50"], ttft_p99=summary["ttft_ms"]["p99"],
                 tpot_p50=summary["tpot_ms"]["p50"], tpot_p99=summary["tpot_ms"]["p99"],
                 user_ttft_p99=summary["user_ttft_ms"]["p99"], slo_attainment=summary["slo_attainment"],
                 goodput=summary["goodput_req_per_s"], wall_s=summary["wall_s"])
    if sessions:
        brief.update(gate_wait_p50=summary["gate_wait_ms"]["p50"], gate_wait_p99=summary["gate_wait_ms"]["p99"],
                     session_e2e_p50=summary["session_e2e_ms"]["p50"], sessions_per_s=summary["sessions_per_s"])
    print(json.dumps(brief), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--workload", choices=["agent", "chat"], default="agent")
    p.add_argument("--model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--min-requests", type=int, default=256)
    p.add_argument("--turns", type=int, default=8)
    p.add_argument("--prefix", type=int, default=2000, help="共享 system prompt + 工具定义的 token 数")
    p.add_argument("--task", type=int, default=100, help="每个会话的任务描述 token 数")
    p.add_argument("--tool-result", type=int, default=300, help="每轮工具返回结果的 token 数")
    p.add_argument("--output", type=int, default=100, help="每轮模型输出的 token 数")
    p.add_argument("--gap-min", type=float, default=0.5)
    p.add_argument("--gap-max", type=float, default=1.5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--arrival-rate", type=float, default=0.0, help="开环：会话平均到达速率（个/秒），0 表示闭环")
    p.add_argument("--n-sessions", type=int, default=96, help="开环：会话总数")
    p.add_argument("--gate", type=int, default=0, help="会话级闸门：同时活跃会话上限，0 表示不限制")
    p.add_argument("--slo-ttft-ms", type=float, default=500.0, help="SLO：用户感知 TTFT 上限（毫秒）")
    p.add_argument("--no-warmup", dest="warmup", action="store_false")
    p.add_argument("--tag", required=True)
    p.add_argument("--out", default="results/phase1")
    asyncio.run(bench(p.parse_args()))


if __name__ == "__main__":
    main()
