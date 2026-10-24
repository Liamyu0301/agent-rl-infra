"""沙箱执行压测：裸进程（无隔离基线）vs 冷启动（每次 docker run）vs 预热的一次性容器池。

闭环：concurrency 个工作协程，每个执行完一段代码立刻执行下一段，直到总数达到 n。
代码是 GSM8K 工具调用里典型的短计算，因此测到的基本是沙箱本身的开销。
用法：.venv/bin/python sandbox/bench_sandbox.py --mode pool --concurrency 32 [--pool-size 32]
"""
import argparse
import asyncio
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sandbox.pool import SandboxPool, WorkerPool, cleanup_all, run_cold, run_subprocess  # noqa: E402

CODE = "total_serve = 60\ncartons = total_serve // 15\nprint(cartons * 4.0)\n"


def pct(xs, p):
    xs = sorted(xs)
    return xs[max(0, math.ceil(p / 100 * len(xs)) - 1)] if xs else None


def cpu_ticks():
    """/proc/stat 第一行：(忙碌 tick, 总 tick)，用来算整机 CPU 利用率。"""
    v = [int(x) for x in open("/proc/stat").readline().split()[1:]]
    idle = v[3] + v[4]
    return sum(v) - idle, sum(v)


async def main(args):
    pool = None
    if args.mode in ("pool", "workers"):
        cls = SandboxPool if args.mode == "pool" else WorkerPool
        pool = cls(size=args.pool_size or args.concurrency, timeout=10)
        t0 = time.perf_counter()
        await pool.start()
        warm_s = time.perf_counter() - t0
        run = pool.run
    elif args.mode == "cold":
        run, warm_s = run_cold, 0.0
    else:
        run, warm_s = run_subprocess, 0.0

    n = max(args.min_tasks, args.concurrency * 8)
    remaining = n
    results = []

    async def worker():
        nonlocal remaining
        while remaining > 0:
            remaining -= 1
            t0 = time.perf_counter()
            r = await run(CODE, timeout=10)
            r["latency_s"] = time.perf_counter() - t0
            results.append(r)

    busy0, tot0 = cpu_ticks()
    t0 = time.perf_counter()
    await asyncio.gather(*(worker() for _ in range(args.concurrency)))
    wall = time.perf_counter() - t0
    busy1, tot1 = cpu_ticks()
    if pool:
        await pool.close()
    await asyncio.sleep(1)
    leftover = await cleanup_all()

    lat = [r["latency_s"] * 1000 for r in results]
    wait = [r["wait_s"] * 1000 for r in results]
    wrong = sum(not r["ok"] or r["output"].strip() != "16.0" for r in results)
    summary = {
        "mode": args.mode, "concurrency": args.concurrency, "pool_size": pool.size if pool else None,
        "n": len(results), "failures": wrong, "pool_warmup_s": round(warm_s, 2), "wall_s": round(wall, 2),
        "exec_per_s": round(len(results) / wall, 1),
        "latency_ms": {"p50": round(pct(lat, 50), 1), "p90": round(pct(lat, 90), 1), "p99": round(pct(lat, 99), 1)},
        "pool_wait_ms": {"p50": round(pct(wait, 50), 1), "p99": round(pct(wait, 99), 1),
                         "share_waited": round(sum(w > 1 for w in wait) / len(wait), 3)},
        "host_cpu_util": round((busy1 - busy0) / max(1, tot1 - tot0), 3),
        "leftover_containers": leftover,
    }
    os.makedirs(args.out, exist_ok=True)
    tag = f"{args.mode}_c{args.concurrency}" + (f"_p{args.pool_size}" if args.pool_size else "")
    with open(os.path.join(args.out, f"{tag}.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["subprocess", "cold", "pool", "workers"], required=True)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--pool-size", type=int, default=0, help="默认等于并发数")
    ap.add_argument("--min-tasks", type=int, default=200)
    ap.add_argument("--out", default="results/phase2/bench")
    asyncio.run(main(ap.parse_args()))
