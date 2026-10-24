"""微基准：向长驻容器 docker exec 执行代码的吞吐（容器不销毁、不补充），用来决定沙箱改造方向。"""
import argparse
import asyncio
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sandbox.pool import HARDEN, _docker, _exec_capped, cleanup_all  # noqa: E402
from sandbox.bench_sandbox import CODE, cpu_ticks, pct  # noqa: E402


async def main(args):
    ids = []
    for _ in range(args.concurrency):
        rc, cid, err = await _docker("run", "-d", *HARDEN, "python:3.12-slim", "sleep", "infinity")
        assert rc == 0, err
        ids.append(cid)
    free = asyncio.Queue()
    for cid in ids:
        free.put_nowait(cid)
    n = max(200, args.concurrency * 8)
    remaining, lat, fails = n, [], 0

    async def worker():
        nonlocal remaining, fails
        while remaining > 0:
            remaining -= 1
            cid = await free.get()
            t0 = time.perf_counter()
            r = await _exec_capped(["docker", "exec", "-i", cid, "python3", "-I", "-"], CODE, 10, 4096,
                                   lambda: _docker("rm", "-f", cid))
            lat.append((time.perf_counter() - t0) * 1000)
            fails += (not r["ok"]) or r["output"].strip() != "16.0"
            free.put_nowait(cid)

    b0, t0c = cpu_ticks()
    t0 = time.perf_counter()
    await asyncio.gather(*(worker() for _ in range(args.concurrency)))
    wall = time.perf_counter() - t0
    b1, t1c = cpu_ticks()
    await cleanup_all()
    print(json.dumps({"mode": "docker_exec_longlived", "concurrency": args.concurrency, "n": len(lat), "failures": fails,
                      "exec_per_s": round(len(lat) / wall, 1),
                      "latency_ms": {"p50": round(pct(lat, 50), 1), "p99": round(pct(lat, 99), 1)},
                      "host_cpu_util": round((b1 - b0) / max(1, t1c - t0c), 3)}), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--concurrency", type=int, default=8)
    asyncio.run(main(ap.parse_args()))
