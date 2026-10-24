"""沙箱安全测试套件：每一项攻击都必须被拦截，且宿主不受影响、测完没有残留容器。

用法：.venv/bin/python sandbox/security_test.py [--mode pool|cold|workers]
workers 模式用大小为 1 的长驻容器池，强制所有用例落在同一个容器里，用来检验执行之间的串扰。
"""
import argparse
import asyncio
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sandbox.pool import SandboxPool, WorkerPool, cleanup_all, run_cold  # noqa: E402

TIMEOUT = 3

CASES = [
    # (名称, 代码, 判定函数(结果) -> 是否通过, 说明)
    ("正常执行", "print(1 + 1)",
     lambda r: r["ok"] and r["output"].strip() == "2", "基本功能可用"),
    ("死循环", "while True:\n    pass",
     lambda r: r["timed_out"] and r["elapsed"] < TIMEOUT + 3, "超时后被强制结束"),
    ("进程炸弹", "import os\nwhile True:\n    os.fork()",
     lambda r: not r["ok"] and r["elapsed"] < TIMEOUT + 3, "进程数上限挡住，宿主进程数不受影响"),
    ("内存炸弹", "x = bytearray(2 * 1024 ** 3)\nprint(len(x))",
     lambda r: not r["ok"] and "2147483648" not in r["output"], "超过 512MB 限制被拒绝或被 OOM 杀掉"),
    ("联网", "import urllib.request\nurllib.request.urlopen('http://example.com', timeout=2)",
     lambda r: not r["ok"] and r["elapsed"] < 2.5, "无网络，立即失败"),
    ("写系统目录", "open('/etc/pwned', 'w').write('x')",
     lambda r: not r["ok"], "只读根文件系统"),
    ("读宿主文件", "print(open('/home/liam/agent-rl-infra/ENV_NOTES.md').read()[:50])",
     lambda r: not r["ok"] and "FileNotFoundError" in r["output"], "宿主目录没有挂载进容器"),
    ("超大输出", "print('a' * 10 ** 8)",
     lambda r: r["truncated"] and len(r["output"]) < 5000, "输出被截断，宿主不会被撑爆"),
    ("非 root", "import os\nprint(os.getuid())",
     lambda r: r["ok"] and r["output"].strip() != "0", "以 nobody 身份运行"),
    ("/tmp 可写", "open('/tmp/a.txt', 'w').write('ok')\nprint(open('/tmp/a.txt').read())",
     lambda r: r["ok"] and r["output"].strip() == "ok", "临时目录可用（正常代码需要）"),
]


def host_mem_available_mb():
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable"):
            return int(line.split()[1]) // 1024


def host_pids():
    return len([p for p in os.listdir("/proc") if p.isdigit()])


async def main(mode):
    pool = None
    if mode in ("pool", "workers"):
        pool = SandboxPool(size=4, timeout=TIMEOUT) if mode == "pool" else WorkerPool(size=1, timeout=TIMEOUT)
        await pool.start()
        run = pool.run
    else:
        async def run(code, timeout=TIMEOUT):
            return await run_cold(code, timeout=timeout)

    mem0, pids0 = host_mem_available_mb(), host_pids()
    results, min_mem = [], mem0
    for name, code, check, why in CASES:
        t0 = time.perf_counter()
        r = await run(code, timeout=TIMEOUT)
        r["elapsed"] = time.perf_counter() - t0
        min_mem = min(min_mem, host_mem_available_mb())
        passed = bool(check(r))
        results.append(passed)
        tail = r["output"].strip().splitlines()[-1][:80] if r["output"].strip() else ""
        print(f"{'PASS' if passed else 'FAIL'}  {name:<8} exit={r['exit_code']} timeout={r['timed_out']} "
              f"trunc={r['truncated']} {r['elapsed']:.2f}s | {why} | {tail}", flush=True)

    # 一次性：上一段代码写进 /tmp 的文件，下一段代码必须看不到
    await run("open('/tmp/secret', 'w').write('leak')", timeout=TIMEOUT)
    r = await run("import os\nprint(os.path.exists('/tmp/secret'))", timeout=TIMEOUT)
    iso = r["ok"] and r["output"].strip() == "False"
    results.append(iso)
    print(f"{'PASS' if iso else 'FAIL'}  执行间隔离  上一段写入的文件对下一段不可见 | {r['output'].strip()[-40:]}")

    # 执行之间的串扰：前一次执行的副作用不能影响后一次（workers 模式下两次执行落在同一个容器）
    probe_bg = ("import os\nfound = False\nfor p in os.listdir('/proc'):\n    if p.isdigit():\n"
                "        try:\n            found |= b'sleep\\x001000' in open(f'/proc/{p}/cmdline', 'rb').read()\n"
                "        except OSError:\n            pass\nprint(found)")
    sequences = [
        ("遗留后台进程", "import subprocess\nsubprocess.Popen(['sleep', '1000'])\nprint('spawned')", probe_bg,
         lambda r, dt: r["ok"] and r["output"].strip() == "False", "上一次留下的后台进程必须已被清掉"),
        ("超时后恢复", "while True:\n    pass", "print('alive')",
         lambda r, dt: r["ok"] and dt < 2.0, "上一次超时后，下一次正常执行且不被拖慢"),
        ("OOM 后恢复", "x = bytearray(2 * 1024 ** 3)", "print('alive')",
         lambda r, dt: r["ok"] and dt < 2.0, "上一次被 OOM 杀掉后，下一次正常执行"),
        ("进程炸弹后恢复", "import os\nwhile True:\n    os.fork()", "print('alive')",
         lambda r, dt: r["ok"] and dt < 2.0, "上一次进程炸弹后，下一次正常执行且不被拖慢"),
    ]
    for name, first, second, check, why in sequences:
        await run(first, timeout=TIMEOUT)
        t0 = time.perf_counter()
        r = await run(second, timeout=TIMEOUT)
        dt = time.perf_counter() - t0
        ok = bool(check(r, dt))
        results.append(ok)
        print(f"{'PASS' if ok else 'FAIL'}  {name:<8} 第二次执行 {dt:.2f}s | {why} | {r['output'].strip()[-40:]}", flush=True)

    recycled = getattr(pool, "recycled", None)
    if pool:
        await pool.close()
    await asyncio.sleep(1)
    leftover = await cleanup_all()
    mem1, pids1 = host_mem_available_mb(), host_pids()
    print(json.dumps({"mode": mode, "passed": sum(results), "total": len(results), "recycled_containers": recycled,
                      "host_mem_available_mb": {"before": mem0, "min_during": min_mem, "after": mem1},
                      "host_pids": {"before": pids0, "after": pids1}, "leftover_containers": leftover},
                     ensure_ascii=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["pool", "cold", "workers"], default="pool")
    asyncio.run(main(ap.parse_args().mode))
