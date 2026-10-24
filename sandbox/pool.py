"""代码沙箱：WorkerPool（长驻容器池，推荐）、SandboxPool（预热的一次性容器池），以及冷启动和裸进程两种对照。

v1 实测：一次性池的吞吐上限等于容器创建+销毁速率（本机约 13–14 个/秒），因此改用长驻容器池（见 WorkerPool）。
以下为 SandboxPool 的说明。

设计：池里预先启动 size 个空闲容器；每次执行取走一个，代码跑完（或超时）后立即销毁该容器，
后台补一个新的。这样每段代码都在全新容器里运行（隔离干净），又不用承担冷启动延迟。

每个容器的限制：禁止联网、只读根文件系统（/tmp 为内存盘）、内存 512MB 且无 swap、1 核 CPU、
最多 64 个进程、非 root 用户、去掉全部 capabilities、禁止提权。超时由宿主侧强制执行
（docker exec 自身没有超时），输出超过 max_output 字节即截断并杀掉容器。

注意：Docker 默认运行时 runc 与宿主共享内核；生产级沙箱通常再加 gVisor / Firecracker 微虚拟机。
"""
import asyncio
import os
import sys
import tempfile
import time
import uuid

LABEL = "agent-sandbox=1"
HARDEN = [
    "--network", "none", "--read-only", "--tmpfs", "/tmp:rw,size=64m",
    "--memory", "512m", "--memory-swap", "512m", "--cpus", "1", "--pids-limit", "64",
    "--user", "65534:65534", "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
    "--label", LABEL, "--workdir", "/tmp",
]


async def _docker(*args, timeout=30):
    """调用 docker CLI；任何命令最多等 timeout 秒，防止某个 docker 操作挂住拖垮整个调用方。"""
    p = await asyncio.create_subprocess_exec("docker", *args, stdout=asyncio.subprocess.PIPE,
                                             stderr=asyncio.subprocess.PIPE)
    try:
        out, err = await asyncio.wait_for(p.communicate(), timeout)
    except asyncio.TimeoutError:
        p.kill()
        await p.wait()
        return -1, "", f"docker {args[0]} timed out after {timeout}s"
    return p.returncode, out.decode(errors="replace").strip(), err.decode(errors="replace").strip()


async def _exec_capped(argv, code, timeout, max_output, kill):
    """运行 argv，把 code 从 stdin 喂进去；限制墙钟时间和输出字节数。超限时调用 kill() 结束执行。"""
    t0 = time.perf_counter()
    p = await asyncio.create_subprocess_exec(*argv, stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
                                             stderr=asyncio.subprocess.STDOUT)
    p.stdin.write(code.encode())
    p.stdin.close()
    buf, truncated, timed_out = bytearray(), False, False

    async def stop():
        # 顺序不能反：先杀本地 docker 客户端，再删容器。若客户端活着却不再读输出，dockerd 写输出会阻塞，
        # containerd 要等输出流结束才回收容器进程，docker rm -f 随之卡死（实测卡 3.5 分钟，I/O 背压死锁）
        if p.returncode is None:
            p.kill()
        await kill()

    async def read_all():
        nonlocal truncated
        while chunk := await p.stdout.read(65536):
            room = max_output - len(buf)
            buf.extend(chunk[:room])
            if len(chunk) > room:
                truncated = True
                await stop()
                break
        await p.wait()

    try:
        await asyncio.wait_for(read_all(), timeout)
    except asyncio.TimeoutError:
        timed_out = True
        await stop()
        await p.wait()
    out = buf.decode(errors="replace")
    if truncated:
        out += f"\n[output truncated at {max_output} bytes]"
    if timed_out:
        out += f"\nTimeoutError: code ran longer than {timeout}s"
    return {"output": out, "exit_code": p.returncode, "timed_out": timed_out, "truncated": truncated,
            "ok": p.returncode == 0 and not timed_out and not truncated, "exec_s": time.perf_counter() - t0}


class SandboxPool:
    def __init__(self, size=8, image="python:3.12-slim", timeout=10, max_output=4096):
        self.size, self.image, self.timeout, self.max_output = size, image, timeout, max_output
        self.ready = asyncio.Queue()
        self.spawning = set()
        self.closed = False

    async def _spawn(self):
        rc, cid, err = await _docker("run", "-d", *HARDEN, self.image, "sleep", "infinity")
        if rc != 0:
            raise RuntimeError(f"docker run failed: {err}")
        if self.closed:
            await _docker("rm", "-f", cid)
        else:
            self.ready.put_nowait(cid)

    def _refill(self):
        t = asyncio.ensure_future(self._spawn())
        self.spawning.add(t)
        t.add_done_callback(self.spawning.discard)

    async def start(self):
        await asyncio.gather(*(self._spawn() for _ in range(self.size)))

    async def run(self, code, timeout=None):
        t0 = time.perf_counter()
        cid = await self.ready.get()  # 池空时在这里等待：等待时间即"池空等待"
        wait_s = time.perf_counter() - t0
        self._refill()  # 用一个、补一个
        try:
            r = await _exec_capped(["docker", "exec", "-i", cid, "python3", "-I", "-"], code,
                                   timeout or self.timeout, self.max_output, lambda: _docker("rm", "-f", cid))
        finally:
            t = asyncio.ensure_future(_docker("rm", "-f", cid))  # 一次性：用完即销毁
            self.spawning.add(t)
            t.add_done_callback(self.spawning.discard)
        return {**r, "wait_s": wait_s}

    async def close(self):
        self.closed = True
        if self.spawning:
            await asyncio.gather(*self.spawning, return_exceptions=True)
        ids = []
        while not self.ready.empty():
            ids.append(self.ready.get_nowait())
        if ids:
            await _docker("rm", "-f", *ids)


# 在长驻容器里执行一次代码的包装脚本（作为 python3 -c 的参数传入；用户代码从 stdin 读入）。
# 用户代码在 fork 出的子进程里执行（不再另起解释器：省掉一次 20–30 ms 的 Python 启动），
# 执行完后：杀掉本次执行遗留的所有进程（kill(-1) 不会杀 PID 1 和自身），再清空整个 /tmp。
WRAPPER = r"""
import os, shutil, signal, sys, tempfile, time
code = sys.stdin.read()
d = tempfile.mkdtemp(dir="/tmp")
pid = os.fork()
if pid == 0:
    os.chdir(d)
    rc = 0
    try:
        exec(compile(code, "main.py", "exec"), {"__name__": "__main__"})
    except SystemExit as e:
        rc = e.code if isinstance(e.code, int) else (0 if e.code is None else 1)
    except BaseException:
        import traceback
        traceback.print_exc()
        rc = 1
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(rc)
rc = os.waitstatus_to_exitcode(os.waitpid(pid, 0)[1])
for _ in range(20):
    try:
        os.kill(-1, signal.SIGKILL)
    except ProcessLookupError:
        break
    time.sleep(0.01)
for name in os.listdir("/tmp"):
    p = os.path.join("/tmp", name)
    shutil.rmtree(p, ignore_errors=True) if os.path.isdir(p) else os.remove(p)
sys.exit(rc if rc >= 0 else 128 - rc)
"""
TAINT_EXIT_CODES = {137, 139, -9}  # 被 SIGKILL / 段错误 / OOM 杀掉


class WorkerPool:
    """长驻容器池：容器复用，每次执行前后清理；出现污染（超时、输出超限、被杀）或累计执行 recycle_every 次后换新。

    隔离边界：与宿主之间的隔离和一次性池相同；同一容器内前后两次执行共享内核和容器，
    靠"杀光遗留进程 + 清空 /tmp + 污染即回收"降低串扰。适用威胁模型：执行自家模型生成的代码（半可信），
    不适用于陌生用户提交的任意代码。

    容量规则：每个容器的限制只管单个容器，不管总量。实测 16 个内存炸弹同时运行时宿主可用内存下降 7.9 GB。
    要求 size × 单容器内存上限 ≤ 宿主内存的约 50%（本机 15 GB → 512 MB 时 size ≤ 12，更大的池改用 256 MB）；
    CPU 同理（size × --cpus 不宜远超核数）。
    """

    def __init__(self, size=8, image="python:3.12-slim", timeout=10, max_output=4096, recycle_every=100):
        self.size, self.image, self.timeout, self.max_output = size, image, timeout, max_output
        self.recycle_every = recycle_every
        self.free = asyncio.Queue()
        self.uses = {}
        self.pending = set()
        self.closed = False
        self.recycled = 0

    async def _spawn(self):
        rc, cid, err = await _docker("run", "-d", *HARDEN, self.image, "sleep", "infinity")
        if rc != 0:
            raise RuntimeError(f"docker run failed: {err}")
        if self.closed:
            await _docker("rm", "-f", cid)
            return
        self.uses[cid] = 0
        self.free.put_nowait(cid)

    async def _replace(self, cid):
        self.recycled += 1
        self.uses.pop(cid, None)
        await _docker("rm", "-f", cid)
        await self._spawn()

    def _bg(self, coro):
        t = asyncio.ensure_future(coro)
        self.pending.add(t)
        t.add_done_callback(self.pending.discard)

    async def start(self):
        await asyncio.gather(*(self._spawn() for _ in range(self.size)))

    async def run(self, code, timeout=None):
        t0 = time.perf_counter()
        cid = await self.free.get()  # 所有容器都在忙时在这里等待："容器忙等待"
        wait_s = time.perf_counter() - t0
        r = await _exec_capped(["docker", "exec", "-i", cid, "python3", "-I", "-S", "-c", WRAPPER], code,
                               timeout or self.timeout, self.max_output, lambda: _docker("rm", "-f", cid))
        self.uses[cid] = self.uses.get(cid, 0) + 1
        tainted = r["timed_out"] or r["truncated"] or r["exit_code"] in TAINT_EXIT_CODES
        if tainted or self.uses[cid] >= self.recycle_every:
            self._bg(self._replace(cid))  # 污染即回收：换新容器，期间该槽位不可用
        else:
            self.free.put_nowait(cid)
        return {**r, "wait_s": wait_s, "recycled": tainted}

    async def close(self):
        self.closed = True
        while self.pending:
            await asyncio.gather(*list(self.pending), return_exceptions=True)
        ids = []
        while not self.free.empty():
            ids.append(self.free.get_nowait())
        if ids:
            await _docker("rm", "-f", *ids)


async def run_cold(code, timeout=10, image="python:3.12-slim", max_output=4096):
    """对照组：每次执行都 docker run 一个新容器（冷启动）。"""
    name = f"sbx-cold-{uuid.uuid4().hex[:12]}"
    r = await _exec_capped(["docker", "run", "--rm", "-i", "--name", name, *HARDEN, image, "python3", "-I", "-"],
                           code, timeout, max_output, lambda: _docker("rm", "-f", name))
    return {**r, "wait_s": 0.0}


async def run_subprocess(code, timeout=10, max_output=4096):
    """对照组：第 0 阶段的做法——宿主上直接起 Python 子进程，没有隔离。"""
    tmp = tempfile.mkdtemp()
    proc = {}

    async def kill():
        if "p" in proc and proc["p"].returncode is None:
            proc["p"].kill()

    argv = [sys.executable, "-I", "-"]
    p = await asyncio.create_subprocess_exec(*argv, cwd=tmp, stdin=asyncio.subprocess.PIPE,
                                             stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
    proc["p"] = p
    t0 = time.perf_counter()
    p.stdin.write(code.encode())
    p.stdin.close()
    try:
        out, _ = await asyncio.wait_for(p.communicate(), timeout)
        timed_out = False
    except asyncio.TimeoutError:
        await kill()
        out, timed_out = b"", True
    text = out[:max_output].decode(errors="replace")
    return {"output": text, "exit_code": p.returncode, "timed_out": timed_out, "truncated": len(out) > max_output,
            "ok": p.returncode == 0 and not timed_out, "exec_s": time.perf_counter() - t0, "wait_s": 0.0}


async def cleanup_all():
    """删除所有带沙箱标签的容器（包括异常退出后的残留）。返回删除数量。"""
    _, ids, _ = await _docker("ps", "-aq", "--filter", f"label={LABEL}")
    ids = ids.split()
    if ids:
        await _docker("rm", "-f", *ids)
    return len(ids)
