"""第 2 档结果分析与关卡判断（规则在开机前定好，写在这里，运行中不改）。

  analyze.py g0 RES                 G0：双卡服务端模式能否正确训练；退出码 0 = 通过
  analyze.py profile RES A B        两次运行的分段计时（第 3 步起取平均）+ GPU 利用率，写 RES/profile_A_vs_B.md
  analyze.py g1 RES RUN             G1：按实测每步耗时预测花费，决定正式训练跑几个种子，写 RES/g1.json
  analyze.py fused_check RES RUN    冗余前向优化在服务端模式下是否数值等价；退出码 0 = 等价
"""
import csv
import json
import os
import sys
import time

LOGP = "sampling/sampling_logp_difference/mean"
SKIP = 2  # 前 2 步含编译 / 预热，不计入平均


def jl(path):
    return [json.loads(line) for line in open(path)] if os.path.exists(path) else []


def per_step(res, run, key):
    return {r["step"]: r[key] for r in jl(f"{res}/{run}/trl_log.jsonl") if key in r}


def g0(res):
    v = {"colocate_ok": os.path.exists(f"{res}/g0_colocate06/DONE"),
         "server_ok": os.path.exists(f"{res}/g0_server06/DONE")}
    rc = open(f"{res}/g0/test_rollout_06.rc").read().split()[-1] if os.path.exists(f"{res}/g0/test_rollout_06.rc") else "?"
    v["test_rollout_ok"] = rc == "0"
    c, s = per_step(res, "g0_colocate06", LOGP), per_step(res, "g0_server06", LOGP)
    v["logp_diff_colocate"], v["logp_diff_server"] = c, s
    steps = [k for k in s if k in c and k > 1]
    if steps:
        cm, sm, last = sum(c[k] for k in steps) / len(steps), sum(s[k] for k in steps) / len(steps), max(steps)
        # 服务端没收到新权重时，lr 1e-3 下差异会随步数变大；允许 2 倍 + 0.005 的波动
        v["logp_ok"] = sm <= 2 * cm + 0.005 and s[last] <= 2 * c[last] + 0.005
        v.update(logp_mean_colocate=cm, logp_mean_server=sm)
    else:
        v["logp_ok"] = False
    v["pass"] = all(v[k] for k in ("colocate_ok", "server_ok", "test_rollout_ok", "logp_ok"))
    json.dump(v, open(f"{res}/g0/verdict.json", "w"), indent=2)
    print(json.dumps(v))
    return 0 if v["pass"] else 1


def seg_means(res, run):
    rows = [r for r in jl(f"{res}/{run}/steps.jsonl") if r["step"] > SKIP]
    if not rows:
        return None, {}
    keys = sorted({k for r in rows for k in r.get("segments", {})})
    seg = {k: sum(r.get("segments", {}).get(k, 0.0) for r in rows) / len(rows) for k in keys}
    return sum(r["wall_s"] for r in rows) / len(rows), seg


def gpu_stats(res, run):
    """每张卡的平均利用率、利用率 > 50% 的时间占比、两张卡同时 > 50% 的时间占比。"""
    p = f"{res}/{run}/gpu.csv"
    if not os.path.exists(p):
        return None
    by_t = {}
    for row in csv.reader(open(p)):
        if len(row) < 4:
            continue
        try:
            by_t.setdefault(row[0].strip(), {})[int(row[1])] = float(row[2])
        except ValueError:
            continue
    ts = [u for u in by_t.values() if len(u) >= 2]
    if not ts:
        return None
    out = {}
    for g in sorted(ts[0]):
        out[f"gpu{g}_util_mean"] = sum(u[g] for u in ts) / len(ts)
        out[f"gpu{g}_busy_frac"] = sum(u[g] > 50 for u in ts) / len(ts)
    out["both_busy_frac"] = sum(all(x > 50 for x in u.values()) for u in ts) / len(ts)
    out["samples"] = len(ts)
    return out


def profile(res, a, b):
    lines = [f"| 段 | {a} | {b} |", "|---|---|---|"]
    wa, sa = seg_means(res, a)
    wb, sb = seg_means(res, b)
    for k in sorted(set(sa) | set(sb)):
        f = lambda s, w: f"{s[k]:.2f} s ({100 * s[k] / w:.1f}%)" if k in s and w else "-"  # noqa: E731
        lines.append(f"| {k} | {f(sa, wa)} | {f(sb, wb)} |")
    fw = lambda w: f"{w:.2f} s" if w else "-"  # noqa: E731
    lines.append(f"| **每步墙钟（第 {SKIP + 1} 步起平均）** | {fw(wa)} | {fw(wb)} |")
    for run in (a, b):
        g = gpu_stats(res, run)
        if g:
            lines.append(f"\n{run} GPU：" + "，".join(f"{k}={v:.2f}" if isinstance(v, float) else f"{k}={v}"
                                                      for k, v in g.items()))
    text = "\n".join(lines)
    open(f"{res}/profile_{a}_vs_{b}.md", "w").write(text + "\n")
    print(text)
    return 0


def g1(res, run):
    """预算规则（开机前与用户约定）：按实测每步耗时预测第 2 档总花费，累计不超过 cap_cny 才跑第 2 个种子。"""
    b = json.load(open(os.path.expanduser("~/tier2_budget.json")))
    s_per_step, _ = seg_means(res, run)
    if s_per_step is None:
        dec = {"seeds": [42], "reason": "没有计时数据，保守只跑种子 42"}
    else:
        elapsed_h = (time.time() - b["launch_epoch"]) / 3600
        start_min = 4  # 每次重启 1.7B 服务端（含编译）约 3–4 分钟
        s23_h = (2 * 10 * s_per_step + 2 * start_min * 60) / 3600
        seed_h = (200 * s_per_step + start_min * 60 + 6 * 60) / 3600  # 训练 + 重启服务端 + 评测
        tail_h = 10 / 60                                              # 下载结果、终止
        cny = lambda n: b["prior_cny"] + (elapsed_h + s23_h + n * seed_h + tail_h) * b["usd_per_hour"] * b["cny_per_usd"]  # noqa: E731
        # 一个种子都超上限时不开始正式训练：否则会被保险在训练中途终止，既没有适配器也没有评测
        seeds = [42, 1] if cny(2) <= b["cap_cny"] else [42] if cny(1) <= b["cap_cny"] else []
        reason = {2: "预测累计花费不超过上限", 1: "两个种子会超过上限，只跑种子 42（种子 1 待补）",
                  0: "一个种子也会超过上限，不跑正式训练，等用户决定"}[len(seeds)]
        dec = {"seeds": seeds, "s_per_step": round(s_per_step, 2), "elapsed_h": round(elapsed_h, 2),
               "projected_cny_1seed": round(cny(1)), "projected_cny_2seeds": round(cny(2)), "cap_cny": b["cap_cny"],
               "reason": reason}
    json.dump(dec, open(f"{res}/g1.json", "w"), indent=2, ensure_ascii=False)
    print(json.dumps(dec, ensure_ascii=False))
    return 0


def fused_check(res, run):
    rows = jl(f"{res}/{run}/fused_is_check.jsonl")
    v = {"n": len(rows)}
    if rows:
        v["logps_absdiff_max"] = max(r["logps_absdiff_max"] for r in rows)
        v["is_ratio_absdiff_max"] = max(r["is_ratio_absdiff_max"] for r in rows)
    # 第 3.5 步同卡模式下 logprob 逐位相同、比值差 2.4e-7；这里允许到 1e-5
    v["equivalent"] = bool(rows) and v["logps_absdiff_max"] <= 1e-5 and v["is_ratio_absdiff_max"] <= 1e-5
    json.dump(v, open(f"{res}/{run}/verdict.json", "w"), indent=2)
    print(json.dumps(v))
    return 0 if v["equivalent"] else 1


if __name__ == "__main__":
    cmd, *rest = sys.argv[1:]
    sys.exit({"g0": g0, "profile": profile, "g1": g1, "fused_check": fused_check}[cmd](*rest))
