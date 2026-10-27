"""第 3 档（异步 rollout）关卡判断与汇总（规则在开机前定好，写在这里，运行中不改）。

  analyze.py g0 RES        G0：异步流水线是否正确（sync17 / async17 都跑完 + 落后步数恰好为 1 + 同步与生成不重叠）；退出码 0 = 通过
  analyze.py g1 RES RUN    RUN 每步耗时须小于 sync17（异步确实更快），再按它预测花费，决定种子 42 / 1 能否开跑，写 RES/g1.json
  analyze.py g1b RES       种子 1 开跑前复核：种子 42 准确率 ≥ 78.2% 且花费仍不超上限，写 RES/g1b.json；退出码 0 = 跑
  analyze.py summary RES   汇总表（每步耗时、两卡同时工作占比、IS 屏蔽比例、logprob 差、等待时间），写 RES/summary.md
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tier2"))
from analyze import LOGP, SKIP, gpu_stats, jl, per_step, seg_means  # noqa: E402

ACC_MIN = 0.812 - 0.03  # 第 2 档同步模式种子 42：81.2%
BUDGET = os.path.expanduser("~/tier3_budget.json")


def async_checks(res, run):
    recs = jl(f"{res}/{run}/async.jsonl")
    batches = [r for r in recs if r["event"] == "batch"]
    syncs = [r for r in recs if r["event"] == "sync"]
    stale = [b["staleness"] for b in batches]
    overlaps = sum(1 for s in syncs for b in batches
                   if s["start"] < b["gen_end"] and b["gen_start"] < s["end"])
    return {"batches": len(batches), "syncs": len(syncs),
            "staleness_ok": bool(stale) and stale[0] == 0 and all(x == 1 for x in stale[1:]),
            "staleness_values": sorted(set(stale)), "sync_gen_overlaps": overlaps,
            "wait_s_mean": round(sum(b["wait_s"] for b in batches[1:]) / max(1, len(batches) - 1), 3)}


def g0(res):
    v = {"sync_ok": os.path.exists(f"{res}/sync17/DONE"), "async_ok": os.path.exists(f"{res}/async17/DONE")}
    v.update(async_checks(res, "async17"))
    v["pass"] = v["sync_ok"] and v["async_ok"] and v["staleness_ok"] and v["sync_gen_overlaps"] == 0
    json.dump(v, open(f"{res}/g0.json", "w"), indent=2)
    print(json.dumps(v))
    return 0 if v["pass"] else 1


def _cny(b, hours):
    return b["prior_cny"] + hours * b["usd_per_hour"] * b["cny_per_usd"]


def g1(res, run):
    """预算规则：与第 2 档相同的预测方法（30 步计时 × 1.2 估 200 步平均），累计花费不超过 cap_cny 才开跑。"""
    b = json.load(open(BUDGET))
    measured, _ = seg_means(res, run)
    base, _ = seg_means(res, "sync17")
    if measured is None:
        dec = {"seeds": [], "reason": "没有计时数据，不跑正式训练"}
    elif base is None or measured >= base:
        dec = {"seeds": [], "s_per_step_measured": measured and round(measured, 2),
               "s_per_step_sync17": base and round(base, 2), "reason": "异步不比同步快（或缺同步基线），不跑正式训练"}
    else:
        sps = measured * 1.2
        elapsed_h = (time.time() - b["launch_epoch"]) / 3600
        seed_h = (200 * sps + 4 * 60 + 3 * 60) / 3600  # 训练 + 重启服务端 + 评测（含合并适配器）
        tail_h = 10 / 60
        cny = lambda n: _cny(b, elapsed_h + n * seed_h + tail_h)  # noqa: E731
        seeds = [42, 1] if cny(2) <= b["cap_cny"] else [42] if cny(1) <= b["cap_cny"] else []
        dec = {"seeds": seeds, "s_per_step_measured": round(measured, 2), "s_per_step_sync17": round(base, 2),
               "s_per_step_projected": round(sps, 2),
               "elapsed_h": round(elapsed_h, 2), "projected_cny_1seed": round(cny(1)),
               "projected_cny_2seeds": round(cny(2)), "cap_cny": b["cap_cny"],
               "reason": {2: "预测累计花费不超过上限", 1: "两个种子会超上限，只跑种子 42",
                          0: "一个种子也会超上限，不跑正式训练"}[len(seeds)]}
    json.dump(dec, open(f"{res}/g1.json", "w"), indent=2, ensure_ascii=False)
    print(json.dumps(dec, ensure_ascii=False))
    return 0


def g1b(res):
    b = json.load(open(BUDGET))
    p = f"{res}/eval_amain17_s42/summary.json"
    acc = json.load(open(p))["accuracy"] if os.path.exists(p) else None
    rows = jl(f"{res}/amain17_s42/steps.jsonl")
    sps = sum(r["wall_s"] for r in rows) / len(rows) if rows else None
    elapsed_h = (time.time() - b["launch_epoch"]) / 3600
    proj = _cny(b, elapsed_h + ((200 * sps + 7 * 60) / 3600 if sps else 1) + 10 / 60)
    go = acc is not None and acc >= ACC_MIN and sps is not None and proj <= b["cap_cny"]
    reason = ("通过" if go else "种子 42 没有评测结果" if acc is None
              else f"种子 42 准确率 {acc:.3f} < {ACC_MIN:.3f}，先查原因再决定" if acc < ACC_MIN
              else "预测花费超上限")
    dec = {"run_seed1": go, "acc_s42": acc, "acc_min": round(ACC_MIN, 3), "s_per_step_s42": sps and round(sps, 2),
           "projected_cny": round(proj), "cap_cny": b["cap_cny"], "reason": reason}
    json.dump(dec, open(f"{res}/g1b.json", "w"), indent=2, ensure_ascii=False)
    print(json.dumps(dec, ensure_ascii=False))
    return 0 if go else 1


def summary(res):
    runs = [r for r in ("sync17", "async17", "asyncfused17", "amain17_s42", "amain17_s1") if os.path.isdir(f"{res}/{r}")]
    lines = ["| 运行 | 每步墙钟（第 3 步起平均） | 两卡同时 >50% | GPU0 忙 | GPU1 忙 | IS 屏蔽序列占比 | logprob 差均值 | 等后台生成 s/步 |",
             "|---|---|---|---|---|---|---|---|"]
    for r in runs:
        w, _ = seg_means(res, r)
        g = gpu_stats(res, r) or {}
        steps = [x for x in jl(f"{res}/{r}/steps.jsonl") if x["step"] > SKIP]
        m = [x["is_masked_frac"] for x in steps if "is_masked_frac" in x]
        lp = [v for k, v in per_step(res, r, LOGP).items() if k > SKIP]
        a = async_checks(res, r) if os.path.exists(f"{res}/{r}/async.jsonl") else {}
        f = lambda x, fmt: (fmt % x) if x is not None else "-"  # noqa: E731
        lines.append(f"| {r} | {f(w, '%.2f s')} | {f(g.get('both_busy_frac'), '%.2f')} | {f(g.get('gpu0_busy_frac'), '%.2f')} "
                     f"| {f(g.get('gpu1_busy_frac'), '%.2f')} | {f(sum(m) / len(m) if m else None, '%.3f')} "
                     f"| {f(sum(lp) / len(lp) if lp else None, '%.4f')} | {f(a.get('wait_s_mean'), '%.2f')} |")
    for r in ("eval_amain17_s42", "eval_amain17_s1"):
        p = f"{res}/{r}/summary.json"
        if os.path.exists(p):
            lines.append(f"\n{r}: accuracy={json.load(open(p))['accuracy']}")
    text = "\n".join(lines)
    open(f"{res}/summary.md", "w").write(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    cmd, *rest = sys.argv[1:]
    sys.exit({"g0": g0, "g1": g1, "g1b": g1b, "summary": summary}[cmd](*rest))
