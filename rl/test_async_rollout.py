"""异步 rollout 的流水线自检（不需要 GPU）：用假训练器按 TRL 的调用顺序驱动 rl/async_rollout.py。

TRL 每步的顺序（grpo_trainer.py _generate）：global_step 变了就 sync_weights → rollout_func(prompts) → 训练。
这里生成 / 同步 / 训练都用 sleep 代替，检查：
  1. 数据落后步数：第 1 步 0，之后都是 1；
  2. 权重同步的时间段与后台生成的时间段不重叠（同步前必须等生成结束）；
  3. 返回的 answer 与返回的 prompt 属于同一批（answer 跟着 rollout 走，而不是本步数据集那一列）；
  4. 每步耗时 ≈ max(生成, 训练)，而不是两者之和。
跑两组参数：生成比训练快、生成比训练慢（后者才会在同步时撞上还没结束的生成，能检出漏掉"先等再同步"）。
用法：python rl/test_async_rollout.py   通过时退出码 0
"""
import json
import os
import sys
import tempfile
import time
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import rl.async_rollout as asr  # noqa: E402

SYNC_S, STEPS = 0.05, 6


def check(GEN_S, TRAIN_S):
    print(f"== gen {GEN_S}s, train {TRAIN_S}s")
    log = os.path.join(tempfile.mkdtemp(), "async.jsonl")
    gen_iv = []  # 每次生成的 (开始, 结束)

    def fake_rollout(prompts, trainer):
        t0 = time.time()
        time.sleep(GEN_S)
        gen_iv.append((t0, time.time()))
        return {"prompt_ids": [[p["q"]] for p in prompts], "completion_ids": [[1]] * len(prompts),
                "logprobs": [[0.0]] * len(prompts)}

    trainer = types.SimpleNamespace(state=types.SimpleNamespace(global_step=0), args=None,
                                    processing_class=None, rollout_func=fake_rollout,
                                    vllm_generation=types.SimpleNamespace(sync_weights=lambda: time.sleep(SYNC_S)))
    seen = []

    def gen_and_score(inputs):  # 相当于 TRL 的 _generate_and_score_completions → _generate
        if trainer.state.global_step != trainer._last_loaded_step:
            trainer.vllm_generation.sync_weights()
            trainer._last_loaded_step = trainer.state.global_step
        out = trainer.rollout_func([{"q": x["q"]} for x in inputs], trainer)
        seen.append(([p[0] for p in out["prompt_ids"]], out["answer"]))

    trainer._last_loaded_step = 0  # TRL 第 1 步不同步（服务端已是初始权重）
    trainer._generate_and_score_completions = gen_and_score
    hooks = asr.install(trainer, log)
    t_start = time.time()
    for step in range(STEPS):
        batch = [{"q": step * 10 + i, "answer": -(step * 10 + i)} for i in range(4)]
        trainer._generate_and_score_completions(batch)
        time.sleep(TRAIN_S)
        trainer.state.global_step += 1
    hooks.drain()
    wall = time.time() - t_start

    recs = [json.loads(x) for x in open(log)]
    batches = [r for r in recs if r["event"] == "batch"]
    syncs = [(r["start"], r["end"]) for r in recs if r["event"] == "sync"]
    stale = [b["staleness"] for b in batches]
    ok1 = stale == [0] + [1] * (STEPS - 1)
    overlap = [(s, g) for s in syncs for g in gen_iv if s[0] < g[1] and g[0] < s[1]]
    ok2 = not overlap and len(syncs) == STEPS - 1
    ok3 = all([-q for q in qs] == ans for qs, ans in seen)
    # 第 2 步训练的是 prompts_1 的第二份样本（已知代价），之后每步训练上一步的 prompts
    ok3 = ok3 and seen[0][0] == seen[1][0] and all(seen[t][0][0] == (t - 1) * 10 for t in range(2, STEPS))
    serial = STEPS * (GEN_S + TRAIN_S + SYNC_S) + GEN_S
    bound = GEN_S + STEPS * (max(GEN_S, TRAIN_S) + SYNC_S) + GEN_S  # 第 1 步同步生成 + 每步 max + 最后 drain
    ok4 = wall < bound * 1.1 and wall < serial * 0.85
    print(f"1 staleness {stale} -> {'OK' if ok1 else 'FAIL'}")
    print(f"2 sync/gen overlaps {len(overlap)}, syncs {len(syncs)} -> {'OK' if ok2 else 'FAIL'}")
    print(f"3 answers follow their prompts; trained prompt batches {[s[0][0] for s in seen]} -> {'OK' if ok3 else 'FAIL'}")
    print(f"4 wall {wall:.2f}s vs serial {serial:.2f}s, pipelined bound {bound:.2f}s -> {'OK' if ok4 else 'FAIL'}")
    print("waits", [b["wait_s"] for b in batches])
    return ok1 and ok2 and ok3 and ok4


if __name__ == "__main__":
    ok = [check(0.4, 0.6), check(0.9, 0.5)]
    sys.exit(0 if all(ok) else 1)
