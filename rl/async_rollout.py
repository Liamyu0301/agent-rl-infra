"""第 4 阶段补充：异步 rollout（落后一步的流水线），让 vLLM 卡生成下一批的同时训练卡训练上一批。

第 2 档实测：训推分离但不做异步时，两张卡同时工作的时间 <1%（GPU 0 生成时 GPU 1 等，反之亦然）。
做法：不改 TRL 源码，只包两层：
  rollout_func：第 t 步拿到 prompts_t，在后台线程开始生成 prompts_t，立刻返回上一步在后台生成好的 prompts_{t-1}；
                于是训练卡训练第 t-1 批的同时，vLLM 卡在生成第 t 批。每批数据比当前权重落后 1 步，
                TRL 自带的 vLLM 重要性采样（IS）纠偏用"当前训练模型 vs 生成时的分布"修正这一步的差异。
  sync_weights：TRL 在调 rollout_func 之前同步权重。同步前必须先等后台那批生成结束，
                否则一批数据会前半段用旧权重、后半段用新权重生成，记录的采样 logprob 就对不上任何一个模型。
数据顺序沿用 TRL 自己的采样器（同一种子下与同步模式相同，只是整体错后一步）。
已知的小代价：第 1 步没有上一批可用，同步生成一次并返回，同时在后台再生成一次 prompts_1（第 2 步训练它）；
所以 prompts_1 被训练两次，数据集里最后一批不会被训练。
注意：TRL 传给奖励函数的 prompts 是本步的 prompts_t，与数据差一步；本项目的奖励只读 rollout 返回的
answer / final_text / useful_calls，所以没影响。以后新写的奖励函数不能读 prompts。
TRL 另有一个实验性的 AsyncGRPOTrainer（max_staleness、LoRA 走 vLLM --enable-lora），是另一套训练器，这里没用。
"""
import json
import threading
import time
import types
from concurrent.futures import ThreadPoolExecutor


def install(trainer, log_path, tokenizer_factory=None, seg=None):
    """把 trainer 的 rollout_func 与 vllm_generation.sync_weights 换成异步版本。
    tokenizer_factory：后台线程单独用一个 tokenizer（HF fast tokenizer 并发调用可能报 "Already borrowed"）。
    seg：分段计时字典（--profile），等后台生成的时间记进 seg["gen_wait"]。
    返回 hooks：hooks.sync 是真正的权重同步（分段计时可以只包它，不把等待算进去），hooks.drain() 训练结束后调用。"""
    inner = trainer.rollout_func
    hooks = types.SimpleNamespace(sync=trainer.vllm_generation.sync_weights)
    ex = ThreadPoolExecutor(max_workers=1)
    st = {"pending": None, "answers": None, "sync_n": 0}
    lock = threading.Lock()
    tok = tokenizer_factory() if tokenizer_factory else trainer.processing_class
    # 后台线程看到的 "trainer"：只有 rollout 用到的几个属性，tokenizer 换成独立实例
    proxy = types.SimpleNamespace(processing_class=tok, args=trainer.args, vllm_generation=trainer.vllm_generation)

    def log(rec):
        with lock, open(log_path, "a") as f:
            f.write(json.dumps(rec) + "\n")

    def job(prompts, answers, gen_step):
        t0 = time.time()
        out = inner(prompts, proxy)
        out["answer"] = answers  # 覆盖数据集本步的 answer 列（TRL 把 rollout 的额外字段并入 inputs）
        return out, gen_step, t0, time.time()

    def spawn(prompts):
        st["pending"] = ex.submit(job, prompts, st["answers"], trainer.state.global_step)

    def wait():
        """等后台那批生成结束，返回 (结果, 等了多少秒)；没有在跑的就返回 (None, 0)。"""
        fut, st["pending"] = st["pending"], None
        if fut is None:
            return None, 0.0
        t = time.time()
        res = fut.result()
        w = time.time() - t
        if seg is not None:
            seg["gen_wait"] += w
        return res, w

    def sync_weights_after_join(*a, **kw):
        # TRL 在 rollout_func 之前调这里：先等后台生成结束，再换权重（结果暂存，紧接着的 rollout_func 取用）
        res, w = wait()
        st["ready"], st["wait_s"] = res, w
        t0 = time.time()
        r = hooks.sync(*a, **kw)
        log({"event": "sync", "step": trainer.state.global_step, "n": st["sync_n"], "start": t0, "end": time.time()})
        st["sync_n"] += 1
        return r

    def rollout_func(prompts, _trainer):
        ready, w = st.pop("ready", None), st.pop("wait_s", 0.0)
        if ready is None:  # 本步没触发同步（例如第 1 步 TRL 不同步）：后台若还有在跑的，也要先等它
            ready, w = wait()
        if ready is None:  # 第 1 步：没有上一批，同步生成一次
            t0 = time.time()
            ready = (inner(prompts, proxy), trainer.state.global_step, t0, time.time())
            ready[0]["answer"] = st["answers"]
            w = ready[3] - t0
        spawn(prompts)
        out, gen_step, g0, g1 = ready
        log({"event": "batch", "train_step": trainer.state.global_step, "gen_step": gen_step,
             "staleness": trainer.state.global_step - gen_step, "gen_start": g0, "gen_end": g1,
             "wait_s": round(w, 3), "n": len(out["completion_ids"])})
        return out

    # rollout_func 只拿到 prompts，拿不到 answer：在生成阶段入口记下本步各条数据的标准答案
    gen_and_score = trainer._generate_and_score_completions

    def gen_and_score_with_answers(inputs, *a, **kw):
        st["answers"] = [x["answer"] for x in inputs]
        return gen_and_score(inputs, *a, **kw)

    trainer._generate_and_score_completions = gen_and_score_with_answers
    trainer.rollout_func = rollout_func
    trainer.vllm_generation.sync_weights = sync_weights_after_join

    def drain():
        wait()
        ex.shutdown(wait=True)

    hooks.drain = drain
    return hooks
