"""第 3 阶段：GRPO 训练工具调用 Agent（TRL GRPOTrainer + 自定义多轮 rollout + Docker 长驻容器池）。

用法：.venv/bin/python rl/train_grpo.py --max-steps 10 --out runs/smoke
每步耗时日志：<out>/steps.jsonl（每个优化步的墙钟时间）；rollout 日志：<out>/rollout.jsonl（生成 / 执行代码耗时）。
"""
import argparse
import json
import os
import sys
import time

os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
os.environ.setdefault("TRL_EXPERIMENTAL_SILENCE", "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--max-steps", type=int, default=10)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--num-generations", type=int, default=8)
    ap.add_argument("--per-device-batch", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--max-completion", type=int, default=1024)
    ap.add_argument("--vllm-mem", type=float, default=0.3)
    ap.add_argument("--vllm-mode", choices=["colocate", "server"], default="colocate",
                    help="server=训推分离：生成交给独立的 vLLM 服务（需带 --weight-transfer-config 启动）")
    ap.add_argument("--vllm-server-url", default="http://localhost:8000")
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--save-steps", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42, help="随机种子（影响数据顺序和初始化）；用于重复实验")
    ap.add_argument("--fused-is", choices=["off", "check", "on"], default="off",
                    help="on=把 vLLM IS 纠偏挪进损失、去掉重复前向（rl/fused_is.py）；check=只并行计算并记录差异")
    ap.add_argument("--profile", action="store_true", help="分段计时（每段前后同步 GPU，结果写进 steps.jsonl）")
    ap.add_argument("--out", default="runs/smoke")
    args = ap.parse_args()

    from peft import LoraConfig
    from transformers import TrainerCallback
    from trl import GRPOConfig, GRPOTrainer

    import rl.agent_rollout as ar

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    import collections
    import torch

    seg = collections.defaultdict(float)  # 当前优化步内各段累计耗时（秒）
    state = {"in_gen": False}

    def timed(name, fn):
        """计时包装：前后各同步一次 GPU，保证异步的 CUDA 计算被算进正确的段。"""
        def wrapper(*a, **kw):
            torch.cuda.synchronize()
            t = time.perf_counter()
            try:
                return fn(*a, **kw)
            finally:
                torch.cuda.synchronize()
                seg[name] += time.perf_counter() - t
        return wrapper

    class StepTimer(TrainerCallback):
        """记录每个优化步的墙钟时间和 TRL 打出的日志，写入 steps.jsonl。"""
        def __init__(self):
            self.t = None

        def on_step_begin(self, a, state, control, **kw):
            self.t = time.perf_counter()

        def on_step_end(self, a, state, control, **kw):
            rec = {"step": state.global_step, "wall_s": round(time.perf_counter() - self.t, 2), "t": time.time()}
            if args.profile:
                rec["segments"] = {k: round(v, 3) for k, v in seg.items()}
                seg.clear()
            with open(os.path.join(args.out, "steps.jsonl"), "a") as f:
                f.write(json.dumps(rec) + "\n")

        def on_log(self, a, state, control, logs=None, **kw):
            with open(os.path.join(args.out, "trl_log.jsonl"), "a") as f:
                f.write(json.dumps({"step": state.global_step, **(logs or {})}) + "\n")

    ar._get_pool()  # 预热沙箱池，避免把容器启动时间算进第一步

    config = GRPOConfig(
        output_dir=args.out, max_steps=args.max_steps, learning_rate=args.lr,
        num_generations=args.num_generations, per_device_train_batch_size=args.per_device_batch,
        gradient_accumulation_steps=args.grad_accum, max_completion_length=args.max_completion,
        temperature=1.0, beta=0.0, bf16=True, gradient_checkpointing=True,
        use_vllm=True, vllm_mode=args.vllm_mode, vllm_gpu_memory_utilization=args.vllm_mem,
        vllm_server_base_url=args.vllm_server_url if args.vllm_mode == "server" else None,
        vllm_max_model_length=2048, logging_steps=1, report_to="none",
        save_strategy="steps" if args.save_steps else "no", save_steps=args.save_steps or 500,
        reward_weights=[1.0, 1.0], seed=args.seed,
    )
    trainer_cls, extra = GRPOTrainer, {}
    if args.fused_is != "off":
        from rl.fused_is import FusedISGRPOTrainer
        trainer_cls = FusedISGRPOTrainer
        extra = {"check": args.fused_is == "check", "check_log": os.path.join(args.out, "fused_is_check.jsonl")}
    trainer = trainer_cls(
        model=args.model, args=config, train_dataset=ar.load_gsm8k("train"),
        reward_funcs=[ar.correctness_reward, ar.tool_reward],
        rollout_func=ar.make_rollout_func(log_path=os.path.join(args.out, "rollout.jsonl")),
        peft_config=LoraConfig(r=args.lora_r, lora_alpha=2 * args.lora_r, target_modules="all-linear",
                               task_type="CAUSAL_LM"),
        callbacks=[StepTimer()],
        **extra,
    )
    if args.vllm_mode == "server":
        # 修复服务端模式下 PEFT 权重同步拿到旧权重的问题（第 2 档 G0 发现，CPU 上已复现）：
        # vLLM 的 pack_tensors 先收集参数的视图、凑满缓冲区或迭代器耗尽后才 torch.cat；最后一块是迭代器耗尽时结束的，
        # 而 TRL 的迭代器在耗尽那一刻原地 unmerge LoRA → 最后一块发出去的是未合并的原权重。
        # 交出前先 clone（并等 GPU 拷贝完成，避免与打包所在的 CUDA 流竞争），之后的 unmerge 就不影响已交出的张量
        iter_params = trainer.vllm_generation._iter_named_params

        def iter_params_cloned():
            for name, t in iter_params():
                c = t.clone()
                torch.cuda.synchronize()
                yield name, c

        trainer.vllm_generation._iter_named_params = iter_params_cloned
    if args.profile:
        # 分段：_prepare_inputs 在缓冲区空时触发整轮"生成+打分"（含权重同步、rollout、旧 logprob 重算、算奖励）；
        # training_step 包含 _prepare_inputs，所以 前向+反向 = training_step − prepare_inputs。
        gen_and_score = trainer._generate_and_score_completions

        def gen_and_score_flagged(*a, **kw):
            state["in_gen"] = True
            try:
                return gen_and_score(*a, **kw)
            finally:
                state["in_gen"] = False

        trainer._generate_and_score_completions = timed("gen_and_score_total", gen_and_score_flagged)
        trainer.vllm_generation.sync_weights = timed("sync_weights", trainer.vllm_generation.sync_weights)
        trainer.rollout_func = timed("rollout", trainer.rollout_func)
        trainer._calculate_rewards = timed("rewards", trainer._calculate_rewards)
        logps = trainer._get_per_token_logps_and_entropies

        def logps_split(*a, **kw):
            name = "old_logps_recompute" if state["in_gen"] else "loss_forward_logps"
            return timed(name, logps)(*a, **kw)

        trainer._get_per_token_logps_and_entropies = logps_split
        trainer._prepare_inputs = timed("prepare_inputs", trainer._prepare_inputs)
        trainer.training_step = timed("training_step", trainer.training_step)

    t0 = time.perf_counter()
    trainer.train()
    print(json.dumps({"train_wall_s": round(time.perf_counter() - t0, 1)}))
    if args.save_steps:
        trainer.save_model(os.path.join(args.out, "final"))

    import asyncio
    asyncio.run_coroutine_threadsafe(ar._pool.close(), ar._loop).result()


if __name__ == "__main__":  # vLLM 用 spawn 启动引擎子进程
    main()
