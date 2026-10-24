"""去掉 GRPO 每步一次重复的前向：把 vLLM 重要性采样（IS）纠偏挪到损失计算里，复用损失前向的 logprob。

背景（第 3.5 步实测，每步 21.8 s）：TRL 在 vLLM 模式下，生成阶段会用训练模型对整批回答额外做一次前向，
算"旧 logprob"来求 IS 比值（vLLM 采样分布 vs 训练模型分布），这一次前向占每步 14.5%。
但在 num_iterations == 1、且一批生成的数据在一次优化步内用完（gradient_accumulation_steps 是
steps_per_generation 的整数倍）时，用完这批数据之前参数不变、也没有 dropout，旧 logprob 就等于损失前向算出的
logprob（TRL 源码注释也写明了这一点）。因此：
  生成阶段：临时关闭纠偏开关 → TRL 跳过那次前向（旧 logprob 置为"等于当前值"）；
  损失阶段：损失前向一算出 logprob，就按 TRL 完全相同的公式求 IS 比值，写回 inputs，由 TRL 照常乘进损失。
check=True 时不做替换，只在旁边并行计算并记录两种做法的差异，用来验证数值等价。
"""
import json
import math

import torch
from trl import GRPOTrainer


class FusedISGRPOTrainer(GRPOTrainer):
    def __init__(self, *args, check=False, check_log=None, **kwargs):
        super().__init__(*args, **kwargs)
        a = self.args
        assert self.use_vllm and self.vllm_importance_sampling_correction, "需要开启 vLLM 及其 IS 纠偏"
        assert self.num_iterations == 1, "每批数据只能训练一轮（否则旧 logprob 与当前不同）"
        assert a.gradient_accumulation_steps % a.steps_per_generation == 0, "一批生成数据须在一次优化步内用完"
        assert self.beta == 0.0 and self.loss_type != "vespo", "当前只验证过 beta=0、非 vespo 的损失"
        assert self.off_policy_mask_threshold is None, "OPSM 需要生成阶段的旧 logprob"
        assert self.vllm_importance_sampling_mode == "sequence_mask", "只实现了 sequence_mask 模式"
        self._check, self._check_log, self._cur_inputs = check, check_log, None

    # 生成阶段：关掉开关，TRL 就不会做那次额外前向，也不会生成 importance_sampling_ratio
    def _generate_and_score_completions(self, *args, **kwargs):
        if self._check:
            return super()._generate_and_score_completions(*args, **kwargs)
        self.vllm_importance_sampling_correction = False
        try:
            return super()._generate_and_score_completions(*args, **kwargs)
        finally:
            self.vllm_importance_sampling_correction = True

    def _compute_loss(self, model, inputs):
        self._cur_inputs = inputs
        try:
            return super()._compute_loss(model, inputs)
        finally:
            self._cur_inputs = None

    def _is_ratio(self, logps, inputs):
        """与 TRL 生成阶段完全相同的公式（sequence_mask 模式），只是旧 logprob 换成损失前向的结果。"""
        mask = inputs["completion_mask"]
        if inputs.get("tool_mask") is not None:
            mask = mask * inputs["tool_mask"]
        diff = torch.nan_to_num((logps - inputs["sampling_per_token_logps"]) * mask, nan=0.0)
        ratio = torch.exp(diff.sum(dim=-1, keepdim=True))
        lo = self.vllm_importance_sampling_clip_min if self.vllm_importance_sampling_clip_min is not None else -math.inf
        hi = self.vllm_importance_sampling_clip_max if self.vllm_importance_sampling_clip_max is not None else math.inf
        return ratio.masked_fill((ratio < lo) | (ratio > hi), 0.0)

    def _get_per_token_logps_and_entropies(self, *args, **kwargs):
        out = super()._get_per_token_logps_and_entropies(*args, **kwargs)
        inputs = self._cur_inputs
        if inputs is None:  # 不是损失前向（例如 check 模式下生成阶段的那次前向）
            return out
        fused = self._is_ratio(out[0].detach(), inputs)
        if not self._check:
            inputs["importance_sampling_ratio"] = fused
        elif self._check_log:  # 记录：旧 logprob vs 损失前向 logprob；TRL 的 IS 比值 vs 本实现
            mask = inputs["completion_mask"] * (inputs["tool_mask"] if inputs.get("tool_mask") is not None else 1)
            m = mask.bool()
            d_logps = (inputs["old_per_token_logps"] - out[0].detach()).abs()[m]
            ref = inputs["importance_sampling_ratio"]
            with open(self._check_log, "a") as f:
                f.write(json.dumps({
                    "step": self.state.global_step,
                    "logps_absdiff_max": d_logps.max().item(), "logps_absdiff_mean": d_logps.mean().item(),
                    "is_ratio_absdiff_max": (ref - fused).abs().max().item(),
                    "is_ratio_mean_trl": ref.mean().item(), "is_ratio_mean_fused": fused.mean().item(),
                    "masked_by_trl": int((ref == 0).sum().item()), "masked_by_fused": int((fused == 0).sum().item()),
                }) + "\n")
        return out
