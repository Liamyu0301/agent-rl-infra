| 段 | sync17 | async17 |
|---|---|---|
| gen_and_score_total | 4.40 s (52.3%) | 1.37 s (23.6%) |
| gen_wait | - | 0.03 s (0.5%) |
| loss_forward_logps | 1.14 s (13.6%) | 1.24 s (21.4%) |
| old_logps_recompute | 1.08 s (12.8%) | 1.22 s (21.0%) |
| prepare_inputs | 4.40 s (52.3%) | 1.37 s (23.6%) |
| rewards | 0.00 s (0.0%) | 0.00 s (0.0%) |
| rollout | 3.20 s (38.0%) | 0.00 s (0.0%) |
| sync_weights | 0.09 s (1.1%) | 0.09 s (1.6%) |
| training_step | 8.40 s (99.9%) | 5.78 s (99.5%) |
| **每步墙钟（第 3 步起平均）** | 8.42 s | 5.81 s |

sync17 GPU：gpu0_util_mean=25.98，gpu0_busy_frac=0.26，gpu1_util_mean=57.34，gpu1_busy_frac=0.58，both_busy_frac=0.02，samples=248

async17 GPU：gpu0_util_mean=39.56，gpu0_busy_frac=0.40，gpu1_util_mean=88.29，gpu1_busy_frac=0.91，both_busy_frac=0.37，samples=179
