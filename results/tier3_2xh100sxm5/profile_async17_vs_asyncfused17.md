| 段 | async17 | asyncfused17 |
|---|---|---|
| gen_and_score_total | 1.37 s (23.6%) | 0.29 s (5.3%) |
| gen_wait | 0.03 s (0.5%) | 0.17 s (3.0%) |
| loss_forward_logps | 1.24 s (21.4%) | 1.49 s (27.3%) |
| old_logps_recompute | 1.22 s (21.0%) | - |
| prepare_inputs | 1.37 s (23.6%) | 0.29 s (5.3%) |
| rewards | 0.00 s (0.0%) | 0.00 s (0.0%) |
| rollout | 0.00 s (0.0%) | 0.00 s (0.0%) |
| sync_weights | 0.09 s (1.6%) | 0.09 s (1.7%) |
| training_step | 5.78 s (99.5%) | 5.44 s (99.8%) |
| **每步墙钟（第 3 步起平均）** | 5.81 s | 5.45 s |

async17 GPU：gpu0_util_mean=39.56，gpu0_busy_frac=0.40，gpu1_util_mean=88.29，gpu1_busy_frac=0.91，both_busy_frac=0.37，samples=179

asyncfused17 GPU：gpu0_util_mean=51.82，gpu0_busy_frac=0.51，gpu1_util_mean=81.79，gpu1_busy_frac=0.83，both_busy_frac=0.45，samples=164
