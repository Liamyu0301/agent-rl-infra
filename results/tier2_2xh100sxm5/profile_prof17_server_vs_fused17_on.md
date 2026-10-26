| 段 | prof17_server | fused17_on |
|---|---|---|
| gen_and_score_total | 5.30 s (54.3%) | 3.92 s (49.2%) |
| loss_forward_logps | 1.25 s (12.8%) | 1.14 s (14.4%) |
| old_logps_recompute | 1.20 s (12.3%) | - |
| prepare_inputs | 5.30 s (54.3%) | 3.92 s (49.2%) |
| rewards | 0.00 s (0.0%) | 0.00 s (0.0%) |
| rollout | 3.96 s (40.6%) | 3.78 s (47.5%) |
| sync_weights | 0.10 s (1.1%) | 0.10 s (1.3%) |
| training_step | 9.75 s (99.9%) | 7.95 s (99.8%) |
| **每步墙钟（第 3 步起平均）** | 9.77 s | 7.96 s |

prof17_server GPU：gpu0_util_mean=25.22，gpu0_busy_frac=0.25，gpu1_util_mean=54.40，gpu1_busy_frac=0.55，both_busy_frac=0.00，samples=197

fused17_on GPU：gpu0_util_mean=25.04，gpu0_busy_frac=0.24，gpu1_util_mean=44.47，gpu1_busy_frac=0.45，both_busy_frac=0.00，samples=85
