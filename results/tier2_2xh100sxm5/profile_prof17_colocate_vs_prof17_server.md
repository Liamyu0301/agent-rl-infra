| 段 | prof17_colocate | prof17_server |
|---|---|---|
| gen_and_score_total | 4.83 s (54.5%) | 5.30 s (54.3%) |
| loss_forward_logps | 1.10 s (12.4%) | 1.25 s (12.8%) |
| old_logps_recompute | 1.10 s (12.4%) | 1.20 s (12.3%) |
| prepare_inputs | 4.83 s (54.5%) | 5.30 s (54.3%) |
| rewards | 0.00 s (0.0%) | 0.00 s (0.0%) |
| rollout | 3.50 s (39.6%) | 3.96 s (40.6%) |
| sync_weights | 0.20 s (2.3%) | 0.10 s (1.1%) |
| training_step | 8.85 s (99.9%) | 9.75 s (99.9%) |
| **每步墙钟（第 3 步起平均）** | 8.86 s | 9.77 s |

prof17_colocate GPU：gpu0_util_mean=0.00，gpu0_busy_frac=0.00，gpu1_util_mean=60.60，gpu1_busy_frac=0.62，both_busy_frac=0.00，samples=232

prof17_server GPU：gpu0_util_mean=25.22，gpu0_busy_frac=0.25，gpu1_util_mean=54.40，gpu1_busy_frac=0.55，both_busy_frac=0.00，samples=197
