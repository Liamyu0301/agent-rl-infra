# 第 2 档总流程（实例上，setup 完成后用新登录启动：sg docker -c "bash scripts/tier2/chain.sh"）。
# 关卡：G0 不通过就停（Mac 端看到 g0_fail 会立即终止实例）；各阶段可续跑（已有 DONE / summary.json 的跳过）。
D=$(dirname "$0")
bash "$D/g0.sh" || exit 1
[ -f "$HOME/tier2_markers/g0_pass" ] || [ "$DRY" = 1 ] || exit 1
bash "$D/s22_profile.sh"
bash "$D/s23_fused.sh"
bash "$D/s24_train.sh"
touch "$HOME/tier2_markers/chain_done"
