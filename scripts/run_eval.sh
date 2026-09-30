sleep 3h
cd /workspaces/isaaclab_arena/submodules/workspaces/playground/unified_ball_kick_enhanced
python src/holosoma/holosoma/sim2sim_eval.py --config configs/sim2sim_eval/sweep-ablation-b2.yaml
python src/holosoma/holosoma/sim2sim_eval.py --config configs/sim2sim_eval/sweep-ablation-b3.yaml