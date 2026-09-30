sleep 2.5h
cd /workspaces/isaaclab_arena/submodules/workspaces/playground/unified_ball_kick_enhanced
PYTHONPATH=/workspaces/isaaclab_arena/submodules/workspaces/playground/unified_ball_kick_enhanced/src/holosoma \
HOLOSOMA_SKILLS_CONFIG=configs/skill/skill_016.yaml \
python src/holosoma/holosoma/train_agent.py \
    exp:g1-29dof-unified-fast-sac logger:wandb-ball-kick \
  --training.project UnifiedBallKickingEnhanced \
  --training.name stageC-skill016-h074 \
  --training.checkpoint logs/UnifiedBallKickingEnhanced/20260913_043346-stageB-skill016-h074-locomotion/model_0275000.pt \
  --training.headless=True \
  --training.num-envs 3300 \
  --logger.video.enabled False \
  --algo.config.mujoco-kick-rollout-every-n-saves 5 \
  --algo.config.mujoco-survival-scan-every-n-saves 3 \
  --algo.config.num-learning-iterations 375000


