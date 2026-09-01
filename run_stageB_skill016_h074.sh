#!/usr/bin/env bash
set -euo pipefail

source /root/miniconda3/etc/profile.d/conda.sh
conda activate hssim

cd /workspaces/isaaclab_arena/submodules/workspaces/playground/unified_ball_kick_enhanced

export PYTHONPATH=/workspaces/isaaclab_arena/submodules/workspaces/playground/unified_ball_kick_enhanced/src/holosoma
export HOLOSOMA_SKILLS_CONFIG=configs/skill/skill_016.yaml

exec python src/holosoma/holosoma/train_agent.py \
    exp:g1-29dof-unified-fast-sac logger:wandb-ball-kick \
  --training.project UnifiedBallKickingEnhanced \
  --training.name stageB-skill016-h074 \
  --training.checkpoint logs/UnifiedBallKickingEnhanced/20260830_234103-stageB-skill016-h074-locomotion/model_0230000.pt \
  --training.headless=True \
  --training.num-envs 2048 \
  --logger.video.enabled False \
  --algo.config.mujoco-kick-rollout-every-n-saves 5 \
  --algo.config.mujoco-survival-scan-every-n-saves 3 \
  --algo.config.num-learning-iterations 300000
