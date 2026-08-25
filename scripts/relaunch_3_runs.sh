#!/bin/bash
# 2026-08-16: kills and resumes the 3 healthy multi-skill ball-kicking runs (replay-4, replay-8,
# high-ratio) so they pick up the video-recorder-interval fix (unified_manager.py) and the mujoco
# sim2sim rollout timeout fix (record_mujoco_kick_rollout.py / record_mujoco_locomotion_rollout.py).
# Deliberately excludes low-ratio (PID 3088071), which died from a GPU3 fault and is being left
# stopped for now per your call.
#
# Each run resumes from its own latest checkpoint (saved every 1000 steps -> at most a few hundred
# steps of progress re-done, not lost). Each relaunch creates a NEW log dir + NEW wandb run under
# the same --training.name.
#
# Review the PIDs/checkpoints printed below before it proceeds -- if anything looks stale (e.g. a
# run has already advanced past what's shown, or a PID no longer matches), stop and re-check rather
# than trust this script blindly; it was generated from a point-in-time snapshot.

set -euo pipefail
cd /workspaces/isaaclab_arena/submodules/workspaces/playground/unified_ball_kick_robonaldo

PYTHON=/workspaces/isaaclab_arena/submodules/workspaces/conda_env/hssim/bin/python
export PYTHONPATH=src/holosoma:${PYTHONPATH:-}

declare -A PIDS=(
  [replay4]=3013781
  [replay8]=3031815
  [highratio]=3038737
)
declare -A DIRS=(
  [replay4]="logs/RoboNaldoBallKicking/20260815_122307-stageCB-2skills-l2spweight-0.01-replay-4-locomotion"
  [replay8]="logs/RoboNaldoBallKicking/20260815_124005-stageCB-2skills-l2spweight-0.0-replay-8-locomotion"
  [highratio]="logs/RoboNaldoBallKicking/20260815_124340-stageCB-2skills-high-ratio-locomotion"
)
declare -A NAMES=(
  [replay4]="stageCB-2skills-l2spweight-0.01-replay-4"
  [replay8]="stageCB-2skills-l2spweight-0.0-replay-8"
  [highratio]="stageCB-2skills-high-ratio"
)
declare -A SKILLS_CFG=(
  [replay4]="configs/multi_skills_replay4.yaml"
  [replay8]="configs/multi_skills_replay8.yaml"
  [highratio]="configs/multi_skills_highratio.yaml"
)
declare -A TASK_CFG=(
  [replay4]="configs/task_config_stageD.yaml"
  [replay8]="configs/task_config_stageC1.yaml"
  [highratio]="configs/task_config_stageC1.yaml"
)
declare -A GPU=(
  [replay4]=0
  [replay8]=1
  [highratio]=2
)

echo "=== Current state (verify before proceeding) ==="
for key in replay4 replay8 highratio; do
  pid=${PIDS[$key]}
  latest=$(ls "${DIRS[$key]}"/model_*.pt 2>/dev/null | sort -V | tail -1)
  alive="dead"
  ps -p "$pid" > /dev/null 2>&1 && alive="alive"
  echo "$key: pid=$pid ($alive), latest_checkpoint=$latest"
done
echo
read -p "Proceed with kill + relaunch for all 3? [y/N] " confirm
[[ "$confirm" == "y" || "$confirm" == "Y" ]] || { echo "Aborted."; exit 1; }

for key in replay4 replay8 highratio; do
  pid=${PIDS[$key]}
  dir=${DIRS[$key]}
  name=${NAMES[$key]}
  skills_cfg=${SKILLS_CFG[$key]}
  task_cfg=${TASK_CFG[$key]}
  gpu=${GPU[$key]}

  echo "=== $key ==="
  if ps -p "$pid" > /dev/null 2>&1; then
    echo "Killing pid $pid (SIGTERM)..."
    kill "$pid"
    for i in $(seq 1 30); do
      ps -p "$pid" > /dev/null 2>&1 || break
      sleep 1
    done
    if ps -p "$pid" > /dev/null 2>&1; then
      echo "pid $pid still alive after 30s, sending SIGKILL"
      kill -9 "$pid"
      sleep 2
    fi
  else
    echo "pid $pid already dead, skipping kill"
  fi

  latest=$(ls "$dir"/model_*.pt 2>/dev/null | sort -V | tail -1)
  if [[ -z "$latest" ]]; then
    echo "ERROR: no checkpoint found in $dir, skipping relaunch for $key"
    continue
  fi
  echo "Resuming $key from $latest on GPU $gpu ..."

  CUDA_VISIBLE_DEVICES=$gpu \
  HOLOSOMA_SKILLS_CONFIG=$skills_cfg \
  HOLOSOMA_TASK_CONFIG=$task_cfg \
  nohup "$PYTHON" -m holosoma.train_agent \
    exp:g1-29dof-unified-fast-sac logger:wandb-ball-kick \
    --training.project RoboNaldoBallKicking --training.name "$name" \
    --training.checkpoint "$latest" \
    --training.headless=True --training.num-envs 3300 \
    --algo.config.mujoco-kick-rollout-every-n-saves 5 \
    --algo.config.num-learning-iterations 600000 \
    > "/tmp/relaunch_${key}.log" 2>&1 &

  echo "Launched $key, new pid=$!, log: /tmp/relaunch_${key}.log"
  echo
done

echo "Done. Check each new pid with 'ps aux | grep train_agent' and tail the /tmp/relaunch_*.log files."
