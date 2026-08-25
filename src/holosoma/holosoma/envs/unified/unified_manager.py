from __future__ import annotations

import os
import time
from enum import IntEnum

import torch
from loguru import logger

from holosoma.config_types.reward_tuning import resolve_per_skill_param
from holosoma.config_types.task_config_paths import resolve_task_config_path
from holosoma.envs.base_task.base_task import BaseTask
from holosoma.utils.simulator_config import SimulatorType
from holosoma.utils.torch_utils import torch_rand_float


class TaskMode(IntEnum):
    LOCOMOTION = 0
    KICK = 1


_NAME_TO_MODE = {"locomotion": TaskMode.LOCOMOTION, "kick": TaskMode.KICK}


class UnifiedManager(BaseTask):
    """Env class for a single policy that learns both locomotion (velocity tracking) and
    ball-kicking (motion tracking). Each episode, per env, one task is picked at reset time
    according to configs/skill_mix.yaml's kick_probability — restricted to envs whose terrain is
    flat (see managers/terrain/terms/locomotion.py::env_terrain_is_flat), since kicking needs a
    freely-simulated ball to rest at a fixed configured position.

    Built by combining LeggedRobotLocomotionManager (envs/locomotion/locomotion_manager.py) and
    WholeBodyTrackingManager (envs/wbt/wbt_manager.py) — those two are both thin BaseTask
    subclasses with almost no shared code between them (some blocks are literally copy-pasted).
    This class merges their hooks and adds the one new thing neither has: a per-env task_mode
    buffer that RewardManager/TerminationManager/ObservationManager/CommandManager consult via
    `task_mode_mask()` for any *TermCfg that opts in (see managers/*/manager.py's task_mode field).

    Runs on IsaacSim only (ball spawning is IsaacSim-only, same requirement as WBT).
    """

    BASE_NUM_ENVS = 4096

    def __init__(self, tyro_config, *, device):
        self.init_done = False
        super().__init__(tyro_config, device=device)
        self.init_done = True
        assert not hasattr(self.simulator, "gym"), "UnifiedManager requires IsaacSim — IsaacGym is not supported."

    def _pre_manager_construction_callback(self):
        # 2026-08-15: placeholder skill_id, all zeros -- needs to exist before RewardManager/
        # TerminationManager are constructed (see BaseTask._pre_manager_construction_callback's
        # own docstring for why). _init_buffers() below overwrites this with the same zeros once
        # command_manager also exists -- genuinely idempotent, not just harmless: this early
        # tensor is never read for anything meaningful before then, since the REAL per-env values
        # aren't assigned until _build_task_mode_partition runs at first reset.
        self.skill_id = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

    def _get_task_name(self) -> str:
        return "unified_locomotion_kick"

    # ------------------------------------------------------------------
    # task_mode: the core new mechanism

    def task_mode_mask(self, name: str) -> torch.Tensor:
        """Bool tensor [num_envs]: which envs are currently in the given task mode ("locomotion"
        or "kick"). Consulted by every manager's *TermCfg.task_mode field (see
        managers/reward/manager.py, managers/termination/manager.py,
        managers/observation/manager.py, managers/command/manager.py) — this is the only method
        those managers look for (via hasattr) to decide whether task-mode masking applies at all,
        so every other experiment's env class simply not having this method is what keeps them
        provably unaffected."""
        return self.task_mode == _NAME_TO_MODE[name]

    def trigger_kick(self, env_ids: torch.Tensor | None = None) -> None:
        """Force the given envs (default: all) into kick-mode and reset them immediately — a full
        reset into the kick clip's starting pose, not a live mid-stride switch, matching how
        kick-mode episodes are trained (one task per episode). Used by interactive deployment
        tooling (see eval_interactive.py's 'k'/'kick' terminal command)."""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        self._forced_task_mode[env_ids] = TaskMode.KICK
        self.reset_envs_idx(env_ids)

    def _resample_task_mode(self, env_ids: torch.Tensor) -> None:
        if env_ids.numel() == 0:
            return

        # Clear the flip-tick sentinel for every genuinely-resetting env, unconditionally --
        # see _post_flip_step's own comment (_init_buffers). Harmless no-op for envs that were
        # never post-flip (already -1); essential for envs that WERE, so a stale flip-tick from a
        # PRIOR episode can never leak into this fresh one.
        self._post_flip_step[env_ids] = -1
        # Locomotion -> kick direction's own per-env state: fresh episode always clears it first,
        # mirroring _post_flip_step's own unconditional-clear-then-conditionally-set pattern, so
        # a stale pending/best-residual/entry-tick/fallback state from a PRIOR episode can never
        # leak into a new one. _pre_kick_step is NOT part of _clear_kick_pending (that helper is
        # also called mid-episode on fire/decline, where _pre_kick_step must NOT be touched -- see
        # its own comment in _init_buffers), so it's cleared separately here.
        self._clear_kick_pending(env_ids)
        self._pre_kick_step[env_ids] = -1

        forced = self._forced_task_mode[env_ids]
        has_forced = forced >= 0

        if self.is_evaluating:
            # Deployment: no automatic probability draw. Envs keep whatever task_mode they
            # already had (defaults to LOCOMOTION at construction) unless trigger_kick() forced
            # one explicitly.
            self.task_mode[env_ids[has_forced]] = forced[has_forced]
            self._forced_task_mode[env_ids] = -1
            return

        sampled = self._task_mode_partition[env_ids]
        self.task_mode[env_ids] = torch.where(has_forced, forced, sampled)
        # skill_id is never resampled on its own -- it's fixed for env_ids' whole life by
        # _build_task_mode_partition, just copied out of the fixed partition here on every reset
        # (same pattern task_mode itself uses). Meaningless for envs where task_mode ends up
        # LOCOMOTION; harmless to copy unconditionally.
        self.skill_id[env_ids] = self._skill_id_partition[env_ids]

        # Locomotion -> kick direction (2026-08-13): kick-partitioned envs, NOT explicitly forced
        # (trigger_kick() always means "enter now" -- interactive/deploy tooling expects immediate
        # entry, never a probabilistic delay) and NOT dedicated to wandb kick-video recording
        # (that guarantee -- every recorded episode shows a kick -- would break under a
        # probabilistic pending draw; recording envs keep the reliable immediate-teleport path,
        # matching _pin_recording_env_modes' own pre-existing behavior), draw
        # mid_episode_kick_entry_prob. On a hit: task_mode is overridden back to LOCOMOTION
        # (already what a normal locomotion-mode reset() below does -- see
        # _reset_robot_states_callback) and motion_ids is set NOW, matching skill_id's own
        # just-fixed assignment above -- MotionCommand.reset() is never called for a
        # LOCOMOTION-mode env (task_mode="kick"-tagged reset_terms), so nothing else would set it,
        # and the entry-point search (once eligible, _maybe_enter_kick_from_locomotion) needs to
        # know which skill's approach table to search before it ever runs.
        if self._mid_episode_kick_entry_prob_per_skill is not None or self._mid_episode_kick_entry_prob > 0.0:
            kick_selected = (sampled == TaskMode.KICK) & (~has_forced)
            recording_kick_ids = getattr(self, "_recording_kick_env_skill_ids", None) or {}
            if recording_kick_ids and kick_selected.any():
                recording_tensor = torch.tensor(list(recording_kick_ids.keys()), device=self.device, dtype=env_ids.dtype)
                kick_selected = kick_selected & (~torch.isin(env_ids, recording_tensor))
            if kick_selected.any():
                # 2026-08-15, Tier 3 Group B Wave 2: per-skill probability, gathered by each
                # env_id's own (already-fixed-this-tick, see skill_id assignment above) skill --
                # same single torch.rand(env_ids.shape[0]) draw either way, so this never changes
                # the RNG-consumption order/count versus the scalar path, only the threshold each
                # draw is compared against.
                if self._mid_episode_kick_entry_prob_per_skill is not None:
                    prob_threshold = self._mid_episode_kick_entry_prob_per_skill[self.skill_id[env_ids]]
                else:
                    prob_threshold = self._mid_episode_kick_entry_prob
                draw = torch.rand(env_ids.shape[0], device=self.device) < prob_threshold
                pending_ids = env_ids[kick_selected & draw]
                if pending_ids.numel() > 0:
                    self.task_mode[pending_ids] = TaskMode.LOCOMOTION
                    self._kick_pending[pending_ids] = True
                    motion_command = self.command_manager.get_state("motion_command")
                    if motion_command is not None:
                        motion_command.motion_ids[pending_ids] = self.skill_id[pending_ids]
                        # Increment 4 (D2b, mid_episode_kick_entry_ball_fixed): under the default
                        # (False), the ball gets NO placement here -- D4's original finding still
                        # applies (reset() is task_mode="kick"-tagged, never called for this
                        # LOCOMOTION-mode reset), place_ball_at_entry supplies it later at fire
                        # time instead. Under fixed-ball mode, the ball must be placed NOW (it
                        # will never move again this episode) -- see place_ball_at_reset_pending's
                        # own docstring.
                        if self._mid_episode_kick_entry_ball_fixed:
                            motion_command.place_ball_at_reset_pending(pending_ids)

        self._forced_task_mode[env_ids] = -1
        self._pin_recording_env_modes(env_ids)

    def _pin_recording_env_modes(self, env_ids: torch.Tensor) -> None:
        """Force the envs dedicated to wandb video recording (see
        _setup_task_video_recording) to always run their assigned task (and, for kick recorders,
        their assigned skill) — overriding whatever the random draw above just picked for them.
        Without this, "Training rollout - Kick" would only show a kick clip on the (rare,
        probability-gated) episodes where that particular env happened to be sampled into
        kick-mode — with pinning, every recorded episode is guaranteed to be the right task (and,
        in N-skill mode, the right skill)."""
        recording_env_ids = getattr(self, "_recording_env_ids", None)
        if recording_env_ids:
            loco_id = recording_env_ids.get("locomotion")
            if loco_id is not None:
                match = env_ids == loco_id
                if match.any():
                    self.task_mode[env_ids[match]] = TaskMode.LOCOMOTION

        kick_env_skill_ids = getattr(self, "_recording_kick_env_skill_ids", None)
        if kick_env_skill_ids:
            for kick_env_id, skill_id in kick_env_skill_ids.items():
                match = env_ids == kick_env_id
                if match.any():
                    self.task_mode[env_ids[match]] = TaskMode.KICK
                    # In legacy (non-N-skill) mode skill_id is always 0 for every env already (see
                    # _init_buffers), so writing 0 here is a true no-op -- bit-identical to the
                    # pre-N-skill behavior, which never touched skill_id at all.
                    self.skill_id[env_ids[match]] = skill_id

    def _setup_task_video_recording(self) -> None:
        """Add one video recorder per motion skill (or exactly one generic recorder in legacy
        single-skill mode), each bound to a dedicated, always-flat-terrain env pinned permanently
        to kick-mode AND to that recorder's own skill (see _pin_recording_env_modes) — so the
        "Training rollout" media section shows one locomotion video and one kick video PER SKILL
        every interval, instead of only whatever the single default-recorded env (0) happens to
        be doing that episode (which, at typical kick probabilities/terrain ratios, is usually
        locomotion).

        Without per-skill pinning, N-skill mode would still only ever show ONE skill in
        "Training rollout - Kick" for the entire run (whichever skill the single pinned env
        happened to draw at startup, since skill_id is fixed-for-life) -- the other N-1 skills
        would never appear in wandb at all. This is what fixes that gap.

        No-op if video recording is disabled, or if no flat-terrain env distinct from the primary
        recorder's env is available to dedicate to kicking.

        Idempotent: _init_buffers() (which calls this) runs more than once per env lifetime —
        once directly from BaseTask.__init__, and again every time reset_all() is called (see
        this class's own reset_all() override) — but the extra recorder(s)/camera must only ever
        be created once, so this returns immediately on every call after the first."""
        if getattr(self, "_recording_env_ids", None) is not None:
            return

        primary = self.simulator.video_recorder
        if primary is None or not primary.enabled:
            self._recording_env_ids: dict[str, int] = {}
            self._recording_kick_env_skill_ids: dict[int, int] = {}
            return

        self._recording_env_ids = {"locomotion": primary.config.record_env_id}
        self._recording_kick_env_skill_ids = {}

        flat = self.terrain_manager.get_state("locomotion_terrain").env_terrain_is_flat
        flat_ids = flat.nonzero(as_tuple=False).flatten()
        candidates = flat_ids[flat_ids != primary.config.record_env_id]
        if candidates.numel() == 0:
            logger.warning(
                "UnifiedManager: no flat-terrain env distinct from the primary video-recorded env "
                f"({primary.config.record_env_id}) is available to dedicate to kick-mode recording "
                "— only a locomotion training video will be logged to wandb this run."
            )
            return

        # IsaacSimVideoRecorder is IsaacSim-only -- other backends (MuJoCo Classic/Warp, IsaacGym)
        # have no equivalent second-recorder implementation. Skip the kick-video recording
        # enhancement for them (the PRIMARY recorder set up above still works normally either way)
        # rather than crash -- same pattern as the has_ball guard in managers/command/terms/wbt.py.
        # Checked BEFORE touching _recording_env_ids so state stays consistent on the early return
        # (either both the id(s) and the recorder(s) get set, or neither does).
        if self.simulator.get_simulator_type() != SimulatorType.ISAACSIM:
            logger.warning(
                "UnifiedManager: kick-mode video recording (separate locomotion+kick wandb videos) "
                "is IsaacSim-only -- skipping the extra recorder(s) on this backend. The primary "
                "recorder still logs normally."
            )
            return

        from dataclasses import replace

        from holosoma.config_types.video import CartesianCameraConfig
        from holosoma.simulator.isaacsim.video_recorder import IsaacSimVideoRecorder

        # interval=1 (2026-07-21 finding): VideoConfig.interval counts EPISODE-STARTS of this
        # recorder's OWN pinned env, and locomotion vs. kick reset at wildly different natural
        # rates -- measured on a fresh Stage B run, kick_episode_length ~66 steps (early-term-heavy,
        # matches this project's whole kick-stability focus) vs. locomotion's back-calculated ~2500
        # steps (~40x longer). Both recorders inheriting the SAME interval (default 10) meant
        # locomotion needed ~10x its own already-40x-longer reset cycle -- 25,000+ steps, ~75-90
        # min wall-clock -- before its FIRST video, while kick was producing several per minute. The
        # feature's own intent (see this function's docstring) was comparable per-interval cadence
        # for both, not this. Pin locomotion's own interval to 1 rather than lowering the shared
        # config default, which would also touch every non-unified/non-dual-task experiment that
        # never had this problem: its own reset rate is already low enough that recording every
        # single genuine episode is cheap and gives a video roughly every 2500 steps.
        # "isaacsim_media/" prefix: wandb groups logged media (and scalar) keys into UI panel
        # sections by "/"-delimited prefix, same convention as "train/loss" vs "eval/loss" -- this
        # puts every IsaacSim-recorded video under its own "isaacsim_media" section, separate from
        # the "mujoco_media" section the RoboJuDo/MuJoCo sim2sim rollouts log under (see
        # record_mujoco_kick_rollout.py/record_mujoco_locomotion_rollout.py's WANDB_KEY constants).
        primary.config = replace(primary.config, wandb_key="isaacsim_media/Training rollout - Locomotion", interval=1)

        # Legacy single-skill mode: exactly 1 generic recorder, skill_id 0 (a true no-op pin, see
        # _pin_recording_env_modes) -- bit-identical to the pre-N-skill "kick" recorder. N-skill
        # mode: 1 recorder per configured skill, so every skill gets its own wandb video.
        num_skills = len(self._skill_motion_training_ratios)
        num_recorders = max(num_skills, 1)
        if candidates.numel() < num_recorders:
            logger.warning(
                f"UnifiedManager: only {candidates.numel()} flat-terrain env(s) distinct from the "
                f"primary recorder are available, but {num_recorders} kick-video recorder(s) were "
                f"wanted ({num_skills} configured skill(s)) -- recording only "
                f"{candidates.numel()}, remaining skill(s) will not get a wandb video this run."
            )
            num_recorders = candidates.numel()

        # 2026-08-16 finding: each kick recorder's own capture_frame->_update_camera_position->
        # set_world_poses/set_local_poses chain runs every physics step for the ENTIRE episode it's
        # recording (not just N frames), and at interval=1 (inherited from primary.config above) that
        # means continuously, every kick episode, for the whole run -- profiled at ~18% of collection
        # wall-clock time on a 2-skill (3-recorder) run. Unlike locomotion, kick can afford a much
        # higher interval: kick_episode_length ~66 steps vs locomotion's ~2500 (see the interval=1
        # comment above), so interval=38 gives kick roughly the SAME video-every-~2500-steps cadence
        # as locomotion's interval=1 -- the cadence parity the original feature intended -- while
        # cutting the per-skill recorder's steady-state capture cost by ~38x. Override per-launch via
        # HOLOSOMA_KICK_RECORDER_INTERVAL (same env-var-knob convention as HOLOSOMA_SIM2SIM_LOCK_PATH/
        # HOLOSOMA_ROBOJUDO_PYTHON in record_mujoco_kick_rollout.py) if ~2500 steps/video is too
        # sparse/frequent for a given run -- e.g. `export HOLOSOMA_KICK_RECORDER_INTERVAL=100` before
        # the training launch command for a sparser (cheaper) cadence, or `=10` for more frequent
        # videos at more capture cost. 1 reproduces the old always-record-every-episode behavior.
        KICK_RECORDER_INTERVAL = int(os.environ.get("HOLOSOMA_KICK_RECORDER_INTERVAL", "38"))
        camera = CartesianCameraConfig(offset=[3.5, 3.5, 2.0], target_offset=[1.2, 0.0, 0.3])
        for i in range(num_recorders):
            env_id = int(candidates[i])
            skill_id = i if num_skills > 0 else 0
            # 1-indexed in the wandb key to match the yaml's own motion_skill_1/motion_skill_2/...
            # naming (skill_id is the 0-indexed internal value) -- avoids an off-by-one when
            # comparing a video to its yaml block.
            wandb_key = (
                f"isaacsim_media/Training rollout - Kick - Skill {i + 1}"
                if num_skills > 0
                else "isaacsim_media/Training rollout - Kick"
            )
            self._recording_kick_env_skill_ids[env_id] = skill_id
            recorder = IsaacSimVideoRecorder(
                replace(
                    primary.config,
                    record_env_id=env_id,
                    wandb_key=wandb_key,
                    camera=camera,
                    interval=KICK_RECORDER_INTERVAL,
                ),
                self.simulator,
            )
            self.simulator.add_video_recorder(recorder)

        logger.info(
            f"UnifiedManager: recording locomotion from env {primary.config.record_env_id}; kick "
            f"recorder(s) pinned to skill(s) {list(self._recording_kick_env_skill_ids.values())} on "
            f"env(s) {list(self._recording_kick_env_skill_ids.keys())} (all pinned to their task/"
            "skill permanently) — logged to wandb."
        )

    def _reset_tasks_callback(self, env_ids):
        super()._reset_tasks_callback(env_ids)
        # Must resample before _reset_robot_states_callback/_reset_buffers_callback (both branch
        # on the fresh value) and before command_manager.reset() (called later in
        # BaseTask.reset_envs_idx, filters env_ids per-term using this same fresh value).
        self._resample_task_mode(env_ids)

    # ------------------------------------------------------------------
    # Buffers — union of LeggedRobotLocomotionManager's and WholeBodyTrackingManager's
    # _init_buffers (both are near-identical already; this just merges the small differences:
    # locomotion's _init_counters/lidar_height_offset are additions on top of the shared parts).

    def _init_buffers(self):
        super()._init_buffers()

        self.base_quat = self.simulator.base_quat
        self.need_to_refresh_envs = torch.ones(
            self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False
        )
        self._configure_default_dof_pos()
        self._init_domain_rand_buffers()
        self._init_counters()
        self.lidar_height_offset = getattr(self.robot_config, "lidar_height_offset", 0.5)
        self._sync_scene_env_origins_with_terrain()

        self.task_mode = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        # Which motion skill (0..N-1) a kick-mode env is permanently assigned to -- meaningless
        # where task_mode == LOCOMOTION. Fixed at the same time as task_mode itself, see
        # _build_task_mode_partition/_resample_task_mode. Stays 0 for every env under the legacy
        # single-skill/2-way partition (skill_motion_training_ratios unset), where it's unused.
        self.skill_id = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._forced_task_mode = torch.full((self.num_envs,), -1, dtype=torch.long, device=self.device)
        self._kick_probability = float(getattr(self.command_manager.command_cfg, "kick_probability", 0.5))
        # N-skill mode: a list[float], one entry per motion skill, from the stacked yaml (empty ->
        # legacy 2-way kick_probability partition below, unchanged). See _build_task_mode_partition.
        self._skill_motion_training_ratios = list(
            getattr(self.command_manager.command_cfg, "skill_motion_training_ratios", [])
        )
        # Stage D's post-swing -> locomotion handoff (2026-08-09). Off by default (exact no-op) --
        # see MultiSkillConfig.kick_recovery_locomotion_flip_enabled's own docstring for the full
        # rationale. Same access pattern as _kick_probability above: read once here from
        # command_manager.command_cfg, not from config_values/unified/g1/reward.py's dual-path
        # resolution (that path is for reward-term params at manager-construction time; this flag
        # gates a per-tick task_mode mutation in _update_tasks_callback instead).
        self._kick_recovery_locomotion_flip_enabled = bool(
            getattr(self.command_manager.command_cfg, "kick_recovery_locomotion_flip_enabled", False)
        )
        # Locomotion -> kick direction (the mirror of the flip above), 2026-08-13. See
        # MultiSkillConfig.mid_episode_kick_entry_prob's own docstring and
        # https://claude.ai/code/artifact/53c1da51-d841-4979-8bf8-efd5ea652e06 (decisions D1-D8)
        # for the full design. Same direct-read access pattern as the flip flag above.
        self._mid_episode_kick_entry_prob = float(
            getattr(self.command_manager.command_cfg, "mid_episode_kick_entry_prob", 0.0)
        )
        self._mid_episode_kick_entry_min_steps = int(
            getattr(self.command_manager.command_cfg, "mid_episode_kick_entry_min_steps", 100)
        )
        self._mid_episode_kick_entry_max_residual = float(
            getattr(self.command_manager.command_cfg, "mid_episode_kick_entry_max_residual", 0.0)
        )
        self._pre_kick_decel_steps = float(getattr(self.command_manager.command_cfg, "pre_kick_decel_steps", 0.0))
        self._pre_kick_decel_target = float(getattr(self.command_manager.command_cfg, "pre_kick_decel_target", 0.1))
        self._pre_kick_fallback_timeout_steps = float(
            getattr(self.command_manager.command_cfg, "pre_kick_fallback_timeout_steps", 0.0)
        )
        # Increment 3 (reference-side blend): read here, not in reward.py/termination.py like its
        # ramp/grace siblings, because it's consumed directly by _enter_kick below (passed into
        # MotionCommand.capture_ref_blend), not at reward/termination-manager-construction time.
        self._pre_kick_reference_blend_steps = float(
            getattr(self.command_manager.command_cfg, "pre_kick_reference_blend_steps", 0.0)
        )
        # Increment 4 (D2b -- closes the "at deploy the ball doesn't move" training-scaffold gap):
        # also mirrored onto MotionConfig (config_types/command.py) so MotionCommand.setup() has
        # it at table-build time -- see that field's own docstring for why the two reads exist.
        self._mid_episode_kick_entry_ball_fixed = bool(
            getattr(self.command_manager.command_cfg, "mid_episode_kick_entry_ball_fixed", False)
        )
        # FIX 3 of the observation-side handoff work (2026-08-18): the missing 4th smoothing
        # sibling. Read here (same direct command_cfg access as the three above) and consumed by
        # task_mode_mask_soft below, which ObservationManager calls in place of task_mode_mask.
        # See MultiSkillConfig.pre_kick_obs_ramp_steps's own docstring for the measurement.
        self._pre_kick_obs_ramp_steps = float(
            getattr(self.command_manager.command_cfg, "pre_kick_obs_ramp_steps", 0.0)
        )
        # FIX 6 of the 2026-08-18 observation work: the LOAD-TIME blend, read the same way as its
        # mid-episode siblings above. Consumed by FastSACAgent.load() (not by this class directly)
        # to configure ObservationManager.set_warm_start_blend when a checkpoint's observation
        # config differs from the current one -- see MultiSkillConfig.warm_start_obs_ramp_steps's
        # own docstring for the full mechanism.
        self._warm_start_obs_ramp_steps = float(
            getattr(self.command_manager.command_cfg, "warm_start_obs_ramp_steps", 0.0)
        )
        # FIX 5 -- the kick->locomotion (flip) direction of the same observation discontinuity.
        # Read here rather than in reward.py/termination.py because, like its pre_kick sibling, it
        # is consumed per-tick by this class's own task_mode_mask_soft.
        self._post_flip_obs_ramp_steps = float(
            getattr(self.command_manager.command_cfg, "post_flip_obs_ramp_steps", 0.0)
        )

        # "Simultaneous per-skill task configs" (2026-08-15, Tier 3 Group B Wave 2) -- per-skill
        # [n_skills] tensor counterparts for the 8 Stage D handoff fields read above. Unlike
        # reward.py/termination.py's own per-skill fields (resolved at config-import time into
        # RewardTermCfg/TerminationTermCfg.params_per_skill, gathered generically by those
        # managers), these 8 gate PER-TICK STATE MACHINE TRANSITIONS run directly by this class
        # (_resample_task_mode/_maybe_flip_kick_recovery_to_locomotion/_enter_kick/
        # _maybe_enter_kick_from_locomotion) -- see e.g. kick_recovery_locomotion_flip_enabled's
        # own docstring ("why this reads from command_manager.command_cfg directly") for why that
        # generic mechanism was never a fit here even before per-skill support existed. Resolved
        # independently here (not imported from reward.py/termination.py) for the same
        # each-module-owns-its-own-cheap-idempotent-parse reason those two don't share
        # _skill_task_config_paths with each other either. None (the common case, including every
        # legacy/single-skill run) whenever a field has no genuine per-skill divergence -- every
        # consumer below falls back to the plain scalar attribute unchanged, byte-identical to
        # before this existed.
        _skills = getattr(self.command_manager.command_cfg, "skills", None) or []
        _skill_task_config_paths: list = (
            [resolve_task_config_path(sc.task_config) if sc.task_config is not None else None for sc in _skills]
            if _skills
            else None
        )

        def _per_skill_tensor(field_name: str, base_value, dtype: torch.dtype) -> torch.Tensor | None:
            resolved = resolve_per_skill_param(_skill_task_config_paths, field_name, base_value)
            return torch.tensor(resolved, dtype=dtype, device=self.device) if resolved is not None else None

        self._kick_recovery_locomotion_flip_enabled_per_skill = _per_skill_tensor(
            "kick_recovery_locomotion_flip_enabled", self._kick_recovery_locomotion_flip_enabled, torch.bool
        )
        self._mid_episode_kick_entry_prob_per_skill = _per_skill_tensor(
            "mid_episode_kick_entry_prob", self._mid_episode_kick_entry_prob, torch.float32
        )
        self._mid_episode_kick_entry_min_steps_per_skill = _per_skill_tensor(
            "mid_episode_kick_entry_min_steps", self._mid_episode_kick_entry_min_steps, torch.long
        )
        self._mid_episode_kick_entry_max_residual_per_skill = _per_skill_tensor(
            "mid_episode_kick_entry_max_residual", self._mid_episode_kick_entry_max_residual, torch.float32
        )
        self._pre_kick_decel_steps_per_skill = _per_skill_tensor(
            "pre_kick_decel_steps", self._pre_kick_decel_steps, torch.float32
        )
        self._pre_kick_decel_target_per_skill = _per_skill_tensor(
            "pre_kick_decel_target", self._pre_kick_decel_target, torch.float32
        )
        self._pre_kick_fallback_timeout_steps_per_skill = _per_skill_tensor(
            "pre_kick_fallback_timeout_steps", self._pre_kick_fallback_timeout_steps, torch.float32
        )
        self._pre_kick_reference_blend_steps_per_skill = _per_skill_tensor(
            "pre_kick_reference_blend_steps", self._pre_kick_reference_blend_steps, torch.float32
        )

        # Edge-detection buffer for the raw-clip -> synthetic-recovery-tail boundary crossing
        # (time_steps reaching pre_recovery_motion_end_idx, True->False), same pattern as
        # GaitCommand._was_standing / _ShotTracker._prev_time_steps. Initialized False (not
        # queried from the real motion_command state) so the very first tick after construction
        # can never register a false crossing -- _update_tasks_callback's own episode_length_buf
        # > 1 guard is the second, independent safety net for the same case.
        self._prev_in_raw_clip_phase = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        # 2026-08-12: per-env tick this env crossed the flip boundary (episode_length_buf's value
        # AT the flip, not a duration), sentinel -1 = "not currently post-flip" -- either never
        # kick-partitioned, still pre-flip, or a fresh episode since the last flip. Shared state
        # backing TWO independent, independently-toggleable opt-in mechanisms that both need "how
        # long has it been since this env flipped": post_flip_termination_grace_steps (managers/
        # termination/terms/locomotion.py's contact/low_height graced wrappers) and
        # post_flip_reward_decay_steps (managers/reward/terms/kick_scale_wrappers.py's motion-
        # tracking decay). Both default to 0.0 (exact no-op), so this buffer existing and being
        # maintained unconditionally costs nothing when neither is enabled. Set in
        # _maybe_flip_kick_recovery_to_locomotion at the exact tick of the flip; cleared to -1 in
        # _resample_task_mode on every genuine reset (kick-partitioned or not, so a locomotion-
        # partitioned env's sentinel is never touched by the flip path and a kick env's sentinel
        # never leaks across episodes).
        self._post_flip_step = torch.full((self.num_envs,), -1, dtype=torch.long, device=self.device)
        # Locomotion -> kick direction's own per-env state, mirroring _post_flip_step's shape
        # exactly (mirror mechanism, mirror bookkeeping). _kick_pending: this env reset into
        # LOCOMOTION carrying a pending kick-entry rather than teleporting immediately (see
        # _resample_task_mode). _kick_pending_best_residual/_best_frame: running best-so-far from
        # the entry-point search (D3's turning-point rule -- fire once residual stops improving,
        # not at a fixed threshold), reset to +inf/-1 on every genuine episode reset alongside
        # _kick_pending itself so a stale best from a PRIOR episode can never leak into a fresh
        # one. _pre_kick_step: episode_length_buf's value AT the moment a mid-episode entry
        # actually fires (sentinel -1 = never entered mid-episode this life) -- backs
        # pre_kick_reward_ramp_steps/pre_kick_termination_grace_steps the same way _post_flip_step
        # backs their post-flip counterparts.
        self._kick_pending = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._kick_pending_best_residual = torch.full((self.num_envs,), float("inf"), device=self.device)
        self._kick_pending_best_frame = torch.full((self.num_envs,), -1, dtype=torch.long, device=self.device)
        self._pre_kick_step = torch.full((self.num_envs,), -1, dtype=torch.long, device=self.device)
        # D3/D8's decel-and-retry fallback (only reachable when pre_kick_decel_steps > 0.0): once a
        # kick-pending env's best-so-far residual stops improving but is still worse than
        # mid_episode_kick_entry_max_residual, the env enters this state instead of declining
        # outright. _pre_kick_fallback_active: currently decaying-and-retrying.
        # _pre_kick_fallback_start_step/_start_vx: episode_length_buf and LocomotionCommand vx
        # captured the instant fallback began -- the anchor _maybe_enter_kick_from_locomotion
        # interpolates from, toward pre_kick_decel_target, over pre_kick_decel_steps ticks.
        # _pre_kick_fallback_stale_ticks: count of ticks-in-fallback where the search did NOT
        # improve -- an improving tick is free (matches pre_kick_fallback_timeout_steps' own
        # "EXTENDED, not reset" docstring); compared against pre_kick_fallback_timeout_steps to
        # decide when to give up. All four cleared alongside the rest of the kick-pending state in
        # _clear_kick_pending, both on genuine episode reset and on fire/decline mid-episode.
        self._pre_kick_fallback_active = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._pre_kick_fallback_start_step = torch.full((self.num_envs,), -1, dtype=torch.long, device=self.device)
        self._pre_kick_fallback_start_vx = torch.zeros(self.num_envs, device=self.device)
        self._pre_kick_fallback_stale_ticks = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._build_task_mode_partition()
        self._setup_task_video_recording()
        self._init_kick_episode_stats()

    # ------------------------------------------------------------------
    # KICK-mode episode statistics
    #
    # Why these exist (added 2026-07-20): `average_episode_length` — and the curriculum/penalty
    # scheduler behind it — is fed ONLY locomotion episodes, on purpose (see the comment in
    # _reset_buffers_callback). That is right for the penalty scheduler but it left the project with
    # NO training-time signal for kick stability whatsoever: a Stage-C policy whose kick episodes
    # were dying ~65 ticks after the trigger still logged `average_episode_length` ~= 950, because
    # that number is locomotion-only. The progressive Stage-C kick degradation was consequently
    # invisible for 255k iterations and only surfaced in post-hoc MuJoCo deploy probes. These
    # counters are deliberately READ-ONLY — they feed logging and nothing else, so the penalty
    # scheduler's locomotion-only signal is preserved exactly as before.

    _KICK_EMA_DECAY = 0.99
    # Matches kick_low_height termination's own min_height (config_values/unified/g1/termination.py)
    # -- deliberately the SAME number, so kick_topple_frac below answers "did this episode ever
    # cross the line the fall-termination itself uses" rather than an independently-chosen
    # threshold that would silently disagree with it.
    _KICK_FALL_HEIGHT_THRESHOLD = 0.40
    _KICK_MIN_HEIGHT_SENTINEL = 10.0  # taller than any real G1 base height -- "no reading yet"

    def _init_kick_episode_stats(self) -> None:
        # Idempotent: _init_buffers() runs more than once per env lifetime (BaseTask.__init__ and
        # again on every reset_all()), same pattern as _build_task_mode_partition.
        if getattr(self, "_kick_ep_len_ema", None) is not None:
            return
        self._kick_ep_len_ema = torch.zeros((), dtype=torch.float, device=self.device)  # ended-episode length
        self._kick_ep_term_ema = torch.zeros((), dtype=torch.float, device=self.device)  # terminated, not timed out
        # 2026-07-28 (user-requested aliveness/shooting evaluation logging): same rolling-EMA
        # pattern as the two above, updated at the same episode-end event, so they share
        # _kick_ep_decay_pow's bias correction below (identical decay schedule, no need for
        # separate ones). Read from the SAME _ShotTracker the shooting reward terms already
        # maintain (managers/reward/terms/shooting.py) -- these are the RAW physical quantities
        # (a 0-1 contact rate, a 0-1 goal-success rate, an actual ball speed in m/s), not the
        # reward-shaped/saturated proxies (kick_ball_velocity's Lorentzian, kick_goal_success_
        # burst's one-shot bonus) already logged under Episode/RawEpisode.
        self._kick_ep_hit_ema = torch.zeros((), dtype=torch.float, device=self.device)  # ball actually contacted
        self._kick_ep_success_ema = torch.zeros((), dtype=torch.float, device=self.device)  # reached the target
        self._kick_ep_decay_pow = torch.zeros((), dtype=torch.float, device=self.device)  # EMA bias correction
        # ball_speed gets its OWN decay-power tracker, deliberately separate from the shared one
        # above: unlike len/term/hit/success (every ending-episode batch has a well-defined mean,
        # even if 0), ball speed is only meaningful over episodes that actually made contact -- a
        # batch with zero hits has no mean to fold in at all, so this EMA's own .mul_(d) step (and
        # therefore its own decay-power increment) is SKIPPED on those calls. Sharing the other
        # metrics' decay_pow would desync the bias correction (denominator advancing on calls the
        # numerator didn't actually decay on), silently under-reporting the true recent average.
        self._kick_ep_ball_speed_ema = torch.zeros((), dtype=torch.float, device=self.device)  # max ball speed, m/s
        self._kick_ep_ball_speed_decay_pow = torch.zeros((), dtype=torch.float, device=self.device)

        # 2026-07-28 (user-requested, strict topple-only signal): kick_alive_frac/early_term_frac
        # answer "did the episode end via TERMINATION" -- but kick mode's only two termination
        # terms are bad_tracking (reference-motion error, can fire while still standing) and
        # kick_low_height (an actual fall), so that number CONFLATES choreography mismatches with
        # real falls. This tracks a raw physical fact instead, independent of whether termination
        # fired at all: did base height EVER cross _KICK_FALL_HEIGHT_THRESHOLD during the episode.
        # Per-env running state (_kick_ep_min_height, updated every step in
        # _update_counters_each_step, reset to the sentinel on every env reset in
        # _reset_buffers_callback) feeds this EMA and the mean-min-height one below at episode end.
        # Shares the len/term/hit/success decay_pow -- like those, every ending episode has a
        # well-defined min-height reading by construction (episodes are always >=1 step, so the
        # sentinel is always overwritten by a real reading before being read here).
        self._kick_ep_topple_ema = torch.zeros((), dtype=torch.float, device=self.device)  # min height ever < threshold
        self._kick_ep_min_height_ema = torch.zeros((), dtype=torch.float, device=self.device)  # mean of per-ep min height

        # Per-skill breakdown (2026-07-28, diagnosing an N-skill run's BLENDED kick_alive_frac
        # plateauing well below 1.0 -- e.g. a freshly-introduced skill dragging down an otherwise-
        # mature one, indistinguishable from a genuine regression when only the blended global EMA
        # above is visible). Same EMA-over-ended-episodes mechanism, just indexed by skill_id
        # instead of collapsed across all kick envs. `_kick_num_skills` is 1 in legacy/single-skill
        # mode (skill_id is always 0 there); per-skill and global are then numerically identical,
        # but the "Kick_skills_0" section is still emitted (2026-08-06, user-requested) -- e.g. for
        # dashboard-layout consistency across single- and multi-skill runs, not just to avoid
        # clutter. See _record_kick_episode_ends' and _update_log_dict's `> 0` gates below.
        self._kick_num_skills = max(1, len(getattr(self, "_skill_motion_training_ratios", [])))
        n = self._kick_num_skills
        self._kick_ep_len_ema_sk = torch.zeros(n, dtype=torch.float, device=self.device)
        self._kick_ep_term_ema_sk = torch.zeros(n, dtype=torch.float, device=self.device)
        self._kick_ep_hit_ema_sk = torch.zeros(n, dtype=torch.float, device=self.device)
        self._kick_ep_success_ema_sk = torch.zeros(n, dtype=torch.float, device=self.device)
        self._kick_ep_decay_pow_sk = torch.zeros(n, dtype=torch.float, device=self.device)
        self._kick_ep_ball_speed_ema_sk = torch.zeros(n, dtype=torch.float, device=self.device)
        self._kick_ep_ball_speed_decay_pow_sk = torch.zeros(n, dtype=torch.float, device=self.device)
        self._kick_ep_topple_ema_sk = torch.zeros(n, dtype=torch.float, device=self.device)
        self._kick_ep_min_height_ema_sk = torch.zeros(n, dtype=torch.float, device=self.device)

        # 2026-08-14 (user-requested): is the locomotion->kick handoff mechanism (D1-D8,
        # _maybe_enter_kick_from_locomotion/_enter_kick) actually learnable, and can we see it
        # improving in wandb? The trigger itself (WHEN to enter) is scripted/heuristic, not
        # learned -- but HOW WELL the policy handles whatever locomotion state it was in at
        # trigger time (tracking the kick clip, not falling) is exactly what the reward
        # ramp/termination grace/reference blend exist to shape via RL, so it should show real
        # training progress. Without a split, that signal is invisible: handoff-triggered kick
        # episodes (mid_episode_kick_entry_prob, currently a minority of kick episodes) are pooled
        # into the same kick_alive_frac/kick_episode_length/etc above as ordinary reset-triggered
        # kick episodes, diluting any handoff-specific trend.
        #
        # _kick_ep_is_handoff: per-env, NOT an EMA -- True for the remainder of an episode once
        # _enter_kick fires for that env (the ONLY call site -- both the immediate-fire and D8
        # fallback-fire paths in _maybe_enter_kick_from_locomotion), False otherwise. Reset to
        # False on every genuine episode reset in _reset_buffers_callback, mirroring
        # _kick_ep_min_height's own per-episode reset below.
        self._kick_ep_is_handoff = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        # EMA-over-ended-handoff-episodes, same rolling methodology (and _KICK_EMA_DECAY) as the
        # pooled kick_* stats above, just filtered to _kick_ep_is_handoff episodes at fold-in time
        # (_record_kick_episode_ends) -- own decay_pow since handoff episodes end less often than
        # the pooled total (same "only advance the denominator when there's a real sample" rule
        # _kick_ep_ball_speed_decay_pow already uses).
        self._kick_ep_len_ema_handoff = torch.zeros((), dtype=torch.float, device=self.device)
        self._kick_ep_term_ema_handoff = torch.zeros((), dtype=torch.float, device=self.device)
        self._kick_ep_topple_ema_handoff = torch.zeros((), dtype=torch.float, device=self.device)
        self._kick_ep_decay_pow_handoff = torch.zeros((), dtype=torch.float, device=self.device)

        # Per-env running tracker (NOT an EMA -- reset every episode, see _reset_buffers_callback,
        # updated every step in _update_counters_each_step). Initialized once here to the sentinel;
        # every env gets a real reset via _reset_buffers_callback before its first real episode
        # begins (env.reset_all() at construction), so this initial value is never actually read.
        self._kick_ep_min_height = torch.full(
            (self.num_envs,), self._KICK_MIN_HEIGHT_SENTINEL, dtype=torch.float, device=self.device
        )

    def _record_kick_episode_ends(self, kick_ids: torch.Tensor) -> None:
        """Fold the just-ended kick episodes into the rolling stats.

        Must be called BEFORE episode_length_buf is zeroed — that buffer still holds each ending
        episode's length at this point, and time_out_buf distinguishes "hit the 20s cap" (healthy)
        from "terminated early" (fell / bad tracking), which is the signal we actually care about.
        """
        self._init_kick_episode_stats()  # idempotent; guards a reset firing before _init_buffers ends
        lengths = self.episode_length_buf[kick_ids].float()
        # time_out_buf True => episode ended on the time limit, not a failure.
        terminated = (~self.time_out_buf[kick_ids].bool()).float()
        d = self._KICK_EMA_DECAY
        self._kick_ep_len_ema.mul_(d).add_(lengths.mean() * (1.0 - d))
        self._kick_ep_term_ema.mul_(d).add_(terminated.mean() * (1.0 - d))

        min_heights = self._kick_ep_min_height[kick_ids]
        fell = (min_heights < self._KICK_FALL_HEIGHT_THRESHOLD).float()
        self._kick_ep_topple_ema.mul_(d).add_(fell.mean() * (1.0 - d))
        self._kick_ep_min_height_ema.mul_(d).add_(min_heights.mean() * (1.0 - d))

        # Handoff-only fold-in (see _init_kick_episode_stats' comment) -- skipped when this batch
        # of ended kick episodes happens to contain no handoff-triggered ones, same "only advance
        # the denominator on a real sample" rule as the per-skill/ball-speed EMAs below.
        handoff_mask = self._kick_ep_is_handoff[kick_ids]
        if bool(handoff_mask.any()):
            self._kick_ep_len_ema_handoff.mul_(d).add_(lengths[handoff_mask].mean() * (1.0 - d))
            self._kick_ep_term_ema_handoff.mul_(d).add_(terminated[handoff_mask].mean() * (1.0 - d))
            self._kick_ep_topple_ema_handoff.mul_(d).add_(fell[handoff_mask].mean() * (1.0 - d))
            self._kick_ep_decay_pow_handoff.mul_(d).add_(1.0 - d)

        # Shot tracker may not exist yet (e.g. the very first-ever reset, before any kick-mode
        # reward computation has run to lazily construct it) -- skip this update rather than
        # crash; the length/termination EMAs above are unaffected either way.
        shot_tracker = getattr(self, "_shot_tracker", None)
        if shot_tracker is not None:
            has_kicked = shot_tracker.has_kicked[kick_ids]
            success = shot_tracker.success_latched[kick_ids]
            self._kick_ep_hit_ema.mul_(d).add_(has_kicked.float().mean() * (1.0 - d))
            self._kick_ep_success_ema.mul_(d).add_(success.float().mean() * (1.0 - d))
            # Ball speed only over attempts that actually made contact -- averaging in the 0.0
            # max_ball_speed of every miss would understate how fast a REAL strike sends the ball.
            # Own decay-power increment (see _init_kick_episode_stats' comment): only advances on
            # calls that actually had a hit to fold in, keeping its bias correction consistent.
            if bool(has_kicked.any()):
                mean_speed_of_hits = shot_tracker.max_ball_speed[kick_ids][has_kicked].mean()
                self._kick_ep_ball_speed_ema.mul_(d).add_(mean_speed_of_hits * (1.0 - d))
                self._kick_ep_ball_speed_decay_pow.mul_(d).add_(1.0 - d)

        self._kick_ep_decay_pow.mul_(d).add_(1.0 - d)

        # Per-skill fold-in. A skill with NO ended episodes in this particular call must not have
        # its own decay_pow advanced (same "only advance the denominator when there's a real
        # sample to fold in" rule _kick_ep_ball_speed_decay_pow already uses above) -- otherwise a
        # skill that ends episodes less often than others would have its bias correction quietly
        # desync from how many real batches it actually contributed.
        # 2026-08-06 (user-requested): gate is `> 0`, not `> 1` -- Kick_skills_0 is now populated
        # even in single-skill mode (see _update_log_dict's matching gate for why).
        if self._kick_num_skills > 0:
            skill_ids = self.skill_id[kick_ids]
            for sk in range(self._kick_num_skills):
                sk_mask = skill_ids == sk
                if not bool(sk_mask.any()):
                    continue
                self._kick_ep_len_ema_sk[sk].mul_(d).add_(lengths[sk_mask].mean() * (1.0 - d))
                self._kick_ep_term_ema_sk[sk].mul_(d).add_(terminated[sk_mask].mean() * (1.0 - d))
                self._kick_ep_topple_ema_sk[sk].mul_(d).add_(fell[sk_mask].mean() * (1.0 - d))
                self._kick_ep_min_height_ema_sk[sk].mul_(d).add_(min_heights[sk_mask].mean() * (1.0 - d))
                self._kick_ep_decay_pow_sk[sk].mul_(d).add_(1.0 - d)
                if shot_tracker is not None:
                    sk_ids = kick_ids[sk_mask]
                    sk_has_kicked = shot_tracker.has_kicked[sk_ids]
                    sk_success = shot_tracker.success_latched[sk_ids]
                    self._kick_ep_hit_ema_sk[sk].mul_(d).add_(sk_has_kicked.float().mean() * (1.0 - d))
                    self._kick_ep_success_ema_sk[sk].mul_(d).add_(sk_success.float().mean() * (1.0 - d))
                    if bool(sk_has_kicked.any()):
                        sk_speed = shot_tracker.max_ball_speed[sk_ids][sk_has_kicked].mean()
                        self._kick_ep_ball_speed_ema_sk[sk].mul_(d).add_(sk_speed * (1.0 - d))
                        self._kick_ep_ball_speed_decay_pow_sk[sk].mul_(d).add_(1.0 - d)

    def _kick_stat(self, ema: torch.Tensor, decay_pow: torch.Tensor | None = None) -> torch.Tensor:
        """Bias-corrected EMA read, so early-training values aren't dragged toward the 0 init.

        decay_pow defaults to the shared _kick_ep_decay_pow (every len/term/hit/success update has
        a well-defined batch mean, even 0, every call); pass a metric's own separate tracker (e.g.
        _kick_ep_ball_speed_decay_pow) when that metric's own .mul_(d) step doesn't run on every
        call -- see _record_kick_episode_ends' comment on why ball speed needs its own."""
        pow_tensor = decay_pow if decay_pow is not None else self._kick_ep_decay_pow
        return ema / pow_tensor.clamp_min(1e-8)

    def _build_task_mode_partition(self) -> None:
        """Permanently assign each flat-terrain-eligible env to kick or locomotion ONCE for the
        whole training run, instead of re-rolling kick_probability on every single episode reset.

        Why: kick episodes are drastically shorter than locomotion episodes early in training (an
        undertrained kick policy currently fails within ~5-10 steps of a reset, vs. locomotion's
        ~1000-step episodes), so a PER-EPISODE coin flip gives kick a fair shot at being ASSIGNED
        but not a fair shot at ACCUMULATING TRAINING DATA: measured kick_active_frac stuck around
        0.3-1% even at kick_probability=0.5-0.8, far below the ~20% expected from
        (flat_fraction x kick_probability) — an env that resamples every ~5-10 steps contributes
        far fewer transitions per unit wall-clock time than one running a full ~1000-step episode,
        so kick's aggregate share of the replay buffer ends up nowhere near kick_probability no
        matter how it's set. Partitioning by ENV instead of by EPISODE fixes the allocation at the
        env-count level, immune to episode-length imbalance — matching how the standalone
        kick-only experiment (in effect a 100% partition) gets its full training signal regardless
        of how short its own episodes are early on.

        Idempotent (same pattern as _setup_task_video_recording): _init_buffers() runs more
        than once per env lifetime (once from BaseTask.__init__, again on every reset_all()), but
        the partition must stay fixed for the whole run once decided.

        N-skill extension (self._skill_motion_training_ratios non-empty): the same fixed-for-life-
        per-env rationale above generalizes directly from a 2-way (locomotion/kick) draw to an
        (N+1)-way categorical (locomotion, skill_1, ..., skill_N) — a skill with short early
        episodes needs the same protection kick_probability originally got. Also fixes
        self._skill_id_partition (0..N-1, meaningless where task_mode==LOCOMOTION). When ratios
        is empty (no N-skill config wired in), this reduces to exactly the legacy 2-way path
        below, unchanged."""
        if getattr(self, "_task_mode_partition", None) is not None:
            return

        kick_eligible = self.terrain_manager.get_state("locomotion_terrain").env_terrain_is_flat
        kick_mode_t = torch.full((self.num_envs,), int(TaskMode.KICK), dtype=torch.long, device=self.device)
        loco_mode_t = torch.full((self.num_envs,), int(TaskMode.LOCOMOTION), dtype=torch.long, device=self.device)

        ratios = self._skill_motion_training_ratios
        if ratios:
            loco_ratio = max(1.0 - sum(ratios), 0.0)
            probs = torch.tensor([loco_ratio, *ratios], dtype=torch.float32, device=self.device)
            probs = probs / probs.sum()
            # 0 -> locomotion, i>0 -> skill (i-1). One categorical draw per env, decided once.
            draw = torch.multinomial(probs, self.num_envs, replacement=True)
            draw = torch.where(kick_eligible, draw, torch.zeros_like(draw))  # ineligible -> locomotion
            self._task_mode_partition = torch.where(draw > 0, kick_mode_t, loco_mode_t)
            self._skill_id_partition = (draw - 1).clamp_min(0)
            kick_frac = float((self._task_mode_partition == TaskMode.KICK).float().mean().item())
            target_frac = sum(ratios) * float(kick_eligible.float().mean().item())
            logger.info(
                f"UnifiedManager: task_mode partition fixed for this run (N-skill mode, "
                f"{len(ratios)} skill(s)) — {kick_frac:.3f} of all envs permanently dedicated to a "
                f"motion skill (target ~{target_frac:.3f}), the rest to locomotion."
            )
            return

        roll = torch.rand(self.num_envs, device=self.device) < self._kick_probability
        self._task_mode_partition = torch.where(kick_eligible & roll, kick_mode_t, loco_mode_t)
        self._skill_id_partition = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        kick_frac = float((self._task_mode_partition == TaskMode.KICK).float().mean().item())
        target_frac = self._kick_probability * float(kick_eligible.float().mean().item())
        logger.info(
            f"UnifiedManager: task_mode partition fixed for this run — {kick_frac:.3f} of all envs "
            f"permanently dedicated to kick (target ~{target_frac:.3f}, the rest to locomotion)."
        )

    def _sync_scene_env_origins_with_terrain(self) -> None:
        """Make IsaacLab's scene.env_origins agree with holosoma's own terrain-tile origins.

        holosoma's terrain system (TerrainManager / managers/terrain/terms/locomotion.py) is a
        parallel implementation that never registers itself with IsaacLab's own
        InteractiveScene._terrain slot. IsaacLab's `scene.env_origins` property
        (isaaclab/scene/interactive_scene.py) returns `self._terrain.env_origins` if that slot is
        set, otherwise falls back to `_default_env_origins` — a plain uniform GridCloner spacing
        that has nothing to do with where holosoma actually placed each env's terrain tile.

        Locomotion's own reset path already avoids this trap by reading
        terrain_manager.env_origins directly (see _reset_root_states below). But MotionCommand
        (managers/command/terms/wbt.py's root_pos_w/object_pos_w/reset — used for kick-mode
        placement of both the robot and the ball) reads scene.env_origins directly. For the
        standalone ball-kick/WBT experiments this is harmless by coincidence: they always train on
        terrain_locomotion_plane, an effectively unbounded flat plane, so landing at the "wrong"
        (uniform-grid) coordinates still means standing on valid flat ground. Unified uses
        terrain_unified_mix — a real, bounded, multi-tile-type terrain — where the mismatch
        between the two origin conventions is tens of meters, teleporting kick-mode robots (and
        the ball) clear off their assigned tile into empty space beyond the generated mesh. This
        sync makes MotionCommand's existing, unmodified code pick up the correct origin without
        touching any shared WBT/ball-kick file.
        """
        terrain_origins = self.terrain_manager.get_state("locomotion_terrain").env_origins
        self.simulator.scene.env_origins[:] = terrain_origins

    def _configure_default_dof_pos(self):
        self.default_dof_pos_base = torch.zeros(
            self.num_dof, dtype=torch.float, device=self.device, requires_grad=False
        )
        for i in range(self.num_dofs):
            name = self.dof_names[i]
            if name not in self.robot_config.init_state.default_joint_angles:
                raise ValueError(f"Missing default joint angle for DOF '{name}' in robot configuration.")
            angle = self.robot_config.init_state.default_joint_angles[name]
            self.default_dof_pos_base[i] = angle

        self.default_dof_pos_base = self.default_dof_pos_base.unsqueeze(0)  # (1, num_dof)
        self.default_dof_pos = self.default_dof_pos_base.repeat(self.num_envs, 1).clone()  # (num_envs, num_dof)

    def _init_counters(self):
        self.common_step_counter = 0

    def _update_counters_each_step(self):
        self.common_step_counter += 1
        # Runs after BaseTask._post_physics_step's _refresh_sim_tensors() and before termination/
        # reset processing in the same step -- exactly the window needed so a topple that happens
        # on an episode's LAST step still gets recorded before _record_kick_episode_ends reads it.
        # Updated unconditionally (all envs, not just kick-mode) -- cheap, and simpler than
        # task_mode-gating a per-step op; only ever READ for kick_ids at kick-episode-end anyway.
        self._kick_ep_min_height = torch.minimum(self._kick_ep_min_height, self.simulator.robot_root_states[:, 2])

    def _init_domain_rand_buffers(self):
        # 6-dim (lin+ang), matching WholeBodyTrackingManager's convention. Not task_mode-gated: a
        # push is domain-randomization noise equally valid in both modes (see _push_robots below).
        self.push_robot_vel_buf = torch.zeros(
            self.num_envs, 6, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.record_push_robot_vel_buf = torch.zeros(
            self.num_envs, 6, dtype=torch.float, device=self.device, requires_grad=False
        )
        self._randomize_push_robots = False
        self._max_push_vel = torch.zeros(6, dtype=torch.float32, device=self.device)

    def _setup_robot_body_indices(self):
        # Copied verbatim from LeggedRobotLocomotionManager — needed for the retained `contact`
        # termination + feet_phase/penalty_close_feet_xy locomotion reward terms.
        foot_body_names = [s for s in self.body_names if self.robot_config.foot_body_name in s]
        foot_height_names = [s for s in self.body_names if self.robot_config.foot_height_name in s]

        termination_contact_names = []
        for name in self.robot_config.terminate_after_contacts_on:
            termination_contact_names.extend([s for s in self.body_names if name in s])

        self.feet_indices = torch.zeros(len(foot_body_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i, name in enumerate(foot_body_names):
            self.feet_indices[i] = self.simulator.find_rigid_body_indice(name)

        self.feet_height_indices = torch.zeros(
            len(foot_height_names), dtype=torch.long, device=self.device, requires_grad=False
        )
        for i, name in enumerate(foot_height_names):
            self.feet_height_indices[i] = self.simulator.find_rigid_body_indice(name)

        self.termination_contact_indices = torch.zeros(
            len(termination_contact_names), dtype=torch.long, device=self.device, requires_grad=False
        )
        for i in range(len(termination_contact_names)):
            self.termination_contact_indices[i] = self.simulator.find_rigid_body_indice(termination_contact_names[i])

        if self.robot_config.has_torso:
            self.torso_name = self.robot_config.torso_name
            self.torso_index = self.simulator.find_rigid_body_indice(self.torso_name)

    def set_is_evaluating(self, command=None):
        logger.info("Setting Env is evaluating")
        super().set_is_evaluating()
        commands = self.command_manager.commands
        commands.zero_()
        if command is not None:
            command_tensor = torch.as_tensor(command, device=self.device, dtype=commands.dtype)
            commands[:] = command_tensor.view(1, -1).expand_as(commands)
        gait_state = self.command_manager.get_state("locomotion_gait")
        gait_state.set_eval_mode(True)

    def _setup_simulator_next_task(self):
        pass

    def _setup_simulator_control(self):
        self.simulator.commands = self.command_manager.commands

    def _get_envs_to_refresh(self):
        return self.need_to_refresh_envs.nonzero(as_tuple=False).flatten()

    def _refresh_envs_after_reset(self, env_ids):
        self.simulator.set_actor_root_state_tensor(env_ids, self.simulator.all_root_states)
        self.simulator.set_dof_state_tensor(env_ids, self.simulator.dof_state)
        self.simulator.clear_contact_forces_history(env_ids)
        self.need_to_refresh_envs[env_ids] = False
        self.simulator.refresh_sim_tensors()
        self._pre_compute_observations_callback()

    def _pre_compute_observations_callback(self):
        self.base_quat[:] = self.simulator.base_quat[:]
        self.terrain_manager.update_heights()

    def _update_tasks_callback(self):
        super()._update_tasks_callback()
        self._maybe_flip_kick_recovery_to_locomotion()
        self._maybe_enter_kick_from_locomotion()
        if hasattr(self.simulator, "headless_recording") and self.simulator.headless_recording:
            if hasattr(self.command_manager, "commands"):
                self.simulator.commands = self.command_manager.commands

    def _maybe_flip_kick_recovery_to_locomotion(self) -> None:
        """Stage D's post-swing -> locomotion handoff (2026-08-09; boundary revised twice more the
        same week -- see kick_recovery_locomotion_flip_enabled's own docstring for the full
        history). Called every tick from _update_tasks_callback, AFTER super()'s own call has
        already run command_manager.step() (which advances MotionCommand.time_steps for this
        tick) and BEFORE _compute_observations() runs later this same _post_physics_step() --
        confirmed via BaseTask's own call order that this placement makes task_mode_onehot and
        every task_mode-gated observation term reflect the flip on the SAME tick it fires;
        reward/termination (already computed earlier this tick, before this method runs) reflect
        it starting the NEXT tick -- matching this codebase's existing time_steps-vs-reward
        one-tick relationship, not a new inconsistency.

        Flips task_mode KICK->LOCOMOTION for kick-mode envs the instant time_steps reaches that
        motion's own pre_recovery_motion_end_idx -- the end of the WHOLE authored/raw clip
        (approach + strike + the actor's own captured post-kick follow-through), captured once
        per motion in wbt.py's setup() right before that motion's synthetic recovery-transition +
        static-hold tail gets appended (see that capture site's own comment). Deliberately NOT
        stand_start_idx (narrower -- excludes real captured follow-through footage still worth
        motion-tracking) and NOT motion_end_idx (wider -- includes the synthetic append+hold tail,
        live-tested and reverted: that tail is scripted filler, not real motion, so there's no
        reason to keep clip-tracking it once genuine authored content runs out). Pins the env's
        locomotion command to exact zero for the rest of the episode via LocomotionCommand.
        pin_zero. Deliberately does NOT touch _ShotTracker or the kick_recovery_posture_reward
        family directly -- both go inert on their own once task_mode flips (task_mode_mask gating
        already handles it; see the confirmed-harmless analysis in the Stage D plan), and are not
        re-derived here."""
        if self._kick_recovery_locomotion_flip_enabled_per_skill is None and not self._kick_recovery_locomotion_flip_enabled:
            return

        motion_command = self.command_manager.get_state("motion_command")
        if motion_command is None:
            return

        boundary = motion_command.pre_recovery_motion_end_idx[motion_command.motion_ids]
        in_raw_clip = motion_command.time_steps < boundary
        # 2026-08-15, Tier 3 Group B Wave 2: per-skill divergence means SOME kick envs must be
        # eligible to flip and others not, so the scalar early-return above can no longer gate
        # every env uniformly -- folded into the crossing mask itself instead. When there's no
        # per-skill table this is just `torch.ones(...)` (the early-return above already ruled
        # out the all-False case), matching every crossing env exactly like before this existed.
        if self._kick_recovery_locomotion_flip_enabled_per_skill is not None:
            enabled_mask = self._kick_recovery_locomotion_flip_enabled_per_skill[self.skill_id]
        else:
            enabled_mask = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        crossed = (
            (self.task_mode == TaskMode.KICK)
            & self._prev_in_raw_clip_phase
            & (~in_raw_clip)
            & (self.episode_length_buf > 1)
            & enabled_mask
        )
        env_ids = crossed.nonzero(as_tuple=False).flatten()
        if env_ids.numel() > 0:
            self.task_mode[env_ids] = TaskMode.LOCOMOTION
            locomotion_command = self.command_manager.get_state("locomotion_command")
            if locomotion_command is not None:
                locomotion_command.pin_zero(env_ids)
            # See _post_flip_step's own comment (_init_buffers) for what this backs.
            self._post_flip_step[env_ids] = self.episode_length_buf[env_ids].clone()
        self._prev_in_raw_clip_phase = in_raw_clip.clone()

    def post_flip_steps_since(self) -> torch.Tensor:
        """[num_envs] long tensor: ticks since this env crossed the kick->locomotion flip
        boundary, for envs where that's meaningful (``_post_flip_step >= 0``); 0 everywhere else
        (never kick-partitioned, still pre-flip, or reset since the last flip) -- callers that
        need to distinguish "not applicable" from "just flipped" should check
        ``self._post_flip_step >= 0`` directly rather than relying on this alone. Shared by
        managers/termination/terms/locomotion.py's post-flip-graced contact/low_height and
        managers/reward/terms/kick_scale_wrappers.py's post-flip tracking decay -- one
        computation, two independent consumers, so the "since when" definition can't drift
        between them."""
        is_post_flip = self._post_flip_step >= 0
        return torch.where(
            is_post_flip,
            (self.episode_length_buf - self._post_flip_step).clamp(min=0),
            torch.zeros_like(self.episode_length_buf),
        )

    def pre_kick_steps_since(self) -> torch.Tensor:
        """[num_envs] long tensor: ticks since this env's mid-episode locomotion->kick entry fired
        (``_pre_kick_step >= 0``); 0 everywhere else (never a mid-episode entry, or reset since).
        Mirror of post_flip_steps_since for the opposite boundary -- backs
        pre_kick_reward_ramp_steps/pre_kick_termination_grace_steps the same way that backs
        post_flip_reward_decay_steps/post_flip_termination_grace_steps."""
        is_pre_kick = self._pre_kick_step >= 0
        return torch.where(
            is_pre_kick,
            (self.episode_length_buf - self._pre_kick_step).clamp(min=0),
            torch.zeros_like(self.episode_length_buf),
        )

    def pre_kick_obs_ramp_alpha(self) -> torch.Tensor | None:
        """FIX 3 (2026-08-18). ``None`` when the observation ramp is off -- the exact-no-op
        short-circuit, checked BEFORE touching any buffer so callers never pay for it. Otherwise a
        ``[num_envs]`` float in [0, 1]: how far each env is through its post-mid-episode-entry
        observation ramp. 1.0 means "fully in kick mode" (the pre-existing binary behavior) and is
        what every env that never took a mid-episode entry reads, so only envs genuinely inside
        their ramp window differ from today.

        Keyed off ``_pre_kick_step`` (stamped by ``_enter_kick``), the SAME per-env counter
        ``pre_kick_reward_ramp_steps`` and ``pre_kick_termination_grace_steps`` already run on, so
        the three windows are guaranteed to reference one definition of "since when" and cannot
        drift apart. A kick-mode env that started that way at RESET has ``_pre_kick_step < 0`` and
        correctly reads 1.0 -- there is no transition to smooth there.
        """
        if self._pre_kick_obs_ramp_steps <= 0.0:
            return None
        is_pre_kick = self._pre_kick_step >= 0
        alpha = (self.pre_kick_steps_since().float() / self._pre_kick_obs_ramp_steps).clamp(0.0, 1.0)
        return torch.where(is_pre_kick, alpha, torch.ones_like(alpha))

    def post_flip_obs_ramp_alpha(self) -> torch.Tensor | None:
        """FIX 5 (2026-08-18) -- the OPPOSITE-direction sibling of pre_kick_obs_ramp_alpha, and the
        one that covers where the falls actually are.

        ``kick_recovery_locomotion_flip_enabled`` flips an env KICK->LOCOMOTION the instant
        ``in_kicking_phase`` goes False (at ``stand_start_idx``), pinning its velocity command to
        zero. That flip already has ``post_flip_termination_grace_steps`` and
        ``post_flip_reward_decay_steps``, but -- exactly like the loco->kick entry before FIX 3 --
        BOTH of those smooth what the policy is JUDGED ON, while the OBSERVATION steps in a single
        tick: every ``kick_*`` term goes live->0 and every ``loco_*`` term 0->live.

        Why this matters more than the entry direction: a phase-resolved probe (2026-08-18, 512
        envs, matched protocol) put **76% of teacher-1's falls and 83% of the distilled student's
        skill-1 falls in the post-stand phase** -- i.e. after this flip, not during the kick. And
        pure locomotion is not intrinsically fragile here: a locomotion control arm under identical
        terrain/push/DR toppled 0/1847 episodes. What is hard is ARRIVING in locomotion mode
        discontinuously, from an off-balance post-kick state, with the observation snapping over in
        one tick.

        Returns 1.0 for "fully in the post-flip target mode" so envs that never flipped are
        unaffected, mirroring its sibling. ``None`` when off (exact no-op short-circuit).
        """
        if self._post_flip_obs_ramp_steps <= 0.0:
            return None
        is_post_flip = self._post_flip_step >= 0
        alpha = (self.post_flip_steps_since().float() / self._post_flip_obs_ramp_steps).clamp(0.0, 1.0)
        return torch.where(is_post_flip, alpha, torch.ones_like(alpha))

    def task_mode_mask_soft(self, name: str) -> torch.Tensor:
        """Float ``[num_envs]`` counterpart of ``task_mode_mask``, consulted by ObservationManager
        (and only by it -- reward/termination/command managers keep using the bool version, since
        these fixes are deliberately scoped to the observation channel: the other channels already
        have their own dedicated smoothing fields).

        Returns the bool mask untouched when BOTH ramps are off, so ``.to(obs.dtype) * obs`` is
        bit-for-bit what it was before this method existed -- the exact-no-op guarantee.

        Handles BOTH transition directions, which are mutually exclusive per env at any instant
        (an env is either mid-entry into kick or mid-flip out of it, never both -- ``_enter_kick``
        and the flip stamp different sentinels and the flip only fires for envs already in kick
        mode that have reached ``stand_start_idx``):

        * loco->kick entry (FIX 3): the env IS in kick mode, kick terms fade IN over ``alpha``.
        * kick->loco flip (FIX 5): the env IS in locomotion mode, and the fade runs the other way --
          the kick terms it just left fade OUT rather than vanishing in one tick.

        In both cases the two halves always sum to 1, so the policy never sees a moment with both
        blocks simultaneously dark (which would look like a third, never-trained mode).
        """
        hard = self.task_mode_mask(name)
        entry_alpha = self.pre_kick_obs_ramp_alpha()
        flip_alpha = self.post_flip_obs_ramp_alpha()
        if entry_alpha is None and flip_alpha is None:
            return hard

        out = hard.float()
        if entry_alpha is not None:
            # Only envs currently IN kick mode can be mid-entry (_enter_kick sets both together).
            in_kick = self.task_mode_mask("kick")
            a = entry_alpha if name == "kick" else 1.0 - entry_alpha
            out = torch.where(in_kick, a, out)
        if flip_alpha is not None:
            # Only envs currently in LOCOMOTION mode can be mid-flip, for the mirror-image reason.
            # `_post_flip_step >= 0` alone is not enough: it stays set for the rest of the episode,
            # so without the mode check a long-since-flipped env would keep being re-blended at
            # alpha==1.0 (harmless numerically, but it would mask a genuine future entry ramp).
            in_loco = ~self.task_mode_mask("kick")
            mid_flip = in_loco & (self._post_flip_step >= 0)
            a = flip_alpha if name == "locomotion" else 1.0 - flip_alpha
            out = torch.where(mid_flip, a, out)
        return out

    def _clear_kick_pending(self, env_ids: torch.Tensor) -> None:
        """Reset every buffer backing a mid-episode kick-entry attempt (search state + D8 fallback
        state) to its sentinel -- shared by _resample_task_mode (genuine episode reset) and
        _enter_kick/_decline_kick (mid-episode fire/decline) so the three call sites can never
        drift out of sync with each other. Deliberately does NOT touch _pre_kick_step -- that
        sentinel means something different (see its own comment in _init_buffers) and is managed
        by its own two call sites directly."""
        if env_ids.numel() == 0:
            return
        self._kick_pending[env_ids] = False
        self._kick_pending_best_residual[env_ids] = float("inf")
        self._kick_pending_best_frame[env_ids] = -1
        self._pre_kick_fallback_active[env_ids] = False
        self._pre_kick_fallback_start_step[env_ids] = -1
        self._pre_kick_fallback_start_vx[env_ids] = 0.0
        self._pre_kick_fallback_stale_ticks[env_ids] = 0

    def _enter_kick(self, env_ids: torch.Tensor) -> None:
        """Commit a mid-episode locomotion->kick entry for env_ids, at each env's own
        _kick_pending_best_frame -- re-anchors the reference (MotionCommand.enter_at_frame,
        increment 1), captures the reference-blend snapshot (MotionCommand.capture_ref_blend,
        increment 3, only when pre_kick_reference_blend_steps > 0.0), places the ball at that
        frame's implied position (D2's place_ball_at_entry -- skipped under increment 4's
        mid_episode_kick_entry_ball_fixed, see below), flips task_mode, and stamps _pre_kick_step
        for the ramp/grace/blend consumers. Called from both the immediate-fire and the D8-fallback
        path in _maybe_enter_kick_from_locomotion; deliberately does not touch the locomotion
        command pin -- once task_mode is KICK it's masked/inert regardless (same reasoning
        pin_zero's own docstring gives for why kick-mode commands don't matter), and the NEXT
        genuine reset() or kick->locomotion flip (pin_zero) always overwrites it explicitly before
        it could matter again."""
        if env_ids.numel() == 0:
            return
        motion_command = self.command_manager.get_state("motion_command")
        if motion_command is None:
            return
        frames = self._kick_pending_best_frame[env_ids]
        motion_command.enter_at_frame(env_ids, frames)
        if self._pre_kick_reference_blend_steps_per_skill is not None:
            # 2026-08-15, Tier 3 Group B Wave 2: env_ids here may span skills that want blending
            # and skills that don't -- gather each env's own value and only call
            # capture_ref_blend on the subset that's actually >0.0 (confirmed safe to call on a
            # SUBSET of env_ids: capture_ref_blend's only global state is a sticky
            # _ref_blend_active bool that gates a computation shortcut, not correctness -- see its
            # own docstring in managers/command/terms/wbt.py).
            blend_steps_per_env = self._pre_kick_reference_blend_steps_per_skill[self.skill_id[env_ids]]
            blend_mask = blend_steps_per_env > 0.0
            if blend_mask.any():
                motion_command.capture_ref_blend(env_ids[blend_mask], blend_steps_per_env[blend_mask])
        elif self._pre_kick_reference_blend_steps > 0.0:
            blend_steps = torch.full(
                (env_ids.numel(),), self._pre_kick_reference_blend_steps, device=self.device
            )
            motion_command.capture_ref_blend(env_ids, blend_steps)
        # Increment 4 (D2b): under fixed-ball mode the ball was already placed once, at reset
        # (place_ball_at_reset_pending), and must NOT move again to match the robot -- that's the
        # whole point (a real ball doesn't teleport at deploy). D2's default (ball_fixed False)
        # keeps calling place_ball_at_entry exactly as increments 1-3 already validated.
        if not self._mid_episode_kick_entry_ball_fixed:
            motion_command.place_ball_at_entry(env_ids, frames)
        self.task_mode[env_ids] = TaskMode.KICK
        self._pre_kick_step[env_ids] = self.episode_length_buf[env_ids].clone()
        # This is the ONLY call site that commits a mid-episode locomotion->kick handoff (both the
        # immediate-fire and D8 fallback-fire paths in _maybe_enter_kick_from_locomotion route
        # through here) -- marks the episode for the handoff-specific stats split in
        # _record_kick_episode_ends/_update_log_dict. Cleared on the next genuine reset in
        # _reset_buffers_callback.
        self._kick_ep_is_handoff[env_ids] = True
        self._clear_kick_pending(env_ids)

    def _decline_kick(self, env_ids: torch.Tensor) -> None:
        """Give up on a mid-episode kick entry for env_ids -- the env stays in LOCOMOTION for the
        rest of the episode under ordinary randomized commands. Releases the D8 fallback's decel
        pin (LocomotionCommand.unpin) so step()'s periodic resample resumes; without this the env
        would stay frozen at whatever decel speed the fallback last held for the rest of the
        episode, which nothing else would ever correct."""
        if env_ids.numel() == 0:
            return
        locomotion_command = self.command_manager.get_state("locomotion_command")
        if locomotion_command is not None:
            locomotion_command.unpin(env_ids)
        self._clear_kick_pending(env_ids)

    def _maybe_enter_kick_from_locomotion(self) -> None:
        """Locomotion -> kick direction (2026-08-13), the mirror of
        _maybe_flip_kick_recovery_to_locomotion above. Full design:
        https://claude.ai/code/artifact/53c1da51-d841-4979-8bf8-efd5ea652e06 and memory
        locomotion_to_kick_handoff_design_settled.md (decisions D1-D8). Called every tick from
        _update_tasks_callback, right after the kick->locomotion flip -- same placement
        reasoning: this runs after command_manager.step() has advanced time_steps/commands for
        this tick and before _compute_observations(), so a fire is reflected in
        task_mode_onehot/every task_mode-gated observation on the SAME tick it happens.

        For every env carrying a pending mid-episode entry (_kick_pending, set by
        _resample_task_mode) that has walked at least mid_episode_kick_entry_min_steps ticks: runs
        MotionCommand.search_entry_point (constrained to that env's own fixed-for-life assigned
        skill, D2) every tick and tracks the running best (frame, residual).

        D3 -- fires the INSTANT the best-so-far residual stops improving tick-over-tick, not at a
        fixed threshold (modelling every available clip's own decel-and-search trajectory found an
        INTERIOR minimum -- waiting for a lower threshold can make the match up to 8x worse; see
        the design doc's D3 section for the measured curves). mid_episode_kick_entry_max_residual
        is an ABORT ceiling, not a fire threshold: at the turning point, if the best residual ever
        seen already clears it, fire now; 0.0 (default) = no ceiling, always fires at the first
        turning point.

        D8 -- if the ceiling is exceeded at the turning point and pre_kick_decel_steps > 0.0, the
        env enters a decel-and-retry fallback instead of declining outright: its locomotion command
        decays toward pre_kick_decel_target (reusing LocomotionCommand.pin_zero's own _pinned_zero
        exclusion mechanism via the new general-purpose `pin`, not a new one) while the search keeps
        running every tick; the env fires the moment the (still continuously-updated) best-so-far
        clears the ceiling. pre_kick_fallback_timeout_steps caps how long this can run, EXTENDED
        (not reset) by ticks where the search is still improving -- only non-improving ticks consume
        the timeout budget, so a genuinely converging search is never cut off mid-progress. If
        pre_kick_decel_steps is 0.0 (off), a ceiling-exceeding turning point declines immediately
        instead -- the env keeps walking under its already-assigned ordinary locomotion command for
        the rest of the episode, and may draw a fresh pending attempt next episode.

        Exact no-op end-to-end when mid_episode_kick_entry_prob <= 0.0 (the same gate
        _resample_task_mode uses before _kick_pending can ever become True) -- this method returns
        before touching any tensor."""
        if self._mid_episode_kick_entry_prob_per_skill is None and self._mid_episode_kick_entry_prob <= 0.0:
            return

        # 2026-08-15, Tier 3 Group B Wave 2: _kick_pending can only be True for an env whose OWN
        # skill actually drew a pending entry (already per-skill-gated in _resample_task_mode
        # above), so gathering min_steps by skill_id here is safe even before subsetting to
        # env_ids -- self.skill_id is available [num_envs]-wide regardless of which envs are
        # currently ready.
        if self._mid_episode_kick_entry_min_steps_per_skill is not None:
            min_steps = self._mid_episode_kick_entry_min_steps_per_skill[self.skill_id]
        else:
            min_steps = self._mid_episode_kick_entry_min_steps
        ready = self._kick_pending & (self.episode_length_buf >= min_steps)
        env_ids = ready.nonzero(as_tuple=False).flatten()
        if env_ids.numel() == 0:
            return

        motion_command = self.command_manager.get_state("motion_command")
        if motion_command is None:
            return

        frame, residual = motion_command.search_entry_point(env_ids)

        prev_best = self._kick_pending_best_residual[env_ids]
        searched_before = torch.isfinite(prev_best)
        improved = residual < prev_best
        best_residual = torch.where(improved, residual, prev_best)
        best_frame = torch.where(improved, frame, self._kick_pending_best_frame[env_ids])
        self._kick_pending_best_residual[env_ids] = best_residual
        self._kick_pending_best_frame[env_ids] = best_frame

        was_active = self._pre_kick_fallback_active[env_ids]
        if self._mid_episode_kick_entry_max_residual_per_skill is not None:
            ceiling = self._mid_episode_kick_entry_max_residual_per_skill[self.skill_id[env_ids]]
        else:
            ceiling = self._mid_episode_kick_entry_max_residual
        within_ceiling = (ceiling <= 0.0) | (best_residual <= ceiling)

        # Not-yet-fallback envs whose residual just stopped improving: this is D3's turning point
        # -- decide right now, once, whether to fire, start the D8 fallback, or decline outright.
        turning_point = searched_before & (~improved) & (~was_active)

        self._enter_kick(env_ids[turning_point & within_ceiling])

        def _start_fallback(start_ids: torch.Tensor) -> None:
            if start_ids.numel() == 0:
                return
            self._pre_kick_fallback_active[start_ids] = True
            self._pre_kick_fallback_start_step[start_ids] = self.episode_length_buf[start_ids].clone()
            locomotion_command = self.command_manager.get_state("locomotion_command")
            if locomotion_command is not None and locomotion_command.commands is not None:
                self._pre_kick_fallback_start_vx[start_ids] = locomotion_command.commands[start_ids, 0].clone()

        fallback_start_mask = turning_point & (~within_ceiling)
        fallback_start_ids = env_ids[fallback_start_mask]
        if self._pre_kick_decel_steps_per_skill is not None:
            # 2026-08-15, Tier 3 Group B Wave 2: fallback_start_ids may span skills that want the
            # decel-and-retry fallback and skills that don't -- split by each env's OWN
            # pre_kick_decel_steps rather than the single scalar branch legacy behavior used.
            decel_steps_for_starters = self._pre_kick_decel_steps_per_skill[self.skill_id[fallback_start_ids]]
            wants_fallback = decel_steps_for_starters > 0.0
            _start_fallback(fallback_start_ids[wants_fallback])
            self._decline_kick(fallback_start_ids[~wants_fallback])
        elif self._pre_kick_decel_steps > 0.0:
            _start_fallback(fallback_start_ids)
        else:
            self._decline_kick(fallback_start_ids)

        # Already-in-fallback envs: fire as soon as the continuously-updated best-so-far clears
        # the ceiling; otherwise keep decaying the command and, once the non-improving-ticks
        # budget runs out, decline.
        self._enter_kick(env_ids[was_active & within_ceiling])

        continue_mask = was_active & (~within_ceiling)
        continue_ids = env_ids[continue_mask]
        if continue_ids.numel() > 0:
            continue_stale = (~improved)[continue_mask]
            self._pre_kick_fallback_stale_ticks[continue_ids] = torch.where(
                continue_stale,
                self._pre_kick_fallback_stale_ticks[continue_ids] + 1,
                self._pre_kick_fallback_stale_ticks[continue_ids],
            )

            locomotion_command = self.command_manager.get_state("locomotion_command")
            if locomotion_command is not None and locomotion_command.commands is not None:
                elapsed = (
                    (self.episode_length_buf[continue_ids] - self._pre_kick_fallback_start_step[continue_ids])
                    .clamp(min=0)
                    .float()
                )
                # 2026-08-15, Tier 3 Group B Wave 2: continue_ids are envs ALREADY in fallback
                # (assigned in a prior tick, possibly several) -- safe to re-gather each one's own
                # skill's decel_steps/decel_target here since skill_id is fixed-for-life per env.
                if self._pre_kick_decel_steps_per_skill is not None:
                    decel_steps_for_continuing = self._pre_kick_decel_steps_per_skill[
                        self.skill_id[continue_ids]
                    ].clamp(min=1e-6)
                else:
                    decel_steps_for_continuing = max(self._pre_kick_decel_steps, 1e-6)
                ratio = (elapsed / decel_steps_for_continuing).clamp(max=1.0)
                start_vx = self._pre_kick_fallback_start_vx[continue_ids]
                if self._pre_kick_decel_target_per_skill is not None:
                    target_vx = self._pre_kick_decel_target_per_skill[self.skill_id[continue_ids]]
                else:
                    target_vx = torch.full_like(start_vx, self._pre_kick_decel_target)
                # Decays TOWARD the floor, never away from it -- an env already slower than the
                # target keeps its own (slower) command untouched rather than being sped up.
                vx = torch.where(start_vx > target_vx, start_vx + ratio * (target_vx - start_vx), start_vx)
                values = torch.zeros(continue_ids.numel(), locomotion_command.commands.shape[-1], device=self.device)
                values[:, 0] = vx
                locomotion_command.pin(continue_ids, values)

            if self._pre_kick_fallback_timeout_steps_per_skill is not None:
                # Per-env "has a timeout at all" gate folded into the comparison itself -- a skill
                # with timeout_steps<=0.0 (no cap) must never time out regardless of stale_ticks,
                # same as the scalar `if > 0.0` early-skip did for the whole block before this.
                timeout_per_env = self._pre_kick_fallback_timeout_steps_per_skill[self.skill_id[continue_ids]]
                timed_out = (timeout_per_env > 0.0) & (
                    self._pre_kick_fallback_stale_ticks[continue_ids] > timeout_per_env
                )
                self._decline_kick(continue_ids[timed_out])
            elif self._pre_kick_fallback_timeout_steps > 0.0:
                timed_out = self._pre_kick_fallback_stale_ticks[continue_ids] > self._pre_kick_fallback_timeout_steps
                self._decline_kick(continue_ids[timed_out])

    def _post_compute_observations_callback(self):
        return

    def reset_all(self):
        self._init_buffers()
        motion_command = self.command_manager.get_state("motion_command")
        if motion_command is not None:
            motion_command.init_buffers()
        return super().reset_all()

    # ------------------------------------------------------------------
    # Reset dispatch — branches on task_mode (resampled fresh in _reset_tasks_callback, called
    # just before this in BaseTask.reset_envs_idx_impl)

    def _reset_robot_states_callback(self, env_ids, target_states=None):
        # Kick-mode envs: nothing to do here. MotionCommand.reset() (called later in
        # BaseTask.reset_envs_idx via command_manager.reset(), already filtered to exactly this
        # env's kick-mode subset via CommandTermCfg.task_mode="kick") places their robot pose.
        loco_ids = env_ids[self.task_mode[env_ids] == TaskMode.LOCOMOTION]
        if target_states is not None:
            self._reset_dofs(loco_ids, target_states["dof_states"])
            self._reset_root_states(loco_ids, target_states["root_states"])
        else:
            self._reset_dofs(loco_ids)
            self._reset_root_states(loco_ids)

    def _reset_buffers_callback(self, env_ids, target_buf=None):
        self.need_to_refresh_envs[env_ids] = True
        # 2026-08-09 fix (Stage D): derive loco_ids/kick_ids (below) from the PERMANENT
        # _task_mode_partition, not the live self.task_mode -- for an env whose task_mode flipped
        # mid-episode (kick_recovery_locomotion_flip_enabled), self.task_mode reads LOCOMOTION by
        # the time a timeout reset reaches this method, even though the episode was a real kick
        # attempt. Reading live task_mode here would silently drop that episode from
        # _record_kick_episode_ends (kick_topple_frac/kick_episode_length/kick_ball_hit_rate etc.
        # all lose data) AND fold it into the locomotion curriculum tracker below, contaminating
        # average_episode_length with a hybrid kick+locomotion episode -- exactly what that split
        # exists to prevent. _task_mode_partition never changes post-construction, so it's correct
        # here regardless of whether this episode flipped: unaffected for genuinely-locomotion
        # envs (nothing in v1 ever flips them), and correctly still "kick" for any kick-partitioned
        # env's episode, flipped or not.
        loco_ids = env_ids[self._task_mode_partition[env_ids] == TaskMode.LOCOMOTION]

        if target_buf is not None:
            self.simulator.dof_pos[env_ids] = target_buf["dof_pos"].to(self.simulator.dof_pos.dtype)
            self.simulator.dof_vel[env_ids] = target_buf["dof_vel"].to(self.simulator.dof_vel.dtype)
            self.base_quat[env_ids] = target_buf["base_quat"].to(self.base_quat.dtype)
            self.episode_length_buf[env_ids] = target_buf["episode_length_buf"].to(self.episode_length_buf.dtype)
            self.reset_buf[env_ids] = target_buf["reset_buf"].to(self.reset_buf.dtype)
            self.time_out_buf[env_ids] = target_buf["time_out_buf"].to(self.time_out_buf.dtype)
            self._pending_episode_update_mask[env_ids] = False
            self._pending_episode_lengths[env_ids] = 0
            self._kick_ep_min_height[env_ids] = self._KICK_MIN_HEIGHT_SENTINEL
            self._kick_ep_is_handoff[env_ids] = False
        else:
            # Kick episodes are excluded from the curriculum tracker below, so record them here
            # instead — for LOGGING only. Must run before episode_length_buf is zeroed.
            # (See loco_ids' own comment above -- _task_mode_partition, not live task_mode, for
            # the same reason.)
            kick_ids = env_ids[self._task_mode_partition[env_ids] == TaskMode.KICK]
            if kick_ids.numel() > 0:
                self._record_kick_episode_ends(kick_ids)

            self.episode_length_buf[env_ids] = 0
            self.reset_buf[env_ids] = 1
            # Must run AFTER _record_kick_episode_ends above (which reads the OUTGOING episode's
            # min height and handoff flag) -- starts the new episode's reading fresh.
            self._kick_ep_min_height[env_ids] = self._KICK_MIN_HEIGHT_SENTINEL
            self._kick_ep_is_handoff[env_ids] = False
            # Only feed locomotion-mode episode ends into the curriculum tracker/penalty
            # scheduler — kick episodes end via unrelated termination semantics/durations and
            # would dilute the "robot falls quickly -> reduce penalties" signal otherwise.
            self._pending_episode_update_mask[loco_ids] = True

    def _update_log_dict(self):
        avg = self._get_average_episode_tracker().get_average()
        # NOTE: locomotion-only by construction (see _reset_buffers_callback). Read kick/* below
        # for kick stability — this number says nothing about it.
        self.log_dict["average_episode_length"] = avg.detach().cpu()

        kick_mask = self.task_mode == TaskMode.KICK

        motion_command = self.command_manager.get_state("motion_command")
        if motion_command is not None:
            motion_command.update_metrics()
            self.log_dict.update(motion_command.metrics)

            # The metrics above are averaged over ALL envs, so locomotion envs dominate whenever
            # kick_active_frac is small — a kick-mode tracking blow-up can sit at 0.05 in the
            # headline number while the kick itself is diverging. Re-emit the balance-relevant
            # ones restricted to kick envs.
            if bool(kick_mask.any()):
                for key in ("motion/error_body_pos", "motion/error_body_rot", "motion/error_joint_pos"):
                    val = motion_command.metrics.get(key)
                    if val is not None and val.ndim >= 1 and val.shape[0] == self.num_envs:
                        self.log_dict[f"kick_{key}"] = val[kick_mask].mean().detach().cpu()

                # Per-phase split (2026-07-24, user-requested): kick_motion/* above still mixes
                # swing (authored clip -- single-support/dynamic, kick_recovery_* reward terms
                # OFF) and recovery+hold (synthetic tail, standing-shaped, kick_recovery_* reward
                # terms ON) envs together -- a regression confined to one phase (e.g. swing-phase
                # tracking degrading while hold looks fine, or vice versa) can hide inside a
                # healthy-looking aggregate. in_kicking_phase (renamed from in_swing_phase
                # 2026-07-31, boundary moved to stand_start_idx -- see MotionCommand.setup()) is
                # the SAME per-env signal _kick_recovery_gate (managers/reward/terms/locomotion.py)
                # uses to time-gate the standing-shaping reward -- reused here purely for
                # observability, doesn't change any reward. wandb key names (kick_*_swing/
                # kick_*_hold) are left as-is for dashboard continuity despite the rename.
                in_swing = motion_command.in_kicking_phase
                swing_mask = kick_mask & in_swing
                hold_mask = kick_mask & ~in_swing

                # 2026-08-24 (user-requested): SPLIT "_swing" INTO ITS TWO REAL HALVES. Read the
                # paragraph above carefully -- `in_kicking_phase` is modes 1+2, i.e. the locomotion
                # APPROACH *plus* the strike. So `kick_*_swing` has never been a strike-phase
                # measurement: measured against this project's own clip boundaries the strike is
                # only 21% / 22% / 30% of that window for skill011 / 012 / 013 -- the other 70-79%
                # is the approach WALK. Anyone reading `error_body_pos_swing` as "how far the leg
                # swing diverged from the clip" (we did) is reading a number dominated by walking.
                #
                # `in_strike_phase` (modes 2 only, bounded by strike_start_idx..stand_start_idx --
                # the SAME signal shooting.py gates all 6 shooting reward terms on) is the actual
                # leg-swing window. Emitting both halves separately makes a strike-phase-only
                # tracking regression visible instead of diluted ~4:1 by the approach.
                #
                # Purely observability, exactly like the _swing/_hold split above: reads an
                # existing per-env property, writes only log_dict, touches no reward or
                # termination. `_swing`/`_hold` keys are left emitting unchanged for dashboard
                # continuity -- `_strike` + `_approach` partition `_swing`, they do not replace it.
                in_strike = motion_command.in_strike_phase
                strike_mask = swing_mask & in_strike
                approach_mask = swing_mask & ~in_strike

                for key in ("motion/error_body_pos", "motion/error_body_rot", "motion/error_joint_pos"):
                    val = motion_command.metrics.get(key)
                    if val is None or val.ndim < 1 or val.shape[0] != self.num_envs:
                        continue
                    if bool(swing_mask.any()):
                        self.log_dict[f"kick_{key}_swing"] = val[swing_mask].mean().detach().cpu()
                    if bool(hold_mask.any()):
                        self.log_dict[f"kick_{key}_hold"] = val[hold_mask].mean().detach().cpu()
                    if bool(strike_mask.any()):
                        self.log_dict[f"kick_{key}_strike"] = val[strike_mask].mean().detach().cpu()
                    if bool(approach_mask.any()):
                        self.log_dict[f"kick_{key}_approach"] = val[approach_mask].mean().detach().cpu()

                # Share of kick envs currently mid-strike -- the denominator for reading the
                # _strike metrics above (a small share means they are a noisier estimate), and the
                # direct check that the strike window is being entered at all.
                self.log_dict["kick_strike_active_frac"] = strike_mask.float().mean().detach().cpu()

                # 2026-08-14 (user-requested): is the locomotion->kick handoff mechanism
                # (D1-D8) actually learnable, and can we see it improving? This is the direct
                # answer -- live tracking error for envs CURRENTLY mid a handoff-triggered kick
                # (self._kick_ep_is_handoff, set in _enter_kick), same live-per-tick-mean
                # methodology as kick_*_swing/kick_*_hold above (not EMA-over-ended-episodes --
                # this is meant to read as "how well is the policy tracking the clip right now,
                # given it was just spliced in from an arbitrary locomotion state" and a live
                # snapshot answers that more directly). If this trends down over training while
                # kick_{key} (the pooled, mostly reset-triggered aggregate) stays flat, the
                # handoff-specific skill is the one improving.
                handoff_active_mask = kick_mask & self._kick_ep_is_handoff
                if bool(handoff_active_mask.any()):
                    for key in ("motion/error_body_pos", "motion/error_body_rot", "motion/error_joint_pos"):
                        val = motion_command.metrics.get(key)
                        if val is None or val.ndim < 1 or val.shape[0] != self.num_envs:
                            continue
                        self.log_dict[f"kick_{key}_handoff"] = val[handoff_active_mask].mean().detach().cpu()

        kick_eligible = self.terrain_manager.get_state("locomotion_terrain").env_terrain_is_flat
        self.log_dict["kick_eligible_frac"] = kick_eligible.float().mean().detach().cpu()
        self.log_dict["kick_active_frac"] = kick_mask.float().mean().detach().cpu()
        # Share of ALL envs currently mid a handoff-triggered kick -- the "how much is this
        # mechanism actually firing" sanity check (mid_episode_kick_entry_prob's realized rate),
        # same denominator convention as kick_active_frac above.
        self.log_dict["kick_handoff_active_frac"] = (kick_mask & self._kick_ep_is_handoff).float().mean().detach().cpu()

        # Kick stability, the signal that was previously missing entirely.
        # kick/episode_length     — mean length of ended KICK episodes (vs the locomotion-only
        #                           average_episode_length above). A healthy kick rides the episode
        #                           cap; the degraded Stage-C policies died ~65 ticks post-trigger.
        # kick/early_term_frac    — fraction of kick episodes ending by TERMINATION not timeout.
        #                           This is the one to watch: it goes to 1.0 as the kick collapses.
        self.log_dict["kick_episode_length"] = self._kick_stat(self._kick_ep_len_ema).detach().cpu()
        early_term_frac = self._kick_stat(self._kick_ep_term_ema)
        self.log_dict["kick_early_term_frac"] = early_term_frac.detach().cpu()
        # 2026-07-28 (user-requested): aliveness/shooting evaluation, in directly-readable form --
        # a survival rate and a 0-1 hit/success rate rather than requiring "1 - early_term_frac"
        # mental math, and the RAW ball speed (m/s) actually reached, not the reward-shaped
        # Lorentzian proxy already under Episode/RawEpisode's kick_ball_velocity. All four share
        # the same rolling-EMA-over-ended-episodes methodology as kick_episode_length/
        # kick_early_term_frac above (not a live per-tick snapshot, which would be biased low
        # early in a still-in-progress attempt).
        self.log_dict["kick_alive_frac"] = (1.0 - early_term_frac).detach().cpu()
        self.log_dict["kick_ball_hit_rate"] = self._kick_stat(self._kick_ep_hit_ema).detach().cpu()
        self.log_dict["kick_ball_success_rate"] = self._kick_stat(self._kick_ep_success_ema).detach().cpu()
        self.log_dict["kick_ball_velocity"] = (
            self._kick_stat(self._kick_ep_ball_speed_ema, self._kick_ep_ball_speed_decay_pow).detach().cpu()
        )
        # 2026-07-28 (user-requested, strict topple-only signal): kick_alive_frac conflates two
        # different termination causes (bad_tracking, a reference-motion error that can fire while
        # still standing, vs kick_low_height, an actual fall) -- live measurement on this project's
        # own checkpoints found bad_tracking is the MAJORITY cause (~60-66%), so 1-kick_alive_frac
        # meaningfully overstates how often the robot is actually falling down. kick_topple_frac is
        # the raw physical fact instead: did base height ever cross _KICK_FALL_HEIGHT_THRESHOLD
        # during the episode, independent of whether any termination fired at all. See memory
        # kick_alive_frac_plateau_was_new_skill_introduction for the measurement that motivated this.
        self.log_dict["kick_topple_frac"] = self._kick_stat(self._kick_ep_topple_ema).detach().cpu()
        self.log_dict["kick_min_base_height"] = self._kick_stat(self._kick_ep_min_height_ema).detach().cpu()

        # Handoff-only counterparts (2026-08-14, user-requested), same EMA-over-ended-episodes
        # methodology restricted to handoff-triggered kick episodes (see _record_kick_episode_ends'
        # handoff fold-in and _init_kick_episode_stats' comment on why this split exists). Compare
        # directly against kick_alive_frac/kick_episode_length/kick_topple_frac above -- those are
        # dominated by ordinary reset-triggered kicks, so a gap between the two is exactly the
        # "handoff entries are harder than a clean reset" signal, and a shrinking gap over training
        # is the handoff mechanism's own learning-progress curve. Reads 0 (decay_pow still ~0)
        # until the first handoff-triggered kick episode has actually ended -- expect this later in
        # training than the pooled stats, and noisier throughout, since handoff episodes are a
        # deliberate minority (mid_episode_kick_entry_prob).
        handoff_early_term_frac = self._kick_stat(self._kick_ep_term_ema_handoff, self._kick_ep_decay_pow_handoff)
        self.log_dict["kick_handoff_episode_length"] = (
            self._kick_stat(self._kick_ep_len_ema_handoff, self._kick_ep_decay_pow_handoff).detach().cpu()
        )
        self.log_dict["kick_handoff_early_term_frac"] = handoff_early_term_frac.detach().cpu()
        self.log_dict["kick_handoff_alive_frac"] = (1.0 - handoff_early_term_frac).detach().cpu()
        self.log_dict["kick_handoff_topple_frac"] = (
            self._kick_stat(self._kick_ep_topple_ema_handoff, self._kick_ep_decay_pow_handoff).detach().cpu()
        )
        # Per-skill breakdown of the same 4 metrics. This is what actually distinguishes "a
        # newly-introduced skill is still immature" from "the policy is regressing" when the
        # blended kick_alive_frac plateaus -- see memory
        # kick_alive_frac_plateau_was_new_skill_introduction.
        # 2026-08-06 (user-requested): gate is `> 0`, not `> 1` -- with exactly 1 skill, "Kick_
        # skills_0" is numerically a redundant copy of the global Env/kick_* keys above, but it's
        # still emitted so single-skill and multi-skill runs have the same dashboard layout,
        # rather than the section only appearing once a run happens to add a 2nd skill.
        if self._kick_num_skills > 0:
            term_sk = self._kick_stat(self._kick_ep_term_ema_sk, self._kick_ep_decay_pow_sk)
            len_sk = self._kick_stat(self._kick_ep_len_ema_sk, self._kick_ep_decay_pow_sk)
            hit_sk = self._kick_stat(self._kick_ep_hit_ema_sk, self._kick_ep_decay_pow_sk)
            success_sk = self._kick_stat(self._kick_ep_success_ema_sk, self._kick_ep_decay_pow_sk)
            speed_sk = self._kick_stat(self._kick_ep_ball_speed_ema_sk, self._kick_ep_ball_speed_decay_pow_sk)
            topple_sk = self._kick_stat(self._kick_ep_topple_ema_sk, self._kick_ep_decay_pow_sk)
            min_height_sk = self._kick_stat(self._kick_ep_min_height_ema_sk, self._kick_ep_decay_pow_sk)
            # Key shape is "Kick_skills_{i}::metric_name" -- the "::" is a marker
            # logging_utils.py's post_epoch_logging looks for generically (it has no idea what
            # "kick" or "skill" means) to pull a key OUT of the blanket "Env/" bucket and into its
            # own top-level wandb/tensorboard section instead, via the existing extra_log_dicts
            # mechanism. Net effect: "Kick_skills_0" and "Kick_skills_1" become their own
            # collapsible sections, SIBLING to "Env"/"Loss"/etc, each containing every metric for
            # that one skill -- not nested under "Env/" and not scattered across a flat key list.
            for sk in range(self._kick_num_skills):
                section = f"Kick_skills_{sk}"
                self.log_dict[f"{section}::kick_alive_frac"] = (1.0 - term_sk[sk]).detach().cpu()
                self.log_dict[f"{section}::kick_early_term_frac"] = term_sk[sk].detach().cpu()
                self.log_dict[f"{section}::kick_episode_length"] = len_sk[sk].detach().cpu()
                self.log_dict[f"{section}::kick_ball_hit_rate"] = hit_sk[sk].detach().cpu()
                self.log_dict[f"{section}::kick_ball_success_rate"] = success_sk[sk].detach().cpu()
                self.log_dict[f"{section}::kick_ball_velocity"] = speed_sk[sk].detach().cpu()
                self.log_dict[f"{section}::kick_topple_frac"] = topple_sk[sk].detach().cpu()
                self.log_dict[f"{section}::kick_min_base_height"] = min_height_sk[sk].detach().cpu()
        if bool(kick_mask.any()):
            self.log_dict["kick_base_height"] = (
                self.simulator.robot_root_states[kick_mask, 2].mean().detach().cpu()
            )

    ################ Curriculum #################
    # (identical to LeggedRobotLocomotionManager — copied verbatim)

    def _get_average_episode_tracker(self):
        tracker = self.curriculum_manager.get_term("average_episode_tracker")
        if tracker is None:
            raise RuntimeError("AverageEpisodeLengthTracker is not registered with the curriculum manager.")
        return tracker

    @property
    def average_episode_length(self) -> float:
        avg = self._get_average_episode_tracker().get_average()
        return float(avg.detach().cpu().item())

    # ------------------------------------------------------------------
    # Checkpoint helpers

    def get_checkpoint_state(self) -> dict[str, torch.Tensor | float]:
        state: dict[str, torch.Tensor | float] = {}
        state["average_episode_tracker"] = self._get_average_episode_tracker().state_dict()
        if hasattr(self, "reward_penalty_scale"):
            state["reward_penalty_scale"] = float(self.reward_penalty_scale)
        return state

    def load_checkpoint_state(self, state: dict[str, torch.Tensor | float] | None) -> None:
        if not state:
            return

        tracker_state = state.get("average_episode_tracker")
        if tracker_state is not None:
            tracker = self._get_average_episode_tracker()
            tracker.load_state_dict(tracker_state)
            tracker.suppress_next_update()

        penalty_state = state.get("reward_penalty_scale")
        if penalty_state is not None:
            if isinstance(penalty_state, torch.Tensor):
                self.reward_penalty_scale = float(penalty_state.item())
            else:
                self.reward_penalty_scale = float(penalty_state)

    def synchronize_curriculum_state(self, *, device: str, world_size: int) -> None:
        if world_size <= 1:
            return
        if not torch.distributed.is_available() or not torch.distributed.is_initialized():
            return
        tracker = self._get_average_episode_tracker()
        avg_tensor = tracker.get_average().clone().detach().to(device)
        torch.distributed.broadcast(avg_tensor, src=0)
        tracker.set_average(avg_tensor.to(self.device), suppress_update=False)

        if hasattr(self, "reward_penalty_scale"):
            penalty_tensor = torch.tensor(float(self.reward_penalty_scale), device=device, dtype=torch.float)
            torch.distributed.broadcast(penalty_tensor, src=0)
            self.reward_penalty_scale = float(penalty_tensor.item())

    def _push_robots(self, env_ids):
        """Random pushes the robots. Copied verbatim from WholeBodyTrackingManager (6-dim,
        additive, matching BeyondMimic/IsaacLab's push_by_setting_velocity). Not task_mode-gated:
        a push is domain-randomization noise equally valid in both modes."""
        if len(env_ids) == 0:
            return
        self.need_to_refresh_envs[env_ids] = True
        max_vel_tensor = self._max_push_vel
        if self.randomization_manager is not None:
            state = self.randomization_manager.get_state("push_randomizer_state")
            if state is not None:
                max_vel_tensor = state.max_push_vel.clone().to(self.device)

        if not isinstance(max_vel_tensor, torch.Tensor) or max_vel_tensor.numel() != 6:
            raise ValueError("Unified push velocity vector must have exactly 6 components.")

        rand = torch.rand(len(env_ids), 6, device=self.device) * 2 - 1
        self.push_robot_vel_buf[env_ids] = rand * max_vel_tensor.unsqueeze(0)
        self.record_push_robot_vel_buf[env_ids] = self.push_robot_vel_buf[env_ids].clone()
        self.simulator.robot_root_states[env_ids, 7:13] += self.push_robot_vel_buf[env_ids]
        # Push impulses only take effect in the simulator once we write the mutated root state tensor back.
        self.simulator.set_actor_root_state_tensor_robots(env_ids, self.simulator.robot_root_states)
        self._max_push_vel = max_vel_tensor.clone()

    ################ ENV CALLBACKS #################
    # (locomotion-mode DOF/root reset — copied verbatim from LeggedRobotLocomotionManager, only
    # ever called with the locomotion-mode subset of env_ids, see _reset_robot_states_callback)

    def _reset_dofs(self, env_ids, target_state=None):
        """Resets DOF position and velocities of selected environments.
        Positions are randomly selected within 0.5:1.5 x default positions.
        Velocities are set to zero. If target_state is not None, reset to target_state.
        """
        if len(env_ids) == 0:
            return
        if target_state is not None:
            self.simulator.dof_pos[env_ids] = target_state[..., 0]
            self.simulator.dof_vel[env_ids] = target_state[..., 1]
        else:
            self.simulator.dof_pos[env_ids] = self.default_dof_pos[env_ids] * torch_rand_float(
                0.5, 1.5, (len(env_ids), self.num_dof), device=str(self.device)
            )
            self.simulator.dof_vel[env_ids] = 0.0

    def _reset_root_states(self, env_ids, target_root_states=None):
        """Resets ROOT states position and velocities of selected environments.
        If target_root_states is not None, reset to target_root_states.
        """
        if len(env_ids) == 0:
            return
        if target_root_states is not None:
            self.simulator.robot_root_states[env_ids] = target_root_states
            self.simulator.robot_root_states[env_ids, :3] += self.terrain_manager.get_state(
                "locomotion_terrain"
            ).env_origins[env_ids]
        else:
            self.simulator.robot_root_states[env_ids] = self.base_init_state
            self.simulator.robot_root_states[env_ids, :3] += self.terrain_manager.get_state(
                "locomotion_terrain"
            ).env_origins[env_ids]

            if self.terrain_manager.get_state("locomotion_terrain").custom_origins:
                spawn_cfg = self.terrain_manager.cfg.terrain_term.spawn

                xy_offsets = torch_rand_float(
                    -spawn_cfg.xy_offset_range, spawn_cfg.xy_offset_range, (len(env_ids), 2), device=str(self.device)
                )

                if spawn_cfg.query_terrain_height:
                    current_xy = self.simulator.robot_root_states[env_ids, :2]
                    new_xy = current_xy + xy_offsets

                    terrain_state = self.terrain_manager.get_state("locomotion_terrain")
                    terrain_heights = terrain_state.query_terrain_heights(
                        new_xy,
                        use_grid_sampling=spawn_cfg.use_grid_sampling,
                        grid_size=spawn_cfg.grid_size,
                        grid_spacing=spawn_cfg.grid_spacing,
                    )
                    robot_base_height = self.robot_config.init_state.pos[2]
                    new_z = terrain_heights + robot_base_height

                    new_xyz = torch.cat([new_xy, new_z.unsqueeze(1)], dim=1)
                    self.simulator.robot_root_states[env_ids, :3] = new_xyz
                else:
                    self.simulator.robot_root_states[env_ids, :2] += xy_offsets

            self.simulator.robot_root_states[env_ids, 7:13] = torch_rand_float(
                -0.5, 0.5, (len(env_ids), 6), device=str(self.device)
            )  # [7:10]: lin vel, [10:13]: ang vel

    #########################################################################################################
    ## Debug visualization + kinematic replay (copied verbatim from WholeBodyTrackingManager, minus
    ## the IsaacGym and object-tracking branches — Unified is IsaacSim-only and has no object
    ## tracking — so replay.py keeps working unmodified if pointed at this experiment)
    #########################################################################################################

    def _draw_debug_vis_isaacsim(self):
        motion_command = self.command_manager.get_state("motion_command")
        if motion_command is None:
            return
        real_robot_pos_xyz = motion_command.robot_ref_pos_w.clone()
        real_robot_quat_xyzw = motion_command.robot_ref_quat_w.clone()
        real_robot_quat_wxyz = real_robot_quat_xyzw[:, [3, 0, 1, 2]]
        motion_command.visualization_markers["real_robot"].visualize(real_robot_pos_xyz, real_robot_quat_wxyz)

        motion_robot_pos_xyz = motion_command.ref_pos_w.clone()
        motion_robot_quat_xyzw = motion_command.ref_quat_w.clone()
        motion_robot_quat_wxyz = motion_robot_quat_xyzw[:, [3, 0, 1, 2]]
        motion_command.visualization_markers["motion_robot"].visualize(motion_robot_pos_xyz, motion_robot_quat_wxyz)

        for body_idx, body_names in enumerate(motion_command.motion_cfg.body_names_to_track):
            motion_robot_body_pos_xyz = motion_command.body_pos_w[0, body_idx].clone()
            motion_command.visualization_markers[f"motion_{body_names}"].visualize(
                motion_robot_body_pos_xyz.unsqueeze(0)
            )

    def _draw_debug_vis(self):
        if self.simulator.get_simulator_type() == SimulatorType.ISAACSIM:
            self._draw_debug_vis_isaacsim()

    def step_visualize_motion(self, actions):
        motion_command = self.command_manager.get_state("motion_command")
        dt = 1.0 / float(motion_command.motion.fps)
        motion_command.step()
        print("time_steps: ", motion_command.time_steps[0].item())
        self._draw_debug_vis()

        root_pos = motion_command.root_pos_w.clone()
        root_ori = motion_command.root_quat_w.clone()  # wxyz
        root_lin_vel = motion_command.body_lin_vel_w[:, 0].clone()
        root_ang_vel = motion_command.body_ang_vel_w[:, 0].clone()

        joint_pos = motion_command.joint_pos.clone()
        joint_vel = motion_command.joint_vel.clone()

        env_ids = torch.arange(self.num_envs, device=self.device)
        self.simulator.dof_pos[env_ids] = joint_pos
        self.simulator.dof_vel[env_ids] = joint_vel

        self.simulator.robot_root_states[env_ids, :3] = root_pos
        self.simulator.robot_root_states[env_ids, 3:7] = root_ori
        self.simulator.robot_root_states[env_ids, 7:10] = root_lin_vel
        self.simulator.robot_root_states[env_ids, 10:13] = root_ang_vel

        self.simulator.set_actor_root_state_tensor(env_ids, self.simulator.all_root_states)
        self.simulator.set_dof_state_tensor(env_ids, self.simulator.dof_state)

        self.simulator.scene.write_data_to_sim()
        self.simulator.sim.forward()
        self.simulator.sim.render()
        self.simulator.refresh_sim_tensors()

        time.sleep(dt)

        return motion_command.time_steps[0].item() >= motion_command.motion.time_step_total - 2
