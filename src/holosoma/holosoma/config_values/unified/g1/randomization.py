"""Unified locomotion + ball-kicking randomization preset for the G1 robot.

Reuses g1_29dof_wbt_randomization as a base (friction/CoM/dof-pos-bias startup randomization),
with the changes below on top.

1. push_randomizer_state's push_interval_s is loosened from WBT's [1.0, 3.0]s (tuned for
   post-kick recovery robustness, where frequent perturbation is the point) to [4.0, 8.0]s —
   closer to stock locomotion's [5.0, 10.0]s.

   Verified empirically: a locomotion-only run (kick_probability=0.0, otherwise identical
   pipeline) still showed the robot moving substantially (~2 m/s) under a (0,0,0) velocity
   command after 38k FastSAC iterations, ruling out kick-task interference as the cause. With
   WBT's push schedule applied uniformly, a 10s locomotion episode gets pushed 3-10 times — the
   policy is essentially always mid-recovery from some perturbation during training and may
   never experience enough uninterrupted quiet time to learn genuinely still standing, as
   opposed to a generally push-tolerant gait that's always subtly "ready to react." Push
   magnitude (max_push_vel) is left unchanged since frequency, not magnitude, is the mechanism
   under test.

2. actuator_randomizer_state's enable_pd_gain and setup_action_delay_buffers' enabled are
   flipped on (WBT leaves both off; stock locomotion's own g1_29dof_randomization — which has
   never shown the MuJoCo-only bug below — has both on).

   Root-caused via direct IsaacSim-vs-MuJoCo comparison on a unified checkpoint: after a walk
   command drops to (0,0,0), IsaacSim settles to a static double-support stand in effectively
   every trial while the *same* ONNX checkpoint through MuJoCo deterministically ends up in a
   static one-legged stance. Since the identical network behaves correctly in the simulator it
   was trained in, this points at a MuJoCo/PhysX contact-and-actuator solver discrepancy the
   policy was never trained to be robust to, rather than a reward-shaping problem.

3. SIM2SIM ROBUSTNESS BROADENING for the Stage-C kick (added 2026-07-15). Root cause established
   by exhaustive investigation: the Stage-C kick is robust in IsaacSim (98% of 128 envs survive a
   triggered kick, base dips to 0.51m at the single-support strike and recovers) but collapses in
   MuJoCo sim2sim -- a pure PhysX->MuJoCo engine gap, NOT any single tunable deploy parameter
   (contact stiffness, foot friction, friction cone, starting pose, PD gains, timestep, and the
   ball observation were each manipulated in a faithful MuJoCo harness and none prevented the
   fall; the stance knee stays bent instead of extending, un-saturated, i.e. the policy's
   feedback controller overfit PhysX's contact/integration response). The fix must be training-
   side robustness. This aligns the unified DR with the PROVEN robust recipe: stock G1 locomotion
   (g1_29dof_randomization, which deploys to MuJoCo without this bug) randomizes two axes the
   WBT-inherited unified base does NOT -- robot MASS (link + base) and a wider base CoM. Those are
   added here. RFI (random torque injection) is additionally enabled at a modest level as a
   kick-specific extra: the kick is a single-support balance task far more delicate than
   locomotion (even IsaacSim loses ~2%), so beyond matching stock loco it also needs the policy to
   stop relying on an exact torque->motion mapping. RFI is the more speculative lever -- if the
   warm-started kick struggles to recover early in the retrain, drop rfi_lim to 0.0 (or set
   enabled=False on setup_torque_rfi) to fall back to exactly the proven stock-loco recipe.
   Friction is left as the base [0.3, 1.2]/[0.3, 1.6] material range (already wide; direct
   manipulation showed foot friction is not the gap). See memory stagec-kick-physx-mujoco-gap.
"""

from dataclasses import replace

from holosoma.config_types.multi_skill import load_multi_skill_config, multi_skill_mode_enabled
from holosoma.config_types.randomization import RandomizationManagerCfg, RandomizationTermCfg
from holosoma.config_types.simulator import load_ball_config
from holosoma.config_values.wbt.g1.randomization import g1_29dof_wbt_randomization

# Body-targeted collision-style push (2026-08-27) -- see BodyPushRandomizerState's own docstring
# (managers/randomization/terms/locomotion.py) for why this is a separate mechanism from
# push_randomizer_state above, not a parameter change to it. Read the SAME way every other
# task-config-driven field in this preset family is read (command.py's identical pattern): task
# config wins when HOLOSOMA_TASK_CONFIG/N-skill mode is active, legacy BallConfig otherwise. Module-
# level, same as command.py/reward.py -- resolved once at import time, not per-env.
_multi_skill_cfg = load_multi_skill_config() if multi_skill_mode_enabled() else None
if _multi_skill_cfg is not None:
    _body_push_enabled = _multi_skill_cfg.body_push_enabled
    _body_push_interval_s = [_multi_skill_cfg.body_push_interval_min_s, _multi_skill_cfg.body_push_interval_max_s]
    _body_push_force_range = [_multi_skill_cfg.body_push_force_min, _multi_skill_cfg.body_push_force_max]
    _body_push_duration_s = [_multi_skill_cfg.body_push_duration_min_s, _multi_skill_cfg.body_push_duration_max_s]
    _body_push_vertical_fraction = _multi_skill_cfg.body_push_vertical_fraction
    _body_push_body_names = _multi_skill_cfg.body_push_body_names
else:
    _legacy_ball_cfg = load_ball_config()
    _body_push_enabled = _legacy_ball_cfg.body_push_enabled
    _body_push_interval_s = [_legacy_ball_cfg.body_push_interval_min_s, _legacy_ball_cfg.body_push_interval_max_s]
    _body_push_force_range = [_legacy_ball_cfg.body_push_force_min, _legacy_ball_cfg.body_push_force_max]
    _body_push_duration_s = [_legacy_ball_cfg.body_push_duration_min_s, _legacy_ball_cfg.body_push_duration_max_s]
    _body_push_vertical_fraction = _legacy_ball_cfg.body_push_vertical_fraction
    _body_push_body_names = _legacy_ball_cfg.body_push_body_names

_body_push_params = {
    "enabled": _body_push_enabled,
    "interval_s": _body_push_interval_s,
    "force_range": _body_push_force_range,
    "duration_s": _body_push_duration_s,
    "vertical_fraction": _body_push_vertical_fraction,
}
if _body_push_body_names is not None:
    _body_push_params["body_names"] = list(_body_push_body_names)

g1_29dof_unified_randomization = RandomizationManagerCfg(
    setup_terms={
        **g1_29dof_wbt_randomization.setup_terms,
        "push_randomizer_state": replace(
            g1_29dof_wbt_randomization.setup_terms["push_randomizer_state"],
            params={
                **g1_29dof_wbt_randomization.setup_terms["push_randomizer_state"].params,
                "push_interval_s": [4.0, 8.0],
            },
        ),
        "actuator_randomizer_state": replace(
            g1_29dof_wbt_randomization.setup_terms["actuator_randomizer_state"],
            params={
                **g1_29dof_wbt_randomization.setup_terms["actuator_randomizer_state"].params,
                "enable_pd_gain": True,
                # RFI per-env limit scaling (base rfi_lim set in setup_torque_rfi below). Range
                # matches stock loco's actuator randomizer; enabled here (loco leaves it off).
                "enable_rfi_lim": True,
                "rfi_lim_range": [0.5, 1.5],
            },
        ),
        "setup_action_delay_buffers": replace(
            g1_29dof_wbt_randomization.setup_terms["setup_action_delay_buffers"],
            params={
                **g1_29dof_wbt_randomization.setup_terms["setup_action_delay_buffers"].params,
                "enabled": True,
            },
        ),
        # (3) Widen base CoM to match stock loco's proven range (WBT/unified base only shifts x by
        # +/-0.025; stock loco uses +/-0.05 on all three axes).
        "randomize_base_com_startup": replace(
            g1_29dof_wbt_randomization.setup_terms["randomize_base_com_startup"],
            params={
                **g1_29dof_wbt_randomization.setup_terms["randomize_base_com_startup"].params,
                "base_com_range": {"x": [-0.05, 0.05], "y": [-0.05, 0.05], "z": [-0.05, 0.05]},
            },
        ),
        # (3) Robot MASS randomization -- the biggest axis stock loco has and unified/WBT lacks.
        # link_mass scales each link 90-120%; base adds -1..+3 kg to the torso. Values copied
        # verbatim from the proven g1_29dof_randomization.
        "mass_randomizer": RandomizationTermCfg(
            func="holosoma.managers.randomization.terms.locomotion:randomize_mass_startup",
            params={
                "enable_link_mass": True,
                "link_mass_range": [0.9, 1.2],
                "enable_base_mass": True,
                "added_mass_range": [-1.0, 3.0],
                "enabled": True,
            },
        ),
        # (3) Enable RFI torque injection (base limit). Modest 0.05 (half stock loco's default
        # 0.1) because the single-support kick is delicate; combined with rfi_lim_range [0.5,1.5]
        # above the effective per-env injection is ~2.5-7.5% of each joint's torque limit.
        "setup_torque_rfi": RandomizationTermCfg(
            func="holosoma.managers.randomization.terms.locomotion:setup_torque_rfi",
            params={
                "enabled": True,
                "rfi_lim": 0.05,
            },
        ),
        # (4) JOINT ARMATURE/FRICTION randomization (added 2026-07-15, IsaacSim-only). Mass/CoM/RFI
        # above did NOT close the MuJoCo sim2sim gap (re-tested on the +35k-step broadDR checkpoint:
        # still 0/9 MuJoCo trials survive the kick, unchanged from the pre-broadDR baseline; see
        # memory stagec-kick-physx-mujoco-gap). None of that DR touches how an individual joint
        # responds to a fast torque command -- armature (rotor inertia) and joint-internal friction
        # do, and are exactly the un-randomized axis the original diagnosis flagged as untried. 20%
        # armature scale range and a modest joint-friction range, matching the scale magnitude
        # already proven safe for link mass.
        "joint_armature_randomizer": RandomizationTermCfg(
            func="holosoma.managers.randomization.terms.locomotion:randomize_joint_armature_startup",
            params={
                "armature_range": [0.8, 1.2],
                "joint_friction_range": [0.8, 1.5],
                "enabled": True,
            },
        ),
        # (5) CONTACT SOLVER randomization (added 2026-07-20, MuJoCo-only; inert on IsaacSim).
        # Every axis above perturbs the ROBOT or the friction coefficient -- none perturbs how
        # contact is RESOLVED, so a MuJoCo-trained policy saw one fixed contact response for its
        # entire training and could overfit it for free. Measured consequence (6 matched
        # checkpoints/arm, n=128): the mjwarp-trained policy topples 26.6% in MuJoCo but 100.0% in
        # IsaacSim, and the IsaacSim-trained policy is the exact mirror (68.4% / 99.2%). Both are
        # engine-specialized, neither is robust. geom_solref = [timeconst, dampratio] is the knob
        # that response hangs off (MuJoCo default [0.02, 1.0]); randomizing it is the direct
        # counter to that overfitting, and is the one axis the long parametric-tuning hunt in
        # stagec-kick-sim2sim-model-audit never varied -- that hunt matched parameters BETWEEN
        # engines but always held them CONSTANT during training.
        "contact_solver_randomizer": RandomizationTermCfg(
            func="holosoma.managers.randomization.terms.locomotion:randomize_contact_solver_startup",
            params={
                "timeconst_range": [0.010, 0.040],
                "dampratio_range": [0.70, 1.40],
                "enabled": True,
            },
        ),
        # (6) BODY-TARGETED COLLISION PUSH (added 2026-08-27). See BodyPushRandomizerState's
        # docstring for the full rationale -- addresses two of the three ways an unmodelled real
        # collision differs from the existing root-velocity push_randomizer_state above (location,
        # duration; NOT the third, physical blocking of the recovery motion, which is out of
        # scope). Params sourced from configs/task/task_config_stageB.yaml's body_push_* fields
        # (or the legacy BallConfig counterparts) -- see this module's own top-of-file resolution
        # block. Ships with enabled=False unless that yaml sets body_push_enabled: true, a
        # verified no-op at every other default (bit-identical to before this term existed).
        "body_push_randomizer_state": RandomizationTermCfg(
            func="holosoma.managers.randomization.terms.locomotion:BodyPushRandomizerState",
            params=_body_push_params,
        ),
    },
    reset_terms={
        **g1_29dof_wbt_randomization.reset_terms,
        # Re-assert the RFI injection flag on reset (mirrors stock loco's reset_terms).
        "configure_torque_rfi": RandomizationTermCfg(
            func="holosoma.managers.randomization.terms.locomotion:configure_torque_rfi",
        ),
        # Same class instance as the setup_terms registration above -- this just tags it into the
        # "reset" lifecycle stage too (RandomizationManager dedupes by name, see
        # _register_class_term); params are read once, at the setup_terms registration.
        "body_push_randomizer_state": RandomizationTermCfg(
            func="holosoma.managers.randomization.terms.locomotion:BodyPushRandomizerState",
        ),
        # Per-episode constant bias on kick_ball_pos_b (heading frame). No-op unless
        # BallConfig.observation_bias > 0 (configs/ball*.yaml); see that field's docstring and
        # managers/observation/terms/unified.py::ball_pos_b. Magnitude read from the ball config, so
        # no params here -- toggled entirely by the yaml, keeping every existing run bit-identical.
        "ball_obs_bias_randomizer": RandomizationTermCfg(
            func="holosoma.managers.randomization.terms.locomotion:randomize_ball_obs_bias",
        ),
        # Per-episode frozen/static kick_ball_pos_b (2026-07-24 deployment-robustness training --
        # stuck/dead perception pipeline). No-op unless ball_static_obs_probability > 0
        # (configs/stageB_and_C.yaml's top-level key, or configs/ball*.yaml's
        # ball_static_obs_probability in legacy mode); see that field's docstring and
        # managers/observation/terms/unified.py::ball_pos_b.
        "ball_obs_freeze_randomizer": RandomizationTermCfg(
            func="holosoma.managers.randomization.terms.locomotion:randomize_ball_obs_freeze",
        ),
    },
    step_terms={
        **g1_29dof_wbt_randomization.step_terms,
        # Same instance as the setup_terms registration -- tags the "step" lifecycle stage
        # (increments the schedule counter; see BodyPushRandomizerState.step).
        "body_push_randomizer_state": RandomizationTermCfg(
            func="holosoma.managers.randomization.terms.locomotion:BodyPushRandomizerState",
        ),
        # The actual force application: checks the schedule, samples/ticks/clears disturbances,
        # and writes to the simulator. Registration ORDER matters here, same as
        # push_randomizer_state/apply_pushes above -- dict insertion order is iteration order, and
        # this must run AFTER the state's own step() so the just-incremented counter is what
        # apply_body_pushes compares against the resampled interval.
        "apply_body_pushes": RandomizationTermCfg(
            func="holosoma.managers.randomization.terms.locomotion:apply_body_pushes",
        ),
    },
)

__all__ = ["g1_29dof_unified_randomization"]
