from __future__ import annotations

from dataclasses import field
from typing import Any, List, Union

from pydantic.dataclasses import dataclass


@dataclass(frozen=True)
class OptimizerConfig:
    """Configuration for optimizer settings."""

    _target_: str
    """Target optimizer class (e.g., torch.optim.AdamW)."""

    weight_decay: float = 0.001
    """Weight decay parameter for the optimizer."""


@dataclass(frozen=True)
class LayerConfig:
    """Configuration for neural network layer settings."""

    hidden_dims: List[int] = field(default_factory=lambda: [512, 256, 128])
    """List of hidden layer dimensions."""

    activation: str = "ELU"
    """Activation function name."""

    dropout_prob: float = 0.0
    """Dropout probability."""

    use_layer_norm: bool = False
    """Whether to use layer normalization."""

    encoder_activation: str = "ELU"
    """Activation function name for encoder layers."""

    encoder_output_dim: int | None = None
    """Output dimension for encoder. Only used for encoder modules."""

    encoder_hidden_dims: List[int] | None = None
    """Hidden dimensions for encoder. Only used for encoder modules."""

    encoder_input_name: str = ""
    """Input name for encoder. Only used for encoder modules."""

    input_channels: int = 1
    """Number of input channels. Only used for CNN modules."""

    input_height: int = 1
    """Height of input feature maps. Only used for CNN modules."""

    input_width: int = 1
    """Width of input feature maps. Only used for CNN modules."""

    hidden_channels: tuple[int, ...] | None = None
    """Hidden channel dimensions. Only used for CNN modules."""

    kernel_size: int | tuple[int, ...] = 3
    """Kernel size for convolutions. Only used for CNN modules."""

    stride: int | tuple[int, ...] = 1
    """Stride for convolutions. Only used for CNN modules."""

    padding: str | int | tuple[str | int, ...] = "same"
    """Padding mode for convolutions. Only used for CNN modules."""

    module_input_name: tuple[str, ...] = ()
    """Input names for module. Only used for encoder modules."""


@dataclass(frozen=True)
class ModuleConfig:
    """Configuration for neural network modules."""

    type: str
    """Module type (e.g., MLP)."""

    input_dim: List[str] = field(default_factory=list)
    """Input dimension specification."""

    output_dim: List[str | int] = field(default_factory=list)
    """Output dimension specification."""

    layer_config: LayerConfig = field(default_factory=LayerConfig)
    """Layer configuration settings."""

    min_noise_std: float | None = None
    """Minimum noise standard deviation."""

    min_mean_noise_std: float | None = None
    """Minimum mean noise standard deviation."""


@dataclass(frozen=True)
class PPOModuleDictConfig:
    """Configuration for PPO module dictionary."""

    actor: ModuleConfig
    """Actor module configuration."""

    critic: ModuleConfig
    """Critic module configuration."""


@dataclass(frozen=True)
class PPOConfig:
    """Configuration for PPO algorithm."""

    module_dict: PPOModuleDictConfig
    """PPO module configurations (actor, critic)."""

    num_learning_epochs: int = 8
    """Number of learning epochs per update."""

    num_mini_batches: int = 4
    """Number of mini-batches per epoch."""

    clip_param: float = 0.2
    """PPO clipping parameter."""

    gamma: float = 0.99
    """Discount factor for future rewards."""

    lam: float = 0.95
    """GAE lambda parameter."""

    value_loss_coef: float = 1.0
    """Value loss coefficient."""

    entropy_coef: float = 0.01
    """Entropy coefficient for exploration."""

    actor_learning_rate: float = 1e-5
    """Learning rate for actor network."""

    actor_optimizer: OptimizerConfig = field(default_factory=lambda: OptimizerConfig(_target_="torch.optim.AdamW"))
    """Actor optimizer configuration."""

    critic_learning_rate: float = 1e-5
    """Learning rate for critic network."""

    critic_optimizer: OptimizerConfig = field(default_factory=lambda: OptimizerConfig(_target_="torch.optim.AdamW"))
    """Critic optimizer configuration."""

    max_grad_norm: float = 1.0
    """Maximum gradient norm for clipping."""

    schedule: str = "adaptive"
    """Learning rate schedule type."""

    desired_kl: float = 0.01
    """Desired KL divergence for adaptive learning rate."""

    use_symmetry: bool = False
    """Whether to use symmetry in training."""

    symmetry_actor_coef: float = 1.0
    """Symmetry coefficient for actor."""

    symmetry_critic_coef: float = 0.0
    """Symmetry coefficient for critic."""

    num_steps_per_env: int = 24
    """Number of steps per environment."""

    save_interval: int = 100
    """Interval for saving model checkpoints."""

    load_optimizer: bool = True
    """Whether to load optimizer state."""

    init_noise_std: float = 0.8
    """Initial noise standard deviation."""

    num_learning_iterations: int = 1000000
    """Total number of learning iterations."""

    init_at_random_ep_len: bool = True
    """Whether to initialize at random episode length."""

    empirical_normalization: bool = False
    """Whether to apply empirical normalization to actor and critic observations."""

    eval_callbacks: Any = None
    """Evaluation callbacks configuration."""

    max_actor_learning_rate: float | None = None
    min_actor_learning_rate: float | None = None
    max_critic_learning_rate: float | None = None
    min_critic_learning_rate: float | None = None


@dataclass(frozen=True)
class FastSACConfig:
    num_learning_iterations: int = 25000
    """total timesteps of the experiments"""

    critic_learning_rate: float = 3e-4
    """the learning rate of the critic"""

    actor_learning_rate: float = 3e-4
    """the learning rate for the actor"""

    alpha_learning_rate: float = 3e-4
    """the learning rate for the alpha"""

    buffer_size: int = 1024
    """the replay memory buffer size per environment"""

    num_steps: int = 1
    """the number of steps to use for the multi-step return"""

    gamma: float = 0.97
    """the discount factor gamma"""

    tau: float = 0.125
    """target smoothing coefficient (default: 0.005)"""

    batch_size: int = 8192
    """the batch size of sample from the replay memory"""

    learning_starts: int = 10
    """timestep to start learning"""

    skill_replay_weights: List[float] = field(default_factory=list)
    """Per-skill relative weight on each KICK transition's contribution to the critic and actor
    losses, indexed by `env.skill_id` (0..N-1). Empty (default) = exact no-op, byte-identical to
    before this existed. Locomotion transitions always weight 1.0 and are never rescaled.

    WHY THIS EXISTS -- `motion_training_ratio` silently sets gradient share, not just data share.
    `SimpleReplayBuffer` is a PER-ENV ring buffer (`[n_env, buffer_size, ...]`) and `sample()`
    draws exactly `batch_size` transitions from EVERY env, so a skill's share of every gradient
    batch is EXACTLY its share of envs -- forever, independent of buffer depth. A skill dropped to
    `motion_training_ratio: 0.1` therefore receives 10% of every update for the rest of training,
    which is a textbook recipe for catastrophic forgetting of an already-mature skill. This was
    measured live in this project once already (a mature skill cut 0.7->0.2 regressed to ~3x the
    per-env failure rate of the brand-new skill that replaced it).

    Note what this implies about the alternatives: buffer PREFILL cannot fix it (prefilling at
    ratio 0.1 yields a buffer that is 10% that skill -- more data, identical skew, and per-env
    sampling ignores depth anyway), and `critic_warmup_iters` cannot either (it addresses a
    one-time stale-critic transient at a task/reward switch, not a sustained data proportion).

    This field DECOUPLES gradient share from env share: keep the env partition skewed toward the
    skill you are actively training (fresh on-policy-ish data where you want it) while holding the
    mature skill's gradient share high enough to keep rehearsing it.

    Weights are RELATIVE and are renormalized per batch to mean 1.0 over the sampled transitions,
    so the overall gradient magnitude (and hence the effective learning rate) is preserved -- you
    do not need to retune lr after enabling this. To equalize gradient across skills, set each
    weight proportional to 1/ratio_i: e.g. for `motion_training_ratio` 0.1 / 0.8, use
    `[8.0, 1.0]`. To merely SOFTEN the imbalance rather than remove it (usually the better first
    try -- full equalization also slows the skill you are actively training, which still needs the
    majority of the gradient), use something between 1.0 and 8.0, e.g. `[4.0, 1.0]`.

    Length must equal the number of configured motion skills; validated at construction time.
    Requires an env exposing `skill_id` (a UnifiedManager-family env)."""

    critic_warmup_iters: int = 0
    """Freeze the ACTOR for this many steps at the start of THIS PROCESS -- no actor_optimizer
    steps at all -- while the critic (qnet + qnet_target) keeps updating normally from real
    transitions the frozen actor generates by rollout. 0 (default) = disabled, exact previous
    behavior. Counted from the first `learn()` iteration of the CURRENT process, not from the
    checkpoint's absolute `global_step` (which persists across a `--training.checkpoint` resume) --
    see `FastSACAgent.learn()`'s `warmup_end_step` computation for why raw `global_step` would be
    wrong here (a Stage-C resume from a Stage-B checkpoint saved at e.g. step 275000 would already
    satisfy `global_step >= critic_warmup_iters` on its very first step, and warmup would silently
    never run).

    Why this exists (2026-07-22, user directive): resuming into a changed reward (e.g. Stage B -> C
    turning on `shooting_reward_scale`) leaves the critic's Q-estimates calibrated to the OLD
    reward. FastSAC is off-policy with no PPO-style trust region, so the actor faithfully follows
    this stale critic's gradient and can walk a good policy off a cliff before the critic catches
    up -- the exact mechanism behind the Stage B->C "unlearning" transient (see memory
    stagec_obs_normalizer_shock.md and the shooting_reward_scale_ramp_iters docstring below).
    A reward-magnitude ramp only slows how fast the WRONG signal arrives; it does not make the
    critic correct. Freezing the actor lets the critic re-fit the TRUE post-change value function
    (bootstrapped over enough TD updates to actually converge, not just one batch) before the actor
    is ever allowed to act on it -- so when unfrozen, its very first update is already correct
    rather than chasing a moving target.

    Also serves as a replay-buffer prefill: FastSAC's replay buffer is NOT checkpointed (`save`/
    `load` carry the networks and optimizers, never `rb`), so any resume starts from an empty
    buffer regardless of this setting. Since rollout collection and buffer `extend()` happen
    unconditionally every step (this field only skips the ACTOR's gradient step, never the
    env.step()/rb.extend() that happens earlier in the loop), the warmup window doubles as filling
    the buffer with on-distribution transitions before the actor starts training against them.

    Intended to be used INSTEAD OF `BallConfig.shooting_reward_scale_hold_iters`/
    `shooting_reward_scale_ramp_iters` for the reward-introduction transient, not alongside them:
    set the ball config's ramp/hold to 0 (reward jumps to its full target instantly) so the frozen
    actor's rollout is labeled with the TRUE Stage-C reward from step 0 of warmup, and let this
    field alone control how long the actor stays shielded from it. Combining a nonzero ramp/hold
    WITH this field means the critic spends part of warmup re-learning a reward that's about to
    change again once the ramp finishes -- redundant, not wrong, just not the intended usage.

    Size this by watching `qf_loss`/`qf_max` in wandb flatten under the new reward, not by
    guessing a round number -- too short and the actor unfreezes into a critic that hasn't
    converged yet, defeating the purpose."""

    policy_frequency: int = 4
    """the frequency of training policy (delayed)"""

    num_updates: int = 8
    """the number of updates to perform per step"""

    target_entropy_ratio: float = 0.0
    """the ratio of the target entropy to the number of actions"""

    kick_target_entropy_ratio: float | None = None
    """2026-07-28: opt-in per-task-mode entropy target. None (default) = OFF, exactly today's
    behavior everywhere (a single shared log_alpha/target_entropy scalar for the whole policy,
    fit against the pooled entropy of every sample regardless of task) -- every existing config,
    including every current Unified experiment, is unaffected unless this is explicitly set.

    When set (a float), the agent maintains TWO independent alpha parameters instead of one:
    index 0 for locomotion-mode transitions (still driven by target_entropy_ratio above,
    unchanged), index 1 for kick-mode transitions (driven by THIS value instead). Only meaningful
    for an env exposing `task_mode` (e.g. UnifiedManager) -- the agent raises clearly at
    construction time if set on an env without one, rather than silently doing nothing.

    Why this needs to be a genuine opt-in, not just "safe to enable by default with the same
    ratio": splitting one shared alpha into two independently-optimized parameters is NOT
    numerically equivalent to today's single alpha even when both targets are configured
    identically. Today's single alpha converges so the POOLED (locomotion+kick) average entropy
    matches the target; two separate alphas each converge so THEIR OWN group's average entropy
    matches it -- and locomotion/kick have measurably different baseline entropy characteristics
    (see config_values/unified/g1/experiment.py's target_entropy_ratio=0.0 comment), so the two
    will drift to different equilibrium alpha values even under an unchanged nominal target. This
    is precisely the point of the feature (decoupling exploration by task), but it means turning
    it on is a real, deliberate behavior change, not a free / always-safe default.

    Motivated by real telemetry (run 5yq6yh5a): kick motion-tracking error plateaued for 355k
    steps while policy_entropy kept DECREASING, under a single alpha whose target_entropy_ratio=0.0
    was chosen for locomotion (raising it project-wide to WBT's own tuned 0.5 already measurably
    broke locomotion -- see experiment.py). This field lets kick get its own, separately-tuned
    target without touching locomotion's. Also directly closes a gap `deterministic_loss_weight`'s
    own docstring already flags: "a pure entropy-target change can only fix [sample vs mean gap]
    indirectly, by forcing sample=mean via shrinking sigma everywhere, INCLUDING in task modes
    where a wide sigma was fine" -- this is that shared-sigma coupling, made task-mode-aware."""

    kick_gamma: float | None = None
    """2026-07-30: opt-in per-task-mode discount factor, mirroring kick_target_entropy_ratio's own
    design exactly (same opt-in-only rationale, same task_mode requirement, same "not numerically
    equivalent to a single shared value" caveat -- see that field's docstring for the shared
    background). None (default) = OFF, exactly today's behavior: a single `gamma` scalar applied
    to every sample's TD bootstrap target regardless of task mode.

    When set (a float), kick-mode transitions bootstrap with THIS discount instead of `gamma`
    above; locomotion-mode transitions are unaffected. Only meaningful for an env exposing
    `task_mode` -- same construction-time guard as kick_target_entropy_ratio.

    Motivation: kick mode's effective credit-assignment horizon at gamma=0.97 is
    1/(1-0.97)=33 control steps = 0.67s. A kick clip's authored swing runs ~4.6s and its recovery+
    hold tail another ~3s; a phase-resolved fall probe found roughly a third of falls land in that
    recovery/hold tail, well past a 0.67s bootstrap window -- the algorithm cannot propagate credit
    from a fall back to the swing-phase decision that caused it. Raising kick_gamma (e.g. 0.99 ->
    100-step/2.0s horizon) is a reasoned hypothesis targeting this specific mechanism, distinct from
    every reward-WEIGHT change tried against the same symptom (those change how a transition is
    scored; this changes how far its consequences propagate). NOT yet validated by training.

    Implementation note: only `FastSACAgent`'s TD-bootstrap discount
    (`discount = gamma ** effective_n_steps`) is made per-sample-aware. The replay buffer's own
    internal `gamma` (agents/fast_sac/fast_sac_utils.py::SimpleReplayBuffer, used to combine
    `n_steps` consecutive raw rewards before the bootstrap discount is applied) is UNCHANGED and
    stays at the single global `gamma` -- confirmed inert for this project regardless, since every
    experiment here runs `num_steps=1` (`discounts = gamma**arange(1) = [1.0]`, i.e. that inner
    combination never touches more than the single immediate reward). If a future config raises
    `num_steps` above 1 while this field is also set, that inner combination would still use the
    single global `gamma`, a real (currently inapplicable) limitation worth revisiting then."""

    replay_buffer_sanitize_enabled: bool = False
    """2026-08-10: opt-in NaN/Inf guard at the SimpleReplayBuffer write boundary
    (agents/fast_sac/fast_sac_utils.py::SimpleReplayBuffer.extend). False (default) = OFF, exactly
    today's behavior: observations/rewards/next_observations/critic_observations/
    next_critic_observations are written into the circular buffer exactly as received from the env.

    When True, every one of those tensors is passed through `torch.nan_to_num` (NaN -> 0.0,
    +-Inf -> the dtype's own finite max/min) immediately before being written into the buffer, on
    every `extend()` call, unconditionally (cheap -- a single elementwise pass over already-GPU-
    resident tensors).

    Motivation: a rare per-env physics-solver numerical explosion (a contact/collision-resolution
    edge case in one of thousands of parallel envs, not a reward/curriculum/tuning issue) was
    directly observed corrupting a real Stage C run -- `kick_motion/error_joint_pos_swing` spiked
    to 2.36e8 for exactly one logged tick, and `Loss/qf_loss`/`Loss/qf_max`/`Loss/buffer_rewards`
    went to NaN at that SAME step, staying NaN for ~1100 steps before self-recovering. Termination
    terms (e.g. a joint_pos sanity check, see MultiSkillConfig.joint_pos_sanity_check_enabled) run
    in the SAME tick as reward/observation computation, BEFORE any reset -- so a termination alone
    cannot stop that tick's already-corrupted transition from reaching the buffer; only sanitizing
    at the write boundary itself does. This is a last-line-of-defense guard, not a substitute for
    the termination-side check, which still helps by shortening how long a genuinely broken env
    stays in that state. NOT yet validated by a training run -- ships as an opt-in A/B."""

    deterministic_loss_weight: float = 0.0
    """Weight on an additional actor-loss term that directly evaluates -Q(s, deterministic_action)
    -- deterministic_action being the same tanh(mean)*scale+bias the ONNX export and deployment
    actually run (Actor.forward's own `action` output), as opposed to the entropy-regularized
    sampled-action term SAC normally trains on alone. 0.0 (default) fully disables this --
    identical to prior behavior.

    Motivation: SAC's standard actor loss (log_alpha*log_probs - Q(s, sampled_action)) never once
    evaluates the mean action directly -- nothing in the objective stops the mean from landing
    somewhere bad as long as the SAMPLED-action cloud around it still scores well combined with
    the entropy bonus. This is invisible in training/eval that samples actions, but deployment
    always runs the deterministic mean (see policies/*.py's ONNX export), so a real gap between
    "good sampled actions" and "good mean action" can go completely undetected until deployment.
    Confirmed empirically on a unified locomotion+kick checkpoint: stochastic rollout survived a
    dynamic single-support kick almost to completion while the deterministic rollout collapsed
    within ~0.3s, on the exact same checkpoint.

    A pure entropy-target change (target_entropy_ratio) can only fix this indirectly, by forcing
    sample≈mean via shrinking sigma everywhere (including in task modes where a wide sigma was
    fine) -- this term instead directly pushes the mean itself to be Q-good, without requiring
    sigma to collapse at all, so other task modes' entropy/exploration are left alone. Suggested
    starting point when enabling: 1.0 (equal weight to the existing sampled-action Q term)."""

    l2sp_weight: float = 0.0
    """L2-SP ("L2 toward Starting Point", Xuhong et al. 2018) continual-learning regularizer for
    the ACTOR: after every actor optimizer step, pull each actor parameter a little way back
    toward the value it had in the checkpoint this run resumed from. 0.0 (default) fully disables
    it -- no anchor is ever captured, no extra work runs, bit-identical to prior behavior.

    Motivation (2026-08-15): adding motion skill N+1 on top of a checkpoint that already knows
    skills 1..N trains ONE shared Actor/Critic (there are no per-skill heads) against a UNIFORMLY
    sampled shared replay buffer, so every gradient step moves the weights that encode the older
    skills. Lowering an old skill's motion_training_ratio does NOT protect it -- that knob only
    sets what fraction of envs (hence of the buffer) the skill occupies, so lowering it means LESS
    rehearsal and FASTER forgetting. Retention has to come from the parameter side instead, which
    is what this does.

    Deliberately DECOUPLED (applied directly to the parameters after the optimizer step, exactly
    like AdamW's own weight_decay) rather than added to the actor loss. A loss-side L2 penalty is
    rescaled per-parameter by Adam's second-moment normalization -- the very coupling AdamW exists
    to remove -- so its effective strength would neither be comparable to `weight_decay` nor stay
    stable as gradient statistics drift. In this decoupled form the update is literally weight
    decay toward theta_anchor instead of toward 0, so `weight_decay` (0.001) is a directly
    meaningful reference scale for choosing a value.

    Values to try (see also MultiSkillConfig.l2sp_weight):
      0.0   -- OFF (default). Use this whenever you are training skills jointly from scratch;
               forgetting is a problem created by SEQUENTIAL training, and joint training over all
               skills at once avoids it outright without any of this machinery.
      0.001 -- equal to `weight_decay` above. Almost certainly too weak to change anything;
               useful mainly as a sanity/A-B rung.
      0.01  -- 10x weight_decay. START HERE when you do need it.
      0.1   -- 100x weight_decay; roughly the ratio the L2-SP paper used relative to its own decay.
               Treat as the practical upper end -- if this still shows no retention benefit, the
               problem is not the anchor strength and raising it further will not help.
    Tune it against the per-skill EMAs already logged under `Kick_skills_{i}` (see
    UnifiedManager._kick_ep_success_ema_sk and friends): an older skill's success/topple EMA is
    the retention signal, the new skill's learning curve is the plasticity cost. Pick the LARGEST
    value that has not yet visibly slowed the new skill.

    Requires resuming from a checkpoint: the anchor is captured in `load()`, so enabling this on a
    fresh run (randomly initialized actor, no "starting point" to return to) is a configuration
    error and raises at the start of `learn()` rather than silently regularizing toward noise."""

    num_atoms: int = 101
    """the number of atoms"""

    v_min: float = -20.0
    """the minimum value of the support"""

    v_max: float = 20.0
    """the maximum value of the support"""

    critic_hidden_dim: int = 768
    """the hidden dimension of the critic network"""

    actor_hidden_dim: int = 512
    """the hidden dimension of the actor network"""

    use_symmetry: bool = False
    """whether to use symmetry"""

    alpha_init: float = 0.001
    """the initial value of the alpha"""

    use_autotune: bool = True
    """whether to use autotune for the alpha"""

    use_tanh: bool = True
    """whether to use tanh for the action"""

    export_expected_action: bool = True
    """ONNX export only (no effect on training): export the policy's EXPECTED squashed action
    E[tanh(mu + sigma*eps)] (computed with fixed 8-node Gauss-Hermite quadrature -- fully
    deterministic, no sampling) instead of tanh(mu) as the deployed action.

    Why: tanh is nonlinear, so tanh(mu) != E[tanh(mu + sigma*eps)] whenever sigma isn't tiny --
    and E[tanh(...)] is the effective average action the robot actually experienced throughout
    training, i.e. the behavior the critic and dynamics co-adapted to. Deploying tanh(mu) runs a
    systematically different (more saturated) action that training never evaluated anywhere.
    Confirmed empirically on a unified locomotion+kick checkpoint (sigma ~0.5 pre-tanh at
    convergence): the tanh(mu) deployment collapsed a dynamic single-support kick within ~0.4s,
    100% of episodes, while the SAME checkpoint deployed with the GH expected action survived
    295/300 steps with 88% ball contact -- and standing/locomotion metrics were identical between
    the two rules (the correction is only material where sigma is large; as sigma -> 0 the GH sum
    converges to tanh(mu) exactly, so well-annealed policies are unaffected).

    Kept as a flag (rather than unconditional) only so the previous export behavior remains
    reproducible; there is no known reason to turn it off for deployment."""

    log_std_max: float = 0.0
    """the maximum value of the log std"""

    log_std_min: float = -5.0
    """the minimum value of the log std"""

    compile: bool = True
    """whether to use torch.compile."""

    obs_normalization: bool = True
    """whether to enable observation normalization"""

    use_layer_norm: bool = True
    """whether to use layer normalization"""

    num_q_networks: int = 2
    """number of Q-networks to ensemble"""

    max_grad_norm: float = 0.0
    """the maximum gradient norm"""

    amp: bool = True
    """whether to use amp"""

    amp_dtype: str = "bf16"
    """the dtype of the amp"""

    weight_decay: float = 0.001
    """the weight decay of the optimizer"""

    save_interval: int = 1000
    """the interval to save the model"""

    mujoco_kick_rollout_every_n_saves: int = 5
    """Record TWO single-env MuJoCo sim2sim rollouts on the same cadence, every N checkpoint
    saves, i.e. when global_step % (save_interval * mujoco_kick_rollout_every_n_saves) == 0:
      - a scripted stand -> kick-trigger -> hold clip, kick-only (no locomotion), logged to wandb
        as "Training rollout - MuJoCo Kick" (see record_mujoco_kick_rollout.py).
      - a scripted forward-walk -> stop-and-stand clip with a physical ball present but never
        kicked (2026-07-21 addition), logged as "Training rollout - MuJoCo Walk" (see
        record_mujoco_locomotion_rollout.py) -- covers the locomotion side and the fast-walk-to-
        stop transition that the kick rollout has no signal on.
    Both are an early-warning signal for the PhysX<->MuJoCo sim2sim gap: robust in IsaacSim/PhysX
    (training) while failing in MuJoCo deployment, invisible in the existing IsaacSim-only
    rollouts. They run as two independent subprocesses/threads/locks (see
    `FastSACAgent._mujoco_rollout_gate_open`) -- one is never gated on the other.

    Set to 0 (or any value <= 0) to disable both entirely. Only ever fires for runs where the
    training env's kick_probability > 0 (Stage B/C) -- Stage A never triggers either regardless of
    this setting.

    Each runs out-of-process (a subprocess in a separate conda env with RoboJuDo installed) from
    its own daemon thread spawned in the checkpoint-save block, serialized cluster-wide via its own
    lock file so at most one rollout of EACH type runs at a time across all concurrent training
    runs on this machine (the kick and walk locks are independent of each other -- see
    record_mujoco_locomotion_rollout.py's module docstring for why)."""

    mujoco_kick_rollout_with_ball: bool = True
    """Whether the MuJoCo kick rollout above spawns a real, physically-simulated ball and feeds
    kick_ball_pos_b/kick_target_pos_b observations (2026-07-17 default). As of 2026-07-21, Stage B
    also trains with a real, live (if unrandomized) ball observation (managers/observation/terms/
    unified.py::ball_pos_b) -- so True is now correct for BOTH Stage B and Stage C checkpoints;
    leave this at the default for any current run. Set False only to reproduce the original
    no-ball, zero-observation rollout for a checkpoint trained BEFORE 2026-07-21 under the old
    zero-obs-in-Stage-B scheme, where feeding a real ball here would be out-of-distribution noise."""

    mujoco_survival_scan_every_n_saves: int = 0
    """2026-08-19: N-trial in-distribution MuJoCo sim2sim FALL RATE scan (mujoco_kick_survival_scan.py
    via record_mujoco_survival_scan.py), on the same `global_step % (save_interval * this) == 0`
    cadence as mujoco_kick_rollout_every_n_saves (see FastSACAgent._mujoco_rollout_gate_open) but
    its OWN independent knob/thread/lock -- deliberately not folded into that shared cadence, since
    unlike the fixed 4-rollout video bundle (which the user explicitly asked to keep bundled), this
    produces a wandb SCALAR ("sim2sim/kick_fall_rate"), a different kind of signal a user may want
    on a different cadence (cheaper single trials frequently, or this less-frequently since N
    trials costs ~N x one video rollout's wall-clock).

    WHY THIS EXISTS: Env/kick_topple_frac (the IsaacSim training-time metric) conflates three
    distinct sources of movement -- (1) small-sample EMA noise (each fold is ~1-2 ended episodes,
    weighted equally regardless of batch size; measured sd falls ~1/sqrt(averaging-window), the
    signature of pure sampling noise, not real drift), (2) SAC exploration-noise-driven behavioral
    variance (the IsaacSim rollout uses actor.explore(deterministic=False), not the deployed
    E[tanh] action every checkpoint actually ships with), and (3) termination-censoring (an
    unrelated termination -- e.g. bad_tracking, or the old 1.0N contact term before
    contact_termination_force_threshold -- can end an episode before a fall would have had the
    chance to develop, so kick_topple_frac partly measures how often OTHER terminations fire
    first, not just genuine falls; see that field's own docstring for the measured 78.9%-of-
    episodes example). This scan is immune to all three: deployed deterministic action, a fixed
    hold duration regardless of what an IsaacSim-side termination would have done, and an EXACT
    count over N trials rather than a decaying average -- a second, complementary signal, not a
    replacement (a real, independently-measured PhysX<->MuJoCo contact-resolution gap means this
    will not numerically match the IsaacSim rate -- see memory
    stagec-kick-open-loop-physics-proof).

    Each trial jitters the ball spawn UNIFORMLY IN-DISTRIBUTION -- within the SAME +/- half-range
    training itself draws from (BallConfig.position_randomization on the live env's own attached
    experiment config), never an invented or OOD range, so this measures robustness within what
    the policy actually trained on, not a stress test beyond it. The observed TARGET varies the
    same way via kick_aim_theta (BallConfig/MultiSkillConfig.kick_aim_theta_max_deg) for any
    kick_aim_enabled skill -- the old independent target_randomization jitter was removed
    2026-08-22, once every skill in this project had moved to kick_aim_enabled=True.

    0 (default) = off, exact no-op -- FastSACAgent never imports record_mujoco_survival_scan or
    constructs its thread/queue at this default. Runs out-of-process (a subprocess in the separate
    `robojudo` conda env) from its own daemon thread spawned in the checkpoint-save block, same
    architecture as the kick/walk/handoff rollouts above, serialized via its own lock file
    (independent of theirs -- same "own lock per rollout type" rationale) so it never contends with
    them or with itself across concurrent training runs on this machine. Only ever fires for runs
    where the training env's kick_probability > 0, same eligibility gate as the other three."""

    mujoco_survival_scan_num_trials: int = 32
    """Trials per scan when mujoco_survival_scan_every_n_saves > 0 -- see that field's own
    docstring. Each trial costs roughly one settle+hold cycle (~9.5s of sim time at the defaults,
    a few real seconds of subprocess wall-clock -- CPU-only, no GPU: RoboJuDo's ONNX policy runs
    on CPUExecutionProvider and this scan never constructs a renderer, so it never contends with
    the training process's own GPU usage regardless of N), so N trials costs roughly N x that.

    Raised from 8 (2026-08-19): 8 only resolves a rate to ~12.5% steps, coarse enough that two
    checkpoints measured back-to-back (0/8 vs 2/8 on a real run, stageC-skill2-new-fixes-no-
    strike-pitch at steps 253k/263k) could plausibly be the same underlying rate. 32 gives ~3.1%
    steps. Still not independently tuned beyond that -- raise further if a specific comparison
    needs finer resolution than 3.1%."""

    logging_interval: int = 100
    """the interval to log the metrics"""

    encoder_obs_key: str = "perception_obs"
    """the key of the encoder observation. only valid if use_cnn_encoder is True"""

    encoder_obs_shape: tuple[int, int, int] = (1, 13, 9)
    """the shape of the encoder observation. only valid if use_cnn_encoder is True"""

    use_cnn_encoder: bool = False
    """whether to use CNN for the encoder"""

    actor_obs_keys: List[str] = field(default_factory=lambda: ["actor_obs"])
    critic_obs_keys: List[str] = field(default_factory=lambda: ["critic_obs"])

    eval_callbacks: Any = None
    """Evaluation callbacks configuration."""


@dataclass(frozen=True)
class PPOAlgoConfig:
    """Configuration for algorithm wrapper."""

    _target_: str
    """Target algorithm class."""

    _recursive_: bool
    """Whether to recursively instantiate."""

    config: PPOConfig
    """Algorithm-specific configuration."""


@dataclass(frozen=True)
class FastSACAlgoConfig:
    """Configuration for algorithm wrapper."""

    _target_: str
    """Target algorithm class."""

    _recursive_: bool
    """Whether to recursively instantiate."""

    config: FastSACConfig
    """Algorithm-specific configuration."""


AlgoInitConfig = Union[PPOConfig, FastSACConfig]

AlgoConfig = Union[PPOAlgoConfig, FastSACAlgoConfig]
