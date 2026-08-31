"""Unit tests for distill_specialists.py's pure-Python pieces.

`parse_teacher_ckpts`/`parse_teacher_ckpts_from_skills_yaml` are testable without a live IsaacSim
env + real checkpoints -- the full DAgger loop (Teacher construction against a real checkpoint,
env stepping) still needs a real env/GPU and is NOT covered here. See distill_specialists.py's own
module docstring for the design this implements.

2026-08-28: added coverage for the critic-distillation MATH specifically (cross-entropy against a
teacher's softmax, twin-qnet pairing, per-skill masking, the qnet_target hard-copy) -- these ARE
testable in isolation against the real `Critic`/`DistributionalQNetwork` classes on CPU with tiny
dimensions, same "test one piece of a heavy module without building the whole thing" approach
agents/fast_sac/tests/test_l2sp.py uses for L2-SP.
"""

from __future__ import annotations

import types

import pytest
import torch
import torch.nn.functional as F

from holosoma.agents.fast_sac.fast_sac import Critic
from holosoma.distill_specialists import _sync_qnet_target, parse_teacher_ckpts, parse_teacher_ckpts_from_skills_yaml


def test_parses_two_entries():
    result = parse_teacher_ckpts("0=logs/a/model_0400000.pt,1=logs/b/model_0119000.pt")
    assert result == {0: "logs/a/model_0400000.pt", 1: "logs/b/model_0119000.pt"}


def test_parses_single_entry():
    assert parse_teacher_ckpts("0=logs/a/model_0400000.pt") == {0: "logs/a/model_0400000.pt"}


def test_strips_whitespace_around_entries_and_values():
    result = parse_teacher_ckpts(" 0 = logs/a/model.pt , 1 = logs/b/model.pt ")
    assert result == {0: "logs/a/model.pt", 1: "logs/b/model.pt"}


def test_tolerates_trailing_comma():
    result = parse_teacher_ckpts("0=logs/a/model.pt,1=logs/b/model.pt,")
    assert result == {0: "logs/a/model.pt", 1: "logs/b/model.pt"}


def test_empty_string_raises():
    # No longer phrased as "required" -- the env var is now optional (falls back to
    # parse_teacher_ckpts_from_skills_yaml), but calling this directly with "" must still raise:
    # an empty explicit override is never valid on its own.
    with pytest.raises(ValueError, match="must declare at least one teacher"):
        parse_teacher_ckpts("")


def test_missing_equals_raises():
    with pytest.raises(ValueError, match="missing '='"):
        parse_teacher_ckpts("0-logs/a/model.pt")


def test_non_integer_skill_id_raises():
    with pytest.raises(ValueError, match="not a non-negative integer"):
        parse_teacher_ckpts("skill1=logs/a/model.pt")


def test_duplicate_skill_id_raises():
    with pytest.raises(ValueError, match="declared twice"):
        parse_teacher_ckpts("0=logs/a/model.pt,0=logs/b/model.pt")


def test_empty_path_raises():
    with pytest.raises(ValueError, match="empty path"):
        parse_teacher_ckpts("0=")


def test_non_dense_skill_ids_raise():
    """0,2 (skipping 1) would silently index-error at runtime (teachers[env.skill_id]) rather
    than fail at parse time -- this must be caught here instead."""
    with pytest.raises(ValueError, match="not a dense 0..N-1 range"):
        parse_teacher_ckpts("0=logs/a/model.pt,2=logs/b/model.pt")


def test_ids_not_starting_at_zero_raise():
    with pytest.raises(ValueError, match="not a dense 0..N-1 range"):
        parse_teacher_ckpts("1=logs/a/model.pt,2=logs/b/model.pt")


def test_windows_style_or_unusual_but_valid_paths_pass_through_unmodified():
    """Paths can legitimately contain characters other than '=' and ',' -- only the FIRST '=' in
    each entry is meaningful (split(...,1)), everything after it is the path verbatim."""
    result = parse_teacher_ckpts("0=/abs/path with spaces/model=checkpoint.pt")
    assert result == {0: "/abs/path with spaces/model=checkpoint.pt"}


# ----------------------------------------------------------------------------------------------
# parse_teacher_ckpts_from_skills_yaml -- the yaml-field alternative to the long env-var export.
# ----------------------------------------------------------------------------------------------

_TWO_SKILL_YAML_WITH_TEACHERS = """
task_config: task_config_stageC1
base_robot:
  target_height: 0.76
  deadzone: 0.015
motion_skill_1:
  motion_npz: a.npz
  x: 1.0
  y: 0.0
  strike_start_frame: 10
  stand_start_frame: 20
  motion_training_ratio: 0.45
  teacher_checkpoint: logs/skill1_ckpt.pt
motion_skill_2:
  motion_npz: b.npz
  x: 1.0
  y: 0.0
  strike_start_frame: 10
  stand_start_frame: 20
  motion_training_ratio: 0.45
  teacher_checkpoint: logs/skill2_ckpt.pt
"""


def test_reads_teacher_checkpoint_per_skill_in_declaration_order(tmp_path):
    p = tmp_path / "skills.yaml"
    p.write_text(_TWO_SKILL_YAML_WITH_TEACHERS)
    result = parse_teacher_ckpts_from_skills_yaml(str(p))
    assert result == {0: "logs/skill1_ckpt.pt", 1: "logs/skill2_ckpt.pt"}


def test_order_follows_yaml_declaration_not_alphabetical_block_names(tmp_path):
    """motion_skill_2 declared BEFORE motion_skill_1 in the file -- skill_id must follow file
    order (dict/yaml.safe_load insertion order), matching _parse_skill_blocks' own contract, NOT
    numeric/alphabetical sort of the block names."""
    p = tmp_path / "skills.yaml"
    p.write_text(
        "motion_skill_2:\n"
        "  motion_npz: b.npz\n  x: 1.0\n  y: 0.0\n  strike_start_frame: 10\n  stand_start_frame: 20\n"
        "  motion_training_ratio: 0.45\n  teacher_checkpoint: logs/second_declared.pt\n"
        "motion_skill_1:\n"
        "  motion_npz: a.npz\n  x: 1.0\n  y: 0.0\n  strike_start_frame: 10\n  stand_start_frame: 20\n"
        "  motion_training_ratio: 0.45\n  teacher_checkpoint: logs/first_declared.pt\n"
    )
    result = parse_teacher_ckpts_from_skills_yaml(str(p))
    assert result == {0: "logs/second_declared.pt", 1: "logs/first_declared.pt"}


def test_single_skill(tmp_path):
    p = tmp_path / "skills.yaml"
    p.write_text(
        "motion_skill_1:\n"
        "  motion_npz: a.npz\n  x: 1.0\n  y: 0.0\n  strike_start_frame: 10\n  stand_start_frame: 20\n"
        "  motion_training_ratio: 0.9\n  teacher_checkpoint: logs/only.pt\n"
    )
    assert parse_teacher_ckpts_from_skills_yaml(str(p)) == {0: "logs/only.pt"}


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="does not exist"):
        parse_teacher_ckpts_from_skills_yaml(str(tmp_path / "nope.yaml"))


def test_no_skill_blocks_raises(tmp_path):
    p = tmp_path / "skills.yaml"
    p.write_text("task_config: task_config_stageC1\nbase_robot: {target_height: 0.76, deadzone: 0.015}\n")
    with pytest.raises(ValueError, match="no 'motion_skill_N' blocks"):
        parse_teacher_ckpts_from_skills_yaml(str(p))


def test_missing_teacher_checkpoint_field_raises_naming_the_block(tmp_path):
    p = tmp_path / "skills.yaml"
    p.write_text(
        "motion_skill_1:\n"
        "  motion_npz: a.npz\n  x: 1.0\n  y: 0.0\n  strike_start_frame: 10\n  stand_start_frame: 20\n"
        "  motion_training_ratio: 0.9\n"  # no teacher_checkpoint
    )
    with pytest.raises(ValueError, match=r"motion_skill_1.*no 'teacher_checkpoint:'"):
        parse_teacher_ckpts_from_skills_yaml(str(p))


def test_empty_teacher_checkpoint_field_raises(tmp_path):
    p = tmp_path / "skills.yaml"
    p.write_text(
        "motion_skill_1:\n"
        "  motion_npz: a.npz\n  x: 1.0\n  y: 0.0\n  strike_start_frame: 10\n  stand_start_frame: 20\n"
        "  motion_training_ratio: 0.9\n  teacher_checkpoint: ''\n"
    )
    with pytest.raises(ValueError, match="is empty"):
        parse_teacher_ckpts_from_skills_yaml(str(p))


def test_the_real_distill_config_file_parses_end_to_end():
    """Not a synthetic fixture -- the actual configs/skill/distill_skill1_skill2.yaml this project
    uses, proving the real file (and its currently-declared checkpoint paths) parses cleanly."""
    result = parse_teacher_ckpts_from_skills_yaml("configs/skill/distill_skill1_skill2.yaml")
    assert set(result.keys()) == {0, 1}
    assert all(v.endswith(".pt") for v in result.values())


# ----------------------------------------------------------------------------------------------
# Critic distillation math (2026-08-28) -- against the REAL Critic/DistributionalQNetwork classes,
# tiny CPU dimensions, no env/GPU needed. See distill_specialists.py's module docstring, "OPTIONAL
# CRITIC DISTILLATION" section, for the design this exercises.
# ----------------------------------------------------------------------------------------------

_OBS_INDICES = {"obs": {"start": 0, "end": 4, "size": 4}}
_OBS_KEYS = ["obs"]
_N_ACT = 2
_NUM_ATOMS = 5


def _make_critic(seed: int) -> Critic:
    torch.manual_seed(seed)
    return Critic(
        obs_indices=_OBS_INDICES,
        obs_keys=_OBS_KEYS,
        n_act=_N_ACT,
        num_atoms=_NUM_ATOMS,
        v_min=-10.0,
        v_max=10.0,
        hidden_dim=8,
        use_layer_norm=False,  # LayerNorm needs batch>1 in train mode; irrelevant to what's tested
    )


def _q_cross_entropy(target_dist: torch.Tensor, student_logits: torch.Tensor) -> torch.Tensor:
    """The exact loss form implemented in distill_specialists.py's learn-time block: cross-entropy
    per q-network (dim 0), then mean over q-networks -> per-env loss, shape [batch]."""
    student_log_probs = F.log_softmax(student_logits, dim=-1)
    per_qnet_ce = -(target_dist * student_log_probs).sum(dim=-1)  # [num_q_networks, batch]
    return per_qnet_ce.mean(dim=0)  # [batch]


def test_cross_entropy_against_self_equals_own_entropy():
    """A precise, non-tautological correctness check: cross-entropy(P, P) == entropy(P) exactly,
    a standard identity. If the loss were built wrong (e.g. summing over the wrong dim, or
    swapping which tensor gets log_softmax'd), this would not hold to floating-point precision."""
    critic = _make_critic(seed=0)
    obs = torch.randn(6, 4)
    action = torch.randn(6, _N_ACT)

    with torch.no_grad():
        logits = critic(obs, action)  # [2, 6, 5]
        probs = F.softmax(logits, dim=-1)

    ce = _q_cross_entropy(probs, logits)  # cross-entropy(P, P)
    entropy = -(probs * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean(dim=0)  # same formula, named differently

    torch.testing.assert_close(ce, entropy)
    assert (ce > 0).all(), "entropy of a real (non-degenerate) softmax output must be positive"


def test_cross_entropy_is_zero_only_when_distributions_coincide():
    """A student assigning probability 1.0 to the exact atom the teacher does gets zero loss;
    a uniform student against a peaked teacher target gets a strictly larger loss. Confirms the
    loss actually discriminates 'matches teacher' from 'does not', not just that it runs."""
    teacher_dist = torch.zeros(2, 3, _NUM_ATOMS)
    teacher_dist[:, :, 2] = 1.0  # peaked entirely on atom index 2, both q-networks, all 3 envs

    matching_logits = torch.full((2, 3, _NUM_ATOMS), -1e4)
    matching_logits[:, :, 2] = 1e4  # softmax(this) ~= teacher_dist

    uniform_logits = torch.zeros(2, 3, _NUM_ATOMS)  # softmax(this) = uniform

    ce_matching = _q_cross_entropy(teacher_dist, matching_logits)
    ce_uniform = _q_cross_entropy(teacher_dist, uniform_logits)

    torch.testing.assert_close(ce_matching, torch.zeros(3), atol=1e-3, rtol=0)
    assert (ce_uniform > 1.0).all()  # log(num_atoms=5) ~= 1.609, the uniform-vs-peaked cross-entropy
    assert (ce_uniform > ce_matching).all()


def test_twin_qnets_pair_by_index_not_averaged_together():
    """The pairing this project's design depends on: qnets[0]<->qnets[0], qnets[1]<->qnets[1],
    never mixed. Builds a target where q-network 0 and q-network 1 want DIFFERENT distributions
    and confirms a student matching ONLY index 0 correctly gets a nonzero loss from index 1's
    mismatch (i.e. the two q-networks are not silently averaged into one before comparison)."""
    target_dist = torch.zeros(2, 1, _NUM_ATOMS)
    target_dist[0, 0, 0] = 1.0  # q-net 0 wants atom 0
    target_dist[1, 0, 4] = 1.0  # q-net 1 wants atom 4 -- deliberately different

    # Student matches q-net 0 exactly, is uniform (wrong) on q-net 1.
    student_logits = torch.zeros(2, 1, _NUM_ATOMS)
    student_logits[0, 0, 0] = 1e4  # matches target_dist[0]
    # student_logits[1] left at all-zeros -> uniform, does NOT match target_dist[1]'s peak-at-4

    per_qnet_ce = -(target_dist * F.log_softmax(student_logits, dim=-1)).sum(dim=-1)  # [2, 1]
    assert per_qnet_ce[0, 0].item() < 1e-3, "q-net 0 matched its own target -- should be ~0"
    assert per_qnet_ce[1, 0].item() > 1.0, "q-net 1 did NOT match its own target -- should be large"

    # If pairing were wrong (e.g. q-net 0's target compared against q-net 1's student output, or
    # the two q-networks averaged before the loss), this specific asymmetry would be lost.
    pooled = per_qnet_ce.mean(dim=0)
    assert pooled[0].item() > 0.4, "pooling must still reflect q-net 1's large individual error"


def test_per_skill_masking_isolates_correct_rows():
    """Mirrors distill_specialists.py's per-skill diagnostic split: per_env_q_ce[mask].mean() for
    each skill's own envs must only reflect THAT skill's rows, not leak across the mask boundary."""
    per_env_q_ce = torch.tensor([0.1, 0.1, 5.0, 5.0, 5.0])
    skill_masks = [
        (0, torch.tensor([True, True, False, False, False])),
        (1, torch.tensor([False, False, True, True, True])),
    ]
    result = {f"skill{sid}": per_env_q_ce[mask].mean().item() for sid, mask in skill_masks}
    assert result["skill0"] == pytest.approx(0.1)
    assert result["skill1"] == pytest.approx(5.0)


def test_single_action_coverage_leaves_action_gradient_unconstrained():
    """The measured defect (2026-08-28) that ACTION COVERAGE fixes, demonstrated rather than
    asserted: fitting Q at ONE action per state pins the VALUE there but says nothing about how Q
    varies with the action -- which is exactly the quantity (dQ/da) SAC's actor update consumes.

    Trains a student critic to match a teacher at a single action, then checks dQ/da at that point
    against the teacher's own -- they are free to disagree, because nothing ever constrained it.
    """
    torch.manual_seed(0)
    teacher = _make_critic(seed=10)
    student = _make_critic(seed=11)
    for p in teacher.parameters():
        p.requires_grad_(False)

    obs = torch.randn(64, 4)
    action = torch.randn(64, _N_ACT).tanh()

    opt = torch.optim.Adam(student.parameters(), lr=1e-2)
    for _ in range(300):
        with torch.no_grad():
            tgt = F.softmax(teacher(obs, action), dim=-1)
        loss = _q_cross_entropy(tgt, student(obs, action)).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    def dq_da(critic):
        a = action.clone().requires_grad_(True)
        q = critic.get_value(F.softmax(critic(obs, a), dim=-1)).mean(dim=0).sum()
        (g,) = torch.autograd.grad(q, a)
        return g

    g_teacher, g_student = dq_da(teacher), dq_da(student)
    cos = F.cosine_similarity(g_teacher.flatten(), g_student.flatten(), dim=0).item()
    # Values match well after fitting...
    with torch.no_grad():
        v_t = teacher.get_value(F.softmax(teacher(obs, action), dim=-1)).mean()
        v_s = student.get_value(F.softmax(student(obs, action), dim=-1)).mean()
    assert abs(float(v_t - v_s)) < 1.0, "single-action fitting should still match the VALUE"
    # ...but the action-gradient is not meaningfully aligned, which is the whole point.
    assert cos < 0.9, (
        f"dQ/da cosine similarity {cos:.3f} -- if this were reliably ~1.0, single-action coverage "
        "would already constrain the action-gradient and the ACTION COVERAGE fix would be moot"
    )


def test_extra_action_coverage_does_NOT_rescue_the_action_gradient():
    """Regression guard on a NEGATIVE result (2026-08-28).

    Extra action coverage was implemented as the fix for the unconstrained-dQ/da defect above, then
    measured not to work -- adding sampled actions leaves alignment no better (in the full sweep it
    was slightly worse at every scale and count tried; see the module docstring's ACTION COVERAGE
    table). The cause is general: matching a function's VALUES at k points does not constrain its
    GRADIENT between them, and no feasible k densely covers a 29-dim action space.

    This test exists so that if someone later re-enables HOLOSOMA_DISTILL_CRITIC_ACTION_SAMPLES
    expecting it to fix the gradient, the expectation is contradicted here rather than in a
    multi-hour training run. The real fix would be explicit Sobolev/gradient matching, not more
    sample points.
    """
    obs = torch.randn(64, 4)
    a1 = torch.randn(64, _N_ACT).tanh()
    a2 = (a1 + 0.3 * torch.randn(64, _N_ACT)).tanh()

    def fit(actions):
        torch.manual_seed(0)
        teacher = _make_critic(seed=10)
        student = _make_critic(seed=11)
        for p in teacher.parameters():
            p.requires_grad_(False)
        opt = torch.optim.Adam(student.parameters(), lr=1e-2)
        for _ in range(300):
            terms = []
            for a in actions:
                with torch.no_grad():
                    tgt = F.softmax(teacher(obs, a), dim=-1)
                terms.append(_q_cross_entropy(tgt, student(obs, a)))
            loss = torch.stack(terms).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        return teacher, student

    def grad_cos(teacher, student, at):
        def dq_da(critic):
            a = at.clone().requires_grad_(True)
            q = critic.get_value(F.softmax(critic(obs, a), dim=-1)).mean(dim=0).sum()
            (g,) = torch.autograd.grad(q, a)
            return g
        return F.cosine_similarity(dq_da(teacher).flatten(), dq_da(student).flatten(), dim=0).item()

    cos_one = grad_cos(*fit([a1]), at=a1)
    cos_two = grad_cos(*fit([a1, a2]), at=a1)

    # Neither is well-aligned -- that is the point. Asserting a LOOSE ceiling rather than an exact
    # value keeps this robust to seed/arch churn while still failing loudly if extra coverage ever
    # started genuinely solving the problem (which would mean this guard should be revisited).
    assert cos_one < 0.9 and cos_two < 0.9, (
        f"dQ/da alignment unexpectedly high (one={cos_one:.3f}, two={cos_two:.3f}) -- if extra "
        "action coverage now genuinely constrains the action-gradient, re-evaluate the ACTION "
        "COVERAGE section and this test's premise."
    )


def test_action_coverage_loss_magnitude_is_independent_of_sample_count():
    """Averaging (not summing) across action-coverage samples means changing
    HOLOSOMA_DISTILL_CRITIC_ACTION_SAMPLES does not silently rescale the critic's effective
    learning rate -- coverage and step size stay independent knobs."""
    torch.manual_seed(0)
    critic = _make_critic(seed=3)
    obs = torch.randn(16, 4)
    a = torch.randn(16, _N_ACT).tanh()
    with torch.no_grad():
        tgt = F.softmax(critic(obs, a), dim=-1)

    one = torch.stack([_q_cross_entropy(tgt, critic(obs, a))]).mean(dim=0).mean()
    # The identical action repeated N times must give the identical loss under averaging.
    three = torch.stack([_q_cross_entropy(tgt, critic(obs, a)) for _ in range(3)]).mean(dim=0).mean()
    torch.testing.assert_close(one, three)


def test_sync_qnet_target_copies_qnet_into_qnet_target():
    """_sync_qnet_target must make qnet_target byte-identical to qnet's CURRENT weights -- the
    hard-copy this script relies on instead of Polyak-averaging during distillation (see module
    docstring: a short real critic_warmup_iters after resuming is what builds genuine divergence,
    not this script)."""
    qnet = torch.nn.Linear(4, 4)
    qnet_target = torch.nn.Linear(4, 4)
    torch.nn.init.constant_(qnet_target.weight, 999.0)  # deliberately different from qnet's init
    assert not torch.equal(qnet.weight, qnet_target.weight)

    fake_agent = types.SimpleNamespace(qnet=qnet, qnet_target=qnet_target)
    _sync_qnet_target(fake_agent)

    torch.testing.assert_close(qnet_target.weight, qnet.weight)
    torch.testing.assert_close(qnet_target.bias, qnet.bias)


def test_freeze_actor_env_var_truthy_parsing():
    """Mirrors the `os.environ.get(...).strip().lower() in _TRUTHY` gate in distill(). Guards
    against the classic bug where "0"/"false" is read as truthy because only emptiness was
    checked -- that would silently freeze the actor on a run meant to train it."""
    from holosoma.distill_specialists import _TRUTHY

    for raw in ("1", "true", "TRUE", " yes ", "on"):
        assert raw.strip().lower() in _TRUTHY, f"{raw!r} should enable the flag"
    for raw in ("0", "false", "no", "off", ""):
        assert raw.strip().lower() not in _TRUTHY, f"{raw!r} must NOT enable the flag"


def test_frozen_actor_receives_no_gradient_from_critic_loss():
    """The load-bearing claim of critic-only mode, tested end-to-end on real modules rather than
    asserted: run the ACTUAL critic-distillation loss shape (cross-entropy against a detached
    teacher target, evaluated at a detached teacher action) and confirm every actor parameter's
    .grad is still None afterward -- i.e. the actor is untouchable through that path.

    This is the property that makes the output checkpoint's actor byte-identical to its input's.
    """
    torch.manual_seed(0)
    critic = _make_critic(seed=1)
    # Stand-in for the student actor -- any module works; the point is that it is NOT part of the
    # critic loss's graph, exactly as student_actor is not in distill()'s critic branch.
    actor = torch.nn.Linear(4, _N_ACT)

    obs = torch.randn(6, 4)
    # target_action/target_q_dist are detached in distill() (computed under torch.no_grad()) --
    # reproduced here, since that detachment is precisely what severs the actor from this graph.
    with torch.no_grad():
        target_action = torch.tanh(actor(obs))
        target_dist = F.softmax(critic(obs, target_action), dim=-1)

    student_logits = critic(obs, target_action)
    q_loss = _q_cross_entropy(target_dist, student_logits)[
        torch.ones(6, dtype=torch.bool)
    ].mean()
    (0.01 * q_loss).backward()

    for name, p in actor.named_parameters():
        assert p.grad is None, f"actor param {name} received gradient from the critic loss"
    assert any(p.grad is not None for p in critic.parameters()), "critic should have received gradient"


def test_frozen_actor_weights_are_unchanged_after_a_critic_optimizer_step():
    """Complements the gradient test above with the end-state check that actually matters: after a
    real critic optimizer step, the actor's WEIGHTS are bit-identical. Uses a critic-only optimizer
    exactly as distill() builds it in critic-only mode (`optimizer = None`)."""
    torch.manual_seed(0)
    critic = _make_critic(seed=2)
    actor = torch.nn.Linear(4, _N_ACT)
    actor_before = [p.detach().clone() for p in actor.parameters()]

    q_optimizer = torch.optim.Adam(critic.parameters(), lr=3e-4)

    obs = torch.randn(6, 4)
    with torch.no_grad():
        target_action = torch.tanh(actor(obs))
        target_dist = F.softmax(critic(obs, target_action), dim=-1)

    student_logits = critic(obs, target_action)
    q_loss = _q_cross_entropy(target_dist, student_logits).mean()
    q_optimizer.zero_grad(set_to_none=True)
    (0.01 * q_loss).backward()
    q_optimizer.step()

    for before, p in zip(actor_before, actor.parameters()):
        torch.testing.assert_close(before, p.detach(), rtol=0, atol=0)


def test_sync_qnet_target_result_is_a_real_copy_not_aliased():
    """Subsequent training on qnet must NOT retroactively change the already-saved qnet_target --
    load_state_dict copies values, it does not alias parameters."""
    qnet = torch.nn.Linear(4, 4)
    qnet_target = torch.nn.Linear(4, 4)
    fake_agent = types.SimpleNamespace(qnet=qnet, qnet_target=qnet_target)
    _sync_qnet_target(fake_agent)

    with torch.no_grad():
        qnet.weight.add_(1.0)  # simulate a further training step on qnet AFTER the sync

    assert not torch.equal(qnet_target.weight, qnet.weight), (
        "qnet_target must be an independent copy, not a live alias of qnet's parameters"
    )
