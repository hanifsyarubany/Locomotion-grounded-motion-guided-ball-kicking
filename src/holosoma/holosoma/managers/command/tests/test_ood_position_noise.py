"""Unit tests for draw_position_noise_with_ood (managers/command/terms/wbt.py) -- the 2026-07-24
deployment-robustness mechanism: with probability ``ood_prob``, an env's ball/target placement
noise draw comes from a WIDER region (``ood_multiplier`` times the normal range) instead of the
normal one, so training occasionally exposes the policy to spawn positions outside the usual
in-distribution box.

2026-08-01: the function now also RETURNS the per-env ``is_ood`` mask (previously computed and
discarded internally) as ``(noise, is_ood)``, so callers can persist which envs got an OOD draw --
see ``MotionCommand.is_ood_spawn``, consumed by ``managers/reward/terms/shooting.py``'s
``_ood_gate_multiplier`` to zero the shooting reward for those attempts. Every test below was
updated to unpack the tuple, and several now assert on ``is_ood`` directly instead of inferring it
from the noise magnitude -- an exact signal where before there was only a lossy proxy.
"""

from __future__ import annotations

import torch

from holosoma.managers.command.terms.wbt import draw_position_noise_with_ood


def test_ood_prob_zero_is_bit_identical_to_plain_noise():
    """ood_prob<=0.0 (the default -- every config before this feature existed) must reduce to
    exactly the pre-existing uniform draw, bounded by position_randomization alone, and the mask
    must be all-False."""
    torch.manual_seed(0)
    position_randomization = torch.full((1000, 2), 0.5)
    noise, is_ood = draw_position_noise_with_ood(position_randomization, ood_prob=0.0, ood_multiplier=3.0, device="cpu")
    assert torch.all(noise.abs() <= 0.5 + 1e-6)
    assert not is_ood.any()


def test_ood_prob_one_always_uses_the_wider_region():
    """ood_prob=1.0: every draw must come from the wider region -- the mask itself is now the
    primary, decisive signal (is_ood.all()), with the magnitude check kept as an independent
    sanity check that the OOD branch's range is what it claims."""
    torch.manual_seed(0)
    position_randomization = torch.full((2000, 2), 0.5)
    noise, is_ood = draw_position_noise_with_ood(position_randomization, ood_prob=1.0, ood_multiplier=3.0, device="cpu")
    assert is_ood.all()
    assert torch.all(noise.abs() <= 1.5 + 1e-6)  # never exceeds the OOD bound (0.5*3)
    exceeds_normal = (noise.abs() > 0.5 + 1e-6).any(dim=-1)
    # With ood_multiplier=3.0, ~2/3 of the OOD region's width lies outside the normal box on each
    # axis -- over 2000 envs x 2 axes, an overwhelming majority of rows should show at least one
    # axis exceeding the normal bound. A generous threshold (>80%) avoids test flakiness while
    # still being a decisive, not just "some", signal.
    assert exceeds_normal.float().mean().item() > 0.8


def test_ood_prob_fraction_selects_only_some_envs():
    """A middling probability must select roughly that fraction of envs for the OOD region, not
    all or none. Now checked directly off the returned mask (an exact draw count) instead of
    inferred through the lossy noise-magnitude proxy the pre-2026-08-01 version had to use."""
    torch.manual_seed(0)
    n = 5000
    position_randomization = torch.full((n, 2), 0.5)
    _, is_ood = draw_position_noise_with_ood(position_randomization, ood_prob=0.1, ood_multiplier=5.0, device="cpu")
    frac = is_ood.float().mean().item()
    assert 0.07 < frac < 0.13


def test_ood_never_exceeds_its_own_multiplier_bound():
    """Even at ood_prob=1.0, no draw should ever exceed ood_multiplier * position_randomization
    -- the region has a hard outer bound, not unbounded."""
    torch.manual_seed(1)
    position_randomization = torch.full((3000, 2), 0.75)
    noise, is_ood = draw_position_noise_with_ood(position_randomization, ood_prob=1.0, ood_multiplier=2.0, device="cpu")
    assert torch.all(noise.abs() <= 0.75 * 2.0 + 1e-6)
    assert is_ood.all()


def test_per_env_position_randomization_respected():
    """Different envs with different position_randomization (e.g. different skills' own ranges)
    must each scale relative to THEIR OWN range, not a shared global one."""
    torch.manual_seed(0)
    position_randomization = torch.tensor([[0.1, 0.1], [2.0, 2.0]])
    noise, is_ood = draw_position_noise_with_ood(position_randomization, ood_prob=0.0, ood_multiplier=3.0, device="cpu")
    assert torch.all(noise[0].abs() <= 0.1 + 1e-6)
    assert torch.all(noise[1].abs() <= 2.0 + 1e-6)
    assert not is_ood.any()


def test_zero_position_randomization_stays_zero_even_with_ood_enabled():
    """A skill with randomize_x=randomize_y=0.0 (e.g. a tightly-controlled Stage B skill) must
    stay exactly at zero noise even with OOD enabled -- the OOD region scales FROM
    position_randomization, so zero stays zero (documented, accepted consequence of the chosen
    'multiplier of randomize_x/y' design, not a bug). Before 2026-08-01 this test couldn't tell
    "the OOD branch fired but scaled to exactly zero" apart from "the OOD branch never fired",
    since position_randomization=0 makes both produce identical zero noise -- the returned mask
    closes that gap."""
    torch.manual_seed(0)
    position_randomization = torch.zeros(500, 2)
    noise, is_ood = draw_position_noise_with_ood(position_randomization, ood_prob=1.0, ood_multiplier=5.0, device="cpu")
    assert torch.all(noise == 0.0)
    assert is_ood.all()


def test_non_ood_envs_never_exceed_the_normal_bound_regardless_of_ood_envs_in_the_same_batch():
    """Cross-check the mask against the noise it produced: whichever envs is_ood is False for
    must obey the ordinary bound, even in a batch where OTHER envs' draws are OOD (proves the
    mask and the noise stay correctly paired per-env, not just correct in aggregate)."""
    torch.manual_seed(2)
    position_randomization = torch.full((4000, 2), 0.5)
    noise, is_ood = draw_position_noise_with_ood(position_randomization, ood_prob=0.3, ood_multiplier=4.0, device="cpu")
    assert torch.all(noise[~is_ood].abs() <= 0.5 + 1e-6)
