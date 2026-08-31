"""Unit tests for the critic-warmup alpha gate (2026-08-28).

THE BUG THIS GUARDS. `critic_warmup_iters` freezes the ACTOR for N steps so the critic can
re-fit before the actor acts on it. But only `_update_pol` was gated on `warmup_done`; the alpha
auto-tune step lives in `_update_main` and ran unconditionally. SAC's alpha controller is an
INTEGRATOR -- it raises alpha whenever policy entropy sits below `target_entropy`, expecting the
actor to respond by becoming more stochastic. A frozen actor structurally cannot respond, so the
error never clears and alpha integrates without bound.

Measured on run 20260827_221312 before the fix: alpha/kick 0.0010 -> 0.0106 -> 1.379 -> 14.38 ->
45.59 -> 142.54 across 5000 warmup steps (142,000x). At unfreeze, `alpha*log_probs - qf_value`
was dominated by the entropy term (actor_loss 11.8 -> -4251 in one logging interval), so the
policy's fastest descent direction became "be maximally random": policy_entropy -23.4 -> +74.9,
kick_topple_frac 0.028 -> 1.0000, no recovery.

These tests exercise the integral-windup dynamics directly on a real `torch.optim.AdamW` over a
real `log_alpha` leaf, using SAC's own alpha loss form, WITHOUT building a FastSACAgent (no env,
no sim, no GPU) -- same isolation approach test_l2sp.py uses for L2-SP. What they do NOT cover:
that `learn()` actually threads `actor_frozen=not warmup_done` into `_update_main` (that wiring is
verified by reading the call site; exercising it needs a live env).
"""

from __future__ import annotations

import math

import torch


def _alpha_loss(log_alpha: torch.Tensor, log_probs: torch.Tensor, target_entropy: float) -> torch.Tensor:
    """SAC's alpha loss, matching `_update_main`'s single-group branch verbatim:
    `(-log_alpha.exp() * (next_state_log_probs.detach() + target_entropy)).mean()`."""
    return (-log_alpha.exp() * (log_probs.detach() + target_entropy)).mean()


# The alpha step runs once per PREPARED BATCH, i.e. `num_updates` times per global step -- not
# once. At this project's num_updates=8, a 5000-step warmup is 40,000 alpha optimizer steps, which
# is what turns a per-step lr of 3e-4 into the ~142,000x growth actually observed
# (exp(40000 * 3e-4) ~= 1.6e5). Getting this factor wrong under-predicts the pathology 8x.
_NUM_UPDATES = 8
_WARMUP_GLOBAL_STEPS = 5000


def _run_alpha(steps: int, *, gated: bool, alpha_init: float = 0.001, lr: float = 3e-4) -> float:
    """Simulate `steps` OPTIMIZER updates with a FROZEN actor -- i.e. entropy pinned, never
    responding. Note `steps` is optimizer steps, so callers must multiply global steps by
    `_NUM_UPDATES` (see above).

    `gated=True` reproduces the fix (skip the whole alpha block while frozen); `gated=False`
    reproduces the pre-fix behavior. Returns the final alpha.
    """
    log_alpha = torch.tensor([math.log(alpha_init)], requires_grad=True)
    opt = torch.optim.AdamW([log_alpha], lr=lr, betas=(0.9, 0.95))

    # A frozen actor's entropy does not move, so the controller's error term is CONSTANT --
    # this is precisely what makes it wind up. -29.0 vs a -23.2 target means "entropy too low,
    # raise alpha", held forever.
    log_probs = torch.full((256,), 29.0)
    target_entropy = -23.2

    for _ in range(steps):
        if gated:
            continue  # the fix: no backward, no step, while the actor is frozen
        opt.zero_grad(set_to_none=True)
        loss = _alpha_loss(log_alpha, log_probs, target_entropy)
        loss.backward()
        opt.step()
    return float(log_alpha.exp().item())


def test_ungated_alpha_winds_up_against_a_frozen_actor():
    """Reproduces the bug: with the actor frozen and the alpha step still running, alpha grows
    without bound. This is the failure the gate exists to prevent."""
    final = _run_alpha(_WARMUP_GLOBAL_STEPS * _NUM_UPDATES, gated=False)
    # The real run reached ~142 from 0.001 (a ~142,000x factor) over exactly this many optimizer
    # steps. Assert the ORDER of the pathology rather than a brittle exact value -- the point is
    # that it is catastrophic, not that it lands on a particular number.
    assert final / 0.001 > 1000.0, (
        f"expected runaway growth from 0.001, got {final:.6f} ({final / 0.001:.0f}x) -- if this no "
        "longer winds up, the alpha loss form or optimizer defaults changed and this guard needs "
        "revisiting"
    )


def test_gated_alpha_is_exactly_unchanged_during_warmup():
    """The fix: while the actor is frozen, alpha must not move AT ALL -- not merely move less."""
    final = _run_alpha(_WARMUP_GLOBAL_STEPS * _NUM_UPDATES, gated=True)
    # exp(log(0.001)) is not bit-exactly 0.001 in float32, so compare with tolerance -- the claim
    # is "the optimizer never touched it", not "float round-trips exactly".
    assert abs(final - 0.001) < 1e-9, f"alpha must be untouched during warmup, got {final!r}"


def test_gate_only_suppresses_during_warmup_not_after():
    """The gate must not disable alpha permanently -- that was the `use_autotune=False` workaround,
    which stops windup but also removes entropy regulation for the run's whole remaining life
    (measured on 20260828_085838: action_std 0.0335 -> 0.157, entropy -23.7 -> -7.4, never
    returning to target). Post-warmup, alpha must be free to move again."""
    log_alpha = torch.tensor([math.log(0.001)], requires_grad=True)
    opt = torch.optim.AdamW([log_alpha], lr=3e-4, betas=(0.9, 0.95))
    log_probs = torch.full((256,), 29.0)

    warmup, after = 1000, 200
    for step in range(warmup + after):
        actor_frozen = step < warmup
        if actor_frozen:
            continue
        opt.zero_grad(set_to_none=True)
        _alpha_loss(log_alpha, log_probs, -23.2).backward()
        opt.step()

    assert float(log_alpha.exp()) > 0.001, "alpha must resume updating once the actor unfreezes"


def test_skipping_the_block_avoids_priming_adam_momentum():
    """Why the fix skips the ENTIRE block rather than zeroing the gradient before stepping.

    Adam carries momentum (exp_avg / exp_avg_sq). Accumulating gradients throughout warmup and
    only suppressing the optimizer step would leave a fully primed optimizer that lurches on the
    first post-warmup update -- trading a gradual windup for an instantaneous one.
    """
    # Variant A (what the fix does): never touch the optimizer while frozen.
    la_a = torch.tensor([math.log(0.001)], requires_grad=True)
    opt_a = torch.optim.AdamW([la_a], lr=3e-4, betas=(0.9, 0.95))
    # Variant B (the tempting shortcut): keep accumulating grads, just don't step.
    la_b = torch.tensor([math.log(0.001)], requires_grad=True)
    opt_b = torch.optim.AdamW([la_b], lr=3e-4, betas=(0.9, 0.95))

    log_probs = torch.full((256,), 29.0)
    for _ in range(500):
        _alpha_loss(la_b, log_probs, -23.2).backward()  # grads pile up, no step, no zero_grad

    # B's accumulated gradient is 500x a single step's -- the concrete hazard. (A's is None: it
    # never ran a backward at all, which is exactly the property the fix guarantees.)
    assert la_a.grad is None, "gated variant must never accumulate an alpha gradient"
    grad_b_accumulated = float(la_b.grad.abs().item())

    # Both now take exactly ONE real update, WITHOUT zeroing first -- i.e. B steps on its
    # accumulated pile, which is what the shortcut would actually do.
    for la, opt in ((la_a, opt_a), (la_b, opt_b)):
        _alpha_loss(la, log_probs, -23.2).backward()
        opt.step()

    grad_a_single = float(la_a.grad.abs().item())
    assert grad_b_accumulated > 100 * grad_a_single, (
        f"accumulated gradient ({grad_b_accumulated:.3e}) should dwarf a single step's "
        f"({grad_a_single:.3e}) -- this is why the fix skips the whole block rather than "
        "suppressing only the optimizer step"
    )
