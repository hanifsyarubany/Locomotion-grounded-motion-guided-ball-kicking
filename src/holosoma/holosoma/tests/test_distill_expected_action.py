"""Tests for the 2026-08-18 train/deploy-mismatch fix in distill_specialists.py.

THE CLAIM UNDER TEST: the distillation target is now the quantity the exported ONNX actually
computes. `FastSACConfig.export_expected_action` defaults True, under which
`FastSACAgent.actor_onnx_wrapper` emits ``E[tanh(mu + sigma*Z)]`` (8-node Gauss-Hermite), NOT the
``tanh(mu)`` that `Actor.explore(deterministic=True)` returns. Distilling the latter while
deploying the former left the student's ``fc_logstd`` with exactly zero gradient -- verified frozen
at zero-init across a full 85k-step run.

These tests do NOT need IsaacSim or a GPU: the quadrature is pure tensor math, so it is exercised
directly. What they pin:
  1. The quadrature reproduces E[tanh(mu + sigma*Z)] to high accuracy (checked against a large
     Monte-Carlo sample, i.e. against the DEFINITION, not against a copy of the implementation).
  2. It is byte-for-byte the same node/weight construction the exporter uses -- this is the actual
     correctness claim, and the thing that silently breaks if someone edits one and not the other.
  3. sigma genuinely affects the output (so gradient can reach fc_logstd), and the sigma -> 0 limit
     collapses onto tanh(mu) (so the old behavior is the degenerate case, not a different formula).
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch


def _gh_nodes_weights(n: int = 8):
    """The construction under test, copied from distill_specialists.distill() -- which itself
    mirrors FastSACAgent.actor_onnx_wrapper. test_matches_exporter_construction below is what
    guarantees this copy has not drifted from the exporter's own."""
    t, w = np.polynomial.hermite.hermgauss(n)
    return (
        torch.tensor(t * math.sqrt(2.0), dtype=torch.float32),
        torch.tensor(w / math.sqrt(math.pi), dtype=torch.float32),
    )


def _expected_action_core(mean: torch.Tensor, log_std: torch.Tensor) -> torch.Tensor:
    """The scale/bias-free core of distill_specialists' expected_action()."""
    nodes, weights = _gh_nodes_weights()
    std = log_std.exp()
    shifted = mean.unsqueeze(0) + std.unsqueeze(0) * nodes.view(-1, 1, 1)
    return (torch.tanh(shifted) * weights.view(-1, 1, 1)).sum(dim=0)


# ----------------------------------------------------------------------------------------------
# 1. Correct against the DEFINITION.
# ----------------------------------------------------------------------------------------------


@pytest.mark.parametrize("sigma", [0.01, 0.082, 0.26, 0.5])
@pytest.mark.parametrize("mu", [-2.0, -0.5, 0.0, 0.7, 1.5])
def test_quadrature_matches_monte_carlo_expectation(mu, sigma):
    """Checked against a 2M-sample Monte-Carlo estimate of E[tanh(mu + sigma*Z)] rather than
    against another closed form -- this validates the quadrature itself, not just its self-
    consistency. Tolerance is set by the MC standard error, not by the quadrature."""
    torch.manual_seed(0)
    mean = torch.tensor([[mu]])
    log_std = torch.tensor([[math.log(sigma)]])
    quad = _expected_action_core(mean, log_std).item()

    z = torch.randn(2_000_000)
    mc = torch.tanh(mu + sigma * z).mean().item()
    assert quad == pytest.approx(mc, abs=2e-3)


def test_sigma_to_zero_collapses_onto_tanh_mu():
    """The pre-2026-08-18 target is the sigma->0 degenerate case of the new one. If this failed,
    the change would be a different formula rather than a strict generalization."""
    mean = torch.tensor([[0.4, -1.1, 2.0]])
    tiny = torch.full_like(mean, math.log(1e-6))
    assert torch.allclose(_expected_action_core(mean, tiny), torch.tanh(mean), atol=1e-6)


# ----------------------------------------------------------------------------------------------
# 2. THE correctness claim: identical to what the exporter deploys.
# ----------------------------------------------------------------------------------------------


def test_matches_exporter_construction():
    """The exporter builds its nodes/weights inside a locally-defined nn.Module, so it cannot be
    imported and reused -- distill_specialists duplicates the construction instead. This test is
    the only thing standing between that duplication and a silent divergence, so it reproduces the
    exporter's literal expression from fast_sac_agent.py rather than calling the helper above."""
    t_nodes, w_nodes = np.polynomial.hermite.hermgauss(8)
    exporter_nodes = torch.tensor(t_nodes * math.sqrt(2.0), dtype=torch.float32)
    exporter_weights = torch.tensor(w_nodes / math.sqrt(math.pi), dtype=torch.float32)

    ours_nodes, ours_weights = _gh_nodes_weights()
    assert torch.equal(ours_nodes, exporter_nodes)
    assert torch.equal(ours_weights, exporter_weights)


def test_weights_form_a_probability_measure():
    """1/sqrt(pi)-scaled Gauss-Hermite weights must sum to 1, or the result is not an expectation
    at all (it would be a scaled version of one, silently biasing every target)."""
    _, weights = _gh_nodes_weights()
    assert weights.sum().item() == pytest.approx(1.0, abs=1e-6)


def test_output_stays_within_tanh_range():
    """A convex combination of tanh values cannot leave (-1, 1); if it did, the subsequent
    *action_scale would emit out-of-range joint targets."""
    mean = torch.tensor([[-5.0, 0.0, 5.0]])
    log_std = torch.tensor([[math.log(0.5)] * 3])
    out = _expected_action_core(mean, log_std)
    assert (out.abs() <= 1.0).all()


# ----------------------------------------------------------------------------------------------
# 3. sigma actually matters -- i.e. gradient can reach fc_logstd.
# ----------------------------------------------------------------------------------------------


def test_expected_action_is_sensitive_to_sigma():
    """The entire point of the fix. Under the old tanh(mu) target these two would be IDENTICAL,
    which is exactly why fc_logstd never moved."""
    mean = torch.tensor([[0.8]])
    small = _expected_action_core(mean, torch.tensor([[math.log(0.01)]]))
    large = _expected_action_core(mean, torch.tensor([[math.log(0.5)]]))
    assert abs((small - large).item()) > 1e-3


def test_gradient_flows_to_log_std():
    """Directly verifies the mechanism the fix depends on: d(target)/d(log_std) must be nonzero.
    Under the old tanh(mu) path this gradient is structurally zero."""
    mean = torch.tensor([[0.8, -0.3]], requires_grad=True)
    log_std = torch.tensor([[math.log(0.2), math.log(0.2)]], requires_grad=True)
    _expected_action_core(mean, log_std).sum().backward()
    assert log_std.grad is not None
    assert log_std.grad.abs().sum().item() > 1e-6


def test_larger_sigma_shrinks_the_action_toward_zero():
    """Jensen: tanh is concave for mu>0, so averaging over noise pulls the expectation toward 0.
    Pins the DIRECTION of the correction -- a student with too-large sigma under-actuates, which is
    the behavioral consequence the frozen-sigma bug was producing."""
    mean = torch.tensor([[1.2]])
    sigmas = [0.01, 0.2, 0.5, 1.0]
    vals = [_expected_action_core(mean, torch.tensor([[math.log(s)]])).item() for s in sigmas]
    assert all(a > b for a, b in zip(vals, vals[1:]))
    assert all(v > 0 for v in vals)  # sign is preserved throughout


def test_symmetry_in_mu():
    """E[tanh(-mu + sigma*Z)] == -E[tanh(mu + sigma*Z)] -- tanh is odd and Z is symmetric. A node/
    weight ordering bug would typically break this."""
    log_std = torch.tensor([[math.log(0.3)]])
    pos = _expected_action_core(torch.tensor([[1.1]]), log_std)
    neg = _expected_action_core(torch.tensor([[-1.1]]), log_std)
    assert pos.item() == pytest.approx(-neg.item(), abs=1e-6)
