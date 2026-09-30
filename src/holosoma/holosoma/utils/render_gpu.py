"""Optionally pin the offscreen MuJoCo render subprocess to a specific GPU.

The sim2sim rollout workers (`mujoco_kick_rollout_worker.py`, `mujoco_locomotion_rollout_worker.py`)
build an offscreen GL context to render video frames. On a box also running training + other MuJoCo
eval scans, that context regularly gets starved under load and every rendered frame comes back
solid black -- no exception raised (observed 2026-09-06: entire checkpoints' worth of all-black
`mujoco_media` videos, while the sim itself -- CSV trajectories, survival scans -- was fine).

The primary defence against that is the workers' own black-frame guard (they now abort loudly
instead of shipping a black video) plus the one retry in `record_mujoco_*_rollout.py`.

This module is a secondary, OPT-IN knob on top: if `HOLOSOMA_SIM2SIM_RENDER_GPU` is set, the render
subprocess is forced onto EGL with that value as `MUJOCO_EGL_DEVICE_ID`. It is deliberately NOT
automatic:
  - the default GL backend (GLFW via the box's X server) is the one that actually works here;
    forcing EGL where it's half-configured just trades a black frame for a hard crash;
  - `MUJOCO_EGL_DEVICE_ID` indexes EGL's own `eglQueryDevicesEXT()` list, which on this box has 9
    entries (4 GPUs + Mesa/software devices) and does NOT line up with `nvidia-smi` indices, and
    the PyOpenGL build here won't expose the device-query strings needed to match them up.
So picking a GPU automatically can't be done safely -- an operator who has worked out the right
EGL index for their box sets it explicitly; everyone else gets the working default + the guard.
"""

from __future__ import annotations

import os

RENDER_GPU_OVERRIDE_ENV = "HOLOSOMA_SIM2SIM_RENDER_GPU"


def render_subprocess_env() -> dict[str, str]:
    """A copy of `os.environ`, with `MUJOCO_GL=egl` + `MUJOCO_EGL_DEVICE_ID` added ONLY when
    `HOLOSOMA_SIM2SIM_RENDER_GPU` is set (an EGL device index -- see this module's docstring for why
    it's opt-in and why it's an EGL index, not an nvidia-smi one). Otherwise returns the env
    unchanged so the worker keeps MuJoCo's default backend selection."""
    env = dict(os.environ)
    override = env.get(RENDER_GPU_OVERRIDE_ENV, "").strip()
    if override:
        env["MUJOCO_GL"] = "egl"
        env["MUJOCO_EGL_DEVICE_ID"] = override
    return env
