from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, List

import numpy as np
import torch
from loguru import logger

from holosoma.config_types.command import MotionConfig, NoiseToInitialPoseConfig
from holosoma.envs.wbt.wbt_manager import WholeBodyTrackingManager
from holosoma.managers.command.base import CommandTermBase
from holosoma.managers.command.terms.kick_ankle_pitch_correction import correct_kick_foot_ankle_pitch
from holosoma.managers.command.terms.motion_head_velocity_smoothing import smooth_motion_head_velocities
from holosoma.utils.file_cache import cached_open
from holosoma.utils.motion_fk import DEFAULT_MJCF_RELATIVE_PATH, KinematicTree
from holosoma.utils.path import resolve_data_file_path
from holosoma.utils.rotations import (
    get_euler_xyz,
    quat_apply,
    quat_error_magnitude,
    quat_from_euler_xyz,
    quat_inverse,
    quat_mul,
    quat_rotate,
    slerp,
    yaw_quat,
)
from holosoma.utils.simulator_config import SimulatorType


#########################################################################################################
## MotionLoader and AdaptiveTimestepsSampler
#########################################################################################################
class MotionLoader:
    def __init__(
        self,
        motion_file: str,
        robot_body_names: list[str],
        robot_joint_names: list[str],
        device: str = "cpu",
    ):
        # Resolve the motion file path using importlib.resources
        motion_file = resolve_data_file_path(motion_file)

        logger.info(f"Loading motion file: {motion_file}")
        body_names_in_motion_data, joint_names_in_motion_data = self._load_data_from_motion_npz(motion_file, device)
        body_indexes = self._get_index_of_a_in_b(robot_body_names, body_names_in_motion_data, device)
        joint_indexes = self._get_index_of_a_in_b(robot_joint_names, joint_names_in_motion_data, device)

        self._joint_indexes = joint_indexes
        self._body_indexes = body_indexes
        self.time_step_total = self._joint_pos.shape[0]

    def _get_index_of_a_in_b(self, a_names: List[str], b_names: List[str], device: str = "cpu") -> torch.Tensor:
        indexes = []
        for name in a_names:
            assert name in b_names, f"The specified name ({name}) doesn't exist: {b_names}"
            indexes.append(b_names.index(name))
        return torch.tensor(indexes, dtype=torch.long, device=device)

    # Expected holosoma NPZ keys
    _REQUIRED_KEYS = {
        "fps",
        "joint_pos",
        "joint_vel",
        "body_pos_w",
        "body_quat_w",
        "body_lin_vel_w",
        "body_ang_vel_w",
        "body_names",
        "joint_names",
    }

    def _load_data_from_motion_npz(self, motion_file: str, device: str) -> tuple[list[str], list[str]]:
        with cached_open(motion_file, "rb") as f, np.load(f) as data:
            # Sanity check: warn if not in expected holosoma format
            keys = set(data.files)
            missing = self._REQUIRED_KEYS - keys
            if missing:
                logger.warning(
                    f"Motion NPZ '{motion_file}' is missing expected holosoma keys: {missing}. "
                    f"All motion data should be in holosoma format (with body_names, joint_names, "
                    f"and root DOFs in joint_pos). Convert from TML/BeyondMimic first."
                )
                raise ValueError(
                    f"Unsupported motion format in '{motion_file}': missing keys {missing}. "
                    f"Please convert to holosoma format."
                )

            self.fps = data["fps"]

            body_names = data["body_names"].tolist()
            joint_names = data["joint_names"].tolist()

            joint_pos_raw = data["joint_pos"]
            joint_vel_raw = data["joint_vel"]
            body_pos_w_raw = data["body_pos_w"]
            body_quat_w_raw = data["body_quat_w"]
            body_lin_vel_w_raw = data["body_lin_vel_w"]
            body_ang_vel_w_raw = data["body_ang_vel_w"]

            # Holosoma format: joint_pos includes root DOFs [xyz, wxyz] as first 7 values
            # joint_vel includes root velocity [vel_xyz, vel_wxyz] as first 6 values
            num_joint_cols = joint_pos_raw.shape[1]
            num_vel_cols = joint_vel_raw.shape[1]
            num_bodies = body_pos_w_raw.shape[1]

            if num_joint_cols != len(joint_names) + 7:
                logger.warning(
                    f"Unexpected joint_pos columns: got {num_joint_cols}, expected {len(joint_names) + 7} "
                    f"(= {len(joint_names)} joints + 7 root DOFs). File: {motion_file}"
                )
            if num_vel_cols != len(joint_names) + 6:
                logger.warning(
                    f"Unexpected joint_vel columns: got {num_vel_cols}, expected {len(joint_names) + 6} "
                    f"(= {len(joint_names)} joints + 6 root DOFs). File: {motion_file}"
                )
            if num_bodies != len(body_names):
                logger.warning(
                    f"Body count mismatch: body_pos_w has {num_bodies} bodies but body_names has "
                    f"{len(body_names)}. File: {motion_file}"
                )

            # Strip root DOFs
            self._joint_pos = torch.tensor(joint_pos_raw[:, 7:], dtype=torch.float32, device=device)
            self._joint_vel = torch.tensor(joint_vel_raw[:, 6:], dtype=torch.float32, device=device)

            assert len(joint_names) == self._joint_pos.shape[1], (
                f"Joint names ({len(joint_names)}) != joint_pos columns ({self._joint_pos.shape[1]}) in {motion_file}"
            )
            assert len(body_names) == body_pos_w_raw.shape[1], (
                f"Body names ({len(body_names)}) != body_pos_w bodies ({body_pos_w_raw.shape[1]}) in {motion_file}"
            )

            self._body_pos_w = torch.tensor(body_pos_w_raw, dtype=torch.float32, device=device)

            # NOTE: wxyz after loading from npz
            body_quat_w_wxyz = torch.tensor(body_quat_w_raw, dtype=torch.float32, device=device)  # This is wxyz
            self._body_quat_w = body_quat_w_wxyz[:, :, [1, 2, 3, 0]]  # Change to xyzw

            self._body_lin_vel_w = torch.tensor(body_lin_vel_w_raw, dtype=torch.float32, device=device)
            self._body_ang_vel_w = torch.tensor(body_ang_vel_w_raw, dtype=torch.float32, device=device)

            # add object pos and quat
            self.has_object = "object_pos_w" in data
            if self.has_object:
                self._object_pos_w = torch.tensor(data["object_pos_w"], dtype=torch.float32, device=device)
                # NOTE: wxyz after loading from npz
                object_quat_w = torch.tensor(data["object_quat_w"], dtype=torch.float32, device=device)
                self._object_quat_w = object_quat_w[:, [1, 2, 3, 0]]  # Change to xyzw
                self._object_lin_vel_w = torch.tensor(data["object_lin_vel_w"], dtype=torch.float32, device=device)
            else:
                self._object_pos_w = torch.zeros(0, 3, device=device)
                self._object_quat_w = torch.zeros(0, 4, device=device)
                self._object_lin_vel_w = torch.zeros(0, 3, device=device)
        return body_names, joint_names

    @property
    def joint_pos(self) -> torch.Tensor:
        return self._joint_pos[:, self._joint_indexes]

    @property
    def joint_vel(self) -> torch.Tensor:
        return self._joint_vel[:, self._joint_indexes]

    @property
    def body_pos_w(self) -> torch.Tensor:
        return self._body_pos_w[:, self._body_indexes]

    @property
    def body_quat_w(self) -> torch.Tensor:
        return self._body_quat_w[:, self._body_indexes]

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        return self._body_lin_vel_w[:, self._body_indexes]

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        return self._body_ang_vel_w[:, self._body_indexes]

    @property
    def object_pos_w(self) -> torch.Tensor:
        return self._object_pos_w[:]

    @property
    def object_quat_w(self) -> torch.Tensor:
        return self._object_quat_w[:]

    @property
    def object_lin_vel_w(self) -> torch.Tensor:
        return self._object_lin_vel_w[:]

    @property
    def num_motions(self) -> int:
        return 1

    @property
    def motion_start_idx(self) -> torch.Tensor:
        return torch.tensor([0], dtype=torch.long, device=self._joint_pos.device)

    @property
    def motion_end_idx(self) -> torch.Tensor:
        return torch.tensor([self.time_step_total], dtype=torch.long, device=self._joint_pos.device)

    def extend_with_segments(self, segments: dict[str, torch.Tensor], prepend: bool) -> MotionLoader:
        """Merge interpolated segments with motion data, mutating this MotionLoader."""
        concat_targets = [
            ("joint_pos", "_joint_pos"),
            ("joint_vel", "_joint_vel"),
            ("body_pos", "_body_pos_w"),
            ("body_quat", "_body_quat_w"),
            ("body_lin_vel", "_body_lin_vel_w"),
            ("body_ang_vel", "_body_ang_vel_w"),
        ]
        if self.has_object:
            concat_targets.extend(
                [
                    ("object_pos", "_object_pos_w"),
                    ("object_quat", "_object_quat_w"),
                    ("object_lin_vel", "_object_lin_vel_w"),
                ]
            )

        for seg_key, attr_name in concat_targets:
            existing = getattr(self, attr_name)
            tensors = (segments[seg_key], existing) if prepend else (existing, segments[seg_key])
            setattr(self, attr_name, torch.cat(tensors, dim=0))

        self.time_step_total = self._joint_pos.shape[0]
        return self

    def insert_segment_at_motion_boundary(
        self, motion_idx: int, segments: dict[str, torch.Tensor], *, at_start: bool
    ) -> MotionLoader:
        """Single-clip analog of MultiMotionLoader's method of the same name, so
        MotionCommand's per-motion prepend/append loop can call the identical method regardless
        of loader type. There's only ever one motion here (motion_idx must be 0), so this reduces
        exactly to extend_with_segments."""
        assert motion_idx == 0, f"MotionLoader has exactly one motion; got motion_idx={motion_idx}"
        return self.extend_with_segments(segments, prepend=at_start)


class MultiMotionLoader:
    """Loads multiple NPZ motion files and concatenates them at runtime.

    Tracks per-motion boundaries so environments can sample within individual clips.
    Compatible with the same interface as MotionLoader.
    """

    @classmethod
    def from_dir(
        cls,
        motion_dir: str,
        robot_body_names: list[str],
        robot_joint_names: list[str],
        device: str = "cpu",
    ) -> MultiMotionLoader:
        """Original motion_dir behavior: glob one or more (comma-separated) directories for
        `*.npz`, sorted alphabetically within each directory. Order here is whatever the
        filesystem/glob happens to produce -- fine for the existing "any N clips, no particular
        skill identity" use case, but NOT suitable when the caller needs deterministic, yaml-
        declared skill ordering (index i must mean "skill i", not "whatever sorted alphabetically
        first") -- see the main constructor's `motion_paths` for that case."""
        dirs = [d.strip() for d in motion_dir.split(",")]
        motion_paths = []
        for d in dirs:
            expanded = os.path.expanduser(d)
            files = sorted(str(p) for p in Path(expanded).glob("*.npz"))
            logger.info(f"MultiMotionLoader: found {len(files)} .npz files in {expanded}")
            motion_paths.extend(files)
        assert len(motion_paths) > 0, f"No .npz files found in {motion_dir}"
        return cls(motion_paths, robot_body_names, robot_joint_names, device=device)

    def __init__(
        self,
        motion_paths: list[str],
        robot_body_names: list[str],
        robot_joint_names: list[str],
        device: str = "cpu",
    ):
        """motion_paths: explicit, ORDERED list of .npz file paths. Order is preserved exactly --
        index i becomes motion_id i (motion_start_idx[i]/motion_end_idx[i]) -- unlike from_dir's
        alphabetical glob, this is what a caller assigning skill identity by declaration order
        (e.g. a yaml's motion_skill_1, motion_skill_2, ...) needs."""
        motion_files = list(motion_paths)
        assert len(motion_files) > 0, "MultiMotionLoader: motion_paths must be non-empty"
        logger.info(f"MultiMotionLoader: loading {len(motion_files)} total motion files")

        loaders = []
        skipped = 0
        for mf in motion_files:
            try:
                loader = MotionLoader(mf, robot_body_names, robot_joint_names, device=device)
                loaders.append(loader)
            except (KeyError, AssertionError, ValueError) as e:  # noqa: PERF203
                # Skip files with incompatible format (e.g., missing body_names, wrong body count)
                skipped += 1
                if skipped <= 3:
                    logger.warning(f"MultiMotionLoader: skipping {mf}: {e}")
        if skipped > 3:
            logger.warning(f"MultiMotionLoader: skipped {skipped} files total due to format issues")
        assert len(loaders) > 0, f"No compatible motion files found (skipped {skipped})"

        # Track per-motion boundaries
        lengths = [loader.time_step_total for loader in loaders]
        cumulative = torch.tensor(lengths, dtype=torch.long, device=device).cumsum(dim=0)
        self._motion_start_idx = torch.cat([torch.tensor([0], dtype=torch.long, device=device), cumulative[:-1]])
        self._motion_end_idx = cumulative
        self._num_motions = len(loaders)

        # Concatenate all motion data
        self._joint_pos = torch.cat([ld._joint_pos for ld in loaders], dim=0)
        self._joint_vel = torch.cat([ld._joint_vel for ld in loaders], dim=0)
        self._body_pos_w = torch.cat([ld._body_pos_w for ld in loaders], dim=0)
        self._body_quat_w = torch.cat([ld._body_quat_w for ld in loaders], dim=0)
        self._body_lin_vel_w = torch.cat([ld._body_lin_vel_w for ld in loaders], dim=0)
        self._body_ang_vel_w = torch.cat([ld._body_ang_vel_w for ld in loaders], dim=0)

        # Use indexes from first loader (all loaders share the same robot)
        self._joint_indexes = loaders[0]._joint_indexes
        self._body_indexes = loaders[0]._body_indexes
        self.fps = loaders[0].fps
        self.time_step_total = self._joint_pos.shape[0]

        # Object support: only if ALL motions have objects
        self.has_object = all(ld.has_object for ld in loaders)
        if self.has_object:
            self._object_pos_w = torch.cat([ld._object_pos_w for ld in loaders], dim=0)
            self._object_quat_w = torch.cat([ld._object_quat_w for ld in loaders], dim=0)
            self._object_lin_vel_w = torch.cat([ld._object_lin_vel_w for ld in loaders], dim=0)
        else:
            self._object_pos_w = torch.zeros(0, 3, device=device)
            self._object_quat_w = torch.zeros(0, 4, device=device)
            self._object_lin_vel_w = torch.zeros(0, 3, device=device)

        logger.info(f"MultiMotionLoader: {self._num_motions} motions, {self.time_step_total} total frames")

    @property
    def num_motions(self) -> int:
        return self._num_motions

    @property
    def motion_start_idx(self) -> torch.Tensor:
        return self._motion_start_idx

    @property
    def motion_end_idx(self) -> torch.Tensor:
        return self._motion_end_idx

    @property
    def joint_pos(self) -> torch.Tensor:
        return self._joint_pos[:, self._joint_indexes]

    @property
    def joint_vel(self) -> torch.Tensor:
        return self._joint_vel[:, self._joint_indexes]

    @property
    def body_pos_w(self) -> torch.Tensor:
        return self._body_pos_w[:, self._body_indexes]

    @property
    def body_quat_w(self) -> torch.Tensor:
        return self._body_quat_w[:, self._body_indexes]

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        return self._body_lin_vel_w[:, self._body_indexes]

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        return self._body_ang_vel_w[:, self._body_indexes]

    @property
    def object_pos_w(self) -> torch.Tensor:
        return self._object_pos_w[:]

    @property
    def object_quat_w(self) -> torch.Tensor:
        return self._object_quat_w[:]

    @property
    def object_lin_vel_w(self) -> torch.Tensor:
        return self._object_lin_vel_w[:]

    def insert_segment_at_motion_boundary(
        self, motion_idx: int, segments: dict[str, torch.Tensor], *, at_start: bool
    ) -> MultiMotionLoader:
        """Splice `segments` into the concatenated buffer immediately before `motion_idx`'s
        CURRENT start (at_start=True — a prepend/windup) or immediately after its CURRENT end
        (at_start=False — an append/recovery+hold). Merges the inserted frames into `motion_idx`
        itself (its own span grows by len(segments)) rather than registering a new motion — the
        same "merge, don't add a boundary" convention this class has always used for prepend/
        append, generalized from "only motion 0" / "only the last motion" to any `motion_idx`.
        Shifts every LATER motion's start/end by the inserted frame count so indexing stays
        contiguous.

        Verified reduction: `motion_idx=0, at_start=True` and `motion_idx=num_motions-1,
        at_start=False` produce EXACTLY the old extend_with_segments' index math (see that
        method, kept below as a thin wrapper) — this is a strict generalization, not a behavior
        change, for the two cases that were previously supported.

        Must be called in ASCENDING motion_idx order across a sequence of insertions (motion 0,
        then 1, then 2, ...): each insertion shifts subsequent motions' boundaries, so a later
        motion's insert position must be read AFTER earlier insertions have already updated it."""
        insert_pos = int((self._motion_start_idx if at_start else self._motion_end_idx)[motion_idx].item())

        concat_targets = [
            ("joint_pos", "_joint_pos"),
            ("joint_vel", "_joint_vel"),
            ("body_pos", "_body_pos_w"),
            ("body_quat", "_body_quat_w"),
            ("body_lin_vel", "_body_lin_vel_w"),
            ("body_ang_vel", "_body_ang_vel_w"),
        ]
        if self.has_object:
            concat_targets.extend(
                [
                    ("object_pos", "_object_pos_w"),
                    ("object_quat", "_object_quat_w"),
                    ("object_lin_vel", "_object_lin_vel_w"),
                ]
            )

        added_frames = 0
        for seg_key, attr_name in concat_targets:
            existing = getattr(self, attr_name)
            new_tensor = torch.cat([existing[:insert_pos], segments[seg_key], existing[insert_pos:]], dim=0)
            setattr(self, attr_name, new_tensor)
            if added_frames == 0:
                added_frames = segments[seg_key].shape[0]

        # motion_idx's own span grows (its end moves out by added_frames); every motion AFTER it
        # shifts wholesale (both start and end move out by added_frames). motion_idx's own start
        # is untouched in both cases -- for at_start=True the inserted frames occupy exactly where
        # its old start was, so its start index doesn't change even though new content now begins
        # there; for at_start=False nothing before its end is affected at all.
        self._motion_end_idx[motion_idx:] += added_frames
        self._motion_start_idx[motion_idx + 1 :] += added_frames

        self.time_step_total = self._joint_pos.shape[0]
        return self

    def extend_with_segments(self, segments: dict[str, torch.Tensor], prepend: bool) -> MultiMotionLoader:
        """Backward-compat wrapper. prepend=True -> insert before motion 0's start; prepend=False
        -> insert after the LAST motion's end. Nothing in this codebase calls this directly
        anymore (MotionCommand.setup() now loops insert_segment_at_motion_boundary per motion);
        kept as a documented special case / safety net."""
        motion_idx = 0 if prepend else self._num_motions - 1
        return self.insert_segment_at_motion_boundary(motion_idx, segments, at_start=prepend)


class AdaptiveTimestepsSampler:
    """Prioritizes training on motion segments where the robot fails most often."""

    def __init__(
        self,
        motion_time_step_total: int,
        device: str,
        env_fps: int,
        adaptive_kernel_size: int = 1,
        adaptive_lambda: float = 0.8,
        adaptive_uniform_ratio: float = 0.1,
        adaptive_alpha: float = 0.001,
    ):
        self.device = device
        # length of the motion in rl environment time steps
        self.motion_time_step_total = motion_time_step_total
        # fps of the rl environment
        self.env_fps = env_fps

        self.adaptive_kernel_size = adaptive_kernel_size
        self.adaptive_lambda = adaptive_lambda
        self.adaptive_uniform_ratio = adaptive_uniform_ratio
        self.adaptive_alpha = adaptive_alpha

        # Match BeyondMimic binning: ~1 second bins at env FPS, with +1 tail bin.
        self.num_bins = int(self.motion_time_step_total // max(self.env_fps, 1)) + 1

        # Match BeyondMimic non-causal kernel.
        self.kernel = torch.tensor(
            [self.adaptive_lambda**i for i in range(self.adaptive_kernel_size)],
            device=self.device,
        )
        self.kernel = self.kernel / self.kernel.sum()

        # key data: failure counts
        self.init_buffers()
        # metrics
        self.metrics: dict[str, torch.Tensor] = {}

    def init_buffers(self):
        self.current_bin_failed_count = torch.zeros(self.num_bins, dtype=torch.float, device=self.device)
        self.bin_failed_count = torch.zeros(self.num_bins, dtype=torch.float, device=self.device)

    def update_current_bin_failed_count(self, failed_at_time_step: torch.Tensor):
        """Update the current bin failed count with terminated time steps."""
        failed_bin = torch.clamp(
            (failed_at_time_step * self.num_bins) // max(self.motion_time_step_total, 1),
            0,
            self.num_bins - 1,
        ).long()
        assert failed_bin.min() >= 0 and failed_bin.max() < self.num_bins, "Failed bin is out of range"
        # Accumulate (not overwrite): reset() may be called more than once per env
        # step — once for termination-driven resets and again for clip-ended resets
        # in MotionCommand.step() — before update_bin_failed_count() folds + zeroes
        # this buffer. Overwriting clobbered the earlier wave's failures.
        self.current_bin_failed_count += torch.bincount(failed_bin, minlength=self.num_bins).float()

    def update_bin_failed_count(self):
        """At every rl environment step, update the failed count with the current bin failed count."""
        self.bin_failed_count = (self.adaptive_alpha * self.current_bin_failed_count) + (
            1 - self.adaptive_alpha
        ) * self.bin_failed_count
        self.current_bin_failed_count.zero_()

    @property
    def sampling_probabilities(self) -> torch.Tensor:
        sampling_probabilities = self.bin_failed_count + self.adaptive_uniform_ratio / float(self.num_bins)
        sampling_probabilities = torch.nn.functional.pad(
            sampling_probabilities.unsqueeze(0).unsqueeze(0),
            (0, self.adaptive_kernel_size - 1),  # Non-causal kernel
            mode="replicate",
        )
        sampling_probabilities = torch.nn.functional.conv1d(sampling_probabilities, self.kernel.view(1, 1, -1)).view(-1)
        return sampling_probabilities / sampling_probabilities.sum()

    def sample(self, num_samples: int) -> torch.Tensor:
        sampled_bins = torch.multinomial(self.sampling_probabilities, num_samples, replacement=True)
        # inside of each bin, randomly sample a time step, ignoring the borders
        return (sampled_bins + torch.rand(num_samples, device=self.device)) / self.num_bins

    def sample_global_time_steps(self, num_samples: int) -> torch.Tensor:
        """Sample absolute (global) frame indices in [0, motion_time_step_total).

        The bins live in the GLOBAL concatenated-motion frame space, so a sampled
        phase must be mapped back to a global frame index — NOT reinterpreted as a
        per-motion fraction of an unrelated clip. The caller derives the motion id
        from which clip's [start, end) interval the returned index falls into, so
        the failure-prioritized location stays attached to the motion it came from.
        """
        phase = self.sample(num_samples)
        global_idx = (phase * self.motion_time_step_total).long()
        return global_idx.clamp_(0, self.motion_time_step_total - 1)

    def get_stats(self):
        # Metrics
        prob = self.sampling_probabilities
        H = -(prob * (prob + 1e-12).log()).sum()
        H_norm = H / np.log(max(self.num_bins, 2))  # guard num_bins==1 (log(1)=0 -> nan)
        pmax, imax = prob.max(dim=0)
        self.metrics["sampling_entropy"] = H_norm
        self.metrics["sampling_top1_prob"] = pmax
        self.metrics["sampling_top1_bin"] = imax.float() / self.num_bins


#########################################################################################################
## Helper functions
#########################################################################################################
FAKE_BODY_NAME_ALIASES: dict[str, str] = {
    # Fake foot contact bodies are authored in the URDF purely for height computation.
    # They do not exist in the motion-capture dataset, so we alias them back to the
    # closest real body when indexing into motion data. These are not actually used in training.
    "left_foot_contact_point": "left_ankle_roll_link",
    "right_foot_contact_point": "right_ankle_roll_link",
}


def get_filtered_body_names(body_list: List[str], pattern: str) -> List[str]:
    return [body_name for body_name in body_list if re.match(pattern, body_name)]


def draw_position_noise_with_ood(
    position_randomization: torch.Tensor,
    *,
    ood_prob: float,
    ood_multiplier: float,
    device: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-env uniform xy noise draw for ball/target placement, with 2026-07-24
    deployment-robustness OOD support: normally uniform in ``[-1, 1] * position_randomization``
    (the pre-existing behavior, bit-identical when ``ood_prob <= 0.0``), but with probability
    ``ood_prob`` an env's draw instead comes from the WIDER ``[-ood_multiplier, ood_multiplier] *
    position_randomization`` region -- e.g. behind the robot, too close, too far.

    Not rejection-sampled to guarantee landing outside the normal box (a simple uniform draw over
    the wider region lands outside the normal box on at least one axis the large majority of the
    time, which is judged sufficient -- see ``MotionConfig.ood_spawn_probability``'s own
    docstring for the full rationale/tradeoffs, agreed with the user before implementing this).

    Parameters
    ----------
    position_randomization : torch.Tensor
        [N, 2] per-env half-range (meters) -- the SAME tensor gathered per-motion in ``reset()``
        (``ball_position_randomization_per_motion[motion_ids]`` or the target equivalent).
    ood_prob : float
        Per-env probability of using the OOD region instead of the normal one. <= 0.0 disables
        this entirely (exactly the pre-existing plain-noise behavior).
    ood_multiplier : float
        OOD region half-range, as a multiple of ``position_randomization``. Only meaningful when
        ``ood_prob > 0.0``.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        ``(noise, is_ood)``. ``noise``: [N, 2] draw, ready to add to a nominal local (x, y)
        position (same as this function returned before 2026-08-01). ``is_ood``: [N] bool mask,
        True for envs whose draw came from the OOD region -- all-False whenever ``ood_prob <=
        0.0``. RETURNED (rather than computed and discarded, as before 2026-08-01) so callers can
        persist which envs got an OOD draw this attempt -- see ``MotionCommand.is_ood_spawn``,
        consumed by ``managers/reward/terms/shooting.py``'s ``_ood_gate_multiplier`` to zero the
        shooting reward for those attempts (see ``MotionConfig.ood_spawn_probability``'s own
        docstring for why). The other call site
        (``managers/observation/terms/unified.py``'s ``_draw_independent_frozen_ball_reading``,
        the unrelated ``ball_static_obs_probability`` mechanism) deliberately discards this half
        -- that mechanism stays reward-agnostic.
    """
    n = position_randomization.shape[0]
    noise = (torch.rand(n, 2, device=device) * 2.0 - 1.0) * position_randomization
    is_ood = torch.zeros(n, dtype=torch.bool, device=device)
    if ood_prob > 0.0:
        is_ood = torch.rand(n, device=device) < ood_prob
        if bool(is_ood.any()):
            ood_noise = (torch.rand(n, 2, device=device) * 2.0 - 1.0) * (position_randomization * ood_multiplier)
            noise = torch.where(is_ood.unsqueeze(-1), ood_noise, noise)
    return noise, is_ood


def rsi_span_end_idx(
    motion_end_idx: torch.Tensor,
    pre_recovery_motion_end_idx: torch.Tensor,
    *,
    rsi_scope_to_authored_clip: bool | torch.Tensor,
) -> torch.Tensor:
    """2026-08-05, ported from RoboNaldo (arXiv:2606.11092)'s ``start_time_sampling_fraction`` --
    the per-motion upper bound the uniform-phase RSI draw in ``MotionCommand.reset()`` samples
    against. ``rsi_scope_to_authored_clip=False`` (default): returns ``motion_end_idx`` unchanged
    -- byte-identical to before this field existed, so a mid-clip RSI reset can land anywhere in
    the augmented buffer, including the synthetic recovery-lerp/hold tail spliced on after each
    authored clip. ``True``: returns ``pre_recovery_motion_end_idx`` instead (already computed in
    ``MotionCommand.setup()`` -- the exact frame where a motion's real authored content ends),
    scoping RSI to authored content only. See ``MotionConfig.rsi_scope_to_authored_clip``'s own
    docstring for the full rationale.

    2026-08-15, "simultaneous per-skill task configs": ``rsi_scope_to_authored_clip`` may also be
    a per-env ``[n]`` bool tensor (MotionCommand._resolved_rsi_scope_to_authored_clip, gathered by
    motion_ids) -- ``torch.where`` selects PER-ENV rather than branching once for the whole batch.
    A plain bool still takes the original scalar ternary, byte-identical to before this existed."""
    if torch.is_tensor(rsi_scope_to_authored_clip):
        return torch.where(rsi_scope_to_authored_clip, pre_recovery_motion_end_idx, motion_end_idx)
    return pre_recovery_motion_end_idx if rsi_scope_to_authored_clip else motion_end_idx


def critical_frame_oversample_time_steps(
    time_steps: torch.Tensor,
    *,
    start_idx: torch.Tensor,
    span_end_idx: torch.Tensor,
    strike_start_idx: torch.Tensor,
    oversample_prob: float,
    window: int,
    device: Any,
) -> torch.Tensor:
    """2026-08-05, ported from RoboNaldo (arXiv:2606.11092)'s ``critical_frame_adaptive_sampling``
    -- with probability ``oversample_prob``, replaces each env's already-drawn (plain uniform)
    ``time_steps`` with a fresh uniform draw from a fixed window around that env's assigned
    motion's own ``strike_start_idx`` (``window`` frames either side, clamped to
    ``[start_idx, span_end_idx - 1]``) -- practicing the kick itself, not just the approach.
    ``oversample_prob <= 0.0`` (default): returns ``time_steps`` completely unchanged (the two
    ``torch.rand`` calls below still execute but every result is discarded by the ``roll <
    oversample_prob`` comparison, which is always False -- an exact, not just numerical, no-op on
    the returned values). ``span_end_idx`` should be ``rsi_span_end_idx(...)``'s own return value
    (not the raw ``motion_end_idx``) so this composes correctly with
    ``MotionConfig.rsi_scope_to_authored_clip``: when that's on, the oversampling window can't
    reach into the synthetic tail either. See ``MotionConfig.critical_frame_oversampling_prob``'s
    own docstring for the full rationale."""
    n = time_steps.shape[0]
    window_low = torch.clamp(strike_start_idx - window, min=start_idx)
    window_high = torch.clamp(strike_start_idx + window, max=span_end_idx - 1)
    window_span = (window_high - window_low).clamp(min=0).float()
    window_draw = window_low + (torch.rand(n, device=device) * window_span).long()
    oversample_roll = torch.rand(n, device=device) < oversample_prob
    return torch.where(oversample_roll, window_draw, time_steps)


class MotionCommand(CommandTermBase):
    def __init__(self, cfg: Any, env: WholeBodyTrackingManager):
        super().__init__(cfg, env)

        self._env = env
        # self.motion_cfg: MotionConfig = cfg.params["motion_config"]
        # TODO(jchen):temporary fix for motion_config being a dict after tyro.cli
        if isinstance(cfg.params["motion_config"], MotionConfig):
            self.motion_cfg = cfg.params["motion_config"]
        else:
            self.motion_cfg = MotionConfig(**cfg.params["motion_config"])
        self.init_pose_cfg: NoiseToInitialPoseConfig = self.motion_cfg.noise_to_initial_pose

    def setup(self) -> None:
        self.num_envs = self._env.num_envs
        self.device = self._env.device

        robot_body_names = self._env.simulator._body_list  # type: ignore[attr-defined]
        robot_body_names_alias = [FAKE_BODY_NAME_ALIASES.get(bn, bn) for bn in robot_body_names]

        robot_joint_names = self._env.simulator.dof_names  # type: ignore[attr-defined]

        # 1. load motion data
        assert self.motion_cfg.motion_file or self.motion_cfg.motion_dir or self.motion_cfg.motion_files, (
            "One of motion_file, motion_dir, or motion_files must be set in MotionConfig"
        )
        self.motion: MotionLoader | MultiMotionLoader
        if self.motion_cfg.motion_files:
            # Explicit, ORDERED list of per-skill clips (e.g. from the stacked N-motion-skill
            # yaml) -- index i becomes motion_id i, unlike motion_dir's alphabetical glob, which
            # is what lets skill_1/skill_2/... in the yaml map deterministically onto motion_ids.
            self.motion = MultiMotionLoader(
                self.motion_cfg.motion_files,
                robot_body_names_alias,
                robot_joint_names,
                device=self.device,
            )
        elif self.motion_cfg.motion_dir:
            self.motion = MultiMotionLoader.from_dir(
                self.motion_cfg.motion_dir,
                robot_body_names_alias,
                robot_joint_names,
                device=self.device,
            )
        else:
            self.motion = MotionLoader(
                self.motion_cfg.motion_file,
                robot_body_names_alias,
                robot_joint_names,
                device=self.device,
            )

        # Store body and joint indexes for interpolation
        self._body_indexes_in_motion = self.motion._body_indexes
        self._joint_indexes_in_motion = self.motion._joint_indexes

        # Static, pre-augmentation per-motion raw clip length (frame count in the ORIGINAL npz,
        # before any prepend/append is spliced in) -- captured once, here, before either
        # augmentation loop touches self.motion. Needed below to convert scrubbed strike/stand
        # clip-local frame indices (relative to the raw npz, exactly what motion_clip_scrubber.py
        # displays) into absolute buffer indices, since after a prepend is inserted
        # motion_start_idx[_m] points at the START OF THE PREPEND, not raw frame 0 (see
        # insert_segment_at_motion_boundary's own docstring) -- this value never changes again
        # after this point, unlike motion_start_idx/motion_end_idx which keep shifting as later
        # insertions land.
        self._raw_motion_frame_count = (self.motion.motion_end_idx - self.motion.motion_start_idx).clone()

        # Per-motion prepend/recovery/hold durations: each resolves to the per-skill list
        # (motion_prepend_duration_s / motion_recovery_duration_s / motion_hold_duration_s, e.g.
        # from the stacked N-motion-skill yaml) when non-empty, else broadcasts the single-clip
        # scalar default to every motion -- see _resolve_per_motion_durations. The broadcast case
        # is what makes an N=1 config bit-identical to the pre-existing single-scalar behavior.
        num_motions = self.motion.num_motions
        self._resolved_prepend_duration_s = self._resolve_per_motion_durations(
            self.motion_cfg.motion_prepend_duration_s, self.motion_cfg.default_pose_prepend_duration_s, num_motions
        )
        self._resolved_recovery_duration_s = self._resolve_per_motion_durations(
            self.motion_cfg.motion_recovery_duration_s, self.motion_cfg.default_pose_append_duration_s, num_motions
        )
        self._resolved_hold_duration_s = self._resolve_per_motion_durations(
            self.motion_cfg.motion_hold_duration_s, self.motion_cfg.post_transition_hold_duration_s, num_motions
        )
        # "Simultaneous per-skill task configs" (2026-08-15) -- same broadcast helper, same
        # bit-identical-when-empty contract, for motion_head_velocity_smoothing_frames. See
        # MotionConfig.motion_head_velocity_smoothing_frames_per_motion's own docstring for why
        # this is a genuinely per-CLIP (not just per-training-regime) knob.
        self._resolved_head_velocity_smoothing_frames = self._resolve_per_motion_durations(
            self.motion_cfg.motion_head_velocity_smoothing_frames_per_motion,
            self.motion_cfg.motion_head_velocity_smoothing_frames,
            num_motions,
        )

        # Strike/stand clip-local boundaries (video_011/video_012-scrubbed via
        # motion_clip_scrubber.py): REQUIRED, no global-default broadcast (unlike the duration
        # fields above) -- see MotionConfig field docstrings for why. None (both lists empty)
        # means legacy/single-clip mode: no scrubbed boundaries configured,
        # in_kicking_phase/in_strike_phase fall back to a verified no-op below.
        self._resolved_strike_start_frame = (
            list(self.motion_cfg.motion_strike_start_frame) if self.motion_cfg.motion_strike_start_frame else None
        )
        self._resolved_stand_start_frame = (
            list(self.motion_cfg.motion_stand_start_frame) if self.motion_cfg.motion_stand_start_frame else None
        )
        for _name, _resolved in (
            ("motion_strike_start_frame", self._resolved_strike_start_frame),
            ("motion_stand_start_frame", self._resolved_stand_start_frame),
        ):
            if _resolved is not None and len(_resolved) != num_motions:
                raise ValueError(
                    f"{_name} has {len(_resolved)} entries but {num_motions} motions loaded -- must be "
                    f"empty (legacy mode) or exactly length {num_motions}."
                )
        if (self._resolved_strike_start_frame is None) != (self._resolved_stand_start_frame is None):
            raise ValueError(
                "motion_strike_start_frame and motion_stand_start_frame must be both empty or both "
                "populated -- one without the other is a config mistake, not a valid partial state."
            )

        # "Simultaneous per-skill task configs" (2026-08-15) -- 4 RSI/reset-time sampling fields,
        # each None (no per-skill override, reset() uses the plain scalar unchanged) unless a
        # non-empty per-motion list is set. Deliberately NOT the _resolve_per_motion_durations
        # broadcast helper (unlike motion_head_velocity_smoothing_frames_per_motion above): that
        # helper ALWAYS returns a full-length list even with no override, which would force every
        # reset() call through the tensor/elementwise code path -- fine for 3 of these 4 fields,
        # but start_at_timestep_zero_prob's own >=1.0 fast path deliberately SKIPS a torch.rand
        # draw entirely, a real RNG-consumption-order difference a broadcast tensor would erase
        # even with no genuine per-skill divergence. None-unless-populated preserves that fast
        # path exactly whenever no skill actually overrides it -- the overwhelmingly common case.
        for _name in (
            "start_at_timestep_zero_prob_per_motion",
            "rsi_scope_to_authored_clip_per_motion",
            "critical_frame_oversampling_prob_per_motion",
            "critical_frame_sampling_window_per_motion",
        ):
            _per_motion = getattr(self.motion_cfg, _name)
            if _per_motion and len(_per_motion) != num_motions:
                raise ValueError(
                    f"{_name} has {len(_per_motion)} entries but {num_motions} motions loaded -- must "
                    f"be empty (no per-skill override) or exactly length {num_motions}."
                )
        self._resolved_start_at_timestep_zero_prob = (
            torch.tensor(self.motion_cfg.start_at_timestep_zero_prob_per_motion, dtype=torch.float32, device=self.device)
            if self.motion_cfg.start_at_timestep_zero_prob_per_motion
            else None
        )
        self._resolved_rsi_scope_to_authored_clip = (
            torch.tensor(self.motion_cfg.rsi_scope_to_authored_clip_per_motion, dtype=torch.bool, device=self.device)
            if self.motion_cfg.rsi_scope_to_authored_clip_per_motion
            else None
        )
        self._resolved_critical_frame_oversampling_prob = (
            torch.tensor(
                self.motion_cfg.critical_frame_oversampling_prob_per_motion, dtype=torch.float32, device=self.device
            )
            if self.motion_cfg.critical_frame_oversampling_prob_per_motion
            else None
        )
        self._resolved_critical_frame_sampling_window = (
            torch.tensor(
                self.motion_cfg.critical_frame_sampling_window_per_motion, dtype=torch.long, device=self.device
            )
            if self.motion_cfg.critical_frame_sampling_window_per_motion
            else None
        )

        # Maybe prepend interpolated transition from default pose -- once per motion, in ASCENDING
        # order (load-bearing: each insertion shifts subsequent motions' boundaries, so a later
        # motion's insert position must be read after earlier insertions have already updated it;
        # see insert_segment_at_motion_boundary's docstring).
        for _m in range(num_motions):
            self._maybe_add_default_pose_transition(prepend=True, motion_idx=_m)

        # Snapshot each motion's OWN end index -- after any prepend (still "getting into position
        # for the kick", part of the swing), before THAT motion's own append -- so it marks exactly
        # where that motion's real authored clip content ends and the synthetic recovery-to-
        # neutral-pose + static-hold tail begins. Captured directly rather than recomputed from
        # duration/dt so it can never drift from what actually got appended, including that logic's
        # own skip conditions (e.g. a duration too short for dt). Consumed by
        # _kick_recovery_gate (managers/reward/terms/locomotion.py) and
        # _recovery_phase_tracking_multiplier (managers/reward/terms/kick_scale_wrappers.py) via
        # in_kicking_phase/stand_start_idx to gate posture-shaping/tracking-scale to the actual
        # authored clip -- without this, those terms would treat the entire ~3s recovery+hold tail
        # as still "in the clip". Shooting terms (managers/reward/terms/shooting.py) are gated
        # separately, to the narrower in_strike_phase (strike_start_idx..stand_start_idx) -- see
        # those properties below.
        #
        # MUST be captured PER-MOTION, inside this loop, immediately before that motion's own
        # append call -- NOT as one batch .clone() before the whole loop. A single upfront snapshot
        # goes stale for every motion after the first: motion m's boundaries shift when any EARLIER
        # motion (0..m-1) receives its own append/hold insertion later in this same loop, but a
        # value captured before the loop started would never see that shift. Capturing motion m's
        # own end index at the moment this loop reaches m is exactly when all earlier motions'
        # shifts have already landed, and m's own hasn't happened yet -- the correct instant.
        # (Verified via a live 2-skill run: batch-snapshot gave motion 1 a value stale by exactly
        # motion 0's total appended frame count.)
        pre_recovery_list: list[int] = []
        strike_start_list: list[int] = []
        stand_start_list: list[int] = []
        for _m in range(num_motions):
            motion_end = int(self.motion.motion_end_idx[_m].item())
            pre_recovery_list.append(motion_end)

            # raw_clip_start: absolute buffer index of this motion's raw-npz frame 0. motion_end
            # already includes this motion's own prepend (nothing else has been added to it yet at
            # this point in the loop, since this motion's own append hasn't run), so subtracting
            # its static raw length recovers exactly where raw frame 0 now lives -- see the
            # _raw_motion_frame_count docstring above for why this can't just be
            # motion_start_idx[_m] + clip_local_frame.
            raw_len = int(self._raw_motion_frame_count[_m].item())
            raw_clip_start = motion_end - raw_len

            if self._resolved_stand_start_frame is not None:
                strike_frame = self._resolved_strike_start_frame[_m]
                stand_frame = self._resolved_stand_start_frame[_m]
                if not (0 <= strike_frame < stand_frame <= raw_len):
                    motion_name = self.motion_cfg.motion_files[_m] if self.motion_cfg.motion_files else "?"
                    raise ValueError(
                        f"motion {_m} ({motion_name}): strike_start_frame={strike_frame}, "
                        f"stand_start_frame={stand_frame} incompatible with this clip's own raw "
                        f"length ({raw_len} frames) -- require "
                        f"0 <= strike_start_frame < stand_start_frame <= {raw_len}."
                    )
                strike_start_list.append(raw_clip_start + strike_frame)
                stand_start_list.append(raw_clip_start + stand_frame)
            else:
                # Legacy/single-clip mode: no scrubbed boundaries -- in_kicking_phase keeps its
                # pre-existing end boundary (bit-identical to in_swing_phase before this change),
                # and in_strike_phase collapses onto the SAME window (lower bound = this motion's
                # own earliest achievable time_steps, motion_start_idx -- reset() never sets
                # time_steps below it) so it's a verified no-op everywhere it's read.
                strike_start_list.append(int(self.motion.motion_start_idx[_m].item()))
                stand_start_list.append(motion_end)

            self._maybe_add_default_pose_transition(prepend=False, motion_idx=_m)

            # Self-calibrating kick-foot ankle-pitch correction (2026-08-08) -- ONLY when this
            # motion has real scrubbed strike/stand boundaries (the legacy-mode fallback above
            # spans the whole clip, not a real strike window, so there's nothing meaningful to
            # correct there) AND a resolvable per-skill kick_foot. Runs AFTER this motion's own
            # append-transition so motion_end_idx[_m] is fully finalized before slicing. See
            # kick_ankle_pitch_correction.py's own module docstring for the full rationale --
            # user-requested to ship as an automatic in-memory correction, no yaml field.
            if self._resolved_stand_start_frame is not None:
                self._maybe_correct_kick_foot_ankle_pitch(
                    motion_idx=_m, strike_start_idx=strike_start_list[-1], stand_start_idx=stand_start_list[-1]
                )

            # Head-velocity smoothing (2026-08-11, opt-in, default 0 = exact no-op) -- MUST run
            # LAST for this motion, specifically AFTER _maybe_correct_kick_foot_ankle_pitch.
            # That function unconditionally regenerates EVERY velocity array for the whole motion
            # from positions (torch.gradient over joint_pos/body_pos_w, plus so3 derivative for
            # body_ang_vel_w -- kick_ankle_pitch_correction.py lines ~212-216), so ANY velocity-
            # only edit made before it is silently overwritten. Verified live: an earlier version
            # of this call sat before the prepend loop above and was a measured no-op, producing
            # byte-identical buffers with the feature on and off.
            self._maybe_smooth_motion_head_velocities(motion_idx=_m)
        self.pre_recovery_motion_end_idx = torch.tensor(pre_recovery_list, dtype=torch.long, device=self.device)
        self.strike_start_idx = torch.tensor(strike_start_list, dtype=torch.long, device=self.device)
        self.stand_start_idx = torch.tensor(stand_start_list, dtype=torch.long, device=self.device)

        # 2. get the indexes of the root link and the tracked links
        self.ref_body_index = robot_body_names.index(self.motion_cfg.body_name_ref[0])  # int
        self.tracked_body_indexes = self._get_index_of_a_in_b(
            self.motion_cfg.body_names_to_track, robot_body_names, self.device
        )
        self._robot_body_names = robot_body_names  # stashed for _build_entry_search_table below

        # 3. get the name of the object, or indices of the object
        if self.motion.has_object:
            # cache the object_index_in_simulator
            self.object_name = "object"  # hardcoded object name
            self.object_indices_in_simulator = self._env.simulator.get_actor_indices(self.object_name, env_ids=None)

            assert self._env.simulator.get_simulator_type() == SimulatorType.ISAACSIM, (
                "Object is only supported in IsaacSim"
            )

        # 3b. get the name/indices of the kickable ball, if one was spawned into the scene.
        # Unlike `object` above, the ball is never motion-tracked — this is purely so `reset()` can
        # put it back at its configured fixed spawn position at the start of each new episode.
        # `scene.rigid_objects` is an IsaacSim-only bookkeeping dict (RigidObject instances) -- other
        # backends (MuJoCo Classic/Warp, IsaacGym) have no equivalent here, and the ball config is
        # not currently wired into their scene config at all (see
        # config_values/unified/g1/experiment.py's simulator field, which only merges
        # `ball=load_ball_config()` into the isaacsim variant) -- so `has_ball` is unconditionally
        # False for them today, same end state as before this guard existed, just without crashing.
        # 2026-07-19: MuJoCo/mjwarp gained ball support (the ball is registered as an INDIVIDUAL
        # object and reachable through MuJoCoAllRootStatesProxy), so this is no longer IsaacSim-only.
        # The two backends answer "is there a ball?" differently: IsaacSim spawns it as a separate
        # RigidObject in scene.rigid_objects, whereas MuJoCo carries it inside the same MJCF and
        # exposes a `has_ball` flag on the simulator.
        _sim = self._env.simulator
        _sim_type = _sim.get_simulator_type()
        if _sim_type == SimulatorType.ISAACSIM:
            self.has_ball = "ball" in _sim.scene.rigid_objects
        elif _sim_type == SimulatorType.MUJOCO:
            self.has_ball = bool(getattr(_sim, "has_ball", False))
        else:
            self.has_ball = False  # IsaacGym: still unwired
        if self.has_ball:
            self.ball_name = "ball"  # hardcoded ball name, matches isaacsim.py's spawn block
            # Cached once here, same pattern as object_indices_in_simulator above -- backs
            # live_ball_pos_w (increment 4's ball-fixed entry search, D2's own ball_spawn_pos_w
            # only records the last-PLACED target, not the ball's live post-physics position).
            self.ball_indices_in_simulator = self._env.simulator.get_actor_indices(self.ball_name, env_ids=None)
            ball_cfg = self._env.simulator.simulator_config.scene.ball
            self.skill_ball_configs = self.motion_cfg.skill_ball_configs  # empty -> legacy single-ball_cfg mode
            num_motions = self.motion.num_motions

            if self.skill_ball_configs:
                assert len(self.skill_ball_configs) == num_motions, (
                    f"skill_ball_configs has {len(self.skill_ball_configs)} entries but "
                    f"{num_motions} motions were loaded -- one SkillConfig per motion required."
                )
                xy = torch.tensor([[sc.x, sc.y] for sc in self.skill_ball_configs], dtype=torch.float32, device=self.device)
                rand_xy = torch.tensor(
                    [[sc.randomize_x, sc.randomize_y] for sc in self.skill_ball_configs],
                    dtype=torch.float32,
                    device=self.device,
                )
                target_xy = torch.tensor(
                    [sc.resolved_target() for sc in self.skill_ball_configs], dtype=torch.float32, device=self.device
                )
                # 2026-08-22, azimuth-aim refactor. kick_aim_enabled/kick_aim_theta_max_deg live
                # directly on each SkillConfig already (no new MotionConfig per-motion table
                # needed for these two) -- see SkillConfig.kick_aim_enabled's own docstring.
                # nominal_bearing_deg is DERIVED here, once, from each skill's (possibly
                # uncalibrated -- see resolved_nominal_bearing_deg()'s own docstring) target, not
                # read as a separate yaml field.
                kick_aim_enabled_xy = torch.tensor(
                    [sc.kick_aim_enabled for sc in self.skill_ball_configs], dtype=torch.bool, device=self.device
                )
                nominal_bearing_deg_xy = torch.tensor(
                    [sc.resolved_nominal_bearing_deg() for sc in self.skill_ball_configs],
                    dtype=torch.float32,
                    device=self.device,
                )
                kick_aim_theta_max_deg_xy = torch.tensor(
                    [
                        sc.kick_aim_theta_max_deg
                        if sc.kick_aim_theta_max_deg is not None
                        else self.motion_cfg.kick_aim_theta_max_deg
                        for sc in self.skill_ball_configs
                    ],
                    dtype=torch.float32,
                    device=self.device,
                )
            else:
                # Legacy path, broadcast to every motion -- bit-identical to the old single-ball_cfg
                # behavior for N=1 (and for the single-clip MotionLoader case, always N=1).
                xy = torch.tensor([ball_cfg.position[:2]], dtype=torch.float32, device=self.device).expand(num_motions, -1)
                rand_xy = torch.tensor(
                    [ball_cfg.position_randomization], dtype=torch.float32, device=self.device
                ).expand(num_motions, -1)
                target_xy = torch.tensor([ball_cfg.target], dtype=torch.float32, device=self.device).expand(num_motions, -1)
                kick_aim_enabled_xy = torch.tensor(
                    [ball_cfg.kick_aim_enabled], dtype=torch.bool, device=self.device
                ).expand(num_motions)
                nominal_bearing_deg_xy = torch.tensor(
                    [ball_cfg.resolved_nominal_bearing_deg()], dtype=torch.float32, device=self.device
                ).expand(num_motions)
                kick_aim_theta_max_deg_xy = torch.tensor(
                    [ball_cfg.kick_aim_theta_max_deg], dtype=torch.float32, device=self.device
                ).expand(num_motions)

            # Nominal reset state per motion: configured position (z from the scene ball's own
            # radius -- shared physical geometry, not per-skill), identity orientation (xyzw), zero
            # velocity. Per-reset uniform xy noise (position_randomization below) applied on top in
            # reset() -- 0 by default. The policy never observes the ball position directly here
            # (kick_ball_pos_b is a separate, live observation), so any nonzero range is robustness/
            # curriculum pressure for the shooting rewards, not something the policy reacts to
            # within an episode.
            z_col = torch.full((num_motions, 1), float(ball_cfg.radius), dtype=torch.float32, device=self.device)
            rest_cols = torch.tensor(
                [[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]] * num_motions, dtype=torch.float32, device=self.device
            )
            self.ball_reset_state_per_motion = torch.cat([xy, z_col, rest_cols], dim=-1)  # (num_motions, 13)
            # Backward-compat alias for replay.py / check_kick_geometry.py / utils/replay_controls.py
            # (interactive eval/replay tools -- deliberately out of N-skill scope, per design: they
            # stay single-clip, driven by configs/ball.yaml, never the stacked yaml). A VIEW (basic
            # integer indexing), not a copy, so replay_controls.py's in-place slider edits
            # (`ball_reset_state[0] = x`) keep working exactly as before -- for their single-clip
            # use case motion 0 is the only/whole ball config anyway, so this is always exactly the
            # right row, not an approximation.
            self.ball_reset_state = self.ball_reset_state_per_motion[0]
            self.ball_position_randomization_per_motion = rand_xy  # (num_motions, 2)
            # Each env's ACTUAL (post-randomization) spawn position in world frame, refreshed on
            # every reset — consumed by the shooting reward terms (managers/reward/terms/shooting.py)
            # to detect "ball has been kicked" via displacement from its own spawn point. Real
            # values are only ever written in reset() (every env is reset at least once before
            # being stepped); zero-initialized here purely so the attribute exists.
            self.ball_spawn_pos_w = torch.zeros(self.num_envs, 3, dtype=torch.float32, device=self.device)
            # Each env's OOD-spawn flag for its CURRENT attempt -- mirrors ball_spawn_pos_w's
            # per-attempt, always-fully-overwritten semantics (reset() sets this on EVERY
            # reset/clip-replay call inside this has_ball block, both OOD and non-OOD draws, so it
            # never sticks across attempts -- per-ATTEMPT like the _ShotTracker latches, not
            # per-episode). Real values are only ever written in reset(); zero-initialized here
            # purely so the attribute exists. Consumed by managers/reward/terms/shooting.py's
            # _ood_gate_multiplier to zero all 6 shooting reward terms for OOD-spawn attempts
            # (2026-08-01 reversal of the original "leave the reward alone" decision -- see
            # MotionConfig.ood_spawn_probability's own docstring). False forever whenever
            # ood_spawn_probability<=0.0 (the default) -- a verified no-op.
            self.is_ood_spawn = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            # Each env's CURRENT shot target (world-frame xy) -- FIXED at this skill's own nominal
            # env-local target for kick_aim_enabled=False (no randomization; see SkillConfig.
            # kick_aim_enabled's own docstring for why randomize_target_x/y was removed 2026-08-22),
            # or synthesized from the ball's own actual placed position + kick_aim_theta for
            # kick_aim_enabled=True -- consumed by both the shooting reward terms AND the
            # kick_target_pos_b/kick_aim_command observation (managers/observation/terms/
            # unified.py), so the policy is rewarded against exactly the target it observes.
            self.target_nominal_local_per_motion = target_xy  # (num_motions, 2)
            self.target_xy_w = torch.zeros(self.num_envs, 2, dtype=torch.float32, device=self.device)
            # 2026-08-22, azimuth-aim refactor -- see SkillConfig.kick_aim_enabled's own docstring
            # for the full mechanism. kick_aim_theta is this attempt's SAMPLED command (uniform,
            # +/- kick_aim_theta_max_per_motion), drawn once per reset/clip-replay and held for the
            # rest of the attempt (fed to the actor as a CONSTANT, unlike target_xy_w's live
            # per-tick body-frame transform) -- see managers/observation/terms/unified.py's
            # kick_aim_command. Zero-initialized here for envs whose skill has kick_aim_enabled
            # False (never resampled away from 0 for them -- see the three placement paths below).
            self.kick_aim_enabled_per_motion = kick_aim_enabled_xy  # (num_motions,) bool
            self.nominal_bearing_deg_per_motion = nominal_bearing_deg_xy  # (num_motions,) degrees
            self.kick_aim_theta_max_deg_per_motion = kick_aim_theta_max_deg_xy  # (num_motions,) degrees
            self.kick_aim_theta = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
            self.kick_aim_theta_ref_deg = float(self.motion_cfg.kick_aim_theta_ref_deg)
            self.kick_aim_nominal_distance_m = float(self.motion_cfg.kick_aim_nominal_distance_m)

        # Moved here (was right after step 2, before this has_ball block existed to matter):
        # increment 4's optional ball-geometry search columns (_build_entry_search_table, below)
        # need self.ball_reset_state_per_motion, which this block just set up -- gait-only mode
        # (mid_episode_kick_entry_ball_fixed False, the default) doesn't depend on anything from
        # this block and would have worked fine at either position; this ordering is only load-
        # bearing for the ball-geometry case.
        self._build_entry_search_table()

        # 4. get the adaptive timesteps sampler
        if self.motion_cfg.use_adaptive_timesteps_sampler:
            self.adaptive_timesteps_sampler = AdaptiveTimestepsSampler(
                self.motion.time_step_total, self.device, int(1 / (self._env.dt))
            )

        # 5. metrics
        self.metrics: dict[str, torch.Tensor] = {}

        self.init_buffers()

        # 6. visualization markers for isaacsim
        if self._env.viewer and self._env.simulator.get_simulator_type() == SimulatorType.ISAACSIM:
            self._setup_visualization_markers_for_isaacsim()

    def _synthesize_kick_aim_target_local(
        self,
        env_ids: torch.Tensor,
        env_motion_ids: torch.Tensor,
        ball_local_placed: torch.Tensor,
        target_nominal_local: torch.Tensor,
    ) -> torch.Tensor:
        """Return this attempt's LOCAL (pre-``local_xy_to_world``) target position for ``env_ids``
        -- the one shared computation called from all three ball/target placement paths (``reset``,
        ``place_ball_at_entry``, ``place_ball_at_reset_pending``). See ``SkillConfig.
        kick_aim_enabled``'s own docstring for the full mechanism this implements.

        PER-ENV selection between two modes, by that env's assigned skill's kick_aim_enabled:
          * False: ``target_nominal_local`` -- a FIXED point, no randomization at all. (The old
            independent ``randomize_target_x/y`` draw this branch used to make was removed
            2026-08-22, once every skill in this project had moved to kick_aim_enabled=True and it
            had no live consumer left -- see that field's own removal note in config_types/
            multi_skill.py/simulator.py.)
          * True: ``ball_local_placed + D * unit(nominal_bearing_deg + kick_aim_theta)`` -- the
            target is SYNTHESIZED from the ball's own actual placed position (this call's
            ``ball_local_placed`` argument, i.e. already including that same call's position noise/
            OOD draw), so ``target - ball`` is exactly ``D * unit(bearing)`` by construction,
            independent of where the ball's own noise landed. ``kick_aim_theta`` is freshly sampled
            here (uniform, +/- that skill's own kick_aim_theta_max_deg) and stored to
            ``self.kick_aim_theta[env_ids]`` -- held for the rest of the attempt, since it's fed to
            the actor as a CONSTANT command (see managers/observation/terms/unified.py's
            kick_aim_command), unlike target_xy_w's own live per-tick transform.

        No RNG draw happens at all when no loaded skill has kick_aim_enabled=True anywhere
        (self.kick_aim_enabled_per_motion.any() is False) -- the legacy branch is now a pure
        constant. Once at least one skill uses this mechanism, a per-env theta draw happens for
        every env in the batch (even ones whose own skill has it disabled), since a per-env
        conditional sample can't avoid consuming the RNG for the whole batch.
        """
        target_local_legacy = target_nominal_local
        if not bool(self.kick_aim_enabled_per_motion.any()):
            return target_local_legacy

        aim_enabled = self.kick_aim_enabled_per_motion[env_motion_ids]
        theta_max = self.kick_aim_theta_max_deg_per_motion[env_motion_ids]
        theta_sample = (torch.rand(len(env_ids), device=self.device) * 2.0 - 1.0) * theta_max
        theta = torch.where(aim_enabled, theta_sample, torch.zeros_like(theta_sample))
        self.kick_aim_theta[env_ids] = theta

        bearing_rad = torch.deg2rad(self.nominal_bearing_deg_per_motion[env_motion_ids] + theta)
        direction = torch.stack([torch.cos(bearing_rad), torch.sin(bearing_rad)], dim=-1)
        target_local_aim = ball_local_placed + self.kick_aim_nominal_distance_m * direction

        return torch.where(aim_enabled.unsqueeze(-1), target_local_aim, target_local_legacy)

    def reset(self, env_ids: torch.Tensor | None) -> None:
        """called per reset_idx, reset timesteps and robot/object poses."""
        env_ids = self._ensure_index_tensor(env_ids)
        if env_ids.numel() == 0:
            return

        # Clear any mid-episode-kick-entry re-anchor back to identity BEFORE anything else in
        # this method reads a "from motion data" accessor (root_lin_vel_w below, in particular,
        # feeds directly into this reset's own teleport velocity -- must read the raw, un-anchored
        # clip value here, exactly as before this mechanism existed). This reset() call is
        # task_mode="kick"-tagged (CommandManager filters reset_terms by mode), so it runs for
        # every kick-mode teleport reset today -- this line is live, exercised code, not dead code
        # waiting on increment 2. Cheap no-op write when the anchor was already identity (the
        # common case for every env until increment 2's mid-episode entry exists).
        self._ref_anchor_delta_quat[env_ids] = 0.0
        self._ref_anchor_delta_quat[env_ids, 3] = 1.0
        self._ref_anchor_delta_pos[env_ids] = 0.0
        # Increment 3's reference blend (see capture_ref_blend's own docstring): clear the sentinel
        # only -- the captured pos/quat/vel buffers themselves don't need zeroing, since
        # _ref_blend_ratio() returns exactly 1.0 for any env with start_step==-1, making
        # _apply_ref_blend a true `(1-1)*captured + 1*clip == clip` no-op regardless of whatever
        # (finite, never-NaN) values are still sitting in the captured buffers from a prior life.
        self._ref_blend_start_step[env_ids] = -1

        n = env_ids.numel()
        num_motions = self.motion.num_motions

        # N-skill mode: UnifiedManager (and ONLY UnifiedManager -- duck-typed, same idiom as
        # task_mode_mask elsewhere in this file, e.g. step()'s task_mode_mask("kick") check)
        # permanently assigns each kick-mode env to one motion skill for its whole life (see
        # UnifiedManager._build_task_mode_partition). When present, that fixed assignment is used
        # instead of drawing a fresh motion_id below -- every other env class (standalone WBT/
        # BallKicking) has no such attribute, so getattr returns None and behavior is untouched.
        fixed_motion_ids = None
        skill_id_buf = getattr(self._env, "skill_id", None)
        if skill_id_buf is not None:
            fixed_motion_ids = skill_id_buf[env_ids].clamp(0, num_motions - 1)

        # 0. Sample the time steps (and, for the adaptive sampler, the motion id).
        adaptive_global_idx = None
        if self.motion_cfg.use_adaptive_timesteps_sampler:
            # Match BeyondMimic behavior: update failed bins from environments
            # that terminated before this reset, then sample new phases.
            # Gate the failure-stat update on training mode so evaluation episodes
            # don't contaminate the training sampler's failure distribution
            # (the is_evaluating phase-zeroing below only affects sampling, not stats).
            if not self._env.is_evaluating:
                episode_failed = self._env.termination_manager.terminated[env_ids]
                if torch.any(episode_failed):
                    failed_at_time_step = self.time_steps[env_ids][episode_failed]
                    self.adaptive_timesteps_sampler.update_current_bin_failed_count(failed_at_time_step)
            # The sampler bins failures over the GLOBAL concatenated-motion frame
            # axis, so it must return a global frame index here. The motion id is
            # then derived from that index (NOT chosen independently), keeping the
            # failure-prioritized phase attached to the motion it was recorded on.
            adaptive_global_idx = self.adaptive_timesteps_sampler.sample_global_time_steps(n)
            phase = None
        else:
            phase = torch.rand(n, device=self.device)

        if self._env.is_evaluating:
            # Eval forces every env through the uniform/else branch below, which
            # indexes `phase`, so it must be a real zero tensor even when the
            # adaptive sampler left it as None.
            phase = torch.zeros(n, device=self.device)
            adaptive_global_idx = None  # eval starts every env at its motion's first frame

        if adaptive_global_idx is not None:
            # Map global frame index -> (motion_id, time_step). searchsorted on the
            # per-motion end indices yields the clip whose [start, end) contains it.
            motion_ids = torch.searchsorted(self.motion.motion_end_idx, adaptive_global_idx, right=True)
            motion_ids = motion_ids.clamp_(0, num_motions - 1)
            self.motion_ids[env_ids] = motion_ids
            start_idx = self.motion.motion_start_idx[motion_ids]
            end_idx = self.motion.motion_end_idx[motion_ids]
            self.time_steps[env_ids] = adaptive_global_idx.clamp(start_idx, end_idx - 1)
        else:
            # Uniform path (or eval): assign each env to a motion (fixed skill_id if the env class
            # provides one, else a fresh uniform draw), sample a phase within that motion's range.
            if fixed_motion_ids is not None:
                self.motion_ids[env_ids] = fixed_motion_ids
            else:
                self.motion_ids[env_ids] = torch.randint(0, num_motions, (n,), device=self.device)
            start_idx = self.motion.motion_start_idx[self.motion_ids[env_ids]]
            end_idx = self.motion.motion_end_idx[self.motion_ids[env_ids]]

            # 2026-08-05, ported from RoboNaldo (arXiv:2606.11092) -- see rsi_span_end_idx's and
            # critical_frame_oversample_time_steps's own docstrings (module-level functions above
            # this class) for the full rationale. Both default to their exact-no-op values (False
            # / 0.0) if unset, reproducing this block's pre-2026-08-05 behavior exactly.
            #
            # 2026-08-15, "simultaneous per-skill task configs": each falls back to the plain
            # scalar (byte-identical) unless a per-motion table was resolved in setup() -- gathered
            # by this env's OWN assigned motion (self.motion_ids[env_ids]), same indexing already
            # used for start_idx/end_idx/strike_start_idx immediately around this block.
            rsi_scope_to_authored_clip = (
                self._resolved_rsi_scope_to_authored_clip[self.motion_ids[env_ids]]
                if self._resolved_rsi_scope_to_authored_clip is not None
                else self.motion_cfg.rsi_scope_to_authored_clip
            )
            span_end_idx = rsi_span_end_idx(
                end_idx,
                self.pre_recovery_motion_end_idx[self.motion_ids[env_ids]],
                rsi_scope_to_authored_clip=rsi_scope_to_authored_clip,
            )
            motion_len = span_end_idx - start_idx
            self.time_steps[env_ids] = start_idx + (phase * (motion_len - 1).float()).long()

            oversample_prob = (
                self._resolved_critical_frame_oversampling_prob[self.motion_ids[env_ids]]
                if self._resolved_critical_frame_oversampling_prob is not None
                else self.motion_cfg.critical_frame_oversampling_prob
            )
            window = (
                self._resolved_critical_frame_sampling_window[self.motion_ids[env_ids]]
                if self._resolved_critical_frame_sampling_window is not None
                else self.motion_cfg.critical_frame_sampling_window
            )
            self.time_steps[env_ids] = critical_frame_oversample_time_steps(
                self.time_steps[env_ids],
                start_idx=start_idx,
                span_end_idx=span_end_idx,
                strike_start_idx=self.strike_start_idx[self.motion_ids[env_ids]],
                oversample_prob=oversample_prob,
                window=window,
                device=self.device,
            )

        # Handle start_at_timestep_zero_prob (reset to start of assigned motion)
        #
        # 2026-08-15, "simultaneous per-skill task configs": when a per-motion table was resolved
        # in setup(), gather each env's OWN skill's prob and use the single elementwise form below
        # for every env uniformly -- mathematically equivalent to the scalar fast path at prob>=1.0
        # (rand_vals is always in [0, 1), so rand_vals < 1.0 is always True) and at prob<=0.0
        # (always False), but it also always draws torch.rand_like, unlike the scalar fast path's
        # deliberate skip. That RNG-consumption-order difference is exactly why the scalar path
        # below is kept BYTE-IDENTICAL (not merged into this one) for the no-per-skill-override
        # case -- see _resolved_start_at_timestep_zero_prob's own setup()-time comment.
        if self._resolved_start_at_timestep_zero_prob is not None:
            prob = self._resolved_start_at_timestep_zero_prob[self.motion_ids[env_ids]]
            subset = self.time_steps[env_ids]
            rand_vals = torch.rand_like(subset, dtype=torch.float32)
            subset = torch.where(rand_vals < prob, start_idx, subset)
            self.time_steps[env_ids] = subset
        else:
            prob = self.motion_cfg.start_at_timestep_zero_prob
            if prob >= 1.0:
                self.time_steps[env_ids] = start_idx
            elif prob > 0.0:
                subset = self.time_steps[env_ids]
                rand_vals = torch.rand_like(subset, dtype=torch.float32)
                subset = torch.where(rand_vals < prob, start_idx, subset)
                self.time_steps[env_ids] = subset

        # If the motion is at the last timestep, set it to the second last timestep;
        # Otherwise, update_tasks_callback will advance the timestep to the next timestep -> out of bounds error.
        already_last_timestep_mask = self.time_steps[env_ids] >= end_idx - 1
        self.time_steps[env_ids] = torch.where(already_last_timestep_mask, end_idx - 2, self.time_steps[env_ids])

        # 1. Get the root/body poses from the motion data
        root_pos = self.root_pos_w[env_ids].clone()
        root_rot = self.root_quat_w[env_ids].clone()
        root_lin_vel = self.root_lin_vel_w[env_ids].clone()
        root_ang_vel = self.root_ang_vel_w[env_ids].clone()

        dof_pos = self.joint_pos[env_ids].clone()
        dof_vel = self.joint_vel[env_ids].clone()

        # 2. Adding noise
        # 2.1 prepare the noise scale
        dof_pos_noise = self.init_pose_cfg.dof_pos * self.init_pose_cfg.overall_noise_scale  # float
        root_pos_noise = (
            torch.tensor(
                self.init_pose_cfg.root_pos,
                device=self.device,
            )
            * self.init_pose_cfg.overall_noise_scale
        )  # (3,)
        root_rot_noise_rpy = (
            torch.tensor(
                self.init_pose_cfg.root_rot,
                device=self.device,
            )
            * self.init_pose_cfg.overall_noise_scale
        )  # (3,)
        root_vel_noise = (
            torch.tensor(
                self.init_pose_cfg.root_lin_vel,
                device=self.device,
            )
            * self.init_pose_cfg.overall_noise_scale
        )  # (3,)
        root_ang_vel_noise_rpy = (
            torch.tensor(
                self.init_pose_cfg.root_ang_vel,
                device=self.device,
            )
            * self.init_pose_cfg.overall_noise_scale
        )  # (3,)

        # 2.2 Adding noise to dof_pos, root_pos, root_vel, root_ang_vel, root_rot
        # 1.2.1 dof_pos
        target_dof_pos = (
            dof_pos + (torch.rand(dof_pos.shape, device=self.device) - 0.5) * 2 * dof_pos_noise
        )  # (num_envs, num_dofs)
        soft_joint_pos_limits = self._env.simulator.dof_pos_limits  # type: ignore[attr-defined]  # (num_dofs, 2)
        target_dof_pos = torch.clip(target_dof_pos, soft_joint_pos_limits[:, 0], soft_joint_pos_limits[:, 1])

        # 1.2.2 dof_vel no noise
        target_dof_vel = dof_vel

        # 1.2.3 root_pos
        target_root_pos = root_pos + (
            torch.rand(root_pos.shape, device=self.device) - 0.5
        ) * 2 * root_pos_noise.unsqueeze(0)  # (num_envs, 3)

        # 1.2.4 root_rot
        rand_sample_rpy = (torch.rand((len(env_ids), 3), device=self.device) - 0.5) * 2 * root_rot_noise_rpy
        orientations_delta = quat_from_euler_xyz(
            rand_sample_rpy[:, 0], rand_sample_rpy[:, 1], rand_sample_rpy[:, 2]
        )  # (num_envs, 4), xyzw
        target_root_rot = quat_mul(orientations_delta, root_rot, w_last=True)  # (num_envs, 4), xyzw

        # 1.2.5 root_lin_vel
        target_root_lin_vel = root_lin_vel + (
            torch.rand(root_lin_vel.shape, device=self.device) - 0.5
        ) * 2 * root_vel_noise.unsqueeze(0)  # (num_envs, 3)

        # 1.2.6 root_ang_vel
        target_root_ang_vel = root_ang_vel + (
            torch.rand(root_ang_vel.shape, device=self.device) - 0.5
        ) * 2 * root_ang_vel_noise_rpy.unsqueeze(0)  # (num_envs, 3)

        # 3. Set the robot states in simulator
        self._env.simulator.dof_pos[env_ids] = target_dof_pos
        self._env.simulator.dof_vel[env_ids] = target_dof_vel

        self._env.simulator.robot_root_states[env_ids, :3] = target_root_pos
        self._env.simulator.robot_root_states[env_ids, 3:7] = target_root_rot
        self._env.simulator.robot_root_states[env_ids, 7:10] = target_root_lin_vel
        self._env.simulator.robot_root_states[env_ids, 10:13] = target_root_ang_vel

        # 4. Set the object states in simulator
        if self.motion.has_object:
            obj_pos = self.object_pos_w[env_ids]
            obj_ori = self.object_quat_w[env_ids]
            obj_lin_vel = self.object_lin_vel_w[env_ids]

            # 4.2 add noise to the object states
            obj_pos_noise = torch.tensor(
                [self.init_pose_cfg.object_pos],
                device=self.device,
            )
            obj_pos_noise = obj_pos_noise * self.init_pose_cfg.overall_noise_scale  # (3,)
            target_obj_pos = obj_pos + (torch.rand(obj_pos.shape, device=self.device) - 0.5) * 2 * obj_pos_noise

            object_states = torch.cat(
                [target_obj_pos, obj_ori, obj_lin_vel, torch.zeros_like(obj_lin_vel)], dim=-1
            )  # (num_envs, 7)
            # 4.3 set the object states in simulator
            self._env.simulator.set_actor_states([self.object_name], env_ids, object_states)

        # 5. Reset the ball to its (per-env-assigned-skill's) configured spawn position (±
        # configured uniform xy noise) with zero velocity (never motion-tracked). Gathered per-env
        # from the per-motion tables built in setup() via THIS env's motion_ids (set above, in
        # section 0) -- reduces to every env reading the SAME row when skill_ball_configs was
        # empty (legacy broadcast), so this is bit-identical to the old single-ball_cfg behavior
        # whenever there's only one motion.
        if self.has_ball:
            env_motion_ids = self.motion_ids[env_ids]
            ball_states = self.ball_reset_state_per_motion[env_motion_ids].clone()
            position_randomization = self.ball_position_randomization_per_motion[env_motion_ids]
            target_nominal_local = self.target_nominal_local_per_motion[env_motion_ids]

            # ball_reset_state's position is ROBOT-local ("x = forward from the robot, y = lateral"
            # -- see SkillConfig.x/y's docstring), not a world-axis-aligned offset. The robot's own
            # kick-mode spawn pose (root_pos_w/root_quat_w below) comes directly from the reference
            # clip's own captured pelvis position/orientation, which is essentially ARBITRARY per
            # clip (whatever the original motion-capture happened to record) -- not "at env_origin,
            # facing world +X". Naively adding (x, y) to env_origins (the previous behavior) only
            # gave the intended "forward/lateral from the robot" placement for the coincidental case
            # of a clip captured at exactly zero yaw and zero pelvis offset; for every other clip it
            # silently placed the ball somewhere else entirely (verified empirically: a configured
            # (2.84, -0.46) landed at (0.77, 2.49) in the robot's real heading frame for one
            # production clip, and even further off -- including a SIGN FLIP putting the ball
            # behind the robot -- for another). Anchoring to the robot's actual spawn position and
            # rotating by its actual (yaw-only) heading makes "x meters forward" true for every
            # clip, matching MuJoCo/RoboJuDo's convention (robot spawns at a fixed, canonical
            # facing there, so heading_quat reduces to identity and this is bit-identical to the
            # old plain-add behavior in that case).
            #
            # Deliberately does NOT touch root_quat_w / the reference trajectory / the robot's own
            # spawn pose -- those drive the dominant motion-tracking reward
            # (motion_global_ref_position_error_exp et al, managers/reward/terms/wbt.py), which
            # depends on the robot matching the clip's own raw world-frame reference exactly;
            # re-deriving that trajectory to a canonical orientation is a much larger, reward-
            # critical change and out of scope here -- this only corrects WHERE the ball/target
            # land relative to that unchanged robot pose. See local_xy_to_world's own docstring.
            #
            # Per-reset uniform xy randomization (configs/ball.yaml randomize_x/randomize_y, or this
            # env's assigned skill's own randomize_x/y — 0 by default, exact previous behavior).
            # Fires on env resets AND mid-episode clip replays (both funnel through this method), so
            # every kick attempt gets a fresh draw. Per-env now (each env's own skill's range), not
            # a single shared range. Randomization is drawn in the robot's own local frame too, same
            # as the nominal position, so it stays "forward/lateral" regardless of clip orientation.
            #
            # Anchored to target_root_pos/target_root_rot (section 3 above), the robot's ACTUAL,
            # POST-NOISE simulated pose -- not the pre-noise root_pos_w/root_quat_w
            # local_xy_to_world would otherwise default to. See that method's own docstring for why
            # this distinction is load-bearing (NoiseToInitialPoseConfig.root_rot is a real,
            # substantial per-reset yaw perturbation here, not negligible).
            noise, is_ood = draw_position_noise_with_ood(
                position_randomization,
                ood_prob=self.motion_cfg.ood_spawn_probability,
                ood_multiplier=self.motion_cfg.ood_region_multiplier,
                device=self.device,
            )
            ball_local_placed = ball_states[:, :2] + noise
            ball_states[:, :2] = self.local_xy_to_world(
                ball_local_placed, env_ids, robot_pos_w=target_root_pos, robot_quat_w=target_root_rot
            )
            # z (rest height on the ball's own radius) is env-origin-relative, not robot-position-
            # or heading-relative -- untouched by the rotation/robot-anchoring above.
            ball_states[:, 2] = ball_states[:, 2] + self._env.simulator.scene.env_origins[env_ids, 2]
            self.ball_spawn_pos_w[env_ids] = ball_states[:, :3]
            # Full overwrite (not |=): this attempt's OOD-spawn flag, refreshed on every reset()
            # call including mid-episode clip restarts -- per-ATTEMPT, like the _ShotTracker
            # latches, not per-episode. Consumed by managers/reward/terms/shooting.py's
            # _ood_gate_multiplier to zero all 6 shooting reward terms for OOD-spawn attempts.
            self.is_ood_spawn[env_ids] = is_ood
            self._env.simulator.set_actor_states([self.ball_name], env_ids, ball_states)
            # This env's assigned skill's target -- each kick attempt gets its own, observed by the
            # policy via kick_target_pos_b/kick_aim_command. See _synthesize_kick_aim_target_local's
            # own docstring for the per-env legacy-vs-aim selection (kick_aim_enabled).
            target_local = self._synthesize_kick_aim_target_local(
                env_ids, env_motion_ids, ball_local_placed, target_nominal_local
            )
            self.target_xy_w[env_ids] = self.local_xy_to_world(
                target_local, env_ids, robot_pos_w=target_root_pos, robot_quat_w=target_root_rot
            )

    def step(self) -> None:
        """called in _update_tasks_callback of the environment. (after compute_reward, before compute_observations)"""
        # 0. update time steps, all motion joint/body poses are updated automatically with the time steps.
        advance_mask = torch.ones_like(self.time_steps, dtype=torch.bool)

        # UnifiedManager only: freeze the reference entirely for envs currently running the
        # locomotion task this episode (no-op — hasattr is False — for every other env class, so
        # this doesn't change behavior for plain WBT/BallKicking experiments). This also means
        # `ended_env_ids` below naturally never includes locomotion-mode envs, so the mid-step
        # `self.reset(...)` + ball-teleport a few lines down never fires for them either.
        if hasattr(self._env, "task_mode_mask"):
            advance_mask &= self._env.task_mode_mask("kick")

        # Handle freeze_at_timestep_zero_prob: for envs at their motion's start, randomly decide whether to advance
        freeze_prob = self.motion_cfg.freeze_at_timestep_zero_prob
        if freeze_prob > 0.0:
            zero_mask = self.time_steps == self.motion.motion_start_idx[self.motion_ids]
            if zero_mask.any():
                rand_vals = torch.rand(self.num_envs, device=self.device)
                freeze_mask = (rand_vals < freeze_prob) & zero_mask
                advance_mask = advance_mask & ~freeze_mask

        self.time_steps += advance_mask.long()

        # BeyondMimic-style behavior: when the clip ends, resample motion and
        # reset robot/object state without terminating the whole episode.
        per_motion_end = self.motion.motion_end_idx[self.motion_ids]
        ended_env_ids = torch.where(self.time_steps >= per_motion_end)[0]
        if ended_env_ids.numel() > 0:
            self.reset(ended_env_ids)
            # Flush the mutated root/dof state into the simulator so that
            # rigid-body positions are up-to-date for downstream consumers
            # (termination checks, observations, rewards).
            sim = self._env.simulator
            sim.set_actor_root_state_tensor_robots(ended_env_ids, sim.robot_root_states)
            sim.set_dof_state_tensor_robots(ended_env_ids, sim.dof_state)  # type: ignore[attr-defined]
            sim.refresh_sim_tensors()

        # 1. update body_pos_relative_w and body_quat_relative_w
        # definition of body_pos/quat_relative_w:
        # If I take this motion data and adapt it to where my robot currently is
        # (accounting for position(x, y) offset and yaw difference of a reference body),
        # what should each body part's target pose be?

        ## 1.0 get the reference body poses

        # Issue (This is a isaacgym only issue.):
        # ------------------------------------------------------------
        # In isaacgym, immediately after reset (self._env.episode_length_buf == 0), calling
        # simulator.set_actor_root_state_tensor and simulator.set_dof_state_tensor will reset
        # the robot_root_pos_w and robot_root_quat_w successfully.
        # However, the robot_body_pos_w and robot_body_quat_w are not updated successfully,
        # (since kinematic forward has not been applied yet).
        # Therefore, using robot_ref_pos_w and robot_ref_quat_w as reference body poses is not resetted correctly.

        # Solution:
        # ------------------------------------------------------------
        # if episode_length_buf == 0, use robot_root_pos_w and robot_root_quat_w as reference body.
        # else, use configured reference body as reference body.
        use_root = (self._env.episode_length_buf == 0).unsqueeze(1).float()

        ref_pos_w = self.root_pos_w * use_root + self.ref_pos_w * (1 - use_root)
        ref_quat_w = self.root_quat_w * use_root + self.ref_quat_w * (1 - use_root)
        robot_ref_pos_w = self.robot_root_pos_w * use_root + self.robot_ref_pos_w * (1 - use_root)
        robot_ref_quat_w = self.robot_root_quat_w * use_root + self.robot_ref_quat_w * (1 - use_root)

        ## 1.1 repeat to match the number of body parts
        ref_pos_w_repeat = ref_pos_w[:, None, :].repeat(1, len(self.motion_cfg.body_names_to_track), 1)  # type: ignore[arg-type]
        ref_quat_w_repeat = ref_quat_w[:, None, :].repeat(1, len(self.motion_cfg.body_names_to_track), 1)  # type: ignore[arg-type]
        robot_ref_pos_w_repeat = robot_ref_pos_w[:, None, :].repeat(1, len(self.motion_cfg.body_names_to_track), 1)  # type: ignore[arg-type]
        robot_ref_quat_w_repeat = robot_ref_quat_w[:, None, :].repeat(1, len(self.motion_cfg.body_names_to_track), 1)  # type: ignore[arg-type]

        ## 1.2 compute the relative body poses
        delta_quat_w = yaw_quat(
            quat_mul(robot_ref_quat_w_repeat, quat_inverse(ref_quat_w_repeat, w_last=True), w_last=True), w_last=True
        )
        ### 1.2.1 body_quat_relative_w
        self.body_quat_relative_w = quat_mul(delta_quat_w, self.body_quat_w, w_last=True)
        ### 1.2.2 body_pos_relative_w
        delta_pos_w_height = ref_pos_w_repeat - robot_ref_pos_w_repeat
        delta_pos_w_height[..., :2] = 0.0  # adjusting for height differences
        self.body_pos_relative_w = (
            robot_ref_pos_w_repeat
            + delta_pos_w_height
            + quat_apply(delta_quat_w, self.body_pos_w - ref_pos_w_repeat, w_last=True)
        )

        ### 1.3 update the adaptive timesteps sampler (training only — eval episodes
        ### must not decay/fold failure stats into the training sampler).
        if self.motion_cfg.use_adaptive_timesteps_sampler and not self._env.is_evaluating:
            self.adaptive_timesteps_sampler.update_bin_failed_count()

    @property
    def command(self) -> torch.Tensor:
        return torch.cat([self.joint_pos, self.joint_vel], dim=1)

    #########################################################################################
    ## Robot from motion data
    #########################################################################################
    @property
    def joint_pos(self) -> torch.Tensor:
        return self.motion.joint_pos[self.time_steps]

    @property
    def joint_vel(self) -> torch.Tensor:
        return self.motion.joint_vel[self.time_steps]

    @property
    def body_pos_w(self) -> torch.Tensor:
        anchored = self._apply_ref_anchor_pos(
            self.motion.body_pos_w[self.time_steps][:, self.tracked_body_indexes]
            + self._env.simulator.scene.env_origins[:, None, :]
        )
        return self._apply_ref_blend(anchored, self._ref_blend_captured_pos, is_quat=False)

    @property
    def body_quat_w(self) -> torch.Tensor:
        anchored = self._apply_ref_anchor_rot(self.motion.body_quat_w[self.time_steps][:, self.tracked_body_indexes])
        return self._apply_ref_blend(anchored, self._ref_blend_captured_quat, is_quat=True)

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        anchored = self._apply_ref_anchor_vec(self.motion.body_lin_vel_w[self.time_steps][:, self.tracked_body_indexes])
        return self._apply_ref_blend(anchored, self._ref_blend_captured_lin_vel, is_quat=False)

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        anchored = self._apply_ref_anchor_vec(self.motion.body_ang_vel_w[self.time_steps][:, self.tracked_body_indexes])
        return self._apply_ref_blend(anchored, self._ref_blend_captured_ang_vel, is_quat=False)

    @property
    def ref_pos_w(self) -> torch.Tensor:
        return self._apply_ref_anchor_pos(
            self.motion.body_pos_w[self.time_steps, self.ref_body_index] + self._env.simulator.scene.env_origins
        )

    @property
    def ref_quat_w(self) -> torch.Tensor:
        return self._apply_ref_anchor_rot(self.motion.body_quat_w[self.time_steps, self.ref_body_index])

    @property
    def ref_lin_vel_w(self) -> torch.Tensor:
        return self._apply_ref_anchor_vec(self.motion.body_lin_vel_w[self.time_steps, self.ref_body_index])

    @property
    def ref_ang_vel_w(self) -> torch.Tensor:
        return self._apply_ref_anchor_vec(self.motion.body_ang_vel_w[self.time_steps, self.ref_body_index])

    @property
    def root_pos_w(self) -> torch.Tensor:
        return self._apply_ref_anchor_pos(
            self.motion.body_pos_w[self.time_steps, 0] + self._env.simulator.scene.env_origins
        )

    @property
    def root_quat_w(self) -> torch.Tensor:
        return self._apply_ref_anchor_rot(self.motion.body_quat_w[self.time_steps, 0])

    def local_xy_to_world(
        self,
        local_xy: torch.Tensor,
        env_ids: torch.Tensor,
        *,
        robot_pos_w: torch.Tensor | None = None,
        robot_quat_w: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Convert a ROBOT-LOCAL (forward, lateral) offset -- e.g. SkillConfig.x/y, whose
        docstring promises "forward from the robot" -- to a world-frame (x, y), for the given
        envs. Anchored to the robot's ACTUAL spawn position (env_origin PLUS the reference clip's
        own captured pelvis offset -- not env_origin alone) and rotated by its ACTUAL (yaw-only)
        heading (which comes directly from the clip's own captured orientation and is essentially
        arbitrary per clip -- not necessarily facing world +X). This is the single source of truth
        for that transform: reset()'s ball/target placement below, replay_controls.py's
        BallPositionWindow (live slider), and replay.py's restart handler all call this rather
        than re-deriving it, so they can't silently drift apart. Reduces to a bit-identical plain
        env_origin add when the robot's heading happens to be exactly zero-yaw at exactly
        env_origin (MuJoCo/RoboJuDo's convention).

        robot_pos_w/robot_quat_w default to root_pos_w/root_quat_w (the clip's own raw,
        noise-free captured pose) -- correct for replay.py/BallPositionWindow, which set the
        robot's displayed pose directly from these with no randomization (see
        WholeBodyTrackingManager.step_visualize_motion). They are NOT correct during a real
        env.reset()-driven episode reset: this same method's caller in reset() below applies
        per-episode orientation noise (NoiseToInitialPoseConfig.root_rot) to the robot's ACTUAL
        simulated pose before the ball/target get placed, and MUST pass that noisy, final pose
        explicitly here -- reusing the pre-noise root_pos_w/root_quat_w default there would place
        the ball/target relative to a heading the robot doesn't actually end up at (this exact gap
        was caught live: a configured (2.84, -0.46) still landed meters off after rotating by the
        pre-noise heading alone, because overall_noise_scale=1.0 / root_rot=[0.1,0.1,0.2] rad is a
        real, substantial per-reset yaw perturbation, not a negligible one).

        robot_quat_w (and root_quat_w) is XYZW, not wxyz -- MotionLoader converts the npz's raw
        wxyz on load (see this file's `# Change to xyzw` comment at load time). A stale comment
        elsewhere in this file mislabels root_quat_w as wxyz; don't trust it -- w_last=True below
        is deliberate and was verified against the ACTUAL storage conversion, not that comment."""
        if robot_pos_w is None:
            robot_pos_w = self.root_pos_w[env_ids]
        if robot_quat_w is None:
            robot_quat_w = self.root_quat_w[env_ids]
        heading_quat = yaw_quat(robot_quat_w, w_last=True)
        local_xyz = torch.cat([local_xy, torch.zeros_like(local_xy[:, :1])], dim=-1)
        return quat_rotate(heading_quat, local_xyz, w_last=True)[:, :2] + robot_pos_w[:, :2]

    @property
    def root_lin_vel_w(self) -> torch.Tensor:
        return self._apply_ref_anchor_vec(self.motion.body_lin_vel_w[self.time_steps, 0])

    @property
    def root_ang_vel_w(self) -> torch.Tensor:
        return self._apply_ref_anchor_vec(self.motion.body_ang_vel_w[self.time_steps, 0])

    #########################################################################################
    ## Robot from simulator
    #########################################################################################
    @property
    def robot_joint_pos(self) -> torch.Tensor:
        return self._env.simulator.dof_pos  # (num_envs, num_dofs)

    @property
    def robot_joint_vel(self) -> torch.Tensor:
        return self._env.simulator.dof_vel

    @property
    def robot_body_pos_w(self) -> torch.Tensor:
        return self._env.simulator._rigid_body_pos[:, self.tracked_body_indexes, :]

    @property
    def robot_body_quat_w(self) -> torch.Tensor:
        return self._env.simulator._rigid_body_rot[:, self.tracked_body_indexes, :]  # xyzw

    @property
    def robot_body_lin_vel_w(self) -> torch.Tensor:
        return self._env.simulator._rigid_body_vel[:, self.tracked_body_indexes, :]

    @property
    def robot_body_ang_vel_w(self) -> torch.Tensor:
        return self._env.simulator._rigid_body_ang_vel[:, self.tracked_body_indexes, :]

    @property
    def robot_root_pos_w(self) -> torch.Tensor:
        return self._env.simulator.robot_root_states[:, :3]  # type: ignore[attr-defined]

    @property
    def robot_root_quat_w(self) -> torch.Tensor:
        return self._env.simulator.robot_root_states[:, 3:7]  # type: ignore[attr-defined]

    @property
    def robot_root_lin_vel_w(self) -> torch.Tensor:
        return self._env.simulator.robot_root_states[:, 7:10]  # type: ignore[attr-defined]

    @property
    def robot_root_ang_vel_w(self) -> torch.Tensor:
        return self._env.simulator.robot_root_states[:, 10:13]  # type: ignore[attr-defined]

    @property
    def robot_ref_pos_w(self) -> torch.Tensor:
        return self._env.simulator._rigid_body_pos[:, self.ref_body_index, :]

    @property
    def robot_ref_quat_w(self) -> torch.Tensor:
        return self._env.simulator._rigid_body_rot[:, self.ref_body_index, :]  # xyzw

    @property
    def robot_ref_lin_vel_w(self) -> torch.Tensor:
        return self._env.simulator._rigid_body_vel[:, self.ref_body_index, :]

    @property
    def robot_ref_ang_vel_w(self) -> torch.Tensor:
        return self._env.simulator._rigid_body_ang_vel[:, self.ref_body_index, :]

    #########################################################################################
    ## Object from motion data
    #########################################################################################
    @property
    def object_pos_w(self) -> torch.Tensor:
        # Applies env origins, but ideally we should rely on the simulator
        return self.motion.object_pos_w[self.time_steps] + self._env.simulator.scene.env_origins

    @property
    def object_quat_w(self) -> torch.Tensor:
        return self.motion.object_quat_w[self.time_steps]

    @property
    def object_lin_vel_w(self) -> torch.Tensor:
        return self.motion.object_lin_vel_w[self.time_steps]

    #########################################################################################
    ## Object from simulator
    #########################################################################################
    @property
    def simulator_object_pos_w(self) -> torch.Tensor:
        return self._env.simulator.all_root_states[self.object_indices_in_simulator][:, :3]

    @property
    def simulator_object_quat_w(self) -> torch.Tensor:
        return self._env.simulator.all_root_states[self.object_indices_in_simulator][:, 3:7]

    @property
    def simulator_object_lin_vel_w(self) -> torch.Tensor:
        return self._env.simulator.all_root_states[self.object_indices_in_simulator][:, 7:10]

    @property
    def live_ball_pos_w(self) -> torch.Tensor:
        """The kickable ball's LIVE, post-physics world position -- distinct from
        ball_spawn_pos_w (the last position it was explicitly PLACED at; stale the instant physics
        moves the ball, e.g. rolling after a kick). Only meaningful when has_ball is True (no
        guard here -- same no-guard-read precedent ball_rel_at_frame/place_ball_at_entry already
        follow: callers check has_ball first)."""
        return self._env.simulator.all_root_states[self.ball_indices_in_simulator][:, :3]

    #########################################################################################
    ## Methods that does not fit into setup/step/reset pattern
    #########################################################################################

    @property
    def in_kicking_phase(self) -> torch.Tensor:
        """Bool [num_envs]: True from clip start through the end of "swinging mode" (locomotion-
        approach + strike, modes 1+2) -- False once past stand_start_idx (post-kick-standing mode,
        mode 3, and the synthetic recovery/hold tail). Renamed from in_swing_phase (2026-07-31):
        the boundary moved from pre_recovery_motion_end_idx (end of the WHOLE authored clip) to
        stand_start_idx (end of mode 2 only) -- see stand_start_idx's own capture site in setup().
        Consumed by _kick_recovery_gate (managers/reward/terms/locomotion.py),
        _recovery_phase_tracking_multiplier (managers/reward/terms/kick_scale_wrappers.py), and
        BadTracking's opt-in swing-only/swing-widened knobs (managers/termination/terms/wbt.py)."""
        return self.time_steps < self.stand_start_idx[self.motion_ids]

    @property
    def in_strike_phase(self) -> torch.Tensor:
        """Bool [num_envs]: True only during "swinging mode" (mode 2, the actual leg-swing
        strike), i.e. strictly narrower than in_kicking_phase (modes 1+2). Bounded by
        strike_start_idx (locomotion->swing transition) and stand_start_idx (swing->standing
        transition, shared with in_kicking_phase's own end boundary). Legacy/single-clip mode (no
        scrubbed boundaries configured): collapses onto in_kicking_phase's full window
        (strike_start_idx == this motion's own earliest time_steps), a verified no-op. Consumed by
        managers/reward/terms/shooting.py to gate all 6 shooting reward terms strictly to the
        strike itself -- excludes both the locomotion approach (mode 1) and post-kick-standing/
        recovery/hold (mode 3 onward)."""
        return (self.time_steps >= self.strike_start_idx[self.motion_ids]) & (
            self.time_steps < self.stand_start_idx[self.motion_ids]
        )

    def init_buffers(self):
        self.time_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.motion_ids = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.body_pos_relative_w = torch.zeros(
            self.num_envs, len(self.motion_cfg.body_names_to_track), 3, device=self.device
        )  # type: ignore[arg-type]
        self.body_quat_relative_w = torch.zeros(
            self.num_envs, len(self.motion_cfg.body_names_to_track), 4, device=self.device
        )  # type: ignore[arg-type]
        self.body_quat_relative_w[:, :, 0] = 1.0

        # Mid-episode kick-entry re-anchor (2026-08-13, locomotion->kick handoff, increment 1 of
        # 4 -- see MultiSkillConfig.mid_episode_kick_entry_prob's own docstring for the full
        # mechanism). Per-env rigid transform (yaw-only rotation + translation) applied to every
        # "from motion data" accessor below, so a reference entered mid-episode tracks a copy of
        # the clip anchored to the robot's OWN actual pose at the entry tick, not the clip's raw
        # captured world pose -- see enter_at_frame's own docstring for why this is required.
        # Identity by construction (0 translation, identity rotation): every accessor's transform
        # is a no-op at these defaults, verified by _ref_anchor_active below short-circuiting the
        # transform entirely rather than computing an identity rotation every tick.
        self._ref_anchor_delta_quat = torch.zeros(self.num_envs, 4, device=self.device)
        self._ref_anchor_delta_quat[:, 3] = 1.0  # xyzw identity
        self._ref_anchor_delta_pos = torch.zeros(self.num_envs, 3, device=self.device)
        # Sticky: sets True the first time ANY env is ever anchored (enter_at_frame) and never
        # reverts, even after that env resets back to identity -- trades a per-tick no-op branch
        # (cheap: quat_apply/quat_mul on an identity transform) for zero risk of the fast-path
        # incorrectly staying "on" (never happens, sticky only ever goes False->True) or
        # incorrectly reporting "off" while some env's anchor is genuinely non-identity (would
        # happen with a naive `any(anchor != identity)` check recomputed every read). Nothing in
        # this codebase calls enter_at_frame yet, so this stays False -- and every property below
        # takes the exact pre-existing code path -- for the entire lifetime of any run that
        # doesn't use the (not yet built) increment-2 mid-episode-entry mechanism.
        self._ref_anchor_active = False

        # Increment 3's reference-side blend (2026-08-13, locomotion->kick handoff --
        # pre_kick_reference_blend_steps, see MultiSkillConfig's own docstring for the full
        # rationale). Increment 1's anchor above makes the ROOT/ref-body target match the robot's
        # actual pose exactly at a mid-episode entry, but the RELATIVE-body targets (each tracked
        # body's own position/orientation/velocity) still jump straight to the entered clip
        # frame's raw authored values on the very same tick. capture_ref_blend snapshots the
        # robot's OWN live tracked-body pose/velocity at that instant into the 4 buffers below;
        # _apply_ref_blend (used by body_pos_w/body_quat_w/body_lin_vel_w/body_ang_vel_w) then
        # blends FROM that captured snapshot TOWARD the raw clip value over
        # _ref_blend_window_steps ticks, instead of serving the raw clip value immediately.
        # Zero-valued/identity by construction; _ref_blend_active (sticky, mirrors
        # _ref_anchor_active) short-circuits every accessor to the exact pre-existing code path
        # while nothing has ever called capture_ref_blend.
        num_tracked_bodies = len(self.motion_cfg.body_names_to_track)
        self._ref_blend_captured_pos = torch.zeros(self.num_envs, num_tracked_bodies, 3, device=self.device)
        self._ref_blend_captured_quat = torch.zeros(self.num_envs, num_tracked_bodies, 4, device=self.device)
        self._ref_blend_captured_quat[:, :, 3] = 1.0  # xyzw identity
        self._ref_blend_captured_lin_vel = torch.zeros(self.num_envs, num_tracked_bodies, 3, device=self.device)
        self._ref_blend_captured_ang_vel = torch.zeros(self.num_envs, num_tracked_bodies, 3, device=self.device)
        # Sentinel -1 = not currently blending (either never captured, or past the window --
        # ratio() saturates at 1.0 in both cases, so nothing distinguishes "just finished" from
        # "never started" and nothing needs to). Set to episode_length_buf's value at the capture
        # tick; _ref_blend_window_steps (below) is the per-env window length passed into that same
        # call, NOT a global -- stored per-env so a future per-skill window is a pure config change,
        # not a new buffer.
        self._ref_blend_start_step = torch.full((self.num_envs,), -1, dtype=torch.long, device=self.device)
        self._ref_blend_window_steps = torch.zeros(self.num_envs, device=self.device)
        self._ref_blend_active = False

        if self.motion_cfg.use_adaptive_timesteps_sampler:
            self.adaptive_timesteps_sampler.init_buffers()

    def _ref_anchor_broadcast(self, tensor: torch.Tensor, shape_ref: torch.Tensor) -> torch.Tensor:
        """Expand a per-env (num_envs, D) anchor tensor to match shape_ref's leading dims -- either
        (num_envs, D) unchanged, or (num_envs, num_bodies, D) via repeat. Mirrors step()'s own
        ref_quat_w_repeat/robot_ref_pos_w_repeat broadcasting idiom ("1.1 repeat to match the
        number of body parts" above) rather than relying on torch broadcasting, because
        quat_apply/quat_mul require exact shape equality (see utils/rotations.py's own
        `a.reshape(-1, 4)` / `assert a.shape == b.shape`), not implicit broadcast."""
        if shape_ref.dim() == 2:
            return tensor
        return tensor[:, None, :].repeat(1, shape_ref.shape[1], 1)

    def _apply_ref_anchor_pos(self, raw_pos_w: torch.Tensor) -> torch.Tensor:
        """Rotate-then-translate a world position (or per-body batch of them) by the mid-episode
        re-anchor transform. True no-op while _ref_anchor_active is False -- returns raw_pos_w
        unchanged, not even an identity-transform computation."""
        if not self._ref_anchor_active:
            return raw_pos_w
        quat = self._ref_anchor_broadcast(self._ref_anchor_delta_quat, raw_pos_w)
        pos = self._ref_anchor_broadcast(self._ref_anchor_delta_pos, raw_pos_w)
        return quat_apply(quat, raw_pos_w, w_last=True) + pos

    def _apply_ref_anchor_rot(self, raw_quat_w: torch.Tensor) -> torch.Tensor:
        """Compose the mid-episode re-anchor rotation onto a world orientation (or per-body batch).
        True no-op while _ref_anchor_active is False."""
        if not self._ref_anchor_active:
            return raw_quat_w
        quat = self._ref_anchor_broadcast(self._ref_anchor_delta_quat, raw_quat_w)
        return quat_mul(quat, raw_quat_w, w_last=True)

    def _apply_ref_anchor_vec(self, raw_vec_w: torch.Tensor) -> torch.Tensor:
        """Rotate (no translation) a world-frame direction vector -- linear/angular velocity --
        by the mid-episode re-anchor transform. True no-op while _ref_anchor_active is False."""
        if not self._ref_anchor_active:
            return raw_vec_w
        quat = self._ref_anchor_broadcast(self._ref_anchor_delta_quat, raw_vec_w)
        return quat_apply(quat, raw_vec_w, w_last=True)

    def _ref_blend_ratio(self) -> torch.Tensor:
        """[num_envs] float in [0, 1]: 0 = fully at the captured actual pose, 1 = fully at the raw
        clip target, linear in ticks since capture_ref_blend was called. 1.0 (NOT 0.0) for any env
        with start_step==-1 (never captured this life, or blending disabled entirely) -- makes
        _apply_ref_blend an exact `clip_val` pass-through for those envs regardless of whatever
        (finite) values are sitting in the captured buffers."""
        is_blending = self._ref_blend_start_step >= 0
        elapsed = (self._env.episode_length_buf - self._ref_blend_start_step).clamp(min=0).float()
        window = self._ref_blend_window_steps.clamp(min=1e-6)
        ratio = (elapsed / window).clamp(max=1.0)
        return torch.where(is_blending, ratio, torch.ones_like(ratio))

    def _apply_ref_blend(self, clip_val: torch.Tensor, captured: torch.Tensor, *, is_quat: bool) -> torch.Tensor:
        """Blend a per-tracked-body accessor's raw (already anchor-transformed) clip value toward
        `captured` (the robot's own real pose/velocity snapshotted at entry by capture_ref_blend),
        by _ref_blend_ratio() -- linear interpolation for positions/velocities, slerp for
        orientations. True no-op while _ref_blend_active is False -- returns clip_val unchanged,
        not even a ratio=1.0 computation."""
        if not self._ref_blend_active:
            return clip_val
        ratio = self._ref_blend_ratio()
        if is_quat:
            num_bodies = clip_val.shape[1]
            ratio_flat = ratio.repeat_interleave(num_bodies).unsqueeze(-1)
            blended = slerp(captured.reshape(-1, 4), clip_val.reshape(-1, 4), ratio_flat)
            return blended.view(clip_val.shape)
        ratio_b = ratio.view(-1, 1, 1)
        return captured * (1.0 - ratio_b) + clip_val * ratio_b

    def capture_ref_blend(self, env_ids: torch.Tensor, blend_steps: torch.Tensor) -> None:
        """Increment 3 of the locomotion->kick handoff (pre_kick_reference_blend_steps; see that
        field's own docstring in MultiSkillConfig for the full rationale): snapshot env_ids' OWN
        live tracked-body pose/velocity as the blend-FROM anchor, and start the ramp toward the
        raw clip target over `blend_steps` ticks (per-env tensor, broadcastable to a single global
        value today -- stored per-env so a future per-skill window is a config change, not a new
        buffer). Called from UnifiedManager._enter_kick alongside enter_at_frame/
        place_ball_at_entry, ONLY when pre_kick_reference_blend_steps > 0.0 -- callers must not
        call this at the field's 0.0 off-default, since doing so would set _ref_blend_active True
        and pay the (harmless but pointless) per-tick blend-ratio computation for the rest of the
        run.

        Reads robot_body_pos_w/robot_body_quat_w/robot_body_lin_vel_w/robot_body_ang_vel_w -- the
        LIVE simulator state for env_ids' tracked bodies, exactly the quantities body_pos_w/
        body_quat_w/body_lin_vel_w/body_ang_vel_w are being compared against by the 4 reward terms
        this exists to smooth (motion_relative_body_position_error_exp/_orientation_error_exp,
        motion_global_body_lin_vel/_ang_vel) -- so the blend starts at literally zero error
        against the robot's real current state, same "give the target a smooth glide-in instead
        of a cliff" rationale as increment 1's own root/ref-body anchor."""
        if env_ids.numel() == 0:
            return
        self._ref_blend_captured_pos[env_ids] = self.robot_body_pos_w[env_ids]
        self._ref_blend_captured_quat[env_ids] = self.robot_body_quat_w[env_ids]
        self._ref_blend_captured_lin_vel[env_ids] = self.robot_body_lin_vel_w[env_ids]
        self._ref_blend_captured_ang_vel[env_ids] = self.robot_body_ang_vel_w[env_ids]
        self._ref_blend_start_step[env_ids] = self._env.episode_length_buf[env_ids].clone()
        self._ref_blend_window_steps[env_ids] = blend_steps
        self._ref_blend_active = True

    def _build_entry_search_table(self) -> None:
        """Precomputed per-skill approach-window gait feature table for the locomotion->kick
        mid-episode entry-point search (increment 2 of
        https://claude.ai/code/artifact/53c1da51-d841-4979-8bf8-efd5ea652e06, decisions D1-D8;
        memory locomotion_to_kick_handoff_design_settled.md). Built once here at setup() from
        each motion's OWN [motion_start_idx, strike_start_idx) window -- exactly the walking-
        approach content a mid-episode entry needs to be compared against, never the strike
        itself.

        5 features per frame, identical to the offline diagnostic already validated against all 4
        available clips before this was built: reference-body (ref_body_index, this project's g1
        config: torso_link) height, body-frame vx/vy (world velocity rotated into the ref body's
        own yaw-only heading -- matching enter_at_frame's own yaw-only convention), yaw rate, and
        a continuous stance-foot indicator (left ankle z minus right ankle z). Reads
        self.motion.body_pos_w/body_quat_w/body_lin_vel_w/body_ang_vel_w directly (already in
        this project's full robot-body order, see MotionLoader's own _body_indexes construction)
        -- NOT the outer MotionCommand.body_pos_w property, which is per-env live state, not
        per-motion authored content.

        Increment 4 (mid_episode_kick_entry_ball_fixed, D2b): when True (AND has_ball -- a
        defensive AND, not just an assumption, since a misconfigured non-ball setup must still
        produce a well-formed table rather than a width mismatch against _live_entry_features), 2
        more columns are appended -- the CLIP's own implied robot-to-ball offset at each approach-
        window frame (ball_rel_at_frame, root-anchored, same convention reset()'s own ball
        placement already uses), pooled-z-scored together with the 5 gait columns exactly like
        them, no separate weighting. The resolved decision (_entry_search_use_ball_geometry) is
        stored so _live_entry_features can't independently disagree about the table's width.
        False (the default) leaves this at exactly 5 columns, byte-identical to before this
        increment existed.

        Padded to the longest skill's approach length; a per-skill validity mask keeps shorter
        skills' padding from ever being selected (masked to +inf distance in search_entry_point).
        Normalization stats are POOLED across every skill's real frames -- per-skill
        normalization would make residuals incomparable across skills and silently break the
        skill-constrained search (D2's own explicit warning: never substitute a different,
        easier-to-enter skill under a caller's request)."""
        left_idx, right_idx = self._get_index_of_a_in_b(
            ["left_ankle_roll_link", "right_ankle_roll_link"], self._robot_body_names, self.device
        )
        num_motions = self.motion.num_motions
        approach_lens = (self.strike_start_idx - self.motion.motion_start_idx).clamp(min=1)
        max_len = int(approach_lens.max().item())

        self._entry_search_use_ball_geometry = bool(self.motion_cfg.mid_episode_kick_entry_ball_fixed) and getattr(
            self, "has_ball", False
        )
        num_feats = 7 if self._entry_search_use_ball_geometry else 5

        table = torch.zeros(num_motions, max_len, num_feats, device=self.device)
        valid = torch.zeros(num_motions, max_len, dtype=torch.bool, device=self.device)
        all_feats = []

        for m in range(num_motions):
            start = int(self.motion.motion_start_idx[m].item())
            length = int(approach_lens[m].item())
            frames = torch.arange(start, start + length, device=self.device)

            pos = self.motion.body_pos_w[frames]
            quat = self.motion.body_quat_w[frames]
            lin_vel = self.motion.body_lin_vel_w[frames]
            ang_vel = self.motion.body_ang_vel_w[frames]

            ref_pos = pos[:, self.ref_body_index]
            ref_quat = quat[:, self.ref_body_index]
            ref_lin_vel_w = lin_vel[:, self.ref_body_index]
            ref_ang_vel_w = ang_vel[:, self.ref_body_index]

            yaw_only = yaw_quat(ref_quat, w_last=True)
            body_lin_vel_local = quat_apply(quat_inverse(yaw_only, w_last=True), ref_lin_vel_w, w_last=True)

            gait_feats = [
                ref_pos[:, 2],
                body_lin_vel_local[:, 0],
                body_lin_vel_local[:, 1],
                ref_ang_vel_w[:, 2],
                pos[:, left_idx, 2] - pos[:, right_idx, 2],
            ]
            if self._entry_search_use_ball_geometry:
                ball_rel = self.ball_rel_at_frame(torch.full_like(frames, m), frames)  # (length, 2)
                gait_feats += [ball_rel[:, 0], ball_rel[:, 1]]
            feats = torch.stack(gait_feats, dim=-1)
            table[m, :length] = feats
            valid[m, :length] = True
            all_feats.append(feats)

        pooled = torch.cat(all_feats, dim=0)
        self._entry_search_table = table
        self._entry_search_valid = valid
        self._entry_search_feat_mean = pooled.mean(dim=0)
        self._entry_search_feat_std = pooled.std(dim=0).clamp(min=1e-6)
        self._entry_search_left_ankle_idx = left_idx
        self._entry_search_right_ankle_idx = right_idx

    def _live_entry_features(self, env_ids: torch.Tensor) -> torch.Tensor:
        """The same features as _build_entry_search_table (5, or 7 when
        _entry_search_use_ball_geometry), computed from the LIVE simulator state of env_ids right
        now -- reads _rigid_body_* directly (full robot-body order, same space ref_body_index/the
        ankle indices were resolved into), not the tracked-body-restricted robot_body_pos_w/
        robot_ref_pos_w properties, since the ankle links are not guaranteed to be in
        body_names_to_track."""
        sim = self._env.simulator
        ref_pos = sim._rigid_body_pos[env_ids, self.ref_body_index, :]  # type: ignore[attr-defined]
        ref_quat = sim._rigid_body_rot[env_ids, self.ref_body_index, :]  # type: ignore[attr-defined]
        ref_lin_vel_w = sim._rigid_body_vel[env_ids, self.ref_body_index, :]  # type: ignore[attr-defined]
        ref_ang_vel_w = sim._rigid_body_ang_vel[env_ids, self.ref_body_index, :]  # type: ignore[attr-defined]
        left_z = sim._rigid_body_pos[env_ids, self._entry_search_left_ankle_idx, 2]  # type: ignore[attr-defined]
        right_z = sim._rigid_body_pos[env_ids, self._entry_search_right_ankle_idx, 2]  # type: ignore[attr-defined]

        yaw_only = yaw_quat(ref_quat, w_last=True)
        body_lin_vel_local = quat_apply(quat_inverse(yaw_only, w_last=True), ref_lin_vel_w, w_last=True)

        gait_feats = [ref_pos[:, 2], body_lin_vel_local[:, 0], body_lin_vel_local[:, 1], ref_ang_vel_w[:, 2], left_z - right_z]
        if self._entry_search_use_ball_geometry:
            # Root-anchored (body 0), matching ball_rel_at_frame's own convention exactly -- NOT
            # ref_body_index, the same root-vs-torso distinction enter_at_frame's own docstring
            # warns about for a different reason (see place_ball_at_entry's docstring: ball
            # placement is a SEPARATE, root-anchored convention from the ref-body-anchored reward
            # terms).
            root_pos = sim._rigid_body_pos[env_ids, 0, :2]  # type: ignore[attr-defined]
            root_quat = sim._rigid_body_rot[env_ids, 0, :]  # type: ignore[attr-defined]
            root_heading = yaw_quat(root_quat, w_last=True)
            ball_pos = self.live_ball_pos_w[env_ids]
            delta_world = ball_pos[:, :2] - root_pos
            ball_rel_local = quat_apply(
                quat_inverse(root_heading, w_last=True),
                torch.cat([delta_world, torch.zeros_like(delta_world[:, :1])], dim=-1),
                w_last=True,
            )[:, :2]
            gait_feats += [ball_rel_local[:, 0], ball_rel_local[:, 1]]

        return torch.stack(gait_feats, dim=-1)

    def search_entry_point(self, env_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Deterministic weighted nearest-neighbour entry-point search (D2/D3): for each env in
        env_ids, finds the frame within THAT env's OWN fixed-for-life assigned skill's approach
        window (motion_ids[env_ids] -- never searched across the whole library, D2's own
        constraint) whose gait matches its current live state best.

        Returns (best_frame, residual): best_frame is an ABSOLUTE clip-local frame index (ready
        to pass straight to enter_at_frame); residual is the z-scored Euclidean distance to that
        frame (0 = exact match, larger = worse -- same units mid_episode_kick_entry_max_residual
        is expressed in). No learned parameters, zero new RL hyperparameters -- a precomputed
        table plus one batched cdist-equivalent per call."""
        live = self._live_entry_features(env_ids)
        live_z = (live - self._entry_search_feat_mean) / self._entry_search_feat_std

        motion_ids = self.motion_ids[env_ids]
        table_z = (self._entry_search_table[motion_ids] - self._entry_search_feat_mean) / self._entry_search_feat_std
        valid = self._entry_search_valid[motion_ids]

        dist = torch.norm(table_z - live_z.unsqueeze(1), dim=-1)
        dist = torch.where(valid, dist, torch.full_like(dist, float("inf")))
        residual, best_offset = dist.min(dim=1)

        best_frame = self.motion.motion_start_idx[motion_ids] + best_offset
        return best_frame, residual

    def ball_rel_at_frame(self, motion_ids: torch.Tensor, frames: torch.Tensor) -> torch.Tensor:
        """D2: robot-local (forward, lateral) offset from the ROOT's pose AT `frames` to where
        the configured ball spawn sits for that motion -- root (body 0), not ref_body_index,
        matching local_xy_to_world's/reset()'s own established ball-placement convention exactly
        (a separate, pre-existing anchor choice from enter_at_frame's own ref_body_index fix for
        reward-relevant terms; see local_xy_to_world's own docstring).

        Mechanism: ball_reset_state_per_motion[:, :2] is the configured (sc.x, sc.y) -- "forward,
        lateral from the robot" relative to the clip's OWN frame-0 root pose, by construction (the
        exact quantity reset()'s own teleport-time placement consumes via local_xy_to_world). This
        computes the ball's implied FIXED target world position from that frame-0 anchor, then
        re-expresses it relative to the clip's OWN root pose at the REQUESTED frame instead --
        purely within the clip's own coordinate content, no per-clip authoring, no robot/env state
        touched yet.

        Returns (len(motion_ids), 2). Feed straight into local_xy_to_world alongside the robot's
        ACTUAL entry pose (the one final step this method deliberately leaves to the caller) to
        get the world-frame placement -- mirroring exactly how reset() places the ball for a
        frame-0 teleport entry, just anchored at an arbitrary frame instead."""
        frame0 = self.motion.motion_start_idx[motion_ids]
        root_pos_frame0 = self.motion.body_pos_w[frame0, 0, :2]
        root_quat_frame0 = self.motion.body_quat_w[frame0, 0]
        heading_frame0 = yaw_quat(root_quat_frame0, w_last=True)

        local_xy_frame0 = self.ball_reset_state_per_motion[motion_ids, :2]
        ball_world_xy = (
            quat_apply(
                heading_frame0,
                torch.cat([local_xy_frame0, torch.zeros_like(local_xy_frame0[:, :1])], dim=-1),
                w_last=True,
            )[:, :2]
            + root_pos_frame0
        )

        root_pos_here = self.motion.body_pos_w[frames, 0, :2]
        root_quat_here = self.motion.body_quat_w[frames, 0]
        heading_here = yaw_quat(root_quat_here, w_last=True)
        delta_world = ball_world_xy - root_pos_here
        return quat_apply(
            quat_inverse(heading_here, w_last=True),
            torch.cat([delta_world, torch.zeros_like(delta_world[:, :1])], dim=-1),
            w_last=True,
        )[:, :2]

    def place_ball_at_entry(self, env_ids: torch.Tensor, frames: torch.Tensor) -> None:
        """D2: place the ball (and draw a fresh target) at a mid-episode kick entry, mirroring
        reset()'s own ball-placement block (section 5 above) exactly except for two deliberate
        differences: the robot-local offset comes from ball_rel_at_frame(frames) instead of the
        raw frame-0 ball_reset_state_per_motion, and the robot's pose is read LIVE from the
        simulator (this env's actual current pose, having just walked here) instead of a
        freshly-teleported target_root_pos/quat. Keeps the SAME position_randomization/OOD-spawn
        noise reset() applies, so the kick policy sees the exact tolerance it already trained
        under rather than a suspiciously exact placement -- D2's own explicit design decision.

        No-op if this skill has no ball (has_ball False) or env_ids is empty. Does not place the
        robot or touch time_steps -- purely the ball+target side of a mid-episode entry, called
        alongside enter_at_frame, not instead of it."""
        if not self.has_ball or env_ids.numel() == 0:
            return
        motion_ids = self.motion_ids[env_ids]
        local_xy = self.ball_rel_at_frame(motion_ids, frames)

        sim = self._env.simulator
        robot_pos_w = sim.robot_root_states[env_ids, :3]  # type: ignore[attr-defined]
        robot_quat_w = sim.robot_root_states[env_ids, 3:7]  # type: ignore[attr-defined]

        ball_states = self.ball_reset_state_per_motion[motion_ids].clone()
        position_randomization = self.ball_position_randomization_per_motion[motion_ids]
        target_nominal_local = self.target_nominal_local_per_motion[motion_ids]

        noise, is_ood = draw_position_noise_with_ood(
            position_randomization,
            ood_prob=self.motion_cfg.ood_spawn_probability,
            ood_multiplier=self.motion_cfg.ood_region_multiplier,
            device=self.device,
        )
        ball_local_placed = local_xy + noise
        ball_states[:, :2] = self.local_xy_to_world(
            ball_local_placed, env_ids, robot_pos_w=robot_pos_w, robot_quat_w=robot_quat_w
        )
        ball_states[:, 2] = ball_states[:, 2] + self._env.simulator.scene.env_origins[env_ids, 2]
        self.ball_spawn_pos_w[env_ids] = ball_states[:, :3]
        self.is_ood_spawn[env_ids] = is_ood
        self._env.simulator.set_actor_states([self.ball_name], env_ids, ball_states)

        target_local = self._synthesize_kick_aim_target_local(
            env_ids, motion_ids, ball_local_placed, target_nominal_local
        )
        self.target_xy_w[env_ids] = self.local_xy_to_world(
            target_local, env_ids, robot_pos_w=robot_pos_w, robot_quat_w=robot_quat_w
        )

    def place_ball_at_reset_pending(self, env_ids: torch.Tensor) -> None:
        """Increment 4 (mid_episode_kick_entry_ball_fixed, D2b -- closing the "at deploy the ball
        doesn't move" gap): place the ball (and draw a fresh target) for a kick-pending env AT
        RESET TIME, anchored to the CLIP's OWN frame-0 canonical root pose
        (self.motion.body_pos_w/quat_w[motion_start_idx, 0] + env_origins) -- NOT the robot's
        actual pose, which for a kick-pending env (diverted to LOCOMOTION, never teleported) isn't
        anywhere in particular relative to the clip. Mirrors reset()'s own ball-placement block
        exactly otherwise (same draw_position_noise_with_ood, same local_xy_to_world call, same
        frame-0 ball_reset_state_per_motion offset -- NOT ball_rel_at_frame, since there is no
        "entry frame" yet; this IS the frame-0 placement, just anchored to the clip's own pose
        instead of a freshly-teleported robot's).

        The ball then STAYS here for the rest of the episode: fixed-ball mode never calls
        place_ball_at_entry (see UnifiedManager._enter_kick's own guard), so nothing moves it
        again until this env's next reset() (a genuine new episode) calls this same placement
        afresh. Called from UnifiedManager._resample_task_mode alongside the pending env's
        motion_ids assignment -- ONLY when mid_episode_kick_entry_ball_fixed is True; callers must
        not call this otherwise, since a normal (ball-at-handoff) pending env deliberately gets NO
        ball placement at reset (D4's original finding: reset() is task_mode="kick"-tagged, never
        called for a LOCOMOTION-mode reset; place_ball_at_entry supplies the placement later,
        at fire time, instead).

        No-op if this skill has no ball (has_ball False) or env_ids is empty. Does not place the
        robot or touch time_steps."""
        if not self.has_ball or env_ids.numel() == 0:
            return
        motion_ids = self.motion_ids[env_ids]
        frame0 = self.motion.motion_start_idx[motion_ids]
        anchor_pos = self.motion.body_pos_w[frame0, 0] + self._env.simulator.scene.env_origins[env_ids]
        anchor_quat = self.motion.body_quat_w[frame0, 0]

        ball_states = self.ball_reset_state_per_motion[motion_ids].clone()
        position_randomization = self.ball_position_randomization_per_motion[motion_ids]
        target_nominal_local = self.target_nominal_local_per_motion[motion_ids]

        noise, is_ood = draw_position_noise_with_ood(
            position_randomization,
            ood_prob=self.motion_cfg.ood_spawn_probability,
            ood_multiplier=self.motion_cfg.ood_region_multiplier,
            device=self.device,
        )
        ball_local_placed = ball_states[:, :2] + noise
        ball_states[:, :2] = self.local_xy_to_world(
            ball_local_placed, env_ids, robot_pos_w=anchor_pos, robot_quat_w=anchor_quat
        )
        ball_states[:, 2] = ball_states[:, 2] + self._env.simulator.scene.env_origins[env_ids, 2]
        self.ball_spawn_pos_w[env_ids] = ball_states[:, :3]
        self.is_ood_spawn[env_ids] = is_ood
        self._env.simulator.set_actor_states([self.ball_name], env_ids, ball_states)

        # NOTE: this path anchors to the CLIP'S OWN frame-0 pose (anchor_pos/anchor_quat above),
        # NOT the robot's actual current pose -- see this method's own docstring. Under
        # kick_aim_enabled, that means kick_aim_theta=0 points along the CLIP's canonical heading,
        # not necessarily wherever the robot ends up facing after walking here. Accepted as-is for
        # now (mirrors reset()'s own anchor choice for this fixed-ball mode); revisit if
        # mid_episode_kick_entry_ball_fixed is used together with kick_aim_enabled in practice.
        target_local = self._synthesize_kick_aim_target_local(
            env_ids, motion_ids, ball_local_placed, target_nominal_local
        )
        self.target_xy_w[env_ids] = self.local_xy_to_world(
            target_local, env_ids, robot_pos_w=anchor_pos, robot_quat_w=anchor_quat
        )

    def enter_at_frame(self, env_ids: torch.Tensor, time_steps: torch.Tensor) -> None:
        """Mid-episode kick entry (2026-08-13, increment 1 of the locomotion->kick handoff plan
        -- https://claude.ai/code/artifact/53c1da51-d841-4979-8bf8-efd5ea652e06). Sets time_steps
        for env_ids to the given per-env absolute clip-local frame index, and captures a per-env
        rigid re-anchor transform so every reference accessor below tracks a copy of the clip
        translated and yaw-rotated to START at the robot's own actual current pose at this frame,
        rather than the clip's raw captured world pose.

        Deliberately NOT a reset() variant: writes NO simulator state whatsoever (dof_pos, dof_vel,
        or any of the four root-state slices) -- reset() IS the teleport this mechanism exists to
        avoid entering. motion_ids is deliberately left untouched: kick-partitioned envs already
        have a fixed-for-life skill assignment from UnifiedManager._build_task_mode_partition, so
        entry only ever needs to pick a FRAME within that env's own already-assigned clip, not a
        different clip.

        Why this is needed at all -- and why it does NOT need to touch body_pos_w/body_quat_w's
        already-correct consumer: step()'s own body_pos_relative_w/body_quat_relative_w computation
        ("1.2.2" above) already re-expresses body targets relative to the robot's ACTUAL CURRENT
        ref pose, recomputed fresh every tick from the robot's real orientation -- so the 6
        relative-body motion-tracking terms are already robust to any entry offset, self-correcting
        continuously, with no changes needed here (this was verified algebraically: applying the
        SAME rigid transform to both body_pos_w and ref_pos_w leaves their difference, and hence
        body_pos_relative_w, invariant up to the transform's own rotation -- which step()'s own
        per-tick delta_quat_w computation already applies independently). What is NOT
        self-correcting: motion_global_ref_position_error_exp / _orientation_error_exp (raw
        ref_pos_w/ref_quat_w compared directly against the robot's actual pose, sigma=0.3m on
        position -- saturates to zero reward at any real entry offset, pulling the robot toward the
        clip's ORIGINAL world path like a teleport implemented as reward) and
        motion_global_body_lin_vel/_ang_vel (raw body_lin_vel_w/ang_vel_w, world-frame vectors with
        no yaw correction at all). This method's transform fixes exactly those, and is applied
        uniformly to every accessor for simplicity/auditability rather than cherry-picking -- proven
        harmless for the already-correct relative-body terms above.

        Z handling: delta_quat is constructed via yaw_quat (see below), a pure yaw-axis rotation
        that leaves a vector's Z-component invariant by construction -- so delta_pos's Z component
        reduces automatically to a plain height offset (robot's actual entry height minus the
        clip's own captured height at the entry frame), matching body_pos_relative_w's own existing
        "delta_pos_w_height" convention (height anchored to the robot's actual current height, not
        the clip's captured one) with no special-casing required.

        Anchor body: ref_body_index (this project's g1 config: "torso_link"), NOT body index 0
        (pelvis/root) -- a real bug caught by live IsaacSim verification, invisible to the unit
        tests (their single-body fakes made index-0 and ref_body_index the same body by
        construction). motion_global_ref_position_error_exp/_orientation_error_exp -- the terms
        this whole mechanism exists to protect -- read ref_pos_w/ref_quat_w, which are built from
        ref_body_index, not body 0. body_pos_relative_w's own self-correcting computation
        (step(), "1.2.2") ALSO reads ref_pos_w/ref_quat_w, falling back to root_pos_w/root_quat_w
        only when episode_length_buf==0 -- never true for a mid-episode entry. So ref_body_index is
        the anchor point every live consumer actually needs; anchoring to the root instead left a
        real residual (measured live: 5.4cm, the pelvis-to-torso rigid offset) at the entry tick.
        root_pos_w/root_quat_w's only remaining direct consumers (reset()'s own teleport-velocity
        assignment, and step()'s episode-start fallback) are both reset()-time-only and never reached
        by a mid-episode entry, so leaving them un-anchored is correct, not an oversight.

        No caller exists yet (increment 2, not yet implemented) -- this method is unreachable, and
        _ref_anchor_active stays False, for the entire lifetime of any run today."""
        if env_ids.numel() == 0:
            return
        robot_ref_pos = self.robot_ref_pos_w[env_ids]
        robot_ref_quat = self.robot_ref_quat_w[env_ids]
        clip_ref_pos = (
            self.motion.body_pos_w[time_steps, self.ref_body_index] + self._env.simulator.scene.env_origins[env_ids]
        )
        clip_ref_quat = self.motion.body_quat_w[time_steps, self.ref_body_index]

        delta_quat = yaw_quat(
            quat_mul(robot_ref_quat, quat_inverse(clip_ref_quat, w_last=True), w_last=True), w_last=True
        )
        delta_pos = robot_ref_pos - quat_apply(delta_quat, clip_ref_pos, w_last=True)

        self._ref_anchor_delta_quat[env_ids] = delta_quat
        self._ref_anchor_delta_pos[env_ids] = delta_pos
        self._ref_anchor_active = True
        self.time_steps[env_ids] = time_steps

    def sample_authored_clip_frames(self, env_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Draw a uniformly random (motion_id, GLOBAL frame index) per env, restricted to AUTHORED
        clip content -- ``motion_start_idx .. pre_recovery_motion_end_idx`` -- for
        UnifiedManager's kick-state locomotion initialization (see
        MultiSkillConfig.kick_state_init_prob).

        Deliberately excludes the synthetic recovery-transition + static-hold tail past
        pre_recovery_motion_end_idx: that tail is scripted filler in which the robot is already
        standing near-nominal, so sampling it would mostly reproduce the ordinary locomotion
        reset pose and add no state diversity -- the entire point of this mechanism is the
        off-balance, momentum-carrying poses that only real captured kick footage contains. Same
        boundary, and the same reasoning, as ``rsi_scope_to_authored_clip``'s own authored-content
        scoping and as ``kick_recovery_locomotion_flip_enabled``'s flip boundary.

        Uniform over MOTIONS as well as frames: a locomotion-partitioned env's ``skill_id`` is
        meaningless (UnifiedManager._build_task_mode_partition only assigns it meaningfully to
        kick-partitioned envs), so the caller has no per-env skill to inherit and every clip
        should contribute recovery states equally.

        Returns ``(motion_ids, frames)``, both ``[len(env_ids)]`` long tensors; ``frames`` are
        GLOBAL indices into the concatenated motion buffers, directly usable against
        ``self.motion.joint_pos``/``body_pos_w``/etc. Touches NO per-env state."""
        n = env_ids.numel()
        if n == 0:
            empty = torch.zeros(0, dtype=torch.long, device=self.device)
            return empty, empty.clone()
        motion_ids = torch.randint(0, self.motion.num_motions, (n,), device=self.device, dtype=torch.long)
        start = self.motion.motion_start_idx[motion_ids]
        end = self.pre_recovery_motion_end_idx[motion_ids]
        # clamp(min=1): a degenerate motion whose authored span is empty/inverted would otherwise
        # make the width-scaled draw below produce a frame outside its own clip.
        width = (end - start).clamp(min=1)
        offset = (torch.rand(n, device=self.device) * width.float()).long().clamp(max=width - 1)
        return motion_ids, start + offset

    def teleport_to_frames(self, env_ids: torch.Tensor, frames: torch.Tensor) -> None:
        """Write robot dof + root state into the simulator for ``env_ids`` from the motion data at
        the given GLOBAL ``frames``, applying the SAME per-reset init-pose noise
        ``reset()``'s own teleport applies (``init_pose_cfg``) so these envs land
        in-distribution with every other reset rather than exactly on-clip.

        Deliberately mirrors -- and is deliberately separate from -- ``reset()``'s own teleport
        block (section "2.2/3" there). Separate because ``reset()`` is registered
        ``task_mode="kick"``-tagged (CommandManager filters reset_terms by mode; see reset()'s own
        docstring), so it is structurally unreachable for the LOCOMOTION-partitioned envs this
        method exists to serve, and because reset() also owns RSI frame selection, ball placement
        and ref-anchor clearing -- none of which a locomotion env wants. This method touches NO
        per-env MotionCommand state at all (not ``motion_ids``, not ``time_steps``, not the ref
        anchor): a locomotion-mode env must keep its motion-tracking state meaningless/inert, and
        writing it would leak a kick reference into an env whose kick-tagged observation and
        reward terms are masked off anyway.

        Root state is body index 0, matching ``root_pos_w``/``root_lin_vel_w``'s own convention."""
        if env_ids.numel() == 0:
            return
        dof_pos = self.motion.joint_pos[frames]
        dof_vel = self.motion.joint_vel[frames]
        root_pos = self.motion.body_pos_w[frames, 0] + self._env.simulator.scene.env_origins[env_ids]
        root_rot = self.motion.body_quat_w[frames, 0]
        root_lin_vel = self.motion.body_lin_vel_w[frames, 0]
        root_ang_vel = self.motion.body_ang_vel_w[frames, 0]

        scale = self.init_pose_cfg.overall_noise_scale
        dof_pos_noise = self.init_pose_cfg.dof_pos * scale
        root_pos_noise = torch.tensor(self.init_pose_cfg.root_pos, device=self.device) * scale
        root_rot_noise_rpy = torch.tensor(self.init_pose_cfg.root_rot, device=self.device) * scale
        root_vel_noise = torch.tensor(self.init_pose_cfg.root_vel, device=self.device) * scale
        root_ang_vel_noise_rpy = torch.tensor(self.init_pose_cfg.root_ang_vel, device=self.device) * scale

        target_dof_pos = dof_pos + (torch.rand(dof_pos.shape, device=self.device) - 0.5) * 2 * dof_pos_noise
        soft_joint_pos_limits = self._env.simulator.dof_pos_limits  # type: ignore[attr-defined]
        target_dof_pos = torch.clip(target_dof_pos, soft_joint_pos_limits[:, 0], soft_joint_pos_limits[:, 1])

        target_root_pos = root_pos + (
            torch.rand(root_pos.shape, device=self.device) - 0.5
        ) * 2 * root_pos_noise.unsqueeze(0)
        rand_sample_rpy = (torch.rand((env_ids.numel(), 3), device=self.device) - 0.5) * 2 * root_rot_noise_rpy
        orientations_delta = quat_from_euler_xyz(
            rand_sample_rpy[:, 0], rand_sample_rpy[:, 1], rand_sample_rpy[:, 2]
        )
        target_root_rot = quat_mul(orientations_delta, root_rot, w_last=True)
        target_root_lin_vel = root_lin_vel + (
            torch.rand(root_lin_vel.shape, device=self.device) - 0.5
        ) * 2 * root_vel_noise.unsqueeze(0)
        target_root_ang_vel = root_ang_vel + (
            torch.rand(root_ang_vel.shape, device=self.device) - 0.5
        ) * 2 * root_ang_vel_noise_rpy.unsqueeze(0)

        self._env.simulator.dof_pos[env_ids] = target_dof_pos
        self._env.simulator.dof_vel[env_ids] = dof_vel
        self._env.simulator.robot_root_states[env_ids, :3] = target_root_pos
        self._env.simulator.robot_root_states[env_ids, 3:7] = target_root_rot
        self._env.simulator.robot_root_states[env_ids, 7:10] = target_root_lin_vel
        self._env.simulator.robot_root_states[env_ids, 10:13] = target_root_ang_vel

    def update_metrics(self):
        """Update the metrics. After action, before step() is called."""
        self.metrics["motion/error_ref_pos"] = torch.norm(self.ref_pos_w - self.robot_ref_pos_w, dim=-1)
        self.metrics["motion/error_ref_rot"] = quat_error_magnitude(self.ref_quat_w, self.robot_ref_quat_w)
        self.metrics["motion/error_ref_lin_vel"] = torch.norm(self.ref_lin_vel_w - self.robot_ref_lin_vel_w, dim=-1)
        self.metrics["motion/error_ref_ang_vel"] = torch.norm(self.ref_ang_vel_w - self.robot_ref_ang_vel_w, dim=-1)

        self.metrics["motion/error_body_pos"] = torch.norm(
            self.body_pos_relative_w - self.robot_body_pos_w, dim=-1
        ).mean(dim=-1)

        self.metrics["motion/error_body_rot"] = quat_error_magnitude(
            self.body_quat_relative_w, self.robot_body_quat_w
        ).mean(dim=-1)

        self.metrics["motion/error_body_lin_vel"] = torch.norm(
            self.body_lin_vel_w - self.robot_body_lin_vel_w, dim=-1
        ).mean(dim=-1)
        self.metrics["motion/error_body_ang_vel"] = torch.norm(
            self.body_ang_vel_w - self.robot_body_ang_vel_w, dim=-1
        ).mean(dim=-1)

        self.metrics["motion/error_joint_pos"] = torch.norm(self.joint_pos - self.robot_joint_pos, dim=-1)
        self.metrics["motion/error_joint_vel"] = torch.norm(self.joint_vel - self.robot_joint_vel, dim=-1)

        if self.motion_cfg.use_adaptive_timesteps_sampler:
            self.adaptive_timesteps_sampler.get_stats()
            self.metrics["motion/adaptive_timesteps_sampler_entropy"] = self.adaptive_timesteps_sampler.metrics[
                "sampling_entropy"
            ]
            self.metrics["motion/adaptive_timesteps_sampler_top1_prob"] = self.adaptive_timesteps_sampler.metrics[
                "sampling_top1_prob"
            ]
            self.metrics["motion/adaptive_timesteps_sampler_top1_bin"] = self.adaptive_timesteps_sampler.metrics[
                "sampling_top1_bin"
            ]

    #########################################################################################
    ## Internal helpers
    #########################################################################################
    @staticmethod
    def _resolve_per_motion_durations(per_motion: list[float], global_default: float, num_motions: int) -> list[float]:
        """Empty per_motion list -> broadcast global_default to every motion (bit-identical to
        the old single-scalar behavior, including for N=1). Length-num_motions list -> used as-is,
        one value per motion in declaration order. Anything else (wrong length) is a config
        mistake -- raise clearly rather than silently misassigning one skill's duration to
        another."""
        if not per_motion:
            return [global_default] * num_motions
        if len(per_motion) != num_motions:
            raise ValueError(
                f"Per-motion duration list has {len(per_motion)} entries but there are "
                f"{num_motions} motions loaded -- must be empty (broadcasts the scalar default) "
                f"or exactly length {num_motions}."
            )
        return list(per_motion)

    def _maybe_add_default_pose_transition(self, *, prepend: bool, motion_idx: int) -> None:
        """Shared path for optionally inserting default-pose interpolation before/after
        motion_idx's own clip. Duration comes from the per-motion resolved list
        (self._resolved_prepend_duration_s / self._resolved_recovery_duration_s, set once in
        setup() from MotionConfig.motion_prepend_duration_s / motion_recovery_duration_s, or
        broadcast from the single-clip scalar default when that list is empty -- see
        _resolve_per_motion_durations)."""
        enabled = self.motion_cfg.enable_default_pose_prepend if prepend else self.motion_cfg.enable_default_pose_append
        if not enabled:
            return

        duration = (
            self._resolved_prepend_duration_s[motion_idx] if prepend else self._resolved_recovery_duration_s[motion_idx]
        )
        if duration <= 0.0:
            return

        num_steps = round(duration / self._env.dt)
        if num_steps <= 1:
            logger.warning(
                "Default pose {} duration {}s is too short for dt {}; skipping augmentation (motion {}).",
                "prepend" if prepend else "append",
                duration,
                self._env.dt,
                motion_idx,
            )
            return

        default_state = self._build_default_pose_state(motion_idx, use_motion_end=not prepend)

        action = "prepend" if prepend else "append"
        log_str = f"{action} {num_steps} interpolated frames ({duration}s) from default pose to motion {motion_idx}"
        try:
            self._add_transition_to_motion(default_state, num_steps, prepend=prepend, motion_idx=motion_idx)
            logger.info(log_str)
        except Exception as exc:
            logger.error(f"Failed to {action} default pose transition (motion {motion_idx}): {exc}")
            raise RuntimeError(
                f"Critical error during motion interpolation setup: {exc}\n"
                "This indicates a mismatch in tensor dimensions during interpolation. "
                "Please check that the motion file and robot configuration are compatible."
            ) from exc

        if not prepend:
            self._maybe_add_post_transition_hold(default_state, motion_idx=motion_idx)

    def _maybe_add_post_transition_hold(self, default_state: dict[str, torch.Tensor], motion_idx: int) -> None:
        """Append a genuinely static hold (fixed pose, zero velocity) right after motion_idx's own
        default-pose transition — unlike the transition itself, this does not move the reference at
        all, so it rewards sustained balance/stability rather than tracking a moving target. Only
        meaningful right after `_add_transition_to_motion(..., prepend=False, motion_idx=motion_idx)`,
        since it holds at whatever pose that transition targeted (`default_state`)."""
        hold_duration = self._resolved_hold_duration_s[motion_idx]
        if hold_duration <= 0.0:
            return

        hold_steps = round(hold_duration / self._env.dt)
        if hold_steps <= 1:
            logger.warning(
                "post_transition_hold_duration_s {}s is too short for dt {}; skipping hold (motion {}).",
                hold_duration,
                self._env.dt,
                motion_idx,
            )
            return

        dtype = self.motion._joint_pos.dtype
        device = self.device
        default_motion_state = self._default_motion_state(default_state, dtype=dtype, device=device)
        try:
            # start == target: every interpolated frame is identical to the default pose, with
            # zero joint/body velocity (per `_build_default_pose_state`) — a true static hold.
            self._build_and_apply_transition(
                start_state=default_motion_state,
                target_state=default_motion_state,
                num_steps=hold_steps,
                prepend=False,
                drop_first=True,
                drop_last=False,
                dtype=dtype,
                device=device,
                motion_idx=motion_idx,
            )
            logger.info(f"append {hold_steps} static hold frames ({hold_duration}s) at default pose (motion {motion_idx})")
        except Exception as exc:
            logger.error(f"Failed to append post-transition hold (motion {motion_idx}): {exc}")
            raise RuntimeError(
                f"Critical error during post-transition hold setup: {exc}\n"
                "This indicates a mismatch in tensor dimensions during interpolation. "
                "Please check that the motion file and robot configuration are compatible."
            ) from exc

    def _maybe_smooth_motion_head_velocities(self, *, motion_idx: int) -> None:
        """Opt-in in-memory smoothing of motion_idx's own LEADING AUTHORED velocity frames
        (2026-08-11) -- see motion_head_velocity_smoothing.py's module docstring for the full
        measured rationale. Off by default (motion_head_velocity_smoothing_frames=0 -> no-op).

        Called once per motion from setup()'s own loop, LAST for that motion -- after both its
        prepend/append splices AND _maybe_correct_kick_foot_ankle_pitch, which regenerates every
        velocity array from positions and would otherwise overwrite this edit (see that call
        site's own comment; this ordering was established by a live no-op measurement, not
        assumed).

        Because the prepend has already been spliced by now, motion_start_idx no longer points at
        the clip's own first CAPTURED frame -- the synthetic prepend occupies exactly that slot
        (insert_segment_at_motion_boundary merges the inserted frames into this motion and leaves
        its start index unchanged). The prepend length is therefore added back here so the ramp
        lands on real captured content rather than on synthetic windup frames, which are already
        smooth by construction and are not what this feature exists to fix."""
        # 2026-08-15: per-motion resolved (see setup()'s own _resolved_head_velocity_smoothing_
        # frames comment) -- broadcasts the single scalar to every motion when no per-skill
        # override is set, bit-identical to the old getattr(self.motion_cfg, ...) read.
        n_frames = int(self._resolved_head_velocity_smoothing_frames[motion_idx])
        if n_frames <= 0:
            return

        prepend_len = 0
        if self.motion_cfg.enable_default_pose_prepend:
            prepend_duration = self._resolved_prepend_duration_s[motion_idx]
            if prepend_duration > 0.0:
                steps = round(prepend_duration / self._env.dt)
                # Mirrors _maybe_add_default_pose_transition's own skip condition exactly -- a
                # duration too short for dt inserts NOTHING, so no offset must be applied either.
                if steps > 1:
                    prepend_len = steps

        motion_start = int(self.motion.motion_start_idx[motion_idx].item())
        motion_end = int(self.motion.motion_end_idx[motion_idx].item())
        head_start = motion_start + prepend_len

        smooth_motion_head_velocities(
            joint_vel=self.motion._joint_vel[head_start:motion_end],
            body_lin_vel_w=self.motion._body_lin_vel_w[head_start:motion_end],
            body_ang_vel_w=self.motion._body_ang_vel_w[head_start:motion_end],
            n_frames=n_frames,
        )
        logger.info(
            f"motion_head_velocity_smoothing: ramped first {n_frames} authored velocity frame(s) "
            f"of motion {motion_idx} from zero -- absolute frames "
            f"[{head_start}, {head_start + n_frames}) (motion_start={motion_start}, "
            f"prepend_len={prepend_len})"
        )

    def _maybe_correct_kick_foot_ankle_pitch(self, *, motion_idx: int, strike_start_idx: int, stand_start_idx: int) -> None:
        """Self-calibrating in-memory kick-foot ankle-pitch correction (2026-08-08) -- see
        kick_ankle_pitch_correction.py's own module docstring for the full rationale (a live
        diagnostic found the authored clip's own ankle trajectory is genuinely toe-up exactly
        where real contact happens, making foot_strike_pitch's reward unwinnable). Called once per
        motion from setup()'s own per-motion loop, after that motion's strike/stand boundaries and
        its own append-transition are both finalized (so motion_end_idx[motion_idx] won't move
        again and the correction operates on this motion's FULL, final frame range).

        Resolves kick_foot from skill_ball_configs (N-skill mode) -- skipped, not defaulted, when
        that's empty (legacy single-clip mode has no per-motion kick_foot source reachable from
        here) or when the resolved name doesn't exist in this MJCF (should never happen for a real
        G1 clip, but fails loud rather than silently skipping a real config error). Also skipped,
        per-skill, when that skill's own SkillConfig.kick_ankle_pitch_correction_enabled is False
        (2026-08-08, user-requested toggle -- default True, so this stays a no-op change for every
        existing skills.yaml that doesn't set the field at all)."""
        skill_configs = getattr(self.motion_cfg, "skill_ball_configs", None)
        if not skill_configs or motion_idx >= len(skill_configs):
            return
        skill_cfg = skill_configs[motion_idx]
        if not getattr(skill_cfg, "kick_ankle_pitch_correction_enabled", True):
            logger.info(f"kick_ankle_pitch_correction: disabled for motion {motion_idx} (skill config), skipping")
            return
        kick_foot = skill_cfg.kick_foot

        if not hasattr(self, "_kick_ankle_pitch_correction_tree"):
            mjcf_path = resolve_data_file_path(DEFAULT_MJCF_RELATIVE_PATH)
            self._kick_ankle_pitch_correction_tree = KinematicTree(mjcf_path, device=self.device)
        tree = self._kick_ankle_pitch_correction_tree

        motion_start = int(self.motion.motion_start_idx[motion_idx].item())
        motion_end = int(self.motion.motion_end_idx[motion_idx].item())
        local_strike = strike_start_idx - motion_start
        local_stand = stand_start_idx - motion_start
        fps = float(np.asarray(self.motion.fps).reshape(-1)[0])

        correct_kick_foot_ankle_pitch(
            joint_pos=self.motion._joint_pos[motion_start:motion_end],
            body_pos_w=self.motion._body_pos_w[motion_start:motion_end],
            body_quat_w=self.motion._body_quat_w[motion_start:motion_end],
            joint_vel=self.motion._joint_vel[motion_start:motion_end],
            body_lin_vel_w=self.motion._body_lin_vel_w[motion_start:motion_end],
            body_ang_vel_w=self.motion._body_ang_vel_w[motion_start:motion_end],
            dt=1.0 / fps,
            kick_foot=kick_foot,
            strike_start_idx=local_strike,
            stand_start_idx=local_stand,
            tree=tree,
        )

    def _motion_boundary_frame(self, motion_idx: int, *, at_start: bool) -> int:
        """Absolute frame index (into the whole concatenated buffer) of motion_idx's own current
        start (at_start=True) or last frame (at_start=False) -- i.e. exactly where a prepend or
        append transition for THIS motion should anchor to, generalizing the old hardcoded
        "frame 0" / "frame -1" (which only ever meant "motion 0's start" / "the last loaded
        motion's end") to any motion in the list. Read fresh each call -- correct even mid-loop,
        since earlier insertions in the same setup() pass may have already shifted these
        boundaries (see insert_segment_at_motion_boundary's ascending-order requirement)."""
        if at_start:
            return int(self.motion.motion_start_idx[motion_idx].item())
        return int(self.motion.motion_end_idx[motion_idx].item()) - 1

    def _build_default_pose_state(self, motion_idx: int, use_motion_end: bool = False) -> dict[str, torch.Tensor]:
        """Build the state dict representing the robot's default standing pose, anchored to
        motion_idx's own clip.

        By default, anchor root pos/yaw to that motion's own start; when use_motion_end is True,
        anchor to that motion's own end.
        """
        init_state = self._env.robot_config.init_state
        joint_pos = self._env.default_dof_pos_base.squeeze(0).to(self.device)
        joint_vel = torch.zeros_like(joint_pos)

        init_root_quat = torch.tensor(init_state.rot, dtype=torch.float32, device=self.device).unsqueeze(0)
        init_roll, init_pitch, _ = get_euler_xyz(init_root_quat, w_last=True)

        frame_idx = self._motion_boundary_frame(motion_idx, at_start=not use_motion_end)

        # Assume the pelvis is the first in robot_body_names
        motion_root_pos = self.motion.body_pos_w[frame_idx, 0].to(self.device)
        motion_root_quat = self.motion.body_quat_w[frame_idx, 0].to(self.device).unsqueeze(0)
        _, _, motion_yaw = get_euler_xyz(motion_root_quat, w_last=True)

        # Keep z from init config but adopt the clip's x,y at the chosen anchor frame.
        default_root_pos = torch.tensor(
            [motion_root_pos[0], motion_root_pos[1], init_state.pos[2]],
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)
        # Keep roll/pitch from init config but adopt the clip's yaw at the chosen anchor frame.
        default_root_quat = quat_from_euler_xyz(
            init_roll.squeeze(0),
            init_pitch.squeeze(0),
            motion_yaw.squeeze(0),
        )
        default_root_lin_vel = torch.tensor(init_state.lin_vel, dtype=torch.float32, device=self.device)
        default_root_ang_vel = torch.tensor(init_state.ang_vel, dtype=torch.float32, device=self.device)

        body_states = self._capture_body_states(
            joint_pos,
            joint_vel,
            default_root_pos,
            default_root_quat,
            default_root_lin_vel,
            default_root_ang_vel,
        )

        default_body_pos = self._map_robot_bodies_to_motion_order(body_states["pos"])
        default_body_quat = self._map_robot_bodies_to_motion_order(body_states["quat"])
        default_body_lin_vel = self._map_robot_bodies_to_motion_order(body_states["lin_vel"])
        default_body_ang_vel = self._map_robot_bodies_to_motion_order(body_states["ang_vel"])

        if self.motion.has_object:
            object_pos = self.motion._object_pos_w[frame_idx].to(self.device)
            object_quat = self.motion._object_quat_w[frame_idx].to(self.device)
            object_lin_vel = self.motion._object_lin_vel_w[frame_idx].to(self.device)
        else:
            object_pos = torch.zeros(0, 3, device=self.device, dtype=torch.float32)
            object_quat = torch.zeros(0, 4, device=self.device, dtype=torch.float32)
            object_lin_vel = torch.zeros(0, 3, device=self.device, dtype=torch.float32)

        return {
            "joint_pos": joint_pos.clone(),
            "joint_vel": joint_vel,
            "root_pos": default_root_pos,
            "root_quat": default_root_quat,
            "root_lin_vel": default_root_lin_vel,
            "root_ang_vel": default_root_ang_vel,
            "body_pos": default_body_pos,
            "body_quat": default_body_quat,
            "body_lin_vel": default_body_lin_vel,
            "body_ang_vel": default_body_ang_vel,
            "object_pos": object_pos,
            "object_quat": object_quat,
            "object_lin_vel": object_lin_vel,
        }

    def _add_transition_to_motion(
        self, default_state: dict[str, torch.Tensor], num_steps: int, prepend: bool, motion_idx: int
    ) -> None:
        """Add interpolated frames either before or after motion_idx's own clip data."""
        assert self._body_indexes_in_motion is not None
        assert self._joint_indexes_in_motion is not None

        if num_steps <= 0:
            return

        device = self.device
        dtype = self.motion._joint_pos.dtype

        default_motion_state = self._default_motion_state(default_state, dtype=dtype, device=device)
        anchor_frame = self._motion_boundary_frame(motion_idx, at_start=prepend)
        motion_state = self._motion_state(anchor_frame, dtype=dtype, device=device)

        start_state = default_motion_state if prepend else motion_state
        target_state = motion_state if prepend else default_motion_state
        drop_first, drop_last = (False, True) if prepend else (True, False)

        self._build_and_apply_transition(
            start_state=start_state,
            target_state=target_state,
            num_steps=num_steps,
            prepend=prepend,
            drop_first=drop_first,
            drop_last=drop_last,
            dtype=dtype,
            device=device,
            motion_idx=motion_idx,
        )

    def _slerp_quat_sequence(self, start: torch.Tensor, end: torch.Tensor, alphas: torch.Tensor) -> torch.Tensor:
        """Spherically interpolate quaternions across multiple time steps."""
        if alphas.numel() == 0:
            return start.new_zeros((0,) + start.shape)

        num_steps = alphas.shape[0]
        start_expand = start.unsqueeze(0).expand(num_steps, -1, -1)
        end_expand = end.unsqueeze(0).expand(num_steps, -1, -1)
        alpha_flat = alphas.repeat_interleave(start.shape[0]).unsqueeze(-1)
        blended = slerp(
            start_expand.reshape(-1, 4),
            end_expand.reshape(-1, 4),
            alpha_flat,
        )
        return blended.view(num_steps, start.shape[0], 4)

    def _capture_body_states_mujoco_fk(
        self,
        joint_pos: torch.Tensor,
        joint_vel: torch.Tensor,
        root_pos: torch.Tensor,
        root_quat: torch.Tensor,
        root_lin_vel: torch.Tensor,
        root_ang_vel: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Forward-kinematics capture on the MuJoCo backend's single-env CPU model.

        Returns body pos/quat/lin_vel/ang_vel in the SAME layout and conventions the IsaacSim path
        returns, so callers need no branching:
          - poses relative to the env origin (root_model is a single-env template, so its origin
            is already 0 -- no offset to add or subtract, unlike the IsaacSim path),
          - quaternions in holosoma's w_last (xyzw) order, converted from MuJoCo's native wxyz,
          - body rows ordered by `simulator.body_names` (which excludes MuJoCo's `world` body).
        Restores qpos/qvel on the way out so the caller's sim state is untouched.
        """
        import mujoco

        simulator = self._env.simulator
        model, data = simulator.root_model, simulator.root_data
        assert model is not None and data is not None, "MuJoCo backend has no root_model/root_data"

        qpos_backup = data.qpos.copy()
        qvel_backup = data.qvel.copy()
        try:
            ra, va = simulator.robot_qpos_addr, simulator.robot_qvel_addr
            rp = root_pos.detach().flatten().cpu().numpy()
            rq = root_quat.detach().flatten().cpu().numpy()  # xyzw (holosoma)
            data.qpos[ra : ra + 3] = rp[:3]
            data.qpos[ra + 3 : ra + 7] = [rq[3], rq[0], rq[1], rq[2]]  # -> wxyz (MuJoCo)
            data.qvel[va : va + 3] = root_lin_vel.detach().flatten().cpu().numpy()[:3]
            data.qvel[va + 3 : va + 6] = root_ang_vel.detach().flatten().cpu().numpy()[:3]

            jp = joint_pos.detach().flatten().cpu().numpy()
            jv = joint_vel.detach().flatten().cpu().numpy()
            for i, qadr in enumerate(simulator.dof_qpos_addrs):
                data.qpos[qadr] = jp[i]
            for i, vadr in enumerate(simulator.dof_qvel_addrs):
                data.qvel[vadr] = jv[i]

            mujoco.mj_forward(model, data)

            n = len(simulator.body_names)
            pos = torch.zeros(n, 3, dtype=torch.float32, device=self.device)
            quat = torch.zeros(n, 4, dtype=torch.float32, device=self.device)
            lin = torch.zeros(n, 3, dtype=torch.float32, device=self.device)
            ang = torch.zeros(n, 3, dtype=torch.float32, device=self.device)
            vel6 = np.zeros(6, dtype=np.float64)
            for i, name in enumerate(simulator.body_names):
                bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
                if bid < 0:  # tolerate the backend's name cleaning (e.g. stripped prefixes)
                    bid = next(
                        (
                            b
                            for b in range(model.nbody)
                            if simulator._get_clean_name(model.body(b).name) == name
                        ),
                        -1,
                    )
                assert bid >= 0, f"body '{name}' not found in the MuJoCo model"
                pos[i] = torch.as_tensor(data.xpos[bid], dtype=torch.float32, device=self.device)
                w, x, y, z = data.xquat[bid]  # MuJoCo wxyz
                quat[i] = torch.tensor([x, y, z, w], dtype=torch.float32, device=self.device)
                # mj_objectVelocity returns [angular(3); linear(3)]; flg_local=0 -> world frame,
                # matching the IsaacSim path's world-frame _rigid_body_vel/_rigid_body_ang_vel.
                mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, bid, vel6, 0)
                ang[i] = torch.as_tensor(vel6[:3].copy(), dtype=torch.float32, device=self.device)
                lin[i] = torch.as_tensor(vel6[3:].copy(), dtype=torch.float32, device=self.device)
        finally:
            data.qpos[:] = qpos_backup
            data.qvel[:] = qvel_backup
            mujoco.mj_forward(model, data)

        return {"pos": pos, "quat": quat, "lin_vel": lin, "ang_vel": ang}

    def _capture_body_states(
        self,
        joint_pos: torch.Tensor,
        joint_vel: torch.Tensor,
        root_pos: torch.Tensor,
        root_quat: torch.Tensor,
        root_lin_vel: torch.Tensor,
        root_ang_vel: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Capture body states by temporarily setting the robot state in the simulator."""
        simulator = self._env.simulator
        sim_type = simulator.get_simulator_type()
        if sim_type == SimulatorType.MUJOCO:
            # MuJoCo/mjwarp path (2026-07-19). The IsaacSim path below works by writing state into
            # the sim and reading body poses back out, which relies on write_state_updates() running
            # forward kinematics -- IsaacGym's does not, and neither does holosoma's MuJoCo backend
            # (its GPU state push is not an FK call), so that trick silently returns stale poses.
            # But this is pure kinematics on ONE env at SETUP time (building the default-pose
            # prepend/append frames in _build_default_pose_state), so it does not need the parallel
            # GPU sim at all: the backend already keeps a single-env CPU model/data pair
            # (root_model/root_data) and already drives it with mujoco.mj_forward elsewhere
            # (e.g. set_ball_position, _set_initial_joint_angles). Reuse that directly.
            return self._capture_body_states_mujoco_fk(
                joint_pos, joint_vel, root_pos, root_quat, root_lin_vel, root_ang_vel
            )
        assert sim_type == SimulatorType.ISAACSIM, (
            f"Default-pose interpolation supports IsaacSim and MuJoCo; got {sim_type}. "
            "IsaacGym write_state_updates does not run FK."
        )
        env_id = 0
        env_origin = simulator.scene.env_origins[env_id].to(self.device)

        root_backup = simulator.robot_root_states[env_id].clone()
        dof_pos_backup = simulator.dof_pos[env_id].clone()
        dof_vel_backup = simulator.dof_vel[env_id].clone()

        try:
            simulator.robot_root_states[env_id, :3] = root_pos + env_origin
            simulator.robot_root_states[env_id, 3:7] = root_quat
            simulator.robot_root_states[env_id, 7:10] = root_lin_vel
            simulator.robot_root_states[env_id, 10:13] = root_ang_vel
            simulator.dof_pos[env_id] = joint_pos
            simulator.dof_vel[env_id] = joint_vel

            simulator.set_actor_root_state_tensor_robots()
            simulator.set_dof_state_tensor_robots()
            simulator.write_state_updates()
            simulator.refresh_sim_tensors()

            body_pos = (simulator._rigid_body_pos[env_id] - env_origin).clone()
            body_quat = simulator._rigid_body_rot[env_id].clone()
            body_lin_vel = simulator._rigid_body_vel[env_id].clone()
            body_ang_vel = simulator._rigid_body_ang_vel[env_id].clone()
        finally:
            simulator.robot_root_states[env_id] = root_backup
            simulator.dof_pos[env_id] = dof_pos_backup
            simulator.dof_vel[env_id] = dof_vel_backup
            simulator.set_actor_root_state_tensor_robots()
            simulator.set_dof_state_tensor_robots()
            simulator.write_state_updates()
            simulator.refresh_sim_tensors()

        return {
            "pos": body_pos,
            "quat": body_quat,
            "lin_vel": body_lin_vel,
            "ang_vel": body_ang_vel,
        }

    def _map_robot_bodies_to_motion_order(self, robot_tensor: torch.Tensor) -> torch.Tensor:
        """Map robot body tensor to motion data order using body indexes."""
        assert self._body_indexes_in_motion is not None
        num_motion_bodies = self.motion._body_pos_w.shape[1]
        motion_shape = (num_motion_bodies,) + robot_tensor.shape[1:]
        motion_tensor = torch.zeros(motion_shape, device=robot_tensor.device, dtype=robot_tensor.dtype)
        motion_tensor[self._body_indexes_in_motion] = robot_tensor
        return motion_tensor

    def _map_robot_joints_to_motion_order(
        self, robot_tensor: torch.Tensor, num_motion_joints: int | None = None
    ) -> torch.Tensor:
        """Map robot joint tensor to motion data order using joint indexes."""
        assert self._joint_indexes_in_motion is not None
        if num_motion_joints is None:
            num_motion_joints = self.motion._joint_pos.shape[1]
        motion_shape = robot_tensor.shape[:-1] + (num_motion_joints,)
        motion_tensor = torch.zeros(motion_shape, device=robot_tensor.device, dtype=robot_tensor.dtype)
        motion_tensor[..., self._joint_indexes_in_motion] = robot_tensor
        return motion_tensor

    def _motion_state(self, idx: int, dtype: torch.dtype, device: torch.device) -> dict[str, torch.Tensor]:
        """Slice motion tensors at a given index into a state dict."""
        state = {
            "joint_pos": self.motion._joint_pos[idx].to(device=device, dtype=dtype),
            "joint_vel": self.motion._joint_vel[idx].to(device=device, dtype=dtype),
            "body_pos": self.motion._body_pos_w[idx].to(device=device, dtype=dtype),
            "body_quat": self.motion._body_quat_w[idx].to(device=device, dtype=dtype),
            "body_lin_vel": self.motion._body_lin_vel_w[idx].to(device=device, dtype=dtype),
            "body_ang_vel": self.motion._body_ang_vel_w[idx].to(device=device, dtype=dtype),
        }
        if self.motion.has_object:
            state["object_pos"] = self.motion._object_pos_w[idx].to(device=device, dtype=dtype)
            state["object_quat"] = self.motion._object_quat_w[idx].to(device=device, dtype=dtype)
            state["object_lin_vel"] = self.motion._object_lin_vel_w[idx].to(device=device, dtype=dtype)
        return state

    def _default_motion_state(
        self, default_state: dict[str, torch.Tensor], dtype: torch.dtype, device: torch.device
    ) -> dict[str, torch.Tensor]:
        """Map default robot-state tensors into motion order for interpolation."""
        state = {
            "joint_pos": self._map_robot_joints_to_motion_order(
                default_state["joint_pos"].to(device=device, dtype=dtype),
                num_motion_joints=self.motion._joint_pos.shape[1],
            ),
            "joint_vel": self._map_robot_joints_to_motion_order(
                default_state["joint_vel"].to(device=device, dtype=dtype),
                num_motion_joints=self.motion._joint_vel.shape[1],
            ),
            "body_pos": default_state["body_pos"].to(device=device, dtype=dtype),
            "body_quat": default_state["body_quat"].to(device=device, dtype=dtype),
            "body_lin_vel": default_state["body_lin_vel"].to(device=device, dtype=dtype),
            "body_ang_vel": default_state["body_ang_vel"].to(device=device, dtype=dtype),
        }
        if self.motion.has_object:
            state["object_pos"] = default_state["object_pos"].to(device=device, dtype=dtype)
            state["object_quat"] = default_state["object_quat"].to(device=device, dtype=dtype)
            state["object_lin_vel"] = default_state["object_lin_vel"].to(device=device, dtype=dtype)
        return state

    def _build_transition_segments(
        self,
        start: dict[str, torch.Tensor],
        target: dict[str, torch.Tensor],
        alphas: torch.Tensor,
        alphas_joint: torch.Tensor,
        alphas_body: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Linearly/spherically interpolate between start and target states."""

        def _lerp(a: torch.Tensor, b: torch.Tensor, view: torch.Tensor) -> torch.Tensor:
            return a.unsqueeze(0) + view * (b - a).unsqueeze(0)

        segments = {
            "joint_pos": _lerp(start["joint_pos"], target["joint_pos"], alphas_joint),
            "joint_vel": _lerp(start["joint_vel"], target["joint_vel"], alphas_joint),
            "body_pos": _lerp(start["body_pos"], target["body_pos"], alphas_body),
            "body_lin_vel": _lerp(start["body_lin_vel"], target["body_lin_vel"], alphas_body),
            "body_ang_vel": _lerp(start["body_ang_vel"], target["body_ang_vel"], alphas_body),
            "body_quat": self._slerp_quat_sequence(start["body_quat"], target["body_quat"], alphas),
        }

        if self.motion.has_object:
            segments["object_pos"] = _lerp(start["object_pos"], target["object_pos"], alphas_joint)
            segments["object_lin_vel"] = _lerp(start["object_lin_vel"], target["object_lin_vel"], alphas_joint)
            segments["object_quat"] = self._slerp_quat_sequence(
                start["object_quat"].unsqueeze(0), target["object_quat"].unsqueeze(0), alphas
            ).squeeze(1)

        return segments

    def _apply_transition_segments(self, segments: dict[str, torch.Tensor], prepend: bool, motion_idx: int) -> None:
        """Splice interpolated segments into motion_idx's own span, either prepending (windup) or
        appending (recovery/hold) -- same method name on both MotionLoader and MultiMotionLoader,
        so this call site doesn't need to know which loader type is in use."""
        self.motion = self.motion.insert_segment_at_motion_boundary(motion_idx, segments, at_start=prepend)

    def _build_and_apply_transition(
        self,
        start_state: dict[str, torch.Tensor],
        target_state: dict[str, torch.Tensor],
        num_steps: int,
        prepend: bool,
        drop_first: bool,
        drop_last: bool,
        dtype: torch.dtype,
        device: torch.device,
        motion_idx: int,
    ) -> None:
        """Shared interpolation path for prepend/append transitions, applied to motion_idx's own
        span."""
        if num_steps <= 0:
            return

        alphas = torch.linspace(0.0, 1.0, steps=num_steps + 1, device=device, dtype=dtype)
        if drop_first:
            alphas = alphas[1:]
        if drop_last:
            alphas = alphas[:-1]
        if alphas.numel() == 0:
            return

        alphas_joint = alphas.view(num_steps, 1)
        alphas_body = alphas.view(num_steps, 1, 1)

        segments = self._build_transition_segments(start_state, target_state, alphas, alphas_joint, alphas_body)
        self._apply_transition_segments(segments, prepend=prepend, motion_idx=motion_idx)

    def _setup_visualization_markers_for_isaacsim(self):
        from isaaclab.markers import VisualizationMarkers
        from isaaclab.markers.config import FRAME_MARKER_CFG, RAY_CASTER_MARKER_CFG

        visualization_markers_cfg = FRAME_MARKER_CFG.replace(
            prim_path="/Visuals/Command/real_robot",
        )
        visualization_markers_cfg.markers["frame"].scale = (0.2, 0.2, 0.2)
        real_robot_visualizer = VisualizationMarkers(visualization_markers_cfg)

        visualization_markers_cfg = FRAME_MARKER_CFG.replace(
            prim_path="/Visuals/Command/motion_robot",
        )
        visualization_markers_cfg.markers["frame"].scale = (0.2, 0.2, 0.2)
        motion_robot_visualizer = VisualizationMarkers(visualization_markers_cfg)
        self.visualization_markers = {
            "real_robot": real_robot_visualizer,
            "motion_robot": motion_robot_visualizer,
        }

        for body_names in self.motion_cfg.body_names_to_track:
            visualization_markers_cfg = RAY_CASTER_MARKER_CFG.replace(
                prim_path=f"/Visuals/Command/motion_robot_body/motion_{body_names}",
            )
            visualization_markers_cfg.markers["hit"].radius = 0.03
            visualization_markers_cfg.markers["hit"].visual_material.diffuse_color = (0.0, 1.0, 0.0)
            self.visualization_markers[f"motion_{body_names}"] = VisualizationMarkers(visualization_markers_cfg)

        if self.motion.has_object:
            visualization_markers_cfg = FRAME_MARKER_CFG.replace(
                prim_path="/Visuals/Command/real_object",
            )
            visualization_markers_cfg.markers["frame"].scale = (0.2, 0.2, 0.2)
            real_object_visualizer = VisualizationMarkers(visualization_markers_cfg)

            visualization_markers_cfg = FRAME_MARKER_CFG.replace(
                prim_path="/Visuals/Command/motion_object",
            )
            visualization_markers_cfg.markers["frame"].scale = (0.2, 0.2, 0.2)
            motion_object_visualizer = VisualizationMarkers(visualization_markers_cfg)

            self.visualization_markers["real_object"] = real_object_visualizer
            self.visualization_markers["motion_object"] = motion_object_visualizer

    def _ensure_index_tensor(self, env_ids: torch.Tensor | None) -> torch.Tensor:
        if env_ids is None:
            return torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        if isinstance(env_ids, torch.Tensor):
            return env_ids.to(device=self.device, dtype=torch.long)
        return torch.as_tensor(env_ids, device=self.device, dtype=torch.long)

    def _get_index_of_a_in_b(self, a_names: List[str], b_names: List[str], device: str = "cpu") -> torch.Tensor:
        indexes = []
        for name in a_names:
            assert name in b_names, f"The specified name ({name}) doesn't exist: {b_names}"
            indexes.append(b_names.index(name))
        return torch.tensor(indexes, dtype=torch.long, device=device)
