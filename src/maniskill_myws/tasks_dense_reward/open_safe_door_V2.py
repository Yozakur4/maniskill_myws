from __future__ import annotations

import json
from typing import Any

import importlib.resources as importlib_resources
import numpy as np
import sapien
import torch

from mani_skill.agents.robots import Panda
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.envs.utils import randomization
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table.scene_builder import TableSceneBuilder
from mani_skill.utils.structs.articulation import Articulation
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import SimConfig
from maniskill_myws.task_prompts import TASK_PROMPTS


@register_env("OpenSafeDoor-v1", max_episode_steps=250)
class OpenSafeDoorEnv(BaseEnv):
    """
    Task: open a safe door (asset 101593) with Panda.

    Scene:
      - Panda fixed on a table
      - A safe (URDF from asset 101593) fixed to the table; XY/yaw randomized per episode

    Success (two-stage):
      1) Rotate knob/handle by > handle_turn_threshold (default 90 deg)
      2) Open door by > door_open_threshold (default 60 deg)
    """

    SUPPORTED_REWARD_MODES = ["sparse", "dense", "none"]
    SUPPORTED_ROBOTS = ["panda", "panda_wristcam"]
    agent: Panda
    DEFAULT_TASK_PROMPT = TASK_PROMPTS["OpenSafeDoor-v1"]

    # Joint semantics from mobility_v2.json:
    # - joint_0: door hinge (revolute, limited)
    # - joint_1/joint_2: knobs (continuous)
    DOOR_JOINT_NAME = "joint_0"
    # User requested: use the LEFT knob with handle. According to mobility_v2.json,
    # joint_1 has axis origin x < 0 in the door frame, which corresponds to the left side.
    HANDLE_JOINT_NAME = "joint_1"

    def __init__(
        self,
        *args,
        robot_uids="panda_wristcam",
        robot_init_qpos_noise: float = 0.02,
        # The safe is large; use a tighter placement distribution centered in front of the robot.
        # NOTE: default placement is slightly offset to +Y to leave room for the door to swing open.
        safe_spawn_center_x: float = 0.7,
        safe_spawn_center_y: float = 0,
        safe_spawn_half_size_x: float = 0.06,
        safe_spawn_half_size_y: float = 0.05,
        # Keep yaw noise smaller so the door is less likely to swing into the robot/table.
        safe_yaw_noise: float = np.pi / 12,
        handle_turn_threshold: float = np.pi / 2,
        door_open_threshold: float = np.pi / 3,
        extra_table_half_size_x: float = 1.3,
        extra_table_half_size_y: float = 1.3,
        extra_table_thickness: float = 0.04,
        **kwargs,
    ):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        self.safe_spawn_center_x = float(safe_spawn_center_x)
        self.safe_spawn_center_y = float(safe_spawn_center_y)
        self.safe_spawn_half_size_x = float(safe_spawn_half_size_x)
        self.safe_spawn_half_size_y = float(safe_spawn_half_size_y)
        self.safe_yaw_noise = safe_yaw_noise
        self.handle_turn_threshold = float(handle_turn_threshold)
        self.door_open_threshold = float(door_open_threshold)
        self.extra_table_half_size_x = float(extra_table_half_size_x)
        self.extra_table_half_size_y = float(extra_table_half_size_y)
        self.extra_table_thickness = float(extra_table_thickness)

        # Precompute a z offset so the safe sits on the table surface (z=0).
        bbox_path = importlib_resources.files("maniskill_myws").joinpath(
            "assets/101593/bounding_box.json"
        )
        with importlib_resources.as_file(bbox_path) as p:
            bbox = json.loads(p.read_text())
        self._safe_table_z = float(-bbox["min"][2]) + 1e-3

        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sim_config(self):
        return SimConfig()

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at([-0.55, 0.0, 1.0], [0.0, 0.0, 0.2])
        return [
            CameraConfig("base_camera", pose=pose, width=128, height=128, fov=np.pi / 2)
        ]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([-0.55, 0.0, 1.0], [0.0, 0.0, 0.2])
        return CameraConfig("render_camera", pose=pose, width=512, height=512, fov=1)

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0, 0]))

    def _load_scene(self, options: dict):
        self.scene_builder = TableSceneBuilder(
            self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.scene_builder.build()

        # Add an extra large kinematic tabletop to provide enough workspace area for the large safe.
        # TableSceneBuilder's table surface is at z=0. We create a thin platform at z=0.
        builder = self.scene.create_actor_builder()
        builder.add_box_collision(
            pose=sapien.Pose(p=[0.0, 0.0, self.extra_table_thickness / 2]),
            half_size=(
                self.extra_table_half_size_x,
                self.extra_table_half_size_y,
                self.extra_table_thickness / 2,
            ),
        )
        # Visual: simple box (no mesh dependency)
        builder.add_box_visual(
            pose=sapien.Pose(p=[0.0, 0.0, self.extra_table_thickness / 2]),
            half_size=(
                self.extra_table_half_size_x,
                self.extra_table_half_size_y,
                self.extra_table_thickness / 2,
            ),
            material=sapien.render.RenderMaterial(base_color=[0.25, 0.25, 0.25, 1.0]),
        )
        self.extra_table = builder.build_kinematic(name="extra_table")

        loader = self.scene.create_urdf_loader()
        loader.fix_root_link = True

        asset_dir = importlib_resources.files("maniskill_myws").joinpath("assets/101593")
        with importlib_resources.as_file(asset_dir) as asset_path:
            urdf_path = asset_path / "mobility.urdf"
            self.safe: Articulation = loader.load(
                str(urdf_path),
                name="safe_101593",
                scene_idxs=torch.arange(self.num_envs, dtype=torch.int32),
                package_dir=str(asset_path),
            )

        self.door_joint = self.safe.active_joints_map[self.DOOR_JOINT_NAME]
        self.handle_joint = self.safe.active_joints_map[self.HANDLE_JOINT_NAME]

        # Track cumulative (unwrapped) handle rotation to robustly detect turning > threshold.
        self._handle_qpos_prev = torch.zeros(self.num_envs, device=self.device)
        self._handle_cumulative = torch.zeros(self.num_envs, device=self.device)
        self._success_latched = None

        # Slightly reduce friction to make manipulation easier.
        for j in self.safe.active_joints:
            j.set_friction(0.1)
            j.set_drive_properties(0.0, 0.0)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            self.scene_builder.initialize(env_idx)
            b = len(env_idx)

            # Randomize safe pose on the table (XY and yaw).
            p = torch.zeros((b, 3), device=self.device)
            p[:, 0] = self.safe_spawn_center_x + randomization.uniform(
                -self.safe_spawn_half_size_x,
                self.safe_spawn_half_size_x,
                size=(b,),
                device=self.device,
            )
            p[:, 1] = self.safe_spawn_center_y + randomization.uniform(
                -self.safe_spawn_half_size_y,
                self.safe_spawn_half_size_y,
                size=(b,),
                device=self.device,
            )
            p[:, 2] = self._safe_table_z
            q = randomization.random_quaternions(
                n=b,
                device=self.device,
                lock_x=True,
                lock_y=True,
                bounds=(-self.safe_yaw_noise, self.safe_yaw_noise),
            )
            self.safe.set_pose(Pose.create_from_pq(p, q))

            # Initialize joint positions: door closed, knobs at 0.
            qpos = torch.zeros((b, 3), device=self.device)
            self.safe.set_qpos(qpos)

            # Reset handle cumulative trackers.
            self._handle_qpos_prev[env_idx] = 0.0
            self._handle_cumulative[env_idx] = 0.0

            # Reset success latch.
            if self._success_latched is None:
                self._success_latched = torch.zeros(
                    (self.num_envs,), dtype=torch.bool, device=self.device
                )
            self._success_latched[env_idx] = False

    def evaluate(self):
        door_angle = self.door_joint.qpos  # (B,), starts from 0

        # Update cumulative handle angle.
        handle_qpos = self.handle_joint.qpos
        delta = handle_qpos - self._handle_qpos_prev
        delta = torch.atan2(torch.sin(delta), torch.cos(delta))
        self._handle_cumulative = self._handle_cumulative + delta
        self._handle_qpos_prev = handle_qpos
        handle_angle = torch.abs(self._handle_cumulative)

        # Lazy init for success latch
        if self._success_latched is None:
            self._success_latched = torch.zeros(
                (self.num_envs,), dtype=torch.bool, device=self.device
            )

        handle_turned = handle_angle > self.handle_turn_threshold
        door_open = door_angle > self.door_open_threshold
        current_success = handle_turned & door_open
        self._success_latched |= current_success
        success = self._success_latched

        return dict(
            success=success,
            handle_turned=handle_turned,
            door_open=door_open,
            handle_angle=handle_angle,
            door_angle=door_angle,
        )

    def _get_obs_extra(self, info: dict):
        obs = dict(
            tcp_pose=self.agent.tcp.pose.raw_pose,
            handle_angle=info.get("handle_angle"),
            door_angle=info.get("door_angle"),
        )
        if "state" in self.obs_mode:
            obs["safe_pose"] = self.safe.pose.raw_pose
            obs["safe_qpos"] = self.safe.qpos
        return obs

    def compute_sparse_reward(self, obs: Any, action: torch.Tensor, info: dict):
        return info["success"].to(torch.float32)

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        # Stage 1: Reach handle
        handle_link = self.safe.links_map[self.handle_joint.name]
        handle_pos = handle_link.pose.p
        tcp_to_handle = torch.norm(self.agent.tcp.pose.p - handle_pos, dim=-1)
        reach_reward = 1.0 - torch.tanh(5.0 * tcp_to_handle)

        # Stage 2: Turn handle (smooth progress)
        handle_progress = torch.clamp(
            info["handle_angle"] / self.handle_turn_threshold, 0.0, 1.0
        )
        turn_reward = handle_progress

        # Stage 3: Open door (smooth gate using sigmoid)
        # sigmoid centered at threshold, steep transition over ~20° window
        handle_gate = torch.sigmoid(10.0 * (info["handle_angle"] - self.handle_turn_threshold))
        door_progress = torch.clamp(
            info["door_angle"] / self.door_open_threshold, 0.0, 1.0
        )
        open_reward = handle_gate * door_progress

        reward = 1.0 * reach_reward + 3.0 * turn_reward + 6.0 * open_reward
        return reward


