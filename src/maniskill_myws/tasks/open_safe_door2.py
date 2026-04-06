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


@register_env("OpenSafeDoor-v2", max_episode_steps=250)
class OpenSafeDoor2Env(BaseEnv):
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
    DEFAULT_TASK_PROMPT = TASK_PROMPTS["OpenSafeDoor-v2"]

    # Joint semantics from mobility_v2.json:
    # - joint_door: door hinge (revolute, limited)
    # - joint_bar: handle/knob (revolute)
    # - joint_button: button (prismatic)
    DOOR_JOINT_NAME = "joint_door"
    HANDLE_JOINT_NAME = "joint_bar"
    BUTTON_JOINT_NAME = "joint_button"

    def __init__(
        self,
        *args,
        robot_uids="panda_wristcam",
        robot_init_qpos_noise: float = 0.02,
        # The safe is large; use a tighter placement distribution centered in front of the robot.
        # NOTE: default placement is slightly offset to +Y to leave room for the door to swing open.
        safe_spawn_center_x: float = 0.1,
        safe_spawn_center_y: float = -0.6,
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

        self._safe_table_z = 1e-3

        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sim_config(self):
        return SimConfig()

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at([-0.55, 0.0, 0.45], [0.0, 0.0, 0.15])
        return [
            CameraConfig("base_camera", pose=pose, width=128, height=128, fov=np.pi / 2)
        ]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([-0.7, -0.6, 0.9], [0.0, 0.0, 0.2])
        return CameraConfig("render_camera", pose=pose, width=512, height=512, fov=1)

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0, 0]))

    def _load_scene(self, options: dict):
        self.scene_builder = TableSceneBuilder(
            self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.scene_builder.build()

        # Add extra large table for the big safe.
        # builder = self.scene.create_actor_builder()
        # builder.add_box_collision(
        #     pose=sapien.Pose(p=[0.0, 0.0, self.extra_table_thickness / 2]),
        #     half_size=(
        #         self.extra_table_half_size_x,
        #         self.extra_table_half_size_y,
        #         self.extra_table_thickness / 2,
        #     ),
        # )
        # builder.add_box_visual(
        #     pose=sapien.Pose(p=[0.0, 0.0, self.extra_table_thickness / 2]),
        #     half_size=(
        #         self.extra_table_half_size_x,
        #         self.extra_table_half_size_y,
        #         self.extra_table_thickness / 2,
        #     ),
        #     material=sapien.render.RenderMaterial(base_color=[0.25, 0.25, 0.25, 1.0]),
        # )
        # self.extra_table = builder.build_kinematic(name="extra_table")

        loader = self.scene.create_urdf_loader()
        loader.fix_root_link = True
        loader.disable_self_collisions = True
        asset_dir = importlib_resources.files("maniskill_myws").joinpath("assets/door/urdf")
        with importlib_resources.as_file(asset_dir) as asset_path:
            urdf_path = asset_path / "door.urdf"
            articulations, _ = loader.load_multiple(
                str(urdf_path),
                package_dir=str(asset_path),
            )
        assert len(articulations) == 1
        self.safe: Articulation = articulations[0]

        self.door_joint = self.safe.active_joints_map[self.DOOR_JOINT_NAME]
        self.handle_joint = self.safe.active_joints_map[self.HANDLE_JOINT_NAME]
        self.button_joint = self.safe.active_joints_map[self.BUTTON_JOINT_NAME]

        self.door_closed_angle = float(self.door_joint.get_limits()[0, 0].item())
        self.door_max_open = float(self.door_joint.get_limits()[0, 1].item() - self.door_closed_angle)
        self.bar_handle_closed_angle = float(self.handle_joint.get_limits()[0, 1].item())
        #self.button_pressed_pos = float(self.button_joint.get_limits()[0, 0].item())
        self.button_unpressed_pos = float(self.button_joint.get_limits()[0, 1].item())

        for j in self.safe.active_joints:
            j.set_friction(0.1)
            j.set_drive_properties(0.0, 0.0)

        # start locked, unlock once condition satisfied
        self.door_joint.set_friction(1000.0)

        self._handle_qpos_prev = torch.zeros(self.num_envs, device=self.device)
        self._handle_cumulative = torch.zeros(self.num_envs, device=self.device)
        self._door_released = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            self.scene_builder.initialize(env_idx)
            b = len(env_idx)

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

            qpos = torch.zeros((b, 3), device=self.device)
            qpos[:, 0] = self.door_closed_angle
            qpos[:, 2] = self.button_unpressed_pos
            self.safe.set_qpos(qpos)

            self._handle_qpos_prev[env_idx] = 0.0
            self._handle_cumulative[env_idx] = 0.0
            self._door_released[env_idx] = False
            self.door_joint.set_friction(1000.0)

    def evaluate(self):
        handle_qpos = self.handle_joint.qpos
        delta = handle_qpos - self._handle_qpos_prev
        delta = torch.atan2(torch.sin(delta), torch.cos(delta))
        self._handle_cumulative = self._handle_cumulative + delta
        self._handle_qpos_prev = handle_qpos
        handle_angle = torch.abs(self._handle_cumulative)

        button_qpos = self.button_joint.qpos
        button_pressed = button_qpos >= 0.04
        handle_twisted = handle_angle >= 1.2

        release = handle_twisted | button_pressed
        self._door_released = self._door_released | release

        # hard lock closed until release conditions met
        door_qpos = self.door_joint.qpos
        if not torch.all(self._door_released):
            door_qpos = torch.where(
                self._door_released,
                door_qpos,
                torch.full_like(door_qpos, self.door_closed_angle),
            )
            self.door_joint.qpos = door_qpos
            self.door_joint.set_friction(1000.0)

        if torch.any(self._door_released):
            self.door_joint.set_friction(0.1)

        door_open_amount = torch.clamp(door_qpos - self.door_closed_angle, min=0.0)
        door_open_ratio = torch.clamp(door_open_amount / self.door_max_open, min=0.0, max=1.0)
        door_open = door_open_amount >= self.door_open_threshold

        success = door_open & self._door_released

        return dict(
            success=success,
            handle_angle=handle_angle,
            button_qpos=button_qpos,
            door_open_amount=door_open_amount,
            door_open_ratio=door_open_ratio,
            door_open=door_open,
            door_released=self._door_released,
        )

    def _get_obs_extra(self, info: dict):
        obs = dict(
            tcp_pose=self.agent.tcp.pose.raw_pose,
            handle_angle=info.get("handle_angle"),
            button_qpos=info.get("button_qpos"),
            door_open_amount=info.get("door_open_amount"),
            door_open_ratio=info.get("door_open_ratio"),
            door_released=info.get("door_released"),
        )
        if "state" in self.obs_mode:
            obs["safe_pose"] = self.safe.pose.raw_pose
            obs["safe_qpos"] = self.safe.qpos
        return obs

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        handle_reward = torch.clamp(info["handle_angle"] / 1.2, 0.0, 1.0)
        button_reward = torch.clamp(info["button_qpos"] / 0.04, 0.0, 1.0)
        door_reward = info.get("door_open_ratio", torch.zeros(self.num_envs, device=self.device))
        return 0.3 * handle_reward + 0.3 * button_reward + 0.4 * door_reward

    def compute_sparse_reward(self, obs: Any, action: torch.Tensor, info: dict):
        return info["success"].to(torch.float32)


          