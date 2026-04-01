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


@register_env("TurnGlobeValve-v1", max_episode_steps=200)
class TurnGlobeValveEnv(BaseEnv):
    """
    Tabletop task: turn a fixed globe valve with a Panda arm.

    Scene:
      - Panda mounted at a fixed pose relative to the table
      - Globe valve URDF is loaded as a fixed-root articulation and placed on the table;
        its XY and yaw are randomized each reset

    Success:
      - The handwheel rotates by more than 180 degrees (pi rad), using an unwrapped cumulative angle
    """

    SUPPORTED_REWARD_MODES = ["sparse", "dense", "none"]
    SUPPORTED_ROBOTS = ["panda", "panda_wristcam"]
    agent: Panda
    DEFAULT_TASK_PROMPT = TASK_PROMPTS["TurnGlobeValve-v1"]

    def __init__(
        self,
        *args,
        robot_uids="panda_wristcam",
        robot_init_qpos_noise: float = 0.02,
        valve_xy_noise: float = 0.08,
        valve_yaw_noise: float = np.pi / 6,
        valve_init_qpos_noise: float = np.pi,
        success_threshold: float = np.pi,
        success_dist_thresh: float = 0.05,
        dist_score_threshold: float = 0.5,
        success_hold_steps: int = 5,
        **kwargs,
    ):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        self.valve_xy_noise = valve_xy_noise
        self.valve_yaw_noise = valve_yaw_noise
        self.valve_init_qpos_noise = valve_init_qpos_noise
        self.success_threshold = float(success_threshold)
        self.success_dist_thresh = success_dist_thresh
        self.dist_score_threshold = dist_score_threshold
        self.success_hold_steps = success_hold_steps
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sim_config(self):
        return SimConfig()

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at([-0.45, 0.0, 0.35], [0.0, 0.0, 0.10])
        return [
            CameraConfig("base_camera", pose=pose, width=128, height=128, fov=np.pi / 2)
        ]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([0.6, 0.6, 0.9], [0.0, 0.0, 0.2])
        return CameraConfig("render_camera", pose=pose, width=512, height=512, fov=1)

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0, 0]))

    def _load_scene(self, options: dict):
        self.scene_builder = TableSceneBuilder(
            self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.scene_builder.build()

        loader = self.scene.create_urdf_loader()
        loader.fix_root_link = True

        # Use workspace-shipped asset (works even when ManiSkill is installed via pip).
        base0_dir = importlib_resources.files("maniskill_myws").joinpath(
            "assets/globe_valve/base0"
        )
        with importlib_resources.as_file(base0_dir) as base0_path:
            urdf_path = base0_path / "mobility.urdf"
            self.valve: Articulation = loader.load(
                str(urdf_path),
                name="globe_valve",
                scene_idxs=torch.arange(self.num_envs, dtype=torch.int32),
                package_dir=str(base0_path),
            )
        self.handwheel_joint = self.valve.active_joints_map["handwheel_joint"]

        self._handwheel_qpos_prev = torch.zeros(self.num_envs, device=self.device)
        self._handwheel_cumulative = torch.zeros(self.num_envs, device=self.device)
        self._success_counter = torch.zeros(
            self.num_envs, dtype=torch.int32, device=self.device
        )
        self._success_latched = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )

        # Make the joint easier to turn.
        for j in self.valve.active_joints:
            j.set_friction(0.01)
            j.set_drive_properties(0.0, 0.0)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            self.scene_builder.initialize(env_idx)
            b = len(env_idx)

            # Randomize valve pose on the table (XY and yaw).
            p = torch.zeros((b, 3), device=self.device)
            p[:, :2] = randomization.uniform(
                -self.valve_xy_noise,
                self.valve_xy_noise,
                size=(b, 2),
                device=self.device,
            )
            p[:, 2] = 1e-3  # small clearance above the table
            q = randomization.random_quaternions(
                n=b,
                device=self.device,
                lock_x=True,
                lock_y=True,
                bounds=(-self.valve_yaw_noise, self.valve_yaw_noise),
            )
            self.valve.set_pose(Pose.create_from_pq(p, q))

            # Randomize initial handwheel joint angle.
            qpos0 = randomization.uniform(
                -self.valve_init_qpos_noise,
                self.valve_init_qpos_noise,
                size=(b, 1),
                device=self.device,
            )
            self.valve.set_qpos(qpos0)

            self._handwheel_qpos_prev[env_idx] = qpos0[:, 0]
            self._handwheel_cumulative[env_idx] = 0.0
            self._success_counter[env_idx] = 0
            self._success_latched[env_idx] = False

    def evaluate(self):
        # Unwrap delta angle to accumulate rotation beyond [-pi, pi].
        qpos = self.handwheel_joint.qpos
        delta = qpos - self._handwheel_qpos_prev
        delta = torch.atan2(torch.sin(delta), torch.cos(delta))
        self._handwheel_cumulative = self._handwheel_cumulative + delta
        self._handwheel_qpos_prev = qpos

        valve_rotation = self._handwheel_cumulative
        rotated_enough = torch.abs(valve_rotation) > self.success_threshold

        tcp_xy = self.agent.tcp.pose.p[:, :2]
        valve_xy = self.valve.pose.p[:, :2]
        dist_xy = torch.linalg.norm(tcp_xy - valve_xy, dim=1)
        dist_score = 1.0 - torch.clamp(dist_xy / self.success_dist_thresh, 0.0, 1.0)
        near_enough = dist_score > self.dist_score_threshold

        # Immediate latch: once both conditions met, lock success
        current_success = rotated_enough & near_enough
        self._success_latched |= current_success
        success = self._success_latched

        return {
            "success": success,
            "valve_rotation": valve_rotation,
            "handwheel_qpos": qpos,
        }

    def _get_obs_extra(self, info: dict):
        obs = dict(
            tcp_pose=self.agent.tcp.pose.raw_pose,
            valve_rotation=info.get("valve_rotation", self._handwheel_cumulative),
            handwheel_qpos=info.get("handwheel_qpos", self.handwheel_joint.qpos),
        )
        if "state" in self.obs_mode:
            obs["valve_pose"] = self.valve.pose.raw_pose
        return obs

    def compute_sparse_reward(self, obs: Any, action: torch.Tensor, info: dict):
        return info["success"].to(torch.float32)

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        # Stage 1: Reach valve center
        tcp_pos = self.agent.tcp.pose.p
        valve_pos = self.valve.pose.p
        dist_tcp_valve = torch.norm(tcp_pos - valve_pos, dim=-1)
        reach_reward = 1.0 - torch.tanh(5.0 * dist_tcp_valve)

        # Stage 2: Rotation progress (单向，移除 abs)
        rotation_progress = torch.clamp(
            info["valve_rotation"] / self.success_threshold, 0.0, 1.0
        )

        # 距离门控：更平滑的过渡
        dist_gate = torch.sigmoid(10.0 * (0.10 - dist_tcp_valve))
        rotation_reward = dist_gate * rotation_progress

        # Success bonus: 平滑过渡
        abs_rotation = torch.abs(info["valve_rotation"])
        success_smooth = torch.sigmoid(5.0 * (abs_rotation - 0.9 * self.success_threshold))
        success_bonus = 10.0 * success_smooth

        reward = 1.0 * reach_reward + 8.0 * rotation_reward + success_bonus
        return reward


