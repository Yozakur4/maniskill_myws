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


@register_env("TakeSafetyHook-v1", max_episode_steps=200)
class TakeSafetyHookEnv(BaseEnv):
    """
    Tabletop task: turn a fixed globe valve with a Panda arm.

    Scene:
      - Panda mounted at a fixed pose relative to the table
      - Globe valve URDF is loaded as a fixed-root articulation and placed on the table;
        its XY and yaw are randomized each reset

    Success:
      - The handwheel rotates by more than 180 degrees (pi rad), using an unwrapped cumulative angle
    """

    SUPPORTED_REWARD_MODES = ["sparse", "none"]
    SUPPORTED_ROBOTS = ["panda", "panda_wristcam"]
    agent: Panda
    DEFAULT_TASK_PROMPT = TASK_PROMPTS["TakeSafetyHook-v1"]

    def __init__(
        self,
        *args,
        robot_uids="panda_wristcam",
        robot_init_qpos_noise: float = 0.02,
        hook_xy_noise: float = 0.08,
        hook_yaw_noise: float = np.pi / 6,
        hook_init_qpos_noise: float = 0.1,
        success_threshold: float = np.pi / 4,
        beam_length: float = 0.5,
        beam_radius: float = 0.015,
        beam_x_range: tuple[float, float] = (0.2, 0.6),
        beam_y_range: tuple[float, float] = (-0.25, 0.25),
        beam_z_range: tuple[float, float] = (0.25, 0.45),
        **kwargs,
    ):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        self.hook_xy_noise = hook_xy_noise
        self.hook_yaw_noise = hook_yaw_noise
        self.hook_init_qpos_noise = hook_init_qpos_noise
        self.success_threshold = float(success_threshold)

        self.beam_length = float(beam_length)
        self.beam_radius = float(beam_radius)
        self.beam_x_range = beam_x_range
        self.beam_y_range = beam_y_range
        self.beam_z_range = beam_z_range

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
        loader.fix_root_link = False
        loader.disable_self_collisions = True
        
        # Use workspace-shipped asset (works even when ManiSkill is installed via pip).
        base0_dir = importlib_resources.files("maniskill_myws").joinpath(
            "assets/safety_hook2/urdf"
        )
        with importlib_resources.as_file(base0_dir) as base0_path:
            urdf_path = base0_path / "safety_hook.urdf"
            articulations, _  = loader.load_multiple(
                str(urdf_path),
                package_dir=str(base0_path),
            )
        assert len(articulations) == 1, "URDF should only contain one articulation"
        self.hook = articulations[0]

        self.gate_joint = self.hook.active_joints_map["joint_bar"]
        self.gate_joint.set_drive_properties(150.0, 15.0)
        self.gate_joint.set_drive_target(-0.02)
        self.gate_joint.set_friction(0.1)

        # beam: thin horizontal cylinder in mid-air, static/kinematic
        beam_builder = self.scene.create_actor_builder()
        beam_builder.add_cylinder_collision(
            pose=sapien.Pose(),
            radius=self.beam_radius,
            half_length=self.beam_length / 2,
        )
        beam_builder.add_cylinder_visual(
            pose=sapien.Pose(),
            radius=self.beam_radius,
            half_length=self.beam_length / 2,
            material=sapien.render.RenderMaterial(base_color=[0.5, 0.5, 0.5, 1.0]),
        )
        self.beam = beam_builder.build_kinematic(name="beam")

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            self.scene_builder.initialize(env_idx)
            b = len(env_idx)

            # random beam pose (position + yaw) on each episode
            beam_p = torch.zeros((b, 3), device=self.device)
            beam_p[:, 0] = randomization.uniform(
                self.beam_x_range[0], self.beam_x_range[1], size=(b,), device=self.device
            )
            beam_p[:, 1] = randomization.uniform(
                self.beam_y_range[0], self.beam_y_range[1], size=(b,), device=self.device
            )
            beam_p[:, 2] = randomization.uniform(
                self.beam_z_range[0], self.beam_z_range[1], size=(b,), device=self.device
            )
            beam_q = randomization.random_quaternions(
                n=b,
                device=self.device,
                lock_x=True,
                lock_y=True,
                bounds=(-1/3*np.pi, 1/3*np.pi),
            )
            # set beam in all envs
            self.beam.set_pose(Pose.create_from_pq(beam_p, beam_q))

            # random hook placement on table-area
            hook_p = torch.zeros((b, 3), device=self.device)
            hook_p[:, 0] = randomization.uniform(
                0.0, 0.4, size=(b,), device=self.device
            )
            hook_p[:, 1] = randomization.uniform(
                -0.2, 0.2, size=(b,), device=self.device
            )
            hook_p[:, 2] = 0.2
            hook_q = randomization.random_quaternions(
                n=b,
                device=self.device,
                lock_x=True,
                lock_y=True,
                bounds=(-self.hook_yaw_noise, self.hook_yaw_noise),
            )
            self.hook.set_pose(Pose.create_from_pq(hook_p, hook_q))

            qpos0 = torch.full((b, 1), -0.02, device=self.device)
            self.hook.set_qpos(qpos0)

            self._hook_qpos_prev = qpos0[:, 0].clone()

    def evaluate(self):
        gate_qpos = self.gate_joint.qpos
        # We do a simple proxy for now; later change to beam-hang detection by keypoints.
        progress = torch.clamp((gate_qpos - 1.57) / (2.35 - 1.57), 0.0, 1.0)
        success = progress >= torch.tensor(0.75, device=self.device)

        return {
            "success": success,
            "progress": progress,
            "hook_qpos": gate_qpos,
            "beam_pose": self.beam.pose if hasattr(self.beam, "pose") else None,
        }

    def _get_obs_extra(self, info: dict):
        obs = dict(
            tcp_pose=self.agent.tcp.pose.raw_pose,
            hook_qpos=self.gate_joint.qpos,
            gate_progress=info.get("progress"),
        )
        if "state" in self.obs_mode:
            obs["hook_pose"] = self.hook.pose.raw_pose
            obs["beam_pose"] = self.beam.pose if hasattr(self.beam, "pose") else None
            obs["hook_qpos"] = self.gate_joint.qpos
        return obs

    def compute_sparse_reward(self, obs: Any, action: torch.Tensor, info: dict):
        return info["success"].to(torch.float32)
