from __future__ import annotations

import importlib.resources as importlib_resources
import math

import numpy as np
import sapien
import sapien.render
import torch

from mani_skill.agents.robots import Panda
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.envs.utils import randomization
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table.scene_builder import TableSceneBuilder
from mani_skill.utils.structs.actor import Actor
from mani_skill.utils.structs.articulation import Articulation
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import SimConfig
from maniskill_myws.task_prompts import TASK_PROMPTS

# ---------------------------------------------------------------------------
# Quaternion constants (SAPIEN: w, x, y, z)
# ---------------------------------------------------------------------------
# 90° around X: rotate so hook hangs with Z axis pointing DOWN from the rod
# (hook local Z becomes world -Z when hook is "hanging" with base at bottom).
# We orient the hook so it hangs vertically: local +Z is world +Z (no flip).
# The hook sits BELOW the rod, attached at its top (hinge block area).

# The rod runs along world Y axis.  We need to rotate so the rod-axis (capsule
# local X) becomes world Y: 90° rotation around world Z.
_Q_ROD_Y: list[float] = [math.sqrt(2) / 2, 0.0, 0.0, math.sqrt(2) / 2]   # 90° Z

# Hook orientation when hanging: identity — the hook's local frame already has
# +Z up and +X to the right (gate side), matching world frame.
_Q_IDENTITY: list[float] = [1.0, 0.0, 0.0, 0.0]


@register_env("OpenSafetyHook-v1", max_episode_steps=300)
class OpenSafetyHookEnv(BaseEnv):
    """
    Task: Open a safety hook (carabiner-style) hanging on a horizontal rod.

    Scene (TableSceneBuilder, table top at z=0):
      - Panda (panda_wristcam) fixed at x=-0.615
      - A horizontal steel rod (cylinder, static) positioned ~0.35 m above table.
        Rod runs along world Y axis, centred at x≈0.35.
      - Safety hook articulation: D-shaped frame + screw-gate (1 revolute DOF).
        The hook's base link is kinematically attached (hung) from the rod:
        the URDF root is fixed to the scene so that the top of the frame sits
        just below the rod.

    Geometry summary:
      Frame: 70 mm wide × 24 mm thick × 150 mm tall.
      Gate hinge: at base frame (0.030, 0, 0.140).
      Gate open range: 0 → 70 ° (1.222 rad).
      Gate return spring: implemented via drive with stiffness & damping.

    Success:
      Gate is open past gate_open_threshold AND the hook's gate is disengaged
      from the rod (hook has been pulled away: hook top-of-frame is below
      rod centre-z by more than hook_release_z_drop).

    In practice: robot must push the gate open, then slide the hook OFF the rod.
    """

    SUPPORTED_REWARD_MODES = ["sparse", "none"]
    SUPPORTED_ROBOTS = ["panda", "panda_wristcam"]
    agent: Panda
    DEFAULT_TASK_PROMPT = TASK_PROMPTS["OpenSafetyHook-v1"]

    # ------------------------------------------------------------------
    # Geometry constants
    # ------------------------------------------------------------------
    # Rod
    ROD_RADIUS: float = 0.012          # 12 mm radius steel rod
    ROD_HALF_LENGTH: float = 0.18      # rod spans ±0.18 m in Y

    # Hook frame dims (meters, matching mobility.urdf / meshes)
    HOOK_WIDTH: float = 0.070          # X extent
    HOOK_THICK: float = 0.024          # Y extent
    HOOK_HEIGHT: float = 0.150         # Z extent (base bottom to top)
    # Hinge point in base local frame
    HINGE_LOCAL: list[float] = [0.030, 0.0, 0.140]
    # Gate open threshold for "disengaged" check
    GATE_OPEN_THRESHOLD_RAD: float = math.radians(55.0)  # 55°
    # After opening gate, hook must drop/slide so top is below rod surface
    HOOK_RELEASE_Z_DROP: float = 0.025   # top of hook must fall ≥25 mm below rod bottom

    # Gate return spring (applied via drive in _load_scene)
    GATE_SPRING_STIFFNESS: float = 0.8   # Nm/rad  (equivalent torsion spring)
    GATE_SPRING_DAMPING: float = 0.10    # Nm·s/rad

    def __init__(
        self,
        *args,
        robot_uids: str = "panda_wristcam",
        robot_init_qpos_noise: float = 0.02,
        # Rod centre position (world frame, table top = z=0).
        # Rod is positioned above the table in front of the robot.
        rod_center_x: float = 0.28,
        rod_center_y: float = 0.00,
        rod_center_z: float = 0.32,
        # Spawn randomisation for hook along the rod (Y jitter)
        hook_spawn_y_range: float = 0.05,
        # Thresholds
        gate_open_threshold: float = GATE_OPEN_THRESHOLD_RAD,
        hook_release_z_drop: float = HOOK_RELEASE_Z_DROP,
        **kwargs,
    ):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        self.rod_center_x = float(rod_center_x)
        self.rod_center_y = float(rod_center_y)
        self.rod_center_z = float(rod_center_z)
        self.hook_spawn_y_range = float(hook_spawn_y_range)
        self.gate_open_threshold = float(gate_open_threshold)
        self.hook_release_z_drop = float(hook_release_z_drop)

        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    # ------------------------------------------------------------------
    # Sim / sensor config
    # ------------------------------------------------------------------

    @property
    def _default_sim_config(self):
        return SimConfig()

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(
            eye=[-0.10, 0.50, 0.60],
            target=[0.30, 0.0, 0.30],
        )
        return [
            CameraConfig("base_camera", pose=pose, width=128, height=128, fov=np.pi / 2)
        ]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at(
            eye=[-0.15, 0.65, 0.70],
            target=[0.28, 0.0, 0.28],
        )
        return CameraConfig("render_camera", pose=pose, width=512, height=512, fov=1.0)

    # ------------------------------------------------------------------
    # Scene loading
    # ------------------------------------------------------------------

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0, 0]))

    def _load_scene(self, options: dict):
        self.scene_builder = TableSceneBuilder(
            self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.scene_builder.build()

        # ------------------------------------------------------------------
        # Horizontal rod (static cylinder along Y axis)
        # The hook hangs from this rod.  Rod is placed at (rod_x, 0, rod_z).
        # SAPIEN cylinder axis defaults to X, so we rotate 90° around Z to
        # align the cylinder axis with world Y.
        # ------------------------------------------------------------------
        rod_builder = self.scene.create_actor_builder()
        rod_builder.initial_pose = sapien.Pose(
            p=[self.rod_center_x, self.rod_center_y, self.rod_center_z]
        )

        rod_mat = sapien.render.RenderMaterial()
        rod_mat.base_color = [0.55, 0.55, 0.60, 1.0]
        rod_mat.metallic = 0.8
        rod_mat.roughness = 0.3

        # Rotation: 90° around Z so cylinder axis → world Y
        q_rod = _Q_ROD_Y

        rod_builder.add_cylinder_collision(
            pose=sapien.Pose(p=[0, 0, 0], q=q_rod),
            radius=self.ROD_RADIUS,
            half_length=self.ROD_HALF_LENGTH,
        )
        rod_builder.add_cylinder_visual(
            pose=sapien.Pose(p=[0, 0, 0], q=q_rod),
            radius=self.ROD_RADIUS,
            half_length=self.ROD_HALF_LENGTH,
            material=rod_mat,
        )
        # Wall-mount bracket left
        bracket_mat = sapien.render.RenderMaterial()
        bracket_mat.base_color = [0.40, 0.40, 0.45, 1.0]
        for sign in (-1, +1):
            rod_builder.add_box_collision(
                pose=sapien.Pose(p=[0.0, sign * (self.ROD_HALF_LENGTH + 0.015), 0.0]),
                half_size=[0.018, 0.015, 0.030],
            )
            rod_builder.add_box_visual(
                pose=sapien.Pose(p=[0.0, sign * (self.ROD_HALF_LENGTH + 0.015), 0.0]),
                half_size=[0.018, 0.015, 0.030],
                material=bracket_mat,
            )

        self.rod: Actor = rod_builder.build_static(name="rod")

        # ------------------------------------------------------------------
        # Safety hook articulation (loaded from URDF)
        # The hook is loaded with fix_root_link=False so the base link is
        # a free-floating rigid body (we position it in _initialize_episode).
        # The gate_hinge is the only active joint.
        # ------------------------------------------------------------------
        loader = self.scene.create_urdf_loader()
        loader.fix_root_link = False      # base can move (falls off rod on success)

        asset_dir = importlib_resources.files("maniskill_myws").joinpath(
            "assets/safety_hook"
        )
        with importlib_resources.as_file(asset_dir) as asset_path:
            urdf_path = asset_path / "mobility.urdf"
            self.hook: Articulation = loader.load(
                str(urdf_path),
                name="safety_hook",
                scene_idxs=torch.arange(self.num_envs, dtype=torch.int32),
                package_dir=str(asset_path),
            )

        self.gate_joint = self.hook.active_joints_map["gate_hinge"]

        # Apply return-spring drive: stiffness pulls gate back to q=0
        self.gate_joint.set_drive_properties(
            stiffness=self.GATE_SPRING_STIFFNESS,
            damping=self.GATE_SPRING_DAMPING,
        )
        self.gate_joint.set_drive_target(0.0)   # target: closed

        # Reduce friction slightly (gate should respond to gentle pushes)
        self.gate_joint.set_friction(0.03)

    # ------------------------------------------------------------------
    # Episode initialisation
    # ------------------------------------------------------------------

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            self.scene_builder.initialize(env_idx)
            b = len(env_idx)

            # ----------------------------------------------------------
            # Rod position: fixed per episode (no randomisation on rod).
            # ----------------------------------------------------------
            rod_p = torch.zeros((b, 3), device=self.device)
            rod_p[:, 0] = self.rod_center_x
            rod_p[:, 1] = self.rod_center_y
            rod_p[:, 2] = self.rod_center_z
            rod_q = torch.tensor(_Q_IDENTITY, device=self.device).expand(b, 4)
            self.rod.set_pose(Pose.create_from_pq(rod_p, rod_q))

            # ----------------------------------------------------------
            # Hook pose: hang from rod.
            #
            # The hook's base-link origin is at the bottom of the frame.
            # We want the frame top (z = HOOK_HEIGHT = 0.150) to be just
            # inside the rod (rod centre z = rod_center_z).
            # So base origin z = rod_center_z - HOOK_HEIGHT - ROD_RADIUS + small_gap
            # where the gap (~1 mm) lets the hook loop around the rod.
            # ----------------------------------------------------------
            hook_origin_z = (
                self.rod_center_z
                - self.HOOK_HEIGHT
                + self.ROD_RADIUS          # top of frame arc is at rod centre height
                + 0.001                    # 1 mm clearance so hook doesn't clip
            )

            hook_p = torch.zeros((b, 3), device=self.device)
            hook_p[:, 0] = self.rod_center_x - 0.010   # centre hook under rod (hook is wider on gate side)
            hook_p[:, 1] = self.rod_center_y + randomization.uniform(
                -self.hook_spawn_y_range,
                self.hook_spawn_y_range,
                size=(b,),
                device=self.device,
            )
            hook_p[:, 2] = hook_origin_z

            hook_q = torch.tensor(_Q_IDENTITY, device=self.device).expand(b, 4)
            self.hook.set_pose(Pose.create_from_pq(hook_p, hook_q))

            # Gate starts closed (q=0)
            qpos_init = torch.zeros((b, 1), device=self.device)
            self.hook.set_qpos(qpos_init)
            self.hook.set_qvel(torch.zeros((b, 1), device=self.device))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _hook_top_z(self) -> torch.Tensor:
        """World-frame Z of the top of the hook frame (above hinge). Shape (B,)."""
        # Hook base origin + HOOK_HEIGHT (Z = 0.150 in base local frame)
        return self.hook.pose.p[:, 2] + self.HOOK_HEIGHT

    def _gate_angle(self) -> torch.Tensor:
        """Current gate joint angle (rad). Shape (B,)."""
        return self.gate_joint.qpos[:, 0] if self.gate_joint.qpos.ndim == 2 else self.gate_joint.qpos

    # ------------------------------------------------------------------
    # Evaluate / reward
    # ------------------------------------------------------------------

    def evaluate(self):
        gate_angle = self._gate_angle()           # (B,)
        hook_top_z = self._hook_top_z()           # (B,)

        gate_open = gate_angle > self.gate_open_threshold

        # Rod bottom surface z
        rod_bottom_z = self.rod_center_z - self.ROD_RADIUS
        # Hook top must have fallen at least hook_release_z_drop below rod bottom
        hook_released = hook_top_z < (rod_bottom_z - self.hook_release_z_drop)

        success = gate_open & hook_released

        return dict(
            success=success,
            gate_open=gate_open,
            hook_released=hook_released,
            gate_angle=gate_angle,
            hook_top_z=hook_top_z,
        )

    def _get_obs_extra(self, info: dict):
        obs = dict(
            tcp_pose=self.agent.tcp.pose.raw_pose,
            hook_pose=self.hook.pose.raw_pose,
            rod_pose=self.rod.pose.raw_pose,
        )
        if info.get("gate_angle") is not None:
            obs["gate_angle"] = info["gate_angle"]
        if info.get("hook_top_z") is not None:
            obs["hook_top_z"] = info["hook_top_z"]
        return obs

    def compute_sparse_reward(self, obs, action: torch.Tensor, info: dict):
        return info["success"].to(torch.float32)
