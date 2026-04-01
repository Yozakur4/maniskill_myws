from __future__ import annotations

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
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import SimConfig
from maniskill_myws.task_prompts import TASK_PROMPTS

# Quaternion helpers (SAPIEN convention: w, x, y, z)
# 90° around Y: maps capsule/cylinder X-axis → world Z (vertical)
_Q_Y90 = [math.sqrt(2) / 2, 0.0, math.sqrt(2) / 2, 0.0]

# Solar-panel tilt: 20° around Y-axis in the panel's local frame.
# The plate visual/collision is applied with this extra tilt on top of the
# actor pose.  We bake it into the shape local pose so the actor origin
# stays at the base-bottom (z = 0 on table).
_PANEL_TILT_DEG: float = 20.0
_PANEL_TILT_RAD: float = math.radians(_PANEL_TILT_DEG)
# q_tilt = (cos(10°), 0, sin(10°), 0)  – half-angle = 10°
_Q_TILT_Y = [
    math.cos(math.radians(_PANEL_TILT_DEG / 2)),
    0.0,
    math.sin(math.radians(_PANEL_TILT_DEG / 2)),
    0.0,
]


@register_env("BrushSolarPanel-v1", max_episode_steps=300)
class BrushSolarPanelEnv(BaseEnv):
    """
    Task: Use a brush to clean a solar panel.

    Scene (TableSceneBuilder, table top at z=0):
      - Panda (panda_wristcam) mounted at x=-0.615
      - Brush  : dynamic rigid body, spawned sideways (handle along world-Y)
                 below the robot TCP; head at local x=+0.135, wide (0.14 m)
      - Panel  : static fixture on far side, facing robot (actor yaw 180°)
                 cylinder base (r=0.05 h=0.20) + tilted plate (0.55×0.36×0.012)
                 plate tilted 20° with face toward robot; base is taller to
                 raise the panel clear of table clutter

    Success:
      1) Brush is elevated (grasped off table)
      2) Brush head is in contact with the tilted panel surface
      3) Brush head has slid ≥ slide_distance_threshold from first-contact point
    """

    SUPPORTED_REWARD_MODES = ["sparse", "dense", "none"]
    SUPPORTED_ROBOTS = ["panda", "panda_wristcam"]
    agent: Panda
    DEFAULT_TASK_PROMPT = TASK_PROMPTS["BrushSolarPanel-v1"]

    # ------------------------------------------------------------------ #
    # Geometry constants – kept in sync with _load_scene shapes           #
    # ------------------------------------------------------------------ #
    BRUSH_HANDLE_RADIUS: float = 0.012   # resting z on table = this value
    # Brush is placed with handle along Y axis (횡방향).
    # Head centre in brush local frame is at (BRUSH_HEAD_LOCAL_X, 0, 0).
    BRUSH_HEAD_LOCAL_X:  float = 0.135

    # Panel base: cylinder r=0.05, half_length=0.10  → top of base z=0.20
    PANEL_BASE_TOP_Z:   float = 0.20
    # Plate half-sizes (local, before tilt)
    PANEL_PLATE_HALF_X: float = 0.275   # half of 0.55 m
    PANEL_PLATE_HALF_Y: float = 0.18    # half of 0.36 m
    PANEL_PLATE_HALF_Z: float = 0.006   # half-thickness
    # Plate centre z in actor local frame (before tilt)
    PANEL_PLATE_LOCAL_Z: float = PANEL_BASE_TOP_Z + PANEL_PLATE_HALF_Z  # 0.206
    # Approximate highest z-offset from actor origin (tilted plate).
    # Tallest point: base_top + half_z + half_x*sin(tilt) + half_z*cos(tilt)
    PANEL_TOP_Z_OFFSET: float = (
        PANEL_BASE_TOP_Z
        + PANEL_PLATE_HALF_Z
        + PANEL_PLATE_HALF_X * math.sin(_PANEL_TILT_RAD)
        + PANEL_PLATE_HALF_Z * math.cos(_PANEL_TILT_RAD)
    )
    # Contact detection radius: full plate diagonal + 6 cm margin
    PANEL_CONTACT_RADIUS: float = math.sqrt(PANEL_PLATE_HALF_X**2 + PANEL_PLATE_HALF_Y**2) + 0.06

    def __init__(
        self,
        *args,
        robot_uids: str = "panda_wristcam",
        robot_init_qpos_noise: float = 0.02,
        # Brush spawn: below TCP, placed SIDEWAYS (handle along Y).
        # Kept at y=0.25 to stay clear of the panel footprint entirely.
        # x: 0.08–0.18  y: 0.20–0.30
        brush_spawn_center_x: float = 0.13,
        brush_spawn_center_y: float = 0.25,
        brush_spawn_half_size_x: float = 0.05,
        brush_spawn_half_size_y: float = 0.05,
        # Panel spawn: far-centre of table, facing the robot
        # x: 0.44–0.52  y: -0.05–0.05
        panel_spawn_center_x: float = 0.48,
        panel_spawn_center_y: float = 0.00,
        panel_spawn_half_size_x: float = 0.04,
        panel_spawn_half_size_y: float = 0.05,
        # Slide threshold: brush head must slide ≥ 12 cm from first-contact
        slide_distance_threshold: float = 0.12,
        **kwargs,
    ):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        self.brush_spawn_center_x = float(brush_spawn_center_x)
        self.brush_spawn_center_y = float(brush_spawn_center_y)
        self.brush_spawn_half_size_x = float(brush_spawn_half_size_x)
        self.brush_spawn_half_size_y = float(brush_spawn_half_size_y)
        self.panel_spawn_center_x = float(panel_spawn_center_x)
        self.panel_spawn_center_y = float(panel_spawn_center_y)
        self.panel_spawn_half_size_x = float(panel_spawn_half_size_x)
        self.panel_spawn_half_size_y = float(panel_spawn_half_size_y)
        self.slide_distance_threshold = float(slide_distance_threshold)

        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    # ------------------------------------------------------------------ #
    # Sim / sensor config                                                  #
    # ------------------------------------------------------------------ #

    @property
    def _default_sim_config(self):
        return SimConfig()

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(eye=[0.2, 0.4, 0.6], target=[0.4, 0.0, 0.1])
        return [
            CameraConfig("base_camera", pose=pose, width=128, height=128, fov=np.pi / 2)
        ]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([0.2, 0.55, 0.75], [0.4, 0.0, 0.1])
        return CameraConfig("render_camera", pose=pose, width=512, height=512, fov=1.0)

    # ------------------------------------------------------------------ #
    # Scene loading                                                        #
    # ------------------------------------------------------------------ #

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0, 0]))

    def _load_scene(self, options: dict):
        self.scene_builder = TableSceneBuilder(
            self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.scene_builder.build()

        # ------------------------------------------------------------------ #
        # Brush – dynamic actor (graspable)                                   #
        #   Handle : capsule along X-axis, r=0.012, half_length=0.088        #
        #            total length ≈ 0.20 m                                    #
        #   Head   : box half_size=(0.025, 0.07, 0.01) – wide brush head     #
        #            centre at local x=+0.135                                 #
        #                                                                      #
        # The brush is spawned with a 90° Z-rotation so the handle runs       #
        # along world-Y (横向), making the gripper approach from above easy.  #
        # Origin rests at z = BRUSH_HANDLE_RADIUS = 0.012 on the table.      #
        # ------------------------------------------------------------------ #
        brush_builder = self.scene.create_actor_builder()

        handle_mat = sapien.render.RenderMaterial()
        handle_mat.base_color = [0.45, 0.28, 0.12, 1.0]
        head_mat = sapien.render.RenderMaterial()
        head_mat.base_color = [0.20, 0.20, 0.20, 1.0]

        brush_builder.add_capsule_collision(
            pose=sapien.Pose(p=[0, 0, 0]),
            radius=0.012,
            half_length=0.088,
            density=400,
        )
        brush_builder.add_capsule_visual(
            pose=sapien.Pose(p=[0, 0, 0]),
            radius=0.012,
            half_length=0.088,
            material=handle_mat,
        )
        brush_builder.add_box_collision(
            pose=sapien.Pose(p=[0.135, 0, 0]),
            half_size=[0.025, 0.07, 0.01],
            density=400,
        )
        brush_builder.add_box_visual(
            pose=sapien.Pose(p=[0.135, 0, 0]),
            half_size=[0.025, 0.07, 0.01],
            material=head_mat,
        )
        self.brush: Actor = brush_builder.build(name="brush")

        # ------------------------------------------------------------------ #
        # Solar panel – static actor (fixed fixture on table)                 #
        #   Base  : vertical cylinder r=0.05, half_length=0.06               #
        #           centre at (0, 0, 0.06) → top at z=0.12                   #
        #   Plate : box 0.45×0.30×0.012 tilted 20° around Y                  #
        #           shape local pose = tilt rotation + z-offset               #
        # Cylinder axis defaults to X in SAPIEN → rotate 90° around Y (Z↑). #
        # ------------------------------------------------------------------ #
        panel_builder = self.scene.create_actor_builder()

        base_mat = sapien.render.RenderMaterial()
        base_mat.base_color = [0.35, 0.35, 0.35, 1.0]
        plate_mat = sapien.render.RenderMaterial()
        plate_mat.base_color = [0.04, 0.10, 0.38, 1.0]
        # Slight grid lines via metallic tint
        plate_mat.metallic = 0.3
        plate_mat.roughness = 0.6

        # Base cylinder: taller (half_length=0.10 → total h=0.20 m)
        # Centre at z=0.10, top at z=0.20 = PANEL_BASE_TOP_Z
        panel_builder.add_cylinder_collision(
            pose=sapien.Pose(p=[0, 0, 0.10], q=_Q_Y90),
            radius=0.05,
            half_length=0.10,
            density=2000,
        )
        panel_builder.add_cylinder_visual(
            pose=sapien.Pose(p=[0, 0, 0.10], q=_Q_Y90),
            radius=0.05,
            half_length=0.10,
            material=base_mat,
        )
        # Plate local pose: translate to base-top, then tilt 20° around Y.
        # sapien.Pose: p sets translation, q sets rotation.
        plate_pose = sapien.Pose(
            p=[0.0, 0.0, self.PANEL_PLATE_LOCAL_Z],
            q=_Q_TILT_Y,
        )
        panel_builder.add_box_collision(
            pose=plate_pose,
            half_size=[self.PANEL_PLATE_HALF_X, self.PANEL_PLATE_HALF_Y, self.PANEL_PLATE_HALF_Z],
            density=2000,
        )
        panel_builder.add_box_visual(
            pose=plate_pose,
            half_size=[self.PANEL_PLATE_HALF_X, self.PANEL_PLATE_HALF_Y, self.PANEL_PLATE_HALF_Z],
            material=plate_mat,
        )
        self.solar_panel: Actor = panel_builder.build_static(name="solar_panel")

        # Contact-start tracker (3D, shape num_envs×3); NaN = not yet contacted
        self._brush_contact_start_pos: torch.Tensor | None = None
        # Success latch: once True stays True for the rest of the episode
        self._success_latched: torch.Tensor | None = None

    # ------------------------------------------------------------------ #
    # Episode initialisation                                               #
    # ------------------------------------------------------------------ #

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            self.scene_builder.initialize(env_idx)
            b = len(env_idx)

            # ---------------------------------------------------------------- #
            # Brush – spawn below TCP, handle along world-Y (横向)            #
            #                                                                   #
            # Base orientation: 90° around Z  (w=cos45°, x=0, y=0, z=sin45°) #
            # This rotates the handle from local-X to world-Y, so the Panda   #
            # gripper (approaching from above along world-Z) can easily grab   #
            # the handle from the side.                                        #
            # Noise: ±15° additional yaw so the handle isn't always perfectly #
            # perpendicular.                                                    #
            # Position: x in TCP zone, y offset from panel to avoid overlap.  #
            # ---------------------------------------------------------------- #
            brush_p = torch.zeros((b, 3), device=self.device)
            brush_p[:, 0] = self.brush_spawn_center_x + randomization.uniform(
                -self.brush_spawn_half_size_x, self.brush_spawn_half_size_x,
                size=(b,), device=self.device,
            )
            brush_p[:, 1] = self.brush_spawn_center_y + randomization.uniform(
                -self.brush_spawn_half_size_y, self.brush_spawn_half_size_y,
                size=(b,), device=self.device,
            )
            brush_p[:, 2] = self.BRUSH_HANDLE_RADIUS   # lie flat on table

            # Base: 90° around Z → handle along Y
            s45 = math.sqrt(2) / 2
            q_base_brush = torch.zeros((b, 4), device=self.device)
            q_base_brush[:, 0] = s45   # w = cos(45°)
            q_base_brush[:, 3] = s45   # z = sin(45°)

            # Noise: ±15° Z rotation
            yaw_b = randomization.uniform(
                -np.pi / 12, np.pi / 12, size=(b,), device=self.device
            )
            half_b = yaw_b * 0.5
            q_noise_brush = torch.zeros((b, 4), device=self.device)
            q_noise_brush[:, 0] = torch.cos(half_b)
            q_noise_brush[:, 3] = torch.sin(half_b)

            def quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
                w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
                w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
                return torch.stack([
                    w1*w2 - x1*x2 - y1*y2 - z1*z2,
                    w1*x2 + x1*w2 + y1*z2 - z1*y2,
                    w1*y2 - x1*z2 + y1*w2 + z1*x2,
                    w1*z2 + x1*y2 - y1*x2 + z1*w2,
                ], dim=-1)

            brush_q = quat_mul(q_noise_brush, q_base_brush)
            self.brush.set_pose(Pose.create_from_pq(brush_p, brush_q))

            # ---------------------------------------------------------------- #
            # Solar panel – far side of table, tilted face TOWARD the robot    #
            #                                                                   #
            # The plate is built with _Q_TILT_Y which tilts its +x edge UP.   #
            # Rotating the actor 180° around Z maps actor +x → world -x,      #
            # so the raised edge faces the robot (at world x≈-0.615). ✓       #
            #                                                                   #
            # Base orientation: 180° around Z  (w=0, x=0, y=0, z=1)           #
            # Random yaw noise: ±10° additional Z rotation via quaternion mult #
            # ---------------------------------------------------------------- #
            panel_p = torch.zeros((b, 3), device=self.device)
            panel_p[:, 0] = self.panel_spawn_center_x + randomization.uniform(
                -self.panel_spawn_half_size_x, self.panel_spawn_half_size_x,
                size=(b,), device=self.device,
            )
            panel_p[:, 1] = self.panel_spawn_center_y + randomization.uniform(
                -self.panel_spawn_half_size_y, self.panel_spawn_half_size_y,
                size=(b,), device=self.device,
            )
            panel_p[:, 2] = 0.0   # base bottom flush with table surface

            # Small ±10° yaw noise on top of 180° base orientation
            yaw_angles = randomization.uniform(
                -np.pi / 18, np.pi / 18, size=(b,), device=self.device
            )
            half = yaw_angles * 0.5
            q_noise = torch.zeros((b, 4), device=self.device)
            q_noise[:, 0] = torch.cos(half)   # w
            q_noise[:, 3] = torch.sin(half)   # z

            # Base orientation: 180° around Z → (w=0, x=0, y=0, z=1)
            q_base = torch.zeros((b, 4), device=self.device)
            q_base[:, 3] = 1.0

            panel_q = quat_mul(q_noise, q_base)
            self.solar_panel.set_pose(Pose.create_from_pq(panel_p, panel_q))

            # Reset per-env contact tracker and success latch
            if self._brush_contact_start_pos is None:
                self._brush_contact_start_pos = torch.full(
                    (self.num_envs, 3), float("nan"), device=self.device
                )
            if self._success_latched is None:
                self._success_latched = torch.zeros(
                    (self.num_envs,), dtype=torch.bool, device=self.device
                )
            self._brush_contact_start_pos[env_idx] = float("nan")
            self._success_latched[env_idx] = False

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    def _get_brush_head_world_pos(self) -> torch.Tensor:
        """Return brush-head centre in world frame, shape (num_envs, 3)."""
        # Brush head is at local (BRUSH_HEAD_LOCAL_X, 0, 0) in the brush body frame.
        brush_T = self.brush.pose.to_transformation_matrix()   # (B, 4, 4)
        head_local = torch.tensor(
            [self.BRUSH_HEAD_LOCAL_X, 0.0, 0.0], device=self.device
        )
        # (B, 3, 3) @ (3,) + (B, 3)  →  (B, 3)
        return brush_T[:, :3, :3] @ head_local + brush_T[:, :3, 3]

    def _check_contact(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            in_contact : bool (B,)    – brush head near panel surface & brush elevated
            head_world : float (B, 3) – brush head world position
        """
        head_world = self._get_brush_head_world_pos()
        panel_p = self.solar_panel.pose.p   # (B, 3)

        # 1. Brush must be elevated (well off the table)
        is_elevated = self.brush.pose.p[:, 2] > 0.05

        # 2. XY: brush head within panel contact radius (covers full tilted plate)
        dx = head_world[:, 0] - panel_p[:, 0]
        dy = head_world[:, 1] - panel_p[:, 1]
        xy_in_bounds = (dx * dx + dy * dy) < self.PANEL_CONTACT_RADIUS ** 2

        # 3. Z: brush head within 6 cm of panel's approximate top surface.
        #    PANEL_TOP_Z_OFFSET already accounts for the 20° tilt height gain.
        panel_top_z = panel_p[:, 2] + self.PANEL_TOP_Z_OFFSET
        dz = torch.abs(head_world[:, 2] - panel_top_z)
        z_near = dz < 0.06

        in_contact = is_elevated & xy_in_bounds & z_near
        return in_contact, head_world

    # ------------------------------------------------------------------ #
    # Evaluate / reward                                                    #
    # ------------------------------------------------------------------ #

    def evaluate(self):
        in_contact, head_world = self._check_contact()

        # Lazy init (handles the edge case where evaluate() is called before
        # _initialize_episode, e.g. during env.reset() internal checks)
        if self._brush_contact_start_pos is None:
            self._brush_contact_start_pos = torch.full(
                (self.num_envs, 3), float("nan"), device=self.device
            )
        if self._success_latched is None:
            self._success_latched = torch.zeros(
                (self.num_envs,), dtype=torch.bool, device=self.device
            )

        # Record 3D position of the brush head the FIRST TIME contact is made
        # (NaN → not yet recorded)
        not_started = torch.isnan(self._brush_contact_start_pos[:, 0])
        new_contact = in_contact & not_started
        if new_contact.any():
            self._brush_contact_start_pos[new_contact] = head_world[new_contact]

        # Slide distance = 3D displacement from first-contact position
        # (only meaningful once contact has been established)
        started = ~torch.isnan(self._brush_contact_start_pos[:, 0])
        slide_dist = torch.where(
            started,
            torch.norm(head_world - self._brush_contact_start_pos, dim=-1),
            torch.zeros(self.num_envs, device=self.device),
        )

        has_slid = slide_dist > self.slide_distance_threshold
        # Latch success: once achieved stays True for the rest of the episode
        current_success = in_contact & has_slid
        self._success_latched |= current_success
        success = self._success_latched

        return dict(
            success=success,
            in_contact=in_contact,
            has_slid=has_slid,
            slide_distance=slide_dist,
        )

    def _get_obs_extra(self, info: dict):
        obs = dict(
            tcp_pose=self.agent.tcp.pose.raw_pose,
            brush_pose=self.brush.pose.raw_pose,
            panel_pose=self.solar_panel.pose.raw_pose,
        )
        # info keys are present after the first evaluate() call
        if info.get("in_contact") is not None:
            obs["in_contact"] = info["in_contact"].float()
        if info.get("slide_distance") is not None:
            obs["slide_distance"] = info["slide_distance"]
        return obs

    def compute_sparse_reward(self, obs, action: torch.Tensor, info: dict):
        return info["success"].to(torch.float32)

    def compute_dense_reward(self, obs, action: torch.Tensor, info: dict):
        brush_T = self.brush.pose.to_transformation_matrix()   # (B, 4, 4)
        head_world = self._get_brush_head_world_pos()
        panel_p = self.solar_panel.pose.p
        brush_p = self.brush.pose.p

        # Stage 1 – Reach: TCP approaches brush on table
        tcp_to_brush = torch.norm(self.agent.tcp.pose.p - brush_p, dim=-1)
        reach_reward = 1.0 - torch.tanh(5.0 * tcp_to_brush)

        # Stage 2 – Elevate: smooth sigmoid gate, steep at z=0.05
        lift_score = torch.sigmoid(50.0 * (brush_p[:, 2] - 0.05))

        # Stage 3 – Approach: gated by lift_score
        panel_top_z = panel_p[:, 2] + self.PANEL_TOP_Z_OFFSET
        target_pos = torch.stack([panel_p[:, 0], panel_p[:, 1], panel_top_z], dim=-1)
        head_to_panel = torch.norm(head_world - target_pos, dim=-1)
        approach_reward = lift_score * (1.0 - torch.tanh(3.0 * head_to_panel))

        # Stage 3 – Alignment: brush X-axis vs panel local -X axis (dynamic)
        # brush world X-axis = first column of rotation matrix
        brush_x_axis = brush_T[:, :3, 0]                        # (B, 3)
        # panel world X-axis = first column of panel rotation matrix
        panel_T = self.solar_panel.pose.to_transformation_matrix()
        panel_x_axis = panel_T[:, :3, 0]                        # (B, 3)
        # target direction = panel -X (face toward robot)
        target_dir = -panel_x_axis                              # (B, 3)
        cos_align = (brush_x_axis * target_dir).sum(dim=-1)     # (B,) in [-1,1]
        align_reward = lift_score * (0.5 * cos_align + 0.5)     # mapped to [0,1]

        # Stage 4 – Contact and slide
        contact_reward = info["in_contact"].float()
        slide_reward = torch.clamp(
            info["slide_distance"] / self.slide_distance_threshold, 0.0, 1.0
        )

        reward = (
            0.5 * reach_reward +
            1.0 * lift_score +
            1.5 * approach_reward +
            0.5 * align_reward +
            2.0 * contact_reward +
            5.0 * contact_reward * slide_reward
        )

        return reward
