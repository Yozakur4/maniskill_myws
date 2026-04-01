from __future__ import annotations

import numpy as np

from mani_skill.envs.tasks.tabletop.stack_cube import StackCubeEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.registration import register_env
from maniskill_myws.task_prompts import TASK_PROMPTS


@register_env("StackCube-v2", max_episode_steps=200)
class StackCubeV2Env(StackCubeEnv):
    """
    StackCube variant with standardized sensors for VLA data collection.

    Key differences vs StackCube-v1:
      - Standardized base + side camera for dataset collection.
      - Default robot is panda_wristcam (provides wrist camera).
    """

    SUPPORTED_ROBOTS = ["panda_wristcam", "panda"]
    DEFAULT_TASK_PROMPT = TASK_PROMPTS["StackCube-v2"]

    def __init__(self, *args, robot_uids="panda_wristcam", **kwargs):
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sensor_configs(self):
        base_pose = sapien_utils.look_at(eye=[0.3, 0.0, 0.6], target=[-0.1, 0.0, 0.1])
        side_pose = sapien_utils.look_at(eye=[-0.35, 0.35, 0.35], target=[0.0, 0.0, 0.1])
        return [
            CameraConfig("base_camera", pose=base_pose, width=128, height=128, fov=np.pi / 2),
            CameraConfig("side_camera", pose=side_pose, width=128, height=128, fov=np.pi / 2),
        ]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([0.6, 0.7, 0.6], [0.0, 0.0, 0.35])
        return CameraConfig("render_camera", pose=pose, width=512, height=512, fov=1)