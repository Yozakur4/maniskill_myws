from __future__ import annotations

# Central registry for task prompts.
# Keep this file free of ManiSkill imports so it can be used in conversion scripts.

TASK_PROMPTS: dict[str, str] = {
    "TurnGlobeValve-v1": "turn the globe valve",
    "OpenSafeDoor-v2": "press the button or turn the handle to unlock, then pull open the safe door",
    "StackCube-v2": "stack the red cube on the green cube",
    "BrushSolarPanel-v1": "use the brush to clean the solar panel",
    "OpenSafetyHook-v1": "open the safety hook and remove it from the rod",
}


def get_task_prompt(env_id: str) -> str | None:
    return TASK_PROMPTS.get(env_id)
