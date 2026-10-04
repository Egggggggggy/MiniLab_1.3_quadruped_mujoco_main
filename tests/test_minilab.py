import json
import math

import pytest

from eg.minilab_1_3 import (
    COMMAND_SCHEMA,
    CommandChatWorker,
    CommandExecutionWorker,
    MotionSkillController,
    StructuredCommandParser,
    evaluate_parser,
    load_object_config,
    parse_command,
)


def test_parse_command_accepts_motion_request():
    result = parse_command("walk forward for three seconds, then turn back")
    assert result["accepted"] is True
    assert [item["type"] for item in result["actions"]][:2] == ["move", "turn"]
    assert result["actions"][0]["duration"] > 0


def test_parse_command_preserves_turn_direction_angle_and_order():
    right_first = parse_command("turn 45 degrees right")
    left_first = parse_command("turn 45 degrees left")
    right_last = parse_command("turn right 45 degrees")
    left_last = parse_command("turn left 45 degrees")
    ordered = parse_command("turn 45 degrees then walk 3 seconds")

    assert right_first["actions"][0]["angle_deg"] == 45.0
    assert right_first["actions"][0]["direction"] == "right"
    assert left_first["actions"][0]["direction"] == "left"
    assert right_last["actions"][0]["direction"] == "right"
    assert left_last["actions"][0]["direction"] == "left"
    assert [item["type"] for item in ordered["actions"]] == ["turn", "move"]


def test_parse_command_rejects_non_english_or_empty():
    empty = parse_command("   ")
    invalid = parse_command("fly to the roof")
    assert empty["accepted"] is False
    assert invalid["accepted"] is False


def test_scene_object_config_has_required_objects():
    cfg = load_object_config()
    assert len(cfg["objects"]) >= 3
    classes = {obj["class"] for obj in cfg["objects"]}
    assert len(classes) >= 2
    assert any(obj["color"] == "green" for obj in cfg["objects"])
    assert any(obj["name"].startswith("chair") for obj in cfg["objects"])


def test_task3_evaluation_has_twenty_passing_cases():
    result = evaluate_parser(StructuredCommandParser())
    assert result["total"] >= 20
    assert result["correct"] == result["total"]


def test_chat_worker_keeps_multi_turn_history():
    worker = CommandChatWorker(StructuredCommandParser())
    try:
        first = worker.submit("walk forward for two seconds")
        second = worker.submit("turn left")
        assert first["accepted"] is True
        assert second["accepted"] is True
        assert len(worker.history) == 4
        assert worker.history[0]["role"] == "user"
    finally:
        worker.close()


def test_execution_worker_runs_actions_in_order_and_logs_lifecycle():
    logs = []
    worker = CommandExecutionWorker(MotionSkillController(), logs.append)
    try:
        result = worker.submit([
            {"type": "move", "vx": 0.5, "vy": 0.0, "wz": 0.0, "duration": 0.01},
            {"type": "stop"},
        ])
        assert result["accepted"] is True
        assert result["completed"] == 2
        assert logs == [
            "[EXEC] 1/2 type=move",
            "[DONE] 1/2 type=move",
            "[EXEC] 2/2 type=stop",
            "[DONE] 2/2 type=stop",
        ]
    finally:
        worker.close()


def test_move_uses_heading_at_command_start():
    class FakeData:
        qpos = [0.0, 0.0, 0.0, math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5)]
        qvel = [0.0] * 6

    controller = MotionSkillController(data=FakeData())
    command = controller.move(1.0, 0.0, 0.0, 0.01)
    assert command["start_yaw"] == pytest.approx(math.pi / 2)
    assert command["world_vx"] == pytest.approx(0.0, abs=1e-6)
    assert command["world_vy"] == pytest.approx(1.0, abs=1e-6)
