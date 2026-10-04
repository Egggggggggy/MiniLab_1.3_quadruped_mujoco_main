"""Full MiniLab 1.3 integration scaffold.

This module intentionally combines the project platform base from the existing
MuJoCo runtime-control example with a lightweight command parser, object config,
scene generator, motion skills API, and optional perception pipeline. It is
structured to support the lab workflow without depending on the missing PDF
values or a dedicated GPU.
"""

from __future__ import annotations

import json
import http.client
import math
import os
import queue
import re
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


COMMAND_SCHEMA = {
    "type": "object",
    "properties": {
        "command": {"type": "string"},
        "actions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "type": {"enum": ["move", "turn", "goto_object", "stop", "chat"]},
                    "vx": {"type": "number"},
                    "vy": {"type": "number"},
                    "wz": {"type": "number"},
                    "duration": {"type": "number"},
                    "angle_deg": {"type": "number"},
                    "class": {"type": "string"},
                    "color": {"type": "string"},
                    "text": {"type": "string"},
                },
                "required": ["type"],
            },
        },
    },
    "required": ["command", "actions"],
}

LLM_SYSTEM_PROMPT = """You are a robot command parser. Convert exactly one free-form English
utterance into a JSON object with this shape:
{"command": "original utterance", "actions": [{"type": "move|turn|goto_object|stop|chat", ...}]}

Allowed action fields:
- move: vx, vy, wz, duration
- turn: angle_deg
- goto_object: class, color
- stop: no additional fields
- chat: text

Return JSON only. Support multiple actions in the order requested. Reject empty,
non-English, unsafe, physically impossible, or unsupported requests by returning
an empty actions list. Never invent an object class or color that was not stated;
use a known scene target only when the utterance identifies it clearly.
"""


def load_object_config(path: Optional[str | Path] = None) -> Dict[str, Any]:
    if path is None:
        path = Path(__file__).with_name("minilab_scene.json")
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Object config not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if "objects" not in data:
        raise ValueError("Object config must contain an 'objects' list")
    return data


def _extract_number(text: str) -> Optional[float]:
    matches = re.findall(r"[-+]?\d*\.?\d+", text)
    if not matches:
        return None
    return float(matches[0])


def _extract_color(text: str) -> Optional[str]:
    for color in ["green", "red", "blue", "yellow", "black", "white", "orange"]:
        if color in text:
            return color
    return None


def _extract_object_class(text: str) -> Optional[str]:
    class_map = {
        "chair": "chair",
        "table": "table",
        "ball": "sports ball",
        "basketball": "sports ball",
        "sign": "stop sign",
        "stop": "stop sign",
        "stop sign": "stop sign",
    }
    for key, value in class_map.items():
        if key in text:
            return value
    return None


def _parse_command_local(text: str) -> Dict[str, Any]:
    """Convert a text command into a structured lab-style command list.

    The parser supports motion commands, turn commands, and object-search commands.
    It is intentionally deterministic and works without external API keys. If the
    environment contains an LLM API key, a future extension could replace this
    logic; this function is the local fallback used by the lab implementation.
    """
    raw = (text or "").strip()
    if not raw or not re.search(r"[A-Za-z]", raw):
        return {"accepted": False, "reason": "empty or non-English command", "actions": []}

    lower = raw.lower()
    if any(token in lower for token in ["fly to", "roof", "into the sky", "underwater", "teleport"]):
        return {"accepted": False, "reason": "unsafe or out-of-scope request", "actions": []}

    actions: List[Dict[str, Any]] = []
    clauses = [clause.strip() for clause in re.split(r"\s*(?:,|\bthen\b|\band then\b)\s*", lower) if clause.strip()]
    for clause in clauses:
        if any(token in clause for token in ["walk", "move forward", "go forward", "go straight", "move straight"]):
            seconds = 3.0
            if any(ch.isdigit() for ch in clause):
                seconds = float(_extract_number(clause) or seconds)
            actions.append({
                "type": "move",
                # Match play.py's built-in W key exactly.
                "vx": 1.0,
                "vy": 0.0,
                "wz": 0.0,
                "duration": max(0.5, seconds),
            })
            continue

        if "turn" in clause:
            direction = "left" if "left" in clause else "right" if "right" in clause else "back" if any(
                token in clause for token in ["back", "around"]
            ) else "left"
            angle = 180.0 if direction == "back" else 90.0
            if any(ch.isdigit() for ch in clause):
                angle = float(_extract_number(clause) or angle)
            actions.append({"type": "turn", "angle_deg": angle, "direction": direction})
            continue

        if any(token in clause for token in ["go to", "goto", "search for", "find the", "go to the"]):
            obj_class = _extract_object_class(clause) or "chair"
            color = _extract_color(clause) or "green"
            actions.append({
                "type": "goto_object",
                "class": obj_class,
                "color": color,
                "text": raw,
            })

    if not actions:
        return {"accepted": False, "reason": "unsupported or ambiguous request", "actions": []}

    return {
        "accepted": True,
        "actions": actions,
        "raw": raw,
        "summary": "; ".join(
            f"{item['type']}" if item['type'] != 'goto_object' else f"goto_object({item['class']}, {item['color']})"
            for item in actions
        ),
    }


def _validate_structured_command(result: Any, raw: str) -> Dict[str, Any]:
    """Validate and normalize a parser response against COMMAND_SCHEMA."""
    if not isinstance(result, dict) or not isinstance(result.get("actions"), list):
        raise ValueError("structured response must contain an actions list")
    actions = []
    for action in result["actions"]:
        if not isinstance(action, dict) or action.get("type") not in {
            "move", "turn", "goto_object", "stop", "chat"
        }:
            raise ValueError("structured response contains an invalid action")
        normalized = dict(action)
        if normalized["type"] == "move":
            normalized.setdefault("vx", 1.0)
            normalized.setdefault("vy", 0.0)
            normalized.setdefault("wz", 0.0)
            duration = normalized.get("duration")
            if not isinstance(duration, (int, float)) or duration <= 0:
                normalized["duration"] = max(0.5, _extract_number(raw) or 3.0)
        elif normalized["type"] == "turn":
            if not isinstance(normalized.get("angle_deg"), (int, float)) or normalized["angle_deg"] == 0:
                normalized["angle_deg"] = _extract_number(raw) or 90.0
            if "direction" not in normalized:
                normalized["direction"] = "right" if "right" in raw.lower() else "left"
        actions.append(normalized)
    if not actions:
        return {"accepted": False, "reason": "unsupported or ambiguous request", "actions": []}
    return {
        "accepted": True,
        "actions": actions,
        "raw": raw,
        "summary": "; ".join(
            action["type"] if action["type"] != "goto_object" else
            f"goto_object({action.get('class', 'unknown')}, {action.get('color', 'unknown')})"
            for action in actions
        ),
    }


class StructuredCommandParser:
    """Structured-output parser with an offline fallback and HTTP provider hook."""

    def __init__(self, provider: str = "local", endpoint: Optional[str] = None,
                 api_key: Optional[str] = None, model: Optional[str] = None):
        self.provider = provider.lower()
        provider_suffix = self.provider.upper().replace("-", "_")
        default_endpoints = {
            "openai": "https://api.openai.com/v1/chat/completions",
            "anthropic": "https://api.anthropic.com/v1/messages",
        }
        provider_endpoint = endpoint or os.getenv(f"MINILAB_LLM_API_URL_{provider_suffix}")
        if self.provider == "gemini":
            self.endpoint = provider_endpoint
        else:
            self.endpoint = (provider_endpoint or os.getenv("MINILAB_LLM_API_URL")
                             or default_endpoints.get(self.provider))
        provider_key = os.getenv(f"MINILAB_LLM_API_KEY_{provider_suffix}")
        if self.provider == "openai":
            provider_key = provider_key or os.getenv("OPENAI_API_KEY")
        elif self.provider == "gemini":
            provider_key = provider_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        self.api_key = api_key or provider_key or os.getenv("MINILAB_LLM_API_KEY")
        configured_model = (model or os.getenv(f"MINILAB_LLM_MODEL_{provider_suffix}")
                            or os.getenv("MINILAB_LLM_MODEL", ""))
        self.model = self._normalize_model_name(configured_model)

    @staticmethod
    def _normalize_model_name(model: str) -> str:
        aliases = {
            "gemini 2.5 flash": "gemini-2.5-flash",
            "gemini 2.5 flash-latest": "gemini-2.5-flash",
            "gemini 3 flash preview": "gemini-3-flash-preview",
            "gemini 3.5 flash lite": "gemini-3.5-flash-lite",
        }
        cleaned = model.strip().removeprefix("models/")
        if ":generateContent" in cleaned:
            cleaned = cleaned.split(":generateContent", 1)[0]
        return aliases.get(cleaned.lower(), cleaned)

    def parse(self, text: str, history: Optional[Sequence[Dict[str, str]]] = None) -> Dict[str, Any]:
        raw = (text or "").strip()
        if self.provider == "local":
            return _parse_command_local(raw)
        missing = []
        if not self.api_key:
            missing.append("API key")
        if not self.model:
            missing.append("model name")
        if self.provider == "openai-compatible" and not self.endpoint:
            missing.append("API endpoint")
        if missing:
            fallback = _parse_command_local(raw)
            fallback["provider"] = self.provider
            fallback["llm_used"] = False
            fallback["fallback_reason"] = "missing " + " and ".join(missing)
            return fallback
        try:
            response = self._request_provider(raw, history or [])
            parsed = _validate_structured_command(response, raw)
            parsed["provider"] = self.provider
            parsed["llm_used"] = True
            return parsed
        except (OSError, ValueError, json.JSONDecodeError, urllib.error.URLError,
            http.client.HTTPException) as exc:
            fallback = _parse_command_local(raw)
            fallback["provider"] = self.provider
            fallback["llm_used"] = False
            fallback["fallback_reason"] = str(exc)
            return fallback

    def _request_provider(self, text: str, history: Sequence[Dict[str, str]]) -> Dict[str, Any]:
        if self.provider not in {"openai", "openai-compatible", "anthropic", "gemini"}:
            raise ValueError(f"unsupported provider: {self.provider}")
        system = LLM_SYSTEM_PROMPT + "\nJSON schema reference:\n" + json.dumps(COMMAND_SCHEMA)
        messages = [{"role": "system", "content": system}]
        messages.extend(history)
        messages.append({"role": "user", "content": text})
        if self.provider == "gemini":
            endpoint = self.endpoint or (
                "https://generativelanguage.googleapis.com/v1beta/models/"
                + self.model + ":generateContent"
            )
            payload = {
                "system_instruction": {"parts": [{"text": system}]},
                "contents": [
                    {"role": "user" if item["role"] == "user" else "model",
                     "parts": [{"text": item["content"]}]}
                    for item in messages[1:]
                ],
                "generationConfig": {
                    "temperature": 0,
                    "responseMimeType": "application/json",
                    "responseSchema": COMMAND_SCHEMA,
                },
            }
            endpoint = endpoint + ("&" if "?" in endpoint else "?") + "key=" + (self.api_key or "")
            request_headers = {"Content-Type": "application/json"}
        elif self.provider == "anthropic":
            endpoint = self.endpoint
            payload = {"model": self.model, "max_tokens": 512, "system": system,
                       "messages": [item for item in messages if item["role"] != "system"]}
            request_headers = {"Content-Type": "application/json", "x-api-key": self.api_key or "",
                               "anthropic-version": "2023-06-01"}
        else:
            endpoint = self.endpoint
            payload = {"model": self.model, "messages": messages,
                       "response_format": {"type": "json_object"}}
            request_headers = {"Content-Type": "application/json",
                               "Authorization": f"Bearer {self.api_key or ''}"}
        request = urllib.request.Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers=request_headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace").strip()
            raise ValueError(f"HTTP {exc.code}: {detail[:500]}") from exc
        if self.provider == "gemini":
            content = body["candidates"][0]["content"]["parts"][0]["text"]
        elif self.provider == "anthropic":
            content = body["content"][0]["text"]
        else:
            content = body["choices"][0]["message"]["content"]
        content = content.strip()
        if content.startswith("```"):
            content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content, flags=re.IGNORECASE).strip()
        return json.loads(content)


def parse_command(text: str, parser: Optional[StructuredCommandParser] = None,
                  history: Optional[Sequence[Dict[str, str]]] = None) -> Dict[str, Any]:
    """Parse a command using structured output, retaining the offline API."""
    return (parser or StructuredCommandParser()).parse(text, history)


class CommandChatWorker:
    """Run command parsing in a background thread so simulation stays responsive."""

    def __init__(self, parser: Optional[StructuredCommandParser] = None):
        self.parser = parser or StructuredCommandParser()
        self.history: List[Dict[str, str]] = []
        self._requests: queue.Queue[Optional[Tuple[str, queue.Queue[Dict[str, Any]]]]] = queue.Queue()
        self._thread = threading.Thread(target=self._run, name="minilab-command-chat", daemon=True)
        self._thread.start()

    def submit(self, text: str) -> Dict[str, Any]:
        result_queue: queue.Queue[Dict[str, Any]] = queue.Queue(maxsize=1)
        self._requests.put((text, result_queue))
        return result_queue.get()

    def close(self) -> None:
        self._requests.put(None)
        self._thread.join(timeout=2)

    def _run(self) -> None:
        while True:
            request = self._requests.get()
            if request is None:
                return
            text, result_queue = request
            result = self.parser.parse(text, self.history)
            self.history.extend([
                {"role": "user", "content": text},
                {"role": "assistant", "content": json.dumps(result)},
            ])
            result_queue.put(result)


class CommandExecutionWorker:
    """Execute parsed actions sequentially without blocking the simulation thread."""

    def __init__(self, skills: "MotionSkillController", output=print):
        self.skills = skills
        self.output = output
        self._requests: queue.Queue[Optional[Tuple[List[Dict[str, Any]], queue.Queue[Dict[str, Any]]]]] = queue.Queue()
        self._thread = threading.Thread(target=self._run, name="minilab-command-executor", daemon=True)
        self._thread.start()

    def submit(self, actions: List[Dict[str, Any]]) -> Dict[str, Any]:
        result_queue: queue.Queue[Dict[str, Any]] = queue.Queue(maxsize=1)
        self._requests.put((actions, result_queue))
        return result_queue.get()

    def close(self) -> None:
        self._requests.put(None)
        self._thread.join(timeout=2)

    def _run(self) -> None:
        while True:
            request = self._requests.get()
            if request is None:
                return
            actions, result_queue = request
            completed = 0
            try:
                for index, action in enumerate(actions, start=1):
                    self._execute_action(index, len(actions), action)
                    completed += 1
                result_queue.put({"completed": completed, "total": len(actions), "accepted": True})
            except Exception as exc:
                self.output("[ERROR] action=%d reason=%s" % (completed + 1, exc))
                result_queue.put({"completed": completed, "total": len(actions), "accepted": False, "reason": str(exc)})

    def _execute_action(self, index: int, total: int, action: Dict[str, Any]) -> None:
        action_type = action.get("type")
        self.output("[EXEC] %d/%d type=%s" % (index, total, action_type))
        if action_type == "move":
            self.skills.move(action.get("vx", 0.0), action.get("vy", 0.0),
                             action.get("wz", 0.0), action.get("duration", 0.0))
        elif action_type == "turn":
            angle = float(action.get("angle_deg", 0.0))
            if action.get("direction") == "right":
                angle = -abs(angle)
            elif action.get("direction") == "left":
                angle = abs(angle)
            self.skills.turn(angle)
        elif action_type == "stop":
            self.skills.stop()
        elif action_type == "goto_object":
            self.skills.goto_object(action.get("class", ""), action.get("color", ""))
        elif action_type == "chat":
            pass
        else:
            raise ValueError("unsupported action type: %s" % action_type)
        self.skills.wait_until_idle()
        if self.skills.adjust_pose():
            self.output("[ADJUST] type=%s" % action_type)
            self.skills.wait_until_idle()
        pose_result = self.skills.last_pose_result
        if pose_result and pose_result.get("tracked"):
            self.output("[POSE] type=%s position_error=%.3f yaw_error=%.2fdeg" % (
                action_type, pose_result["position_error"], math.degrees(pose_result["yaw_error"])
            ))
        self.output("[DONE] %d/%d type=%s" % (index, total, action_type))


DEFAULT_EVALUATION_CASES = [
    ("walk forward", True), 
    ("move forward for 2 seconds", True),
    ("turn left", True), 
    ("turn right 45 degrees", True),
    ("turn around", True), 
    ("go to the green chair", True),
    ("find the blue ball", True), 
    ("search for the red stop sign", True),
    ("walk forward then turn left", True), 
    ("go straight for 4 seconds", True),
    ("fly to the roof", False), 
    ("teleport to the chair", False),
    ("", False), 
    ("   ", False), 
    ("你好", False),
    ("go underwater", False), 
    ("walk into the sky", False),
    ("dance with the moon", False), 
    ("turn right then find the yellow chair", True),
    ("move forward and search for the green chair", True),
    ("levitate to the top of the building", False),
    ("patrol the area and find the blue ball", True),
]


def evaluate_parser(parser: StructuredCommandParser,
                    cases: Sequence[Tuple[str, bool]] = DEFAULT_EVALUATION_CASES) -> Dict[str, Any]:
    """Evaluate accepted/rejected classification on the lab's 20-case set."""
    rows = []
    correct = 0
    llm_used = 0
    for text, expected in cases:
        result = parser.parse(text)
        passed = result.get("accepted") is expected
        correct += int(passed)
        llm_used += int(result.get("llm_used", False))
        rows.append({"input": text, "expected": expected, "actual": result.get("accepted"),
                     "passed": passed, "llm_used": result.get("llm_used", False),
                     "fallback_reason": result.get("fallback_reason", "")})
    return {
        "total": len(rows), "correct": correct, "accuracy": correct / len(rows),
        "llm_used": llm_used,
        "rows": rows,
    }


def build_scene_xml(robot_xml_path: Optional[str | Path] = None, output_path: Optional[str | Path] = None) -> Path:
    """Inject a set of static world objects into the default dog robot XML.

    The scene is intentionally simple but valid and ensures the object placements
    needed for Task 2 / Task 4 are reproducible and programmable.
    """
    if robot_xml_path is None:
        robot_xml_path = Path(__file__).with_name("dog").joinpath("xml", "dog.xml")
    robot_path = Path(robot_xml_path).expanduser().resolve()
    if not robot_path.is_file():
        raise FileNotFoundError(f"Robot XML not found: {robot_path}")

    config = load_object_config()
    object_xml = []
    for index, obj in enumerate(config["objects"], start=1):
        name = obj["name"]
        cls_name = obj["class"]
        color = obj["color"]
        x = float(obj.get("x", 0.0))
        y = float(obj.get("y", 0.0))
        z = float(obj.get("z", 0.0))
        if cls_name == "chair":
            object_xml.append(
                f'''
    <body name="{name}" pos="{x} {y} {z}">
      <geom name="{name}_base" type="box" size="0.18 0.18 0.02" pos="0 0 0.02" rgba="{_color_to_rgba(color)}"/>
      <geom name="{name}_seat" type="box" size="0.18 0.18 0.03" pos="0 0 0.40" rgba="{_color_to_rgba(color)}"/>
      <geom name="{name}_back" type="box" size="0.18 0.02 0.22" pos="0 -0.18 0.44" rgba="{_color_to_rgba(color)}"/>
      <geom name="{name}_leg1" type="cylinder" size="0.02 0.20" pos="0.12 0.12 0.20" rgba="0.2 0.2 0.2 1"/>
      <geom name="{name}_leg2" type="cylinder" size="0.02 0.20" pos="-0.12 0.12 0.20" rgba="0.2 0.2 0.2 1"/>
      <geom name="{name}_leg3" type="cylinder" size="0.02 0.20" pos="0.12 -0.12 0.20" rgba="0.2 0.2 0.2 1"/>
      <geom name="{name}_leg4" type="cylinder" size="0.02 0.20" pos="-0.12 -0.12 0.20" rgba="0.2 0.2 0.2 1"/>
    </body>
'''.strip()
            )
        elif cls_name == "sports ball":
            object_xml.append(
                f'''
    <body name="{name}" pos="{x} {y} {z}">
      <geom name="{name}_ball" type="sphere" size="0.12" rgba="{_color_to_rgba(color)}"/>
    </body>
'''.strip()
            )
        elif cls_name == "stop sign":
            object_xml.append(
                f'''
    <body name="{name}" pos="{x} {y} {z}">
      <geom name="{name}_pole" type="cylinder" size="0.02 0.55" pos="0 0 0.45" rgba="0.3 0.3 0.3 1"/>
      <geom name="{name}_sign" type="box" size="0.18 0.02 0.18" pos="0 0.02 1.00" rgba="{_color_to_rgba(color)}"/>
    </body>
'''.strip()
            )
        else:
            object_xml.append(
                f'''
    <body name="{name}" pos="{x} {y} {z}">
      <geom name="{name}_geom" type="box" size="0.15 0.15 0.15" rgba="{_color_to_rgba(color)}"/>
    </body>
'''.strip()
            )

    xml_text = robot_path.read_text(encoding="utf-8")
    marker = "  </worldbody>\n"
    injection = "\n".join(object_xml)
    if marker not in xml_text:
        raise ValueError("Robot XML does not contain the expected worldbody end marker")
    xml_text = xml_text.replace(marker, "    <camera name=\"front\" pos=\"0.8 0 1.2\" xyaxes=\"1 0 0 0 1 0\" fovy=\"60\"/>\n" + injection + "\n  </worldbody>\n")

    if output_path is None:
        output_path = Path(__file__).with_name("minilab_scene.xml")
    output_path = Path(output_path).expanduser().resolve()
    output_path.write_text(xml_text, encoding="utf-8")
    return output_path


def _color_to_rgba(color: str) -> str:
    palette = {
        "green": "0.0 0.75 0.2 1",
        "red": "0.9 0.15 0.15 1",
        "blue": "0.15 0.35 0.95 1",
        "yellow": "0.95 0.85 0.0 1",
        "black": "0.08 0.08 0.08 1",
        "white": "0.96 0.96 0.96 1",
        "orange": "1.0 0.55 0.1 1",
    }
    return palette.get(color.lower(), "0.8 0.8 0.8 1")


class MotionSkillController:
    """Queue-based motion skills for Task 2."""

    def __init__(self, model: Any = None, data: Any = None):
        self.model = model
        self.data = data
        self.pending: List[Dict[str, Any]] = []
        self.lock = threading.Lock()
        self.last_pose_result: Optional[Dict[str, Any]] = None

    def move(self, vx: float, vy: float, wz: float, duration: float) -> Dict[str, Any]:
        start_yaw = self._yaw()
        forward = float(vx)
        lateral = float(vy)
        cmd = {
            "type": "move",
            "vx": forward,
            "vy": lateral,
            "world_vx": forward * math.cos(start_yaw) - lateral * math.sin(start_yaw),
            "world_vy": forward * math.sin(start_yaw) + lateral * math.cos(start_yaw),
            "wz": float(wz),
            "duration": float(duration),
            "start": time.time(),
            "start_yaw": start_yaw,
            "initial_pose": self._pose(),
        }
        if cmd["initial_pose"] is not None:
            cmd["target_pose"] = {
                "x": cmd["initial_pose"]["x"] + cmd["world_vx"] * float(duration),
                "y": cmd["initial_pose"]["y"] + cmd["world_vy"] * float(duration),
                "yaw": cmd["initial_pose"]["yaw"],
            }
        with self.lock:
            self.pending.append(cmd)
        return cmd

    def turn(self, angle_deg: float, target_yaw: Optional[float] = None) -> Dict[str, Any]:
        current_yaw = self._yaw()
        if target_yaw is None:
            target_yaw = current_yaw + math.radians(float(angle_deg))
        cmd = {"type": "turn", "target_yaw": float(target_yaw), "start_yaw": float(current_yaw),
               "angle_deg": float(angle_deg), "start": time.time(), "initial_pose": self._pose()}
        with self.lock:
            self.pending.append(cmd)
        return cmd

    def stop(self) -> None:
        with self.lock:
            self.pending.clear()
        if self.data is not None:
            self.data.qvel[:] = 0.0

    def goto_object(self, object_class: str, color: str) -> Dict[str, Any]:
        """Hook for Task 4 navigation; replace with the detector controller."""
        raise NotImplementedError(
            "goto_object requires the Task 4 detector/navigation controller"
        )

    def wait_until_idle(self, poll_interval: float = 0.01) -> None:
        while True:
            with self.lock:
                if not self.pending:
                    return
            self.update()
            time.sleep(poll_interval)

    def adjust_pose(self, position_tolerance: float = 0.08,
                    yaw_tolerance: float = math.radians(5.0)) -> bool:
        result = self.last_pose_result
        if not result or not result.get("tracked") or result.get("corrected"):
            return False
        result["corrected"] = True
        if result["type"] == "move" and result["position_error"] > position_tolerance:
            final = result["final"]
            target = result["target"]
            world_x = max(-0.3, min(0.3, target["x"] - final["x"]))
            world_y = max(-0.3, min(0.3, target["y"] - final["y"]))
            yaw = final["yaw"]
            local_x = world_x * math.cos(yaw) + world_y * math.sin(yaw)
            local_y = -world_x * math.sin(yaw) + world_y * math.cos(yaw)
            duration = max(0.2, min(1.0, math.hypot(world_x, world_y) / 0.3))
            self.move(local_x, local_y, 0.0, duration)
            return True
        if result["type"] == "turn" and result["yaw_error"] > yaw_tolerance:
            correction = math.degrees(_angle_difference(result["target"]["yaw"], result["final"]["yaw"]))
            self.turn(correction, target_yaw=result["target"]["yaw"])
            return True
        return False

    def update(self) -> None:
        with self.lock:
            if not self.pending:
                return
            item = self.pending[0]
            now = time.time()
            if item["type"] == "move":
                elapsed = now - item["start"]
                if elapsed >= item["duration"]:
                    self.pending.pop(0)
                    if self.data is not None:
                        self.data.qvel[:] = 0.0
                    self._complete_pose(item)
                elif self.data is None:
                    return
                else:
                    self.data.qvel[0] = item["world_vx"]
                    self.data.qvel[1] = item["world_vy"]
                    self.data.qvel[5] = item["wz"]
            elif item["type"] == "turn":
                if self.data is None:
                    if now - item["start"] >= max(0.5, abs(item["angle_deg"]) / 90.0):
                        self.pending.pop(0)
                        self._complete_pose(item)
                    return
                current = self._yaw()
                remaining = _angle_difference(item["target_yaw"], current)
                if abs(remaining) < 0.08:
                    self.pending.pop(0)
                    self._complete_pose(item)
                else:
                    direction = 1.0 if remaining > 0 else -1.0
                    self.data.qvel[5] = 0.5 * direction

    def _pose(self) -> Optional[Dict[str, float]]:
        if self.data is None:
            return None
        return {"x": float(self.data.qpos[0]), "y": float(self.data.qpos[1]), "yaw": self._yaw()}

    def _complete_pose(self, item: Dict[str, Any]) -> None:
        initial = item.get("initial_pose")
        final = self._pose()
        if initial is None or final is None:
            self.last_pose_result = {"tracked": False, "type": item["type"]}
            return
        if item["type"] == "move":
            target = item["target_pose"]
        else:
            target = {"x": initial["x"], "y": initial["y"], "yaw": item["target_yaw"]}
        self.last_pose_result = {
            "tracked": True,
            "type": item["type"],
            "initial": initial,
            "target": target,
            "final": final,
            "position_error": math.hypot(target["x"] - final["x"], target["y"] - final["y"]),
            "yaw_error": abs(_angle_difference(target["yaw"], final["yaw"])),
        }

    def _yaw(self) -> float:
        if self.data is None:
            return 0.0
        quat = np.array(self.data.qpos[3:7], dtype=np.float64)
        w, x, y, z = quat
        return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _angle_difference(target: float, current: float) -> float:
    diff = (target - current + math.pi) % (2.0 * math.pi) - math.pi
    return diff


class SimpleDetector:
    """Fallback CPU detector for object-class and color grounding.

    This does not replace the real YOLO pipeline, but it keeps the lab project
    functional on CPU without external model downloads.
    """

    def __init__(self, labels: Optional[Sequence[str]] = None):
        self.labels = list(labels or ["chair", "sports ball", "stop sign"])

    def detect(self, image_rgb: np.ndarray) -> List[Dict[str, Any]]:
        if image_rgb.size == 0:
            return []
        image_hsv = np.array(image_rgb)
        try:
            import cv2
            hsv = cv2.cvtColor(image_hsv.astype(np.uint8), cv2.COLOR_RGB2HSV)
        except Exception:
            return []

        detections: List[Dict[str, Any]] = []
        color_ranges = {
            "green": ((35, 50, 50), (90, 255, 255)),
            "red": ((0, 50, 50), (15, 255, 255)),
            "blue": ((100, 75, 50), (130, 255, 255)),
            "yellow": ((20, 50, 50), (40, 255, 255)),
        }

        for color_name, (lower, upper) in color_ranges.items():
            lower = np.array(lower, dtype=np.uint8)
            upper = np.array(upper, dtype=np.uint8)
            mask = cv2.inRange(hsv, lower, upper)
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
            for idx in range(1, num_labels):
                area = stats[idx, cv2.CC_STAT_AREA]
                if area < 200:
                    continue
                x, y, w, h, _ = stats[idx]
                bbox = [int(x), int(y), int(x + w), int(y + h)]
                for label in self.labels:
                    if label in ["chair", "sports ball", "stop sign"]:
                        obj_class = label
                        detections.append({
                            "class": obj_class,
                            "color": color_name,
                            "confidence": min(0.99, 0.55 + area / 10000.0),
                            "bbox": bbox,
                        })
                        break
        return detections


class CameraFrameSource:
    """Capture offscreen RGB frames from MuJoCo with a built-in fallback."""

    def __init__(self, model: Any):
        self.model = model
        self.renderer = None
        self._last_frame = None
        try:
            import mujoco
            self.renderer = mujoco.Renderer(model, height=320, width=480)
        except Exception:
            self.renderer = None

    def capture(self, data: Any) -> Optional[np.ndarray]:
        if self.renderer is None:
            return None
        try:
            self.renderer.update_scene(data, camera="front")
            frame = self.renderer.render()
            if frame is None:
                return None
            self._last_frame = np.asarray(frame)
            return self._last_frame.copy()
        except Exception:
            return None


def demo_command_loop() -> None:
    """Small terminal demo of the task flow used for validation and lab testing."""
    print("[CMD] MiniLab 1.3 demo started")
    examples = [
        "walk forward for three seconds, then turn back",
        "go to the green chair",
        "fly to the roof",
    ]
    for command in examples:
        result = parse_command(command)
        if result["accepted"]:
            print("[CMD] accepted actions=%s" % result["summary"])
        else:
            print("[CMD] rejected reason=%s" % result["reason"])


class TerminalChatSession:
    """Non-blocking terminal input, parsing, and ordered motion execution."""

    def __init__(self, parser: Optional[StructuredCommandParser] = None,
                 skills: Optional[MotionSkillController] = None, output=print):
        self.output = output
        self.skills = skills or MotionSkillController()
        self.parser_worker = CommandChatWorker(parser)
        self.execution_worker = CommandExecutionWorker(self.skills, output)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="minilab-terminal-chat", daemon=True)

    def start(self) -> None:
        self.output("[CHAT] MiniLab command chat started; type 'quit' to exit")
        self._thread.start()

    def join(self) -> None:
        self._thread.join()

    def stopped(self) -> bool:
        return self._stop.is_set()

    def close(self) -> None:
        self._stop.set()
        self.parser_worker.close()
        self.execution_worker.close()

    def submit_text(self, text: str) -> None:
        """Submit typed or transcribed text through the same Task 3 pipeline."""
        text = (text or "").strip()
        if not text:
            self.output("[CMD] rejected reason=empty or non-English command")
            return
        if text.lower() in {"quit", "exit"}:
            self._stop.set()
            return
        result = self.parser_worker.submit(text)
        if result.get("llm_used"):
            self.output("[LLM] provider=%s" % result.get("provider", self.parser_worker.parser.provider))
        else:
            self.output("[FALLBACK] provider=%s reason=%s" % (
                result.get("provider", self.parser_worker.parser.provider),
                result.get("fallback_reason", "local parser"),
            ))
        if not result["accepted"]:
            self.output("[CMD] rejected reason=%s" % result["reason"])
            return
        self.output("[CMD] accepted actions=%s" % result["summary"])
        execution = self.execution_worker.submit(result["actions"])
        if not execution["accepted"]:
            self.output("[CMD] execution_failed completed=%d/%d reason=%s" % (
                execution["completed"], execution["total"], execution["reason"]
            ))

    def _run(self) -> None:
        try:
            while not self._stop.is_set():
                try:
                    text = input("you> ").strip()
                except EOFError:
                    return
                self.submit_text(text)
        finally:
            self._stop.set()


class SpeechInputWorker:
    """Capture microphone speech and pass English transcripts to Task 3."""

    def __init__(self, session: TerminalChatSession, output=print, language: str = "en-US"):
        self.session = session
        self.output = output
        self.language = language
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="minilab-speech-input", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2)

    def _run(self) -> None:
        try:
            import speech_recognition as sr
        except ImportError:
            self.output("[STT] unavailable: install with python -m pip install -e .[speech]")
            return
        recognizer = sr.Recognizer()
        try:
            microphone_names = sr.Microphone.list_microphone_names()
            preferred_index = next((index for index, name in enumerate(microphone_names)
                                    if "microphone array" in name.lower()
                                    and "steam" not in name.lower()), None)
            microphone = sr.Microphone(device_index=preferred_index)
            selected_name = (microphone_names[preferred_index]
                             if preferred_index is not None else "Windows default input")
            with microphone:
                self.output("[STT] microphone=%s" % selected_name)
                self.output("[STT] microphone ready; speak an English command")
                recognizer.adjust_for_ambient_noise(microphone, duration=1)
                while not self._stop.is_set() and not self.session.stopped():
                    try:
                        audio = recognizer.listen(microphone, timeout=1, phrase_time_limit=8)
                    except sr.WaitTimeoutError:
                        continue
                    try:
                        transcript = recognizer.recognize_google(audio, language=self.language).strip()
                    except sr.UnknownValueError:
                        self.output("[STT] could not understand speech")
                        continue
                    except sr.RequestError as exc:
                        self.output("[STT] service error: %s" % exc)
                        continue
                    if transcript:
                        self.output("[STT] heard=%s" % transcript)
                        self.session.submit_text(transcript)
        except (OSError, AttributeError) as exc:
            self.output("[STT] microphone unavailable: %s" % exc)


def interactive_chat_loop(parser: Optional[StructuredCommandParser] = None) -> None:
    session = TerminalChatSession(parser)
    session.start()
    try:
        session.join()
    finally:
        session.close()


def print_evaluation(parser: StructuredCommandParser) -> None:
    evaluation = evaluate_parser(parser)
    print("[EVAL] provider=%s cases=%d correct=%d accuracy=%.1f%% llm_responses=%d fallback_responses=%d" % (
        parser.provider, evaluation["total"], evaluation["correct"], evaluation["accuracy"] * 100,
        evaluation["llm_used"], evaluation["total"] - evaluation["llm_used"]
    ))
    fallback_reasons = sorted({row.get("fallback_reason", "") for row in evaluation["rows"] if row.get("fallback_reason")})
    for reason in fallback_reasons[:3]:
        print("[EVAL] fallback_reason=%s" % reason)
    for index, row in enumerate(evaluation["rows"], start=1):
        status = "PASS" if row["passed"] else "FAIL"
        print("[EVAL] %02d %s expected=%s actual=%s input=%r" % (
            index, status, row["expected"], row["actual"], row["input"]
        ))


def compare_providers(provider_names: Sequence[str]) -> None:
    """Run the identical evaluation set against two configured providers."""
    for provider in provider_names:
        print_evaluation(StructuredCommandParser(provider))


if __name__ == "__main__":
    import argparse

    argument_parser = argparse.ArgumentParser(description="MiniLab 1.3 Task 3 command parser")
    argument_parser.add_argument("--chat", action="store_true", help="start the multi-turn terminal chat")
    argument_parser.add_argument("--evaluate", action="store_true", help="run the 20-case parser evaluation")
    argument_parser.add_argument("--compare", nargs=2, metavar=("PROVIDER_A", "PROVIDER_B"),
                                 help="compare two configured providers on the same 20 cases")
    argument_parser.add_argument("--provider", default="local", choices=["local", "openai", "openai-compatible", "anthropic", "gemini"])
    argument_parser.add_argument("--endpoint", help="structured-output API endpoint")
    argument_parser.add_argument("--model", help="provider model name")
    args = argument_parser.parse_args()

    selected_parser = StructuredCommandParser(args.provider, args.endpoint, model=args.model)
    if args.compare:
        compare_providers(args.compare)
    elif args.chat:
        interactive_chat_loop(selected_parser)
    elif args.evaluate:
        print_evaluation(selected_parser)
    else:
        demo_command_loop()
        build_scene_xml()
        print("[INFO] generated demo scene config and XML files")
        print("[INFO] run: python eg/minilab_1_3.py --chat")
        print("[INFO] run: python eg/minilab_1_3.py --evaluate")
