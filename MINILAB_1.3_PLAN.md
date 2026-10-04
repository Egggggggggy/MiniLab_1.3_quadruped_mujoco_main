# EE4705 MiniLab 1.3 Plan

This document summarizes the actual assignment in [EE4705_MiniLab1.3 S1 AY2627.pdf](EE4705_MiniLab1.3%20S1%20AY2627.pdf) and maps it to the current workspace.

## 1. Assignment summary

MiniLab 1.3 is a 15% group assignment titled:

"Command Your Robot Dog: An LLM + YOLO Powered Quadruped"

The objective is to build a simulated quadruped robot that can:

- accept typed English motion commands,
- parse them with an LLM into structured commands,
- execute the actions autonomously using a local locomotion policy,
- detect objects with YOLO from the onboard camera,
- navigate to a target object using color grounding and closed-loop control.

The workflow is a slow cloud LLM planner on top of a fast local control loop.

## 2. Required task split from the PDF

The assignment explicitly states the suggested group split:

- Task 1: ALL
- Task 2: Student A
- Task 3: Student B
- Task 4: Student C
- Task 5: ALL

For a 2-member group:

- Student A leads Task 2
- Student B leads Task 3
- both share Task 4

This is the correct split to use in the report and code ownership.

## 3. What the PDF requires per task

### Task 1: LLM + robot control comparison

- Compare at least three approaches for connecting language models to robots.
- Include structured-output parsing (JSON / function calling) and at least two additional approaches such as SayCan, Code as Policies, ROSGPT / NASA JPL ROSA, RT-2, or VLM-based grounding.
- Explain the two-rate architecture: slow LLM planner, fast local control loop.
- Briefly explain how the quadruped walking policy is obtained through reinforcement learning and sim-to-sim / sim-to-real.
- Report length: about one page + references.
- No software implementation required.

### Task 2: Platform setup, scene and motion skills

- Use the example repository or another platform.
- The recommended default is the repo in this workspace.
- Run the demo in both display modes:
  - native MuJoCo viewer
  - browser panel with --gui
- Drive the robot with the keyboard.
- Switch between at least 3 bundled terrain maps.
- View the 3 onboard cameras.
- Build a custom MJCF scene with at least 3 objects from at least 2 COCO classes, including two objects of the same class in different colors.
- Implement a motion-skills API:
  - move(vx, vy, wz, duration)
  - turn(angle_deg) using actual yaw feedback from simulation
- Provide a demo video for Task 2.

### Task 3: Typed English motion commands via LLM parser

- Use an LLM parser with structured output (JSON schema / function call / schema-constrained decoding).
- Accept free-form English utterances and convert them into a list of commands.
- Support multi-step instructions.
- Reject unsafe, empty, impossible, or non-English commands.
- Maintain a multi-turn terminal chat loop with follow-up history.
- Run the chat loop in a separate thread so the simulation does not block.
- Evaluate on at least 20 utterances using at least two LLM services.
- Video_Task3 must show the terminal and use a command that includes multiple actions and a rejection.

### Task 4: YOLO object search and approach

- Run YOLO on the camera frames.
- Determine object color from HSV / bounding-box pixels.
- Implement goto_object(class, color).
- If the target is not visible, rotate and re-detect.
- Once visible, steer toward the center, walk forward, and stop when all conditions in the PDF are satisfied:
  - detection has the correct class and color,
  - proximity within 0.80 m of the object center in the x-y plane,
  - terminal logging includes [FOUND] class=… color=… t=… d=…
- Must use the same platform, scene and skills from Task 2.
- Evaluate on at least 10 trials with different objects and starting poses.
- Video_Task4 must show typed command, search, detect, found, mission status.

### Task 5: Video and report submission

- Submit one zip file for the group.
- Include report PDF plus source code and install/run instructions.
- Include all scene/object files and LLM prompt files.
- Include three videos:
  - Video_Task2
  - Video_Task3
  - Video_Task4
- Include AI Usage Declaration and contribution split.
- Include platform, Python version, dependencies, and API-key handling procedure.

## 4. Current workspace status

The current workspace already contains the reusable MuJoCo runtime-control example and test suite:

- [README.md](README.md)
- [eg/play.py](eg/play.py)
- [src/runtime_control](src/runtime_control)
- [tests](tests)

The repo already supports the platform setup and motion-control baseline required for Task 2.

## 5. What is still needed to finish MiniLab 1.3

The following parts are not yet present in this workspace and must be built for the actual lab submission:

1. A custom MJCF scene with at least 3 objects from different classes, with color variants.
2. A YOLO-based object detection pipeline.
3. A structured LLM parsing layer using JSON / function calling.
4. A multi-turn command chat loop.
5. A controller that connects detection and navigation to the Task 2 motion skills.
6. A final report and video package per the assignment.

## 6. Recommended next implementation order

1. Validate the example platform in [eg/play.py](eg/play.py) as the Task 2 baseline.
2. Add a custom scene file and object placements for the object-search task.
3. Add a camera frame capture interface that yields RGB frames at a fixed rate.
4. Implement a YOLO detection wrapper with color grounding logic.
5. Implement the LLM parser and terminal chat loop.
6. Connect Task 3 parser output to the Task 2 motion skills.
7. Implement the object-search controller and logging.
8. Create evaluation tables and the final group report.

## 7. Important note

The PDF explicitly requires the custom scene and object setup, so the built-in bundled maps are not sufficient for the final submission. The repo is useful as the platform base, but the final lab still needs additional custom scene and perception logic beyond the generic example.
